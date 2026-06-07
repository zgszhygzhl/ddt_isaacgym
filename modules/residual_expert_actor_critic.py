import torch
import torch.nn as nn
from torch.distributions import Normal


class ResidualExpertActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, base_actor_critic, residual_actor_critic, alpha, freeze_base=True):
        super().__init__()
        self.base_actor_critic = base_actor_critic
        self.residual_actor_critic = residual_actor_critic
        self.alpha = alpha
        self.freeze_base = freeze_base
        # self.imi_flag = getattr(self.residual_actor_critic, "imi_flag", False)
        self.imi_flag = False
        self.distribution = None

        self.last_base_mean = None
        self.last_residual_mean = None
        self.last_final_mean = None

        if self.freeze_base:
            self.base_actor_critic.eval()
            for parameter in self.base_actor_critic.parameters():
                parameter.requires_grad = False

    def get_std(self):
        if hasattr(self.residual_actor_critic, "get_std"):
            return self.residual_actor_critic.get_std()
        return self.residual_actor_critic.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        self.base_actor_critic.reset(dones)
        self.residual_actor_critic.reset(dones)

    def train(self, mode=True):
        super().train(mode)
        self.residual_actor_critic.train(mode)
        if self.freeze_base:
            self.base_actor_critic.eval()
        else:
            self.base_actor_critic.train(mode)
        return self

    def update_distribution(self, obs):
        with torch.no_grad():
            base_mean = self.base_actor_critic.act_inference(obs)
        residual_mean = self.residual_actor_critic.act_inference(obs)
        # residual mean
        delta = self.alpha * residual_mean
        final_mean = base_mean + delta

        # 关键：residual std 也要按 alpha 缩放，并限制范围
        residual_std = self.get_std()
        final_std = self.alpha * residual_std
        final_std = torch.clamp(final_std, min=0.02, max=0.55)

        self.distribution = Normal(final_mean, final_mean * 0.0 + final_std)

        # 用于日志
        self.last_base_mean = base_mean.detach()
        self.last_residual_mean = residual_mean.detach()
        self.last_delta = delta.detach()
        self.last_final_mean = final_mean.detach()
        self.last_final_std = final_std.detach()
        self.last_saturation_ratio = (final_mean.abs() > 0.95).float().mean().detach()

        # 如果后面要加 residual 正则，这个不能 detach
        self.current_delta = delta

    def clamp_action_std(self, min_std=0.02, max_std=0.5):
        if hasattr(self.residual_actor_critic, "clamp_action_std"):
            self.residual_actor_critic.clamp_action_std(min_std, max_std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        self.update_distribution(obs)
        return self.action_mean

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, obs, **kwargs):
        return self.residual_actor_critic.evaluate(obs, **kwargs)

    def evaluate_cost(self, obs, **kwargs):
        return self.residual_actor_critic.evaluate_cost(obs, **kwargs)
