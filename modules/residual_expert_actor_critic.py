import torch
import torch.nn as nn
from torch.distributions import Normal


class ResidualExpertActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        base_actor_critic,
        residual_actor_critic,
        alpha,
        freeze_base=True,
        residual_std_scale=None,
        min_policy_std=0.02,
        max_policy_std=1.0,
    ):
        super().__init__()
        self.base_actor_critic = base_actor_critic
        self.residual_actor_critic = residual_actor_critic
        self.alpha = alpha
        self.freeze_base = freeze_base
        self.residual_std_scale = 1.0 if residual_std_scale is None else residual_std_scale
        self.min_policy_std = min_policy_std
        self.max_policy_std = max_policy_std
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

    def get_residual_std(self):
        if hasattr(self.residual_actor_critic, "get_std"):
            return self.residual_actor_critic.get_std()
        return self.residual_actor_critic.std

    def get_effective_std(self):
        residual_std = self.get_residual_std()
        final_std = self.residual_std_scale * residual_std
        return torch.clamp(final_std, min=self.min_policy_std, max=self.max_policy_std)

    def get_std(self):
        return self.get_effective_std()

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

        # 关键：residual std 使用独立缩放系数，并限制最终执行范围
        residual_std = self.get_residual_std()
        final_std = self.get_effective_std()

        self.distribution = Normal(final_mean, final_mean * 0.0 + final_std)

        # 用于日志
        self.last_base_mean = base_mean.detach()
        self.last_residual_mean = residual_mean.detach()
        self.last_delta = delta.detach()
        self.last_final_mean = final_mean.detach()
        self.last_residual_std = residual_std.detach()
        self.last_final_std = final_std.detach()
        self.last_saturation_ratio = (final_mean.abs() > 0.95).float().mean().detach()

        # 如果后面要加 residual 正则，这个不能 detach
        self.current_delta = delta

    def clamp_action_std(self, min_std=0.02, max_std=1.2):
        if hasattr(self.residual_actor_critic, "clamp_action_std"):
            self.residual_actor_critic.clamp_action_std(self.min_policy_std, self.max_policy_std)

    def set_residual_std(self, value):
        if hasattr(self.residual_actor_critic, "std"):
            with torch.no_grad():
                clamped_value = max(self.min_policy_std, min(value, self.max_policy_std))
                self.residual_actor_critic.std.data.fill_(clamped_value)

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
