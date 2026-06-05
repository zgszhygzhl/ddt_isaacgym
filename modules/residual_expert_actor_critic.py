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
        self.imi_flag = getattr(self.residual_actor_critic, "imi_flag", False)
        self.distribution = None

        self.last_base_mean = None
        self.last_residual_mean = None
        self.last_final_mean = None

        if freeze_base:
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

    def update_distribution(self, obs):
        with torch.no_grad():
            base_mean = self.base_actor_critic.act_inference(obs)
        residual_mean = self.residual_actor_critic.act_inference(obs)
        final_mean = torch.clamp(base_mean + self.alpha * residual_mean, -1.0, 1.0)

        self.last_base_mean = base_mean.detach()
        self.last_residual_mean = residual_mean.detach()
        self.last_final_mean = final_mean.detach()

        residual_std = self.get_std()
        self.distribution = Normal(final_mean, final_mean * 0.0 + residual_std)

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
