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
        residual_delta_clip=None,
        alpha_warmup_steps=0,
        alpha_warmup_min=0.25,
        zero_init_residual=True,
    ):
        super().__init__()
        self.base_actor_critic = base_actor_critic
        self.residual_actor_critic = residual_actor_critic
        self.target_alpha = alpha
        self.alpha = alpha
        self.current_alpha = alpha
        self.freeze_base = freeze_base
        self.residual_std_scale = 1.0 if residual_std_scale is None else residual_std_scale
        self.min_policy_std = min_policy_std
        self.max_policy_std = max_policy_std
        self.residual_delta_clip = residual_delta_clip
        self.alpha_warmup_steps = max(0, int(alpha_warmup_steps))
        self.alpha_warmup_min = max(0.0, min(float(alpha_warmup_min), 1.0))
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

        if zero_init_residual:
            self._zero_init_residual_output()

    def _zero_init_residual_output(self):
        """Start from the frozen base policy instead of a random residual mean."""
        target_modules = [
            getattr(self.residual_actor_critic, "actor_teacher_backbone", None),
            getattr(self.residual_actor_critic, "actor_student_backbone", None),
            getattr(self.residual_actor_critic, "actor", None),
        ]
        for module in target_modules:
            if module is None:
                continue
            linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
            if not linears:
                continue
            last_linear = linears[-1]
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def set_learning_iteration(self, iteration):
        if self.alpha_warmup_steps <= 0:
            self.current_alpha = self.target_alpha
            self.alpha = self.current_alpha
            return
        progress = min(max(float(iteration) / float(self.alpha_warmup_steps), 0.0), 1.0)
        alpha_scale = self.alpha_warmup_min + (1.0 - self.alpha_warmup_min) * progress
        self.current_alpha = self.target_alpha * alpha_scale
        self.alpha = self.current_alpha

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
            for p in self.base_actor_critic.parameters():
                p.requires_grad = False
        else:
            self.base_actor_critic.train(mode)
        return self

    def update_distribution(self, obs):
        with torch.no_grad():
            base_mean = self.base_actor_critic.act_inference(obs)
        residual_mean = self.residual_actor_critic.act_inference(obs)
        delta = self.current_alpha * residual_mean
        if self.residual_delta_clip is not None and self.residual_delta_clip > 0:
            delta = torch.clamp(delta, -self.residual_delta_clip, self.residual_delta_clip)
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
        self.last_delta_norm = torch.norm(delta, dim=-1).mean().detach()
        self.last_current_alpha = torch.as_tensor(self.current_alpha, device=obs.device)

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
