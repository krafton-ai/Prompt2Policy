"""PPO subclass that logs policy distribution diagnostics.

Wraps ``train()`` to snapshot distribution parameters (mean, log_std)
before and after the policy update, then computes:

- **kl_mean_term**: KL contribution from policy mean movement
- **kl_var_term**: KL contribution from variance (log_std) change
- **mean_shift_normalized**: mean shift relative to exploration scale
"""

from __future__ import annotations

import torch as th
from stable_baselines3 import PPO


class PPODiagnostic(PPO):
    """PPO with per-update distribution diagnostic metrics."""

    def train(self) -> None:
        policy = self.policy
        has_log_std = hasattr(policy, "log_std")

        # --- Snapshot BEFORE update ---
        if has_log_std:
            old_log_std = policy.log_std.data.clone()

            # Sample observations from the rollout buffer.
            # obs_tensor is reused after super().train() to compute new_mean
            # on the same inputs. The buffer observations (numpy) are not
            # mutated by train(), so the tensor remains valid.
            buf = self.rollout_buffer
            n_samples = min(512, buf.buffer_size * buf.n_envs)
            obs_all = buf.observations.reshape(-1, *buf.obs_shape)[:n_samples]
            obs_tensor = th.as_tensor(obs_all, device=policy.device)

            with th.no_grad():
                features = policy.extract_features(obs_tensor, policy.features_extractor)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                old_mean = policy.action_net(latent_pi)

        # --- Run the actual PPO update ---
        super().train()

        # --- Compute diagnostics AFTER update ---
        if not has_log_std:
            return

        with th.no_grad():
            new_log_std = policy.log_std.data

            features = policy.extract_features(obs_tensor, policy.features_extractor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            new_mean = policy.action_net(latent_pi)

            old_std = th.exp(old_log_std)
            new_std = th.exp(new_log_std)
            eps = 1e-8

            # KL mean term: 0.5 * Σ (μ_old - μ_new)² / σ_new²
            kl_mean = 0.5 * ((old_mean - new_mean) ** 2 / (new_std**2 + eps)).sum(dim=-1).mean()

            # KL variance term: Σ [log(σ_new/σ_old) + σ_old²/(2σ_new²) - 0.5]
            # eps added to both numerator and denominator to avoid log(0)
            # while preserving the ratio for normal-range std values.
            log_ratio = th.log((new_std + eps) / (old_std + eps))
            var_ratio = old_std**2 / (2 * new_std**2 + eps)
            kl_var = (log_ratio + var_ratio - 0.5).sum(dim=-1).mean()

            # Mean shift normalized: ‖(μ_new - μ_old) / σ_old‖₂
            mean_shift_norm = (
                ((new_mean - old_mean) / (old_std + eps)).pow(2).sum(dim=-1).sqrt().mean()
            )

        self.logger.record("train/kl_mean_term", kl_mean.item())
        self.logger.record("train/kl_var_term", kl_var.item())
        self.logger.record("train/mean_shift_normalized", mean_shift_norm.item())
