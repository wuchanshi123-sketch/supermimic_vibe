"""Differentiable adversarial motion prior for scratch locomotion training.

Feature layout and least-squares style reward follow the local supermimic_AMP
reference. This module has no teacher, checkpoint, or external RSL-RL dependency.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .networks import Normalizer


AMP_BODY_NAMES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
AMP_ANCHOR_NAME = "torso_link"


class AMPFeatureExtractor:
    """195 body features, invariant to global translation and heading.

    Body poses are relative to the torso. Velocities are expressed in each
    body's own frame and are not reduced by the torso velocity.
    """

    feature_dim = 195

    def __init__(self, task):
        self.task = task
        names = list(task.kin.body_names)
        requested = AMP_BODY_NAMES + (AMP_ANCHOR_NAME,)
        missing = [name for name in requested if name not in names]
        if missing:
            raise ValueError(f"Robot is missing AMP bodies: {missing}")
        if len(set(names)) != len(names):
            raise ValueError("AMP body names must be unique")
        self.bodies = [names.index(name) for name in AMP_BODY_NAMES]
        self.anchor = names.index(AMP_ANCHOR_NAME)

    def __call__(self, state: dict, kin=None):
        positions, rotations, velocities, angular = self.task.body_kin(state) if kin is None else kin
        body_rotations = rotations[..., self.bodies, :, :]
        anchor_t = rotations[..., self.anchor, :, :].transpose(-1, -2).unsqueeze(-3)
        relative = (
            anchor_t
            @ (positions[..., self.bodies, :] - positions[..., self.anchor:self.anchor + 1, :]).unsqueeze(-1)
        ).squeeze(-1)
        orientation = (anchor_t @ body_rotations)[..., :, :2]
        body_t = body_rotations.transpose(-1, -2)
        local_velocity = (body_t @ velocities[..., self.bodies, :].unsqueeze(-1)).squeeze(-1)
        local_angular = (body_t @ angular[..., self.bodies, :].unsqueeze(-1)).squeeze(-1)
        return torch.cat(
            (
                relative.flatten(-2),
                orientation.flatten(-3),
                local_velocity.flatten(-2),
                local_angular.flatten(-2),
            ),
            dim=-1,
        )


class ExpertFeaturePairs:
    """Uniform clip sampling followed by an adjacent-frame pair inside that clip."""

    def __init__(self, features, starts, lengths):
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Expert features must be a (frames, features) tensor")
        self.features = features.detach()
        self.starts = torch.as_tensor(starts, device=features.device, dtype=torch.long)
        self.lengths = torch.as_tensor(lengths, device=features.device, dtype=torch.long)
        if (
            self.starts.ndim != 1
            or self.lengths.ndim != 1
            or len(self.starts) == 0
            or self.starts.shape != self.lengths.shape
        ):
            raise ValueError("Expert starts/lengths must be nonempty equal-length vectors")
        if bool((self.starts < 0).any()) or bool((self.lengths < 2).any()):
            raise ValueError("Each expert clip needs a nonnegative start and at least two frames")
        if bool((self.starts + self.lengths > len(features)).any()):
            raise ValueError("Expert clip extends beyond the feature tensor")
        if not bool(torch.isfinite(self.features).all()):
            raise ValueError("Expert features contain non-finite values")
        self.num_clips = len(self.starts)

    def sample_indices(self, batch):
        if batch < 1:
            raise ValueError("Expert batch size must be positive")
        clips = torch.randint(self.num_clips, (batch,), device=self.features.device)
        offsets = (torch.rand(batch, device=self.features.device) * (self.lengths[clips] - 1)).long()
        return self.starts[clips] + offsets

    def sample(self, batch):
        indices = self.sample_indices(batch)
        return self.features[indices], self.features[indices + 1]


class AMPPrior(nn.Module):
    """An independently trained discriminator with a differentiable style reward.

    update() changes parameters and statistics using detached real/expert samples.
    reward() changes neither; callers may freeze this module's parameters while
    retaining gradients through its inputs during the actor's imagined rollout.
    """

    def __init__(
        self,
        feature_dim=195,
        hidden=(1024, 512, 256),
        lr=1e-4,
        gp_weight=10.0,
        style_scale=0.1,
        device="cuda",
    ):
        super().__init__()
        if feature_dim < 1 or any(width < 1 for width in hidden):
            raise ValueError("Discriminator dimensions must be positive")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("Discriminator learning rate must be finite and positive")
        if not math.isfinite(gp_weight) or gp_weight < 0:
            raise ValueError("Gradient-penalty weight must be finite and nonnegative")
        if not math.isfinite(style_scale) or style_scale <= 0:
            raise ValueError("Style scale must be finite and positive")
        self.feature_dim = feature_dim
        self.gp_weight = gp_weight
        self.style_scale = style_scale
        layers = []
        width = 2 * feature_dim
        for next_width in hidden:
            layers.extend((nn.Linear(width, next_width), nn.ReLU()))
            width = next_width
        layers.append(nn.Linear(width, 1))
        self.discriminator = nn.Sequential(*layers)
        self.normalizer = Normalizer(feature_dim)
        self.to(device)
        trunk = [p for layer in self.discriminator[:-1] for p in layer.parameters()]
        self.optimizer = torch.optim.Adam(
            [
                {"params": trunk, "weight_decay": 1e-3},
                {"params": self.discriminator[-1].parameters(), "weight_decay": 0.1},
            ],
            lr=lr,
        )

    def reward(self, f0, f1):
        """Return per-control-step style reward and score, preserving input gradients."""
        with torch.autocast(device_type=f0.device.type, enabled=False):
            pair = torch.cat((self.normalizer(f0.float()), self.normalizer(f1.float())), dim=-1)
            score = self.discriminator(pair).squeeze(-1)
            style = self.style_scale * (1.0 - 0.25 * (score - 1.0).square()).clamp_min(0.0)
        return style, score

    def update(self, policy_f0, policy_f1, expert_f0, expert_f1):
        raw = tuple(x.detach().float() for x in (policy_f0, policy_f1, expert_f0, expert_f1))
        if any(x.ndim != 2 or x.shape[-1] != self.feature_dim or x.shape[0] == 0 for x in raw):
            raise ValueError(f"AMP training samples must have shape (batch, {self.feature_dim})")
        if raw[0].shape != raw[1].shape or raw[2].shape != raw[3].shape:
            raise ValueError("Each AMP transition needs equally shaped endpoints")
        all_raw = torch.cat(raw)
        if not bool(torch.isfinite(all_raw).all()):
            raise FloatingPointError("Non-finite AMP training sample")
        # Fit to raw endpoints, never to already normalized discriminator inputs.
        self.normalizer.update(all_raw)
        self.discriminator.requires_grad_(True)
        self.discriminator.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.enable_grad(), torch.autocast(device_type=all_raw.device.type, enabled=False):
            p0, p1, e0, e1 = (self.normalizer(x).detach() for x in raw)
            policy_score = self.discriminator(torch.cat((p0, p1), -1)).squeeze(-1)
            expert_pair = torch.cat((e0, e1), -1).requires_grad_(True)
            expert_score = self.discriminator(expert_pair).squeeze(-1)
            classification = 0.5 * (
                (expert_score - 1.0).square().mean() + (policy_score + 1.0).square().mean()
            )
            gradient = torch.autograd.grad(
                expert_score.sum(), expert_pair, create_graph=True, retain_graph=True
            )[0]
            penalty = self.gp_weight * gradient.square().sum(-1).mean()
            loss = classification + penalty
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 1.0, error_if_nonfinite=True
            )
            self.optimizer.step()
        names = (
            "discriminator_loss", "gradient_penalty", "policy_score", "expert_score",
            "discriminator_total_loss", "discriminator_grad_norm",
        )
        values = torch.stack(
            (classification.detach(), penalty.detach(), policy_score.detach().mean(),
             expert_score.detach().mean(), loss.detach(), grad_norm.detach())
        ).cpu().tolist()
        self.optimizer.zero_grad(set_to_none=True)
        return dict(zip(names, values))
