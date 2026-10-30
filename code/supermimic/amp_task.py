"""Differentiable command-conditioned locomotion heads for from-scratch AMP.

These heads contain no motion-tracking target, teacher, or discriminator. The
same functions label real simulator transitions and imagined transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .rotations import heading_mat, mtv


COMMAND_LO = (-0.5, -0.5, -1.0)
COMMAND_HI = (1.0, 0.5, 1.0)


@dataclass
class AMPTaskCfg:
    dt: float = 0.02
    history: int = 4
    height_threshold: float = 0.5
    tilt_degrees: float = 70.0
    linear_std: float = 1.0
    angular_std: float = 3.14
    upright_std: float = 0.5
    w_linear: float = 1.0
    w_yaw: float = 1.0
    w_action_rate: float = -0.01
    w_joint_limit: float = -10.0
    w_upright: float = 0.5
    w_foot_slip: float = 0.0
    w_yaw_rate: float = 0.0
    w_foot_landing: float = 0.0
    w_foot_penetration: float = 0.0
    refinement_speed_start: float = 0.5
    refinement_speed_full: float = 0.8
    landing_height: float = 0.02
    landing_width: float = 0.01
    penetration_scale: float = 0.005
    penetration_slop: float = 0.0005
    foot_slip_height_threshold: float = 0.045
    foot_slip_width: float = 0.005
    soft_height_width: float = 0.02
    soft_tilt_width: float = 0.03
    noise_gyro: float = 0.2
    noise_gravity: float = 0.05
    noise_q: float = 0.01
    noise_qd: float = 0.5

    @property
    def max_step_reward(self) -> float:
        """Upper bound on the task part only; AMP style is added separately."""
        return self.dt * (self.w_linear + self.w_yaw + self.w_upright)


def _broadcast_vector(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast per-environment vectors over optional observation-history axes."""
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-2)
    return value.expand(reference.shape[:-1] + (value.shape[-1],))


class AMPTask:
    """Commanded walking task sharing the simulator's G1 kinematic model.

    State layout is the existing dictionary (pos, rot, q, v, w, qd). Observation
    order is gyro(3), gravity(3), command(3), q-q0+bias(29), qd(29), previous
    clean action(29), for 96 dimensions per frame. No reference state is used.
    """

    obs_dim = 96

    def __init__(self, base_task, cfg: AMPTaskCfg | None = None, mj_model=None):
        self.base = base_task
        self.cfg = cfg if cfg is not None else AMPTaskCfg()
        c = self.cfg
        if c.dt <= 0 or c.history < 1:
            raise ValueError("dt must be positive and history must be at least one")
        if not 0 < c.tilt_degrees < 180:
            raise ValueError("tilt_degrees must be between 0 and 180")
        if min(c.linear_std, c.angular_std, c.upright_std,
               c.soft_height_width, c.soft_tilt_width) <= 0:
            raise ValueError("reward scales and soft-termination widths must be positive")
        if min(c.w_linear, c.w_yaw, c.w_upright) < 0:
            raise ValueError("tracking/upright reward weights must be nonnegative")
        if c.w_action_rate > 0 or c.w_joint_limit > 0:
            raise ValueError("action-rate and joint-limit weights must be nonpositive")
        if not math.isfinite(c.w_foot_slip) or c.w_foot_slip > 0:
            raise ValueError("foot-slip weight must be finite and nonpositive")
        if (not math.isfinite(c.foot_slip_height_threshold)
                or c.foot_slip_height_threshold < 0
                or not math.isfinite(c.foot_slip_width) or c.foot_slip_width <= 0):
            raise ValueError("foot-slip height must be nonnegative and width must be positive and finite")
        if min(c.noise_gyro, c.noise_gravity, c.noise_q, c.noise_qd) < 0:
            raise ValueError("noise half-widths must be nonnegative")
        for name in ("w_yaw_rate", "w_foot_landing", "w_foot_penetration"):
            if not math.isfinite(getattr(c, name)) or getattr(c, name) > 0:
                raise ValueError(f"{name} must be finite and nonpositive")
        if (not math.isfinite(c.refinement_speed_start) or c.refinement_speed_start < 0
                or not math.isfinite(c.refinement_speed_full)
                or c.refinement_speed_full <= c.refinement_speed_start):
            raise ValueError("Refinement speed gate requires 0 <= start < full")
        for name in ("landing_width", "penetration_scale"):
            if not math.isfinite(getattr(c, name)) or getattr(c, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("landing_height", "penetration_slop"):
            if not math.isfinite(getattr(c, name)) or getattr(c, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.contact_costs = None
        if c.w_foot_landing != 0 or c.w_foot_penetration != 0:
            if mj_model is None:
                raise ValueError("Foot contact refinement requires the actual MuJoCo model")
            from .amp_contact_reward import FootContactCosts
            self.contact_costs = FootContactCosts.from_model(
                mj_model, self.base.kin.body_names, self.base.q0.device)
        self.torso_index = self.base.kin.body_names.index("torso_link")
        self.foot_indices = (
            [self.base.kin.body_names.index(name) for name in
             ("left_ankle_roll_link", "right_ankle_roll_link")]
            if c.w_foot_slip != 0 else []
        )
        q0 = self.base.q0
        self.obs_noise = torch.cat((
            q0.new_full((3,), c.noise_gyro),
            q0.new_full((3,), c.noise_gravity),
            q0.new_zeros(3),
            q0.new_full((29,), c.noise_q),
            q0.new_full((29,), c.noise_qd),
            q0.new_zeros(29),
        ))

    @property
    def max_step_reward(self) -> float:
        return self.cfg.max_step_reward

    def observation(self, s, previous_clean_action, command, xi):
        """Noise-free observations for (B,...) or (B,H,...) state tensors.

        xi uses the existing [friction, encoder bias(29), COM offset(3)] layout.
        Bias has the same positive sign as Task.obs_clean.
        """
        q = s["q"]
        previous = _broadcast_vector(previous_clean_action, q)
        cmd = _broadcast_vector(command, q)
        bias = _broadcast_vector(xi[..., 1:30], q)
        gyro = mtv(s["rot"], s["w"])
        gravity = -s["rot"][..., 2, :]
        return torch.cat((
            gyro, gravity, cmd, q - self.base.q0 + bias, s["qd"], previous,
        ), dim=-1)

    def sample_noise(self, shape) -> torch.Tensor:
        """Per-frame uniform sensor noise; commands and prior actions are clean."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        sample = torch.rand(shape + (self.obs_dim,), device=self.obs_noise.device,
                            dtype=self.obs_noise.dtype)
        return (2 * sample - 1) * self.obs_noise

    def reward(self, s, action, previous, command, kin=None):
        """Weighted, dt-scaled task reward and its individual components.

        Torso linear velocity is measured in the torso heading frame. Angular
        tracking uses world yaw rate together with torso-local roll/pitch rates,
        matching the reference project's VelocityHeads.
        """
        c = self.cfg
        if kin is None:
            kin = self.base.body_kin(s)
        _, rotations, velocities, angular = kin
        torso_rot = rotations[..., self.torso_index, :, :]
        torso_v = velocities[..., self.torso_index, :]
        torso_w = angular[..., self.torso_index, :]
        cmd = _broadcast_vector(command, s["q"])
        a = _broadcast_vector(action, s["q"])
        ap = _broadcast_vector(previous, s["q"])
        v_local = mtv(heading_mat(torso_rot), torso_v)
        target = torch.cat((cmd[..., :2], torch.zeros_like(cmd[..., :1])), -1)
        linear_error = (v_local - target).square().sum(-1)
        w_local = mtv(torso_rot, torso_w)
        angular_error = ((torso_w[..., 2] - cmd[..., 2]).square()
                         + w_local[..., :2].square().sum(-1))
        gravity_xy_error = s["rot"][..., 2, :2].square().sum(-1)
        action_rate = (a - ap).square().sum(-1)
        joint_limit = (
            (self.base.qmin - s["q"]).clamp_min(0)
            + (s["q"] - self.base.qmax).clamp_min(0)
        ).sum(-1)
        terms = {
            "linear": c.dt * c.w_linear * torch.exp(-linear_error / c.linear_std**2),
            "yaw": c.dt * c.w_yaw * torch.exp(-angular_error / c.angular_std**2),
            "upright": c.dt * c.w_upright * torch.exp(-gravity_xy_error / c.upright_std**2),
            "action_rate": c.dt * c.w_action_rate * action_rate,
            "joint_limit": c.dt * c.w_joint_limit * joint_limit,
        }
        if c.w_yaw_rate != 0 or self.contact_costs is not None:
            # Additional refinement applies only to higher commanded speeds.
            # These are analytic state/geometry costs, not predicted contact forces.
            gate = ((cmd[..., :2].norm(dim=-1) - c.refinement_speed_start)
                    / (c.refinement_speed_full - c.refinement_speed_start)).clamp(0, 1)
            if c.w_yaw_rate != 0:
                terms["yaw_rate_refinement"] = (c.dt * c.w_yaw_rate * gate
                                                * (torso_w[..., 2] - cmd[..., 2]).square())
            if self.contact_costs is not None:
                costs = self.contact_costs.costs(
                    kin, landing_height=c.landing_height, landing_width=c.landing_width,
                    penetration_scale=c.penetration_scale, penetration_slop=c.penetration_slop)
                if c.w_foot_landing != 0:
                    terms["foot_landing"] = c.dt * c.w_foot_landing * gate * costs["landing"]
                if c.w_foot_penetration != 0:
                    terms["foot_penetration"] = c.dt * c.w_foot_penetration * gate * costs["penetration"]
        if c.w_foot_slip != 0:
            # This soft height proxy is differentiable and uses the same FK
            # already supplied for the task reward; it is not a contact sensor.
            height = kin[0][..., self.foot_indices, 2]
            speed_squared = velocities[..., self.foot_indices, :2].square().sum(-1)
            low = torch.sigmoid((c.foot_slip_height_threshold - height) / c.foot_slip_width)
            terms["foot_slip"] = c.dt * c.w_foot_slip * (low * speed_squared).sum(-1)
        return sum(terms.values()), terms

    def termination(self, s):
        """Hard fall label and differentiable fall probability, independent of a clip."""
        c = self.cfg
        height = s["pos"][..., 2]
        up = s["rot"][..., 2, 2]
        threshold = math.cos(math.radians(c.tilt_degrees))
        hard = (height < c.height_threshold) | (up < threshold)
        continuation = (
            torch.sigmoid((height - c.height_threshold) / c.soft_height_width)
            * torch.sigmoid((up - threshold) / c.soft_tilt_width)
        )
        return hard, 1 - continuation
