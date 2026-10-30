"""Optional time-scaled AMP positives, without changing the source motion library.

Only adjacent pairs inside a clip are augmented. A time scale s advances the
source trajectory by s of one frame during one 20 ms control interval and scales
all physical velocities by s. Pose interpolation uses the shortest SO(3) arc.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from .rotations import quat_slerp, quat_to_mat


STATE_KEYS = ("pos", "rot", "q", "v", "w", "qd")


def parse_time_scales(text):
    """Canonical sorted, unique scales; source-speed 1.0 is a valid default."""
    try:
        parts = text.split(",") if isinstance(text, str) else tuple(text)
        if not parts or any(isinstance(part, str) and not part.strip() for part in parts):
            raise ValueError
        scales = tuple(sorted({float(part) for part in parts}))
    except (TypeError, ValueError) as exc:
        raise ValueError("Expert time scales must be a nonempty CSV of finite numbers in (0, 1]") from exc
    if any(not math.isfinite(scale) or not 0 < scale <= 1 for scale in scales):
        raise ValueError("Expert time scales must be finite and in (0, 1]")
    return scales


def parse_clip_prefixes(text):
    """Canonical clip-name prefixes; an empty selection string means all clips."""
    if isinstance(text, str):
        if not text.strip():
            return ()
        parts = text.split(",")
    else:
        try:
            parts = tuple(text)
        except TypeError as exc:
            raise ValueError("Expert clip prefixes must be CSV strings") from exc
    if any(not isinstance(part, str) or not part.strip() for part in parts):
        raise ValueError("Expert clip prefixes must contain nonempty strings")
    return tuple(sorted({part.strip() for part in parts}))


def select_expert_clips(names, prefixes):
    """Select prior-positive clips by full name without changing the library."""
    prefixes = parse_clip_prefixes(prefixes)
    selected = [i for i, name in enumerate(names) if not prefixes or str(name).startswith(prefixes)]
    if not selected:
        raise ValueError(f"No expert clips match prefixes {prefixes!r}")
    return selected


def _finite_state(state):
    if any(not bool(torch.isfinite(state[key]).all()) for key in STATE_KEYS):
        raise FloatingPointError("Non-finite state in AMP expert augmentation")


def _rotation_quaternion(rotation):
    """Largest-component conversion, stable even at a relative pi rotation."""
    m00, m11, m22 = rotation[..., 0, 0], rotation[..., 1, 1], rotation[..., 2, 2]
    m01, m02 = rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m12 = rotation[..., 1, 0], rotation[..., 1, 2]
    m20, m21 = rotation[..., 2, 0], rotation[..., 2, 1]
    absolute = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22,
                            1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1).clamp_min(0).sqrt()
    candidates = torch.stack((
        torch.stack((absolute[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, absolute[..., 1].square(), m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, absolute[..., 2].square(), m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, absolute[..., 3].square()), -1),
    ), -2) / (2 * absolute[..., :, None].clamp_min(1e-8))
    index = absolute.argmax(-1)[..., None, None].expand(absolute.shape[:-1] + (1, 4))
    quaternion = candidates.gather(-2, index).squeeze(-2)
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def time_scaled_states(current, following, scale):
    """Return the two physical states in a time-scaled adjacent expert pair."""
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Expert time scale must be finite and in (0, 1]")
    _finite_state(current)
    _finite_state(following)
    first = {key: current[key] * scale if key in ("v", "w", "qd") else current[key] for key in STATE_KEYS}
    start_quat, end_quat = _rotation_quaternion(current["rot"]), _rotation_quaternion(following["rot"])
    fraction = start_quat.new_full(start_quat.shape[:-1], scale)
    second = {
        "pos": torch.lerp(current["pos"], following["pos"], scale),
        "rot": quat_to_mat(quat_slerp(start_quat, end_quat, fraction)),
        "q": torch.lerp(current["q"], following["q"], scale),
        "v": scale * torch.lerp(current["v"], following["v"], scale),
        "w": scale * torch.lerp(current["w"], following["w"], scale),
        "qd": scale * torch.lerp(current["qd"], following["qd"], scale),
    }
    _finite_state(first)
    _finite_state(second)
    return first, second


def command_style_gate(command, threshold):
    """Command motion score; threshold zero leaves the historical style intact."""
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Style command gate must be finite and nonnegative")
    if threshold == 0:
        return torch.ones_like(command[..., 0])
    motion = command[..., :2].norm(dim=-1) + 0.5 * command[..., 2].abs()
    return (motion / threshold).clamp(0, 1)


class TimeScaledExpertPairs:
    """Precomputed positives, uniform over eligible clips and time scales.

    Eligibility is measured on the unscaled first frame: planar root speed above
    motion_threshold OR absolute world yaw rate above 0.2 rad/s. Terminal clip
    frames cannot be pair starts. Source indices are retained for provenance.
    """

    def __init__(self, lib, features, scales, motion_threshold=0.15, batch_size=4096, clip_ids=None):
        self.scales = tuple(scale for scale in parse_time_scales(scales) if scale < 1)
        if not self.scales:
            raise ValueError("Augmented expert positives require at least one time scale below 1")
        if not math.isfinite(motion_threshold) or motion_threshold < 0 or batch_size < 1:
            raise ValueError("Motion threshold must be nonnegative and batch size positive")
        selected_clips = None
        if clip_ids is not None:
            try:
                if isinstance(clip_ids, torch.Tensor):
                    if clip_ids.ndim != 1:
                        raise ValueError("Selected expert clip IDs must be a one-dimensional sequence")
                    ids = clip_ids.detach().cpu().tolist()
                else:
                    ids = tuple(clip_ids)
            except TypeError as exc:
                raise ValueError("Selected expert clip IDs must be an integer sequence") from exc
            if not ids or any(isinstance(clip, bool) or not isinstance(clip, Integral)
                              or not 0 <= clip < len(lib.starts) for clip in ids):
                raise ValueError("Selected expert clip IDs must be nonempty valid integer indices")
            selected_clips = set(ids)
        self.device = lib.device
        indices, counts, eligible_ids, pair_clip_ids = [], [], [], []
        selected_speeds, selected_yaw, selected_joint_rms = [], [], []
        with torch.no_grad():
            for clip, (start, length) in enumerate(zip(lib.starts.tolist(), lib.lengths.tolist())):
                if selected_clips is not None and clip not in selected_clips:
                    continue
                if length < 2:
                    continue
                candidates = torch.arange(start, start + length - 1, device=self.device)
                current = lib.frame(candidates)
                _finite_state(current)
                speed = current["v"][..., :2].norm(dim=-1)
                yaw = current["w"][..., 2].abs()
                eligible = (speed > motion_threshold) | (yaw > 0.2)
                selected = candidates[eligible]
                if not len(selected):
                    continue
                indices.append(selected)
                counts.append(len(selected))
                eligible_ids.append(clip)
                pair_clip_ids.append(torch.full_like(selected, clip))
                selected_speeds.append(float(speed[eligible].mean()))
                selected_yaw.append(float(yaw[eligible].mean()))
                selected_joint_rms.append(float(current["qd"][eligible].square().mean(-1).sqrt().mean()))
            if not indices:
                raise ValueError("No eligible moving expert pairs at the configured motion threshold")
            self.source_indices = torch.cat(indices)
            self.clip_ids = torch.cat(pair_clip_ids)
            self.eligible_clip_ids = tuple(eligible_ids)
            self.counts = torch.tensor(counts, dtype=torch.long, device=self.device)
            self.starts = torch.cat((self.counts.new_zeros(1), self.counts.cumsum(0)[:-1]))
            first_banks, second_banks = [], []
            for scale in self.scales:
                first_features, second_features = [], []
                for start in range(0, len(self.source_indices), batch_size):
                    index = self.source_indices[start:start + batch_size]
                    first, second = time_scaled_states(lib.frame(index), lib.frame(index + 1), scale)
                    f0, f1 = features(first), features(second)
                    if f0.ndim != 2 or f0.shape != f1.shape or f0.shape[0] != len(index):
                        raise ValueError("Expert feature extractor must return matching (batch, features) tensors")
                    if not bool(torch.isfinite(f0).all() and torch.isfinite(f1).all()):
                        raise FloatingPointError("Non-finite features in AMP expert augmentation")
                    first_features.append(f0.detach())
                    second_features.append(f1.detach())
                first_banks.append(torch.cat(first_features))
                second_banks.append(torch.cat(second_features))
            self.first = torch.stack(first_banks)
            self.second = torch.stack(second_banks)
        mean_speed = sum(selected_speeds) / len(selected_speeds)
        mean_yaw = sum(selected_yaw) / len(selected_yaw)
        mean_joint_rms = sum(selected_joint_rms) / len(selected_joint_rms)
        names = getattr(lib, "names", [str(i) for i in range(len(lib.starts))])
        self.sampling_metadata = {
            "time_scales": list(self.scales), "eligible_pairs": len(self.source_indices),
            "eligible_clips": len(self.eligible_clip_ids),
            "motion_threshold_mps": motion_threshold, "yaw_threshold_radps": 0.2,
            "sampling": "Uniform eligible clip, then uniform eligible pair; independent uniform time scale",
            "pairs_by_clip": [{"clip": clip, "name": str(names[clip]), "pairs": count}
                              for clip, count in zip(eligible_ids, counts)],
            "calibration_clip_uniform_unscaled_mean": {
                "root_planar_speed_mps": mean_speed, "absolute_root_yaw_rate_radps": mean_yaw,
                "joint_velocity_rms_radps": mean_joint_rms,
            },
            "calibration_clip_uniform_first_state_by_scale": [
                {"scale": scale, "root_planar_speed_mps": scale * mean_speed,
                 "absolute_root_yaw_rate_radps": scale * mean_yaw,
                 "joint_velocity_rms_radps": scale * mean_joint_rms}
                for scale in self.scales
            ],
            "feature_bank_bytes": (self.first.numel() + self.second.numel()) * self.first.element_size(),
        }

    @torch.no_grad()
    def sample_indices(self, batch):
        if batch < 1:
            raise ValueError("Expert batch size must be positive")
        clip = torch.randint(len(self.counts), (batch,), device=self.device)
        offset = (torch.rand(batch, device=self.device) * self.counts[clip]).long()
        pair = self.starts[clip] + offset
        scale = torch.randint(len(self.scales), (batch,), device=self.device)
        return scale, pair

    @torch.no_grad()
    def sample(self, batch):
        scale, pair = self.sample_indices(batch)
        return self.first[scale, pair], self.second[scale, pair]


class MixedExpertPairs:
    """Keep the untouched original sampler alongside optional augmented positives."""

    def __init__(self, original, augmented, fraction):
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Augmented expert fraction must be in [0, 1]")
        if fraction > 0 and augmented is None:
            raise ValueError("An augmented sampler is required for a positive mixture fraction")
        self.original, self.augmented, self.fraction = original, augmented, fraction

    @torch.no_grad()
    def sample(self, batch):
        # Preserve original RNG draws and tensors exactly when augmentation is off.
        if self.fraction == 0:
            return self.original.sample(batch)
        if self.fraction == 1:
            return self.augmented.sample(batch)
        first, second = self.original.sample(batch)
        augmented_first, augmented_second = self.augmented.sample(batch)
        choose = torch.rand(batch, device=first.device) < self.fraction
        return (torch.where(choose[:, None], augmented_first, first),
                torch.where(choose[:, None], augmented_second, second))
