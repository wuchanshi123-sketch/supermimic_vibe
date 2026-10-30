"""Fixed expert and gait measurements for AMP checkpoint evaluation.

These measurements do not load the checkpoint's discriminator or its running
normalizer. They complement survival and command error; they are not a gait score.
"""

from __future__ import annotations

import math

import torch

from .amp_prior import AMPFeatureExtractor


def _even_sample(indices, limit):
    if len(indices) <= limit:
        return indices
    return indices[torch.linspace(0, len(indices) - 1, limit, device=indices.device).long()]


def _summary(values):
    values = values.float().flatten()
    return {
        "mean": float(values.mean()),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


class FixedExpertStyle:
    """Nearest adjacent expert pair under frozen, expert-only whitening.

    The last 20% of each clip supplies calibration queries; earlier pairs supply
    the reference bank. These are measurement splits, not withheld training data.
    Distances are RMS over the normalized AMP pair's feature coordinates.
    """

    @torch.no_grad()
    def __init__(self, features, starts, lengths, bank_size=2048, calibration_size=512,
                 std_floor=0.05):
        if bank_size < 1 or calibration_size < 1 or std_floor <= 0:
            raise ValueError("Expert bank sizes and normalization floor must be positive")
        if features.ndim != 2 or not bool(torch.isfinite(features).all()):
            raise ValueError("Expert features must be a finite matrix")
        bank_ids, calibration_ids = [], []
        for start, length in zip(starts.tolist(), lengths.tolist()):
            if length < 3:
                raise ValueError("Expert diagnostics require at least three frames per clip")
            count = length - 1
            split = max(1, min(count - 1, int(count * 0.8)))
            bank_ids.append(torch.arange(start, start + split, device=features.device))
            calibration_ids.append(torch.arange(start + split, start + count, device=features.device))
        bank_ids = _even_sample(torch.cat(bank_ids), bank_size)
        calibration_ids = _even_sample(torch.cat(calibration_ids), calibration_size)
        endpoints = torch.cat((features[bank_ids], features[bank_ids + 1]))
        self.mean = endpoints.mean(0)
        self.std = endpoints.std(0, unbiased=False).clamp_min(std_floor)
        self.bank = self.pairs(features[bank_ids], features[bank_ids + 1])
        calibration = self.distance(features[calibration_ids], features[calibration_ids + 1])
        self.metadata = {
            "version": 1,
            "metric": "nearest expert adjacent-pair RMS after fixed expert-only feature whitening",
            "feature_dimension": features.shape[-1],
            "reference_bank_pairs": len(bank_ids),
            "calibration_pairs": len(calibration_ids),
            "normalization_std_floor": std_floor,
            "split": "per-clip first 80% reference bank; last 20% calibration queries",
            "calibration_is_withheld_from_policy_training": False,
            "calibration_distance": _summary(calibration),
            "checkpoint_discriminator_used": False,
        }

    def pairs(self, f0, f1):
        return torch.cat(((f0 - self.mean) / self.std, (f1 - self.mean) / self.std), -1)

    @torch.no_grad()
    def distance(self, f0, f1):
        pair = self.pairs(f0, f1)
        chunks = []
        for chunk in pair.split(256):
            # Direct differences avoid cancellation for exact bank matches.
            distance = torch.cdist(chunk, self.bank, compute_mode="donot_use_mm_for_euclid_dist")
            chunks.append(distance.min(-1).values / math.sqrt(self.bank.shape[-1]))
        return torch.cat(chunks)

    @classmethod
    @torch.no_grad()
    def from_library(cls, task, lib, bank_size=2048):
        extract = AMPFeatureExtractor(task)
        features = []
        for start in range(0, lib.num_frames, 2048):
            ids = torch.arange(start, min(start + 2048, lib.num_frames), device=lib.device)
            features.append(extract(lib.frame(ids)))
        metric = cls(torch.cat(features), lib.starts, lib.lengths, bank_size=bank_size)
        metric.metadata.update(expert_clips=lib.num_clips, expert_frames=lib.num_frames,
                               control_dt=lib.dt,
                               expert_root_height_m=_summary(lib.root_pos[:, 2]),
                               expert_joint_speed_abs_radps=_summary(lib.qd.abs()))
        return metric


class GaitDiagnostics:
    """Accumulate only first-episode states, including the terminal transition."""

    def __init__(self, env, expert_style=None, style_stride=5, style_samples=4096):
        if style_stride < 1 or style_samples < 1:
            raise ValueError("Style stride and sample count must be positive")
        self.env = env
        self.expert_style = expert_style
        self.style_stride, self.style_samples = style_stride, style_samples
        names = env.task.kin.body_names
        self.torso = names.index("torso_link")
        self.feet = [names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.extract = AMPFeatureExtractor(env.task) if expert_style is not None else None
        self.samples = {}
        self.pairs0, self.pairs1 = [], []
        self.step = 0
        self.initial_pos = None
        self.last_pos = None

    @torch.no_grad()
    def observe(self, before, state, action, previous, active):
        self.step += 1
        ids = active.nonzero().flatten()
        if not len(ids):
            return
        if self.initial_pos is None:
            self.initial_pos = before["pos"].clone()
            self.last_pos = before["pos"].clone()
        self.last_pos[ids] = state["pos"][ids]
        state = {key: value[ids] for key, value in state.items()}
        before = {key: value[ids] for key, value in before.items()}
        kin = self.env.task.body_kin(state)
        positions, rotations, velocities, _ = kin
        action, previous = action[ids], previous[ids]
        ankle_height = positions[:, self.feet, 2]
        ankle_speed = velocities[:, self.feet, :2].norm(dim=-1)
        low, high = ankle_height < 0.05, ankle_height > 0.06
        values = {
            "root_height_m": state["pos"][:, 2],
            "torso_height_m": positions[:, self.torso, 2],
            "root_tilt_degrees": torch.rad2deg(torch.acos(state["rot"][:, 2, 2].clamp(-1, 1))),
            "torso_tilt_degrees": torch.rad2deg(torch.acos(rotations[:, self.torso, 2, 2].clamp(-1, 1))),
            "ankle_height_m": positions[:, self.feet, 2],
            "ankle_vertical_speed_mps": velocities[:, self.feet, 2].abs(),
            "ankle_planar_speed_mps": velocities[:, self.feet, :2].norm(dim=-1),
            "ankle_height_difference_abs_m": (positions[:, self.feet[0], 2] - positions[:, self.feet[1], 2]).abs(),
            "joint_speed_abs_radps": state["qd"].abs(),
            "action_abs": action.abs(),
            "action_step_change_abs": (action - previous).abs(),
            "both_ankles_low_fraction": low.all(-1).float(),
            "both_ankles_high_fraction": high.all(-1).float(),
            "low_ankle_planar_speed_mps": ankle_speed[low],
        }
        for foot, side in enumerate(("left", "right")):
            values.update({
                f"{side}_ankle_height_m": ankle_height[:, foot],
                f"{side}_ankle_planar_speed_mps": ankle_speed[:, foot],
                f"{side}_ankle_vertical_speed_mps": velocities[:, self.feet[foot], 2].abs(),
                f"{side}_ankle_low_fraction": low[:, foot].float(),
                f"{side}_ankle_high_fraction": high[:, foot].float(),
                f"{side}_low_ankle_planar_speed_mps": ankle_speed[:, foot][low[:, foot]],
            })
        for key, value in values.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Non-finite active gait diagnostic: {key}")
            self.samples.setdefault(key, []).append(value.flatten())
        if self.extract is not None and (self.step - 1) % self.style_stride == 0:
            self.pairs0.append(self.extract(before))
            self.pairs1.append(self.extract(state, kin=kin))

    @torch.no_grad()
    def report(self):
        result = {}
        for key, values in self.samples.items():
            value = torch.cat(values)
            result[key] = _summary(value) if value.numel() else None
            if "low_ankle_planar_speed" in key:
                result[key + "_samples"] = value.numel()
        if self.initial_pos is not None:
            displacement = self.last_pos - self.initial_pos
            result["net_planar_displacement_m"] = _summary(displacement[:, :2].norm(dim=-1))
        result["ankle_measurement"] = "ankle-roll body origin, not sole clearance or measured contact"
        result["ankle_ground_proxies"] = {
            "low_height_threshold_m": 0.05, "high_height_threshold_m": 0.06,
            "slip_proxy": "world planar ankle-origin speed conditional on ankle height < 0.05 m",
            "limitation": "kinematic height proxies, not measured contacts or sole slip; tilted feet can differ",
        }
        if self.pairs0:
            f0, f1 = torch.cat(self.pairs0), torch.cat(self.pairs1)
            ids = _even_sample(torch.arange(len(f0), device=f0.device), self.style_samples)
            distance = self.expert_style.distance(f0[ids], f1[ids])
            result["fixed_expert_pair_distance"] = _summary(distance)
            result["fixed_expert_pair_samples"] = len(ids)
        return result
