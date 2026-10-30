"""CPU checks for time-scaled AMP experts, sampling and command style gating.

No simulator, GPU, motion file or training run is required. Run with PYTHONPATH=code.
"""

from __future__ import annotations

import dataclasses
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from supermimic.amp_augmentation import (
    MixedExpertPairs, TimeScaledExpertPairs, command_style_gate,
    parse_time_scales, time_scaled_states,
)
from supermimic.amp_train import AMPTrainer, AMPTrainCfg, reward_config, validate_cfg, validate_resume_config
from supermimic.rotations import axis_angle_to_mat


def example_states(batch=2):
    first = {
        "pos": torch.tensor([1., 2., 3.]).expand(batch, 3).clone(),
        "rot": axis_angle_to_mat(torch.tensor([0., 0., 0.4])).expand(batch, 3, 3).clone(),
        "q": torch.full((batch, 29), 0.2),
        "v": torch.tensor([1., 2., 3.]).expand(batch, 3).clone(),
        "w": torch.tensor([0.1, -0.2, 0.3]).expand(batch, 3).clone(),
        "qd": torch.full((batch, 29), 0.5),
    }
    following = {key: value.clone() for key, value in first.items()}
    following["pos"] += torch.tensor([2., -4., 6.])
    following["rot"] = first["rot"] @ axis_angle_to_mat(torch.tensor([1.2, 0., 0.]))
    following["q"] += 0.8
    following["v"] *= 3
    following["w"] *= 3
    following["qd"] *= 3
    return first, following


class FakeLibrary:
    """Two eligible forward pairs, six turning pairs and one idle clip."""

    def __init__(self):
        self.device = "cpu"
        self.starts = torch.tensor([0, 5, 12])
        self.lengths = torch.tensor([5, 7, 4])
        self.names = ["short_forward", "long_turn", "idle"]
        self.num_frames = 16
        self.num_clips = 3
        index = torch.arange(self.num_frames).float()
        self.data = {
            "pos": torch.stack((index, index * 0, index * 0 + 0.8), -1),
            "rot": axis_angle_to_mat(torch.stack((index * 0, index * 0, index * 0.04), -1)),
            "q": index[:, None].expand(-1, 29).clone() * 0.02,
            "v": torch.zeros(self.num_frames, 3),
            "w": torch.zeros(self.num_frames, 3),
            "qd": torch.full((self.num_frames, 29), 0.5),
        }
        self.data["v"][:2, 0] = 0.3
        self.data["v"][2, 0] = 0.1
        self.data["v"][3, 0] = 0.15  # Strictly greater than the threshold is required.
        self.data["w"][5:11, 2] = 0.3
        # Last frames are deliberately moving, but cannot start a same-clip pair.
        self.data["v"][[4, 11, 15], 0] = 0.8

    def frame(self, index):
        return {key: value[index] for key, value in self.data.items()}


class FakeFeatures:
    feature_dim = 22

    def __call__(self, state):
        return torch.cat((state["pos"], state["rot"].flatten(-2), state["q"][..., :2],
                          state["v"], state["w"], state["qd"][..., :2]), -1)


class MarkerPairs:
    def __init__(self, marker):
        self.marker = float(marker)
        self.features = torch.zeros(1, 2)
        self.calls = 0

    def sample(self, batch):
        self.calls += 1
        first = torch.stack((torch.full((batch,), self.marker), torch.rand(batch)), -1)
        return first, first + torch.tensor([0., 1.])


class ForbiddenPairs:
    def sample(self, batch):
        raise AssertionError("Disabled augmentation must never sample the augmented bank")


class TimeScaleStateTests(unittest.TestCase):
    def test_scale_parser_deduplicates_and_rejects_invalid_scales(self):
        parsed = parse_time_scales("1.0,0.5,0.25,0.5")
        self.assertIsInstance(parsed, tuple)
        self.assertEqual(set(parsed), {1., 0.5, 0.25})
        self.assertEqual(len(parsed), 3)
        for text in ("", " ", "0", "-0.1", "1.01", "nan", "inf", "nope"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_time_scales(text)

    def test_pose_geodesic_and_velocity_interpolation(self):
        current, following = example_states()
        first, second = time_scaled_states(current, following, 0.5)
        for key in ("pos", "rot", "q"):
            torch.testing.assert_close(first[key], current[key], rtol=0, atol=0)
        for key in ("v", "w", "qd"):
            torch.testing.assert_close(first[key], current[key] * 0.5)
            torch.testing.assert_close(second[key], (current[key] + following[key]) * 0.25)
        for key in ("pos", "q"):
            torch.testing.assert_close(second[key], (current[key] + following[key]) * 0.5)
        expected_rotation = current["rot"] @ axis_angle_to_mat(torch.tensor([0.6, 0., 0.]))
        torch.testing.assert_close(second["rot"], expected_rotation, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second["rot"].transpose(-1, -2) @ second["rot"], torch.eye(3).expand(2, 3, 3), atol=1e-6, rtol=1e-6)

    def test_scaling_never_changes_input_states(self):
        current, following = example_states()
        before = [{key: value.clone() for key, value in state.items()} for state in (current, following)]
        time_scaled_states(current, following, 0.25)
        for actual, expected in zip((current, following), before):
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def test_geodesic_half_step_remains_valid_at_and_near_pi(self):
        for angle in (math.pi - 1e-6, math.pi, math.pi + 1e-4):
            with self.subTest(angle=angle):
                current, following = example_states(batch=1)
                current = {key: value.double() for key, value in current.items()}
                following = {key: value.double() for key, value in following.items()}
                current["rot"] = axis_angle_to_mat(torch.tensor([[0., 0., 0.4]], dtype=torch.float64))
                following["rot"] = current["rot"] @ axis_angle_to_mat(torch.tensor([[angle, 0., 0.]], dtype=torch.float64))
                _, halfway = time_scaled_states(current, following, 0.5)
                relative = current["rot"].transpose(-1, -2) @ halfway["rot"]
                target = current["rot"].transpose(-1, -2) @ following["rot"]
                torch.testing.assert_close(relative @ relative, target, rtol=1e-5, atol=1e-5)
                cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
                shortest_angle = min(angle, 2 * math.pi - angle)
                torch.testing.assert_close(cosine, torch.full((1,), math.cos(shortest_angle / 2), dtype=torch.float64), rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(torch.linalg.det(relative), torch.ones(1, dtype=torch.float64), rtol=1e-6, atol=1e-6)

    def test_nonfinite_input_and_invalid_direct_scales_are_rejected(self):
        for key in ("pos", "rot", "q", "v", "w", "qd"):
            with self.subTest(field=key), self.assertRaises((ValueError, FloatingPointError)):
                current, following = example_states()
                following[key].flatten()[0] = float("nan")
                time_scaled_states(current, following, 0.5)
        for scale in (0., -0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                time_scaled_states(*example_states(), scale)

    def test_float32_pi_rotation_handles_principal_and_general_axes(self):
        current, following = example_states(batch=4)
        axes = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.3, -0.8, 0.4]])
        axes = axes / axes.norm(dim=-1, keepdim=True)
        following["rot"] = current["rot"] @ axis_angle_to_mat(math.pi * axes)
        _, halfway = time_scaled_states(current, following, 0.5)
        relative = current["rot"].transpose(-1, -2) @ halfway["rot"]
        target = current["rot"].transpose(-1, -2) @ following["rot"]
        torch.testing.assert_close(relative @ relative, target, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(torch.linalg.det(relative), torch.ones(4), rtol=3e-5, atol=3e-5)


class ExpertBankTests(unittest.TestCase):
    def setUp(self):
        self.lib = FakeLibrary()
        self.features = FakeFeatures()

    def make_bank(self):
        return TimeScaledExpertPairs(self.lib, self.features, (1., 0.5, 0.25), motion_threshold=0.15, batch_size=3)

    def test_only_adjacent_moving_or_turning_pairs_enter_the_bank(self):
        bank = self.make_bank()
        expected = torch.tensor([0, 1, 5, 6, 7, 8, 9, 10])
        torch.testing.assert_close(bank.source_indices.sort().values, expected)
        self.assertEqual(tuple(bank.first.shape), (2, 8, self.features.feature_dim))
        self.assertEqual(tuple(bank.second.shape), (2, 8, self.features.feature_dim))
        # Position x encodes source frame, so every first endpoint must retain it.
        torch.testing.assert_close(bank.first[..., 0], bank.source_indices.float()[None].expand(2, -1))
        self.assertFalse(bool(torch.isin(bank.source_indices, torch.tensor([4, 11, 15])).any()))
        self.assertEqual(set(bank.clip_ids.tolist()), {0, 1})

    def test_bank_creation_does_not_mutate_motion_data(self):
        before = {key: value.clone() for key, value in self.lib.data.items()}
        self.make_bank()
        for key, value in self.lib.data.items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_clip_and_scale_sampling_are_uniform_despite_unequal_pair_counts(self):
        bank = self.make_bank()
        torch.manual_seed(72)
        scale, index = bank.sample_indices(30000)
        self.assertTrue(bool(((scale >= 0) & (scale < 2)).all()))
        self.assertTrue(bool(((index >= 0) & (index < 8)).all()))
        first_clip = float((bank.clip_ids[index] == 0).float().mean())
        first_scale = float((scale == 0).float().mean())
        self.assertTrue(0.46 < first_clip < 0.54, first_clip)
        self.assertTrue(0.46 < first_scale < 0.54, first_scale)
        first, second = bank.sample(20)
        self.assertEqual(tuple(first.shape), (20, self.features.feature_dim))
        self.assertEqual(first.shape, second.shape)
        self.assertTrue(bool(torch.isfinite(first).all() and torch.isfinite(second).all()))


class MixedPairTests(unittest.TestCase):
    def test_disabled_augmentation_preserves_original_samples_and_rng_exactly(self):
        original = MarkerPairs(0.)
        torch.manual_seed(12)
        before = torch.get_rng_state().clone()
        mixed = MixedExpertPairs(original, ForbiddenPairs(), 0.)
        torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
        expected = original.sample(32)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(12)
        actual = mixed.sample(32)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_bernoulli_mixture_keeps_endpoints_together(self):
        torch.manual_seed(31)
        mixed = MixedExpertPairs(MarkerPairs(0.), MarkerPairs(1.), 0.25)
        first, second = mixed.sample(20000)
        fraction = float((first[:, 0] == 1).float().mean())
        self.assertTrue(0.235 < fraction < 0.265, fraction)
        torch.testing.assert_close(second[:, 0], first[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(second[:, 1] - first[:, 1], torch.ones(20000))

    def test_full_augmentation_uses_augmented_endpoints(self):
        first, second = MixedExpertPairs(MarkerPairs(0.), MarkerPairs(1.), 1.).sample(64)
        torch.testing.assert_close(first[:, 0], torch.ones(64), rtol=0, atol=0)
        torch.testing.assert_close(second[:, 0], torch.ones(64), rtol=0, atol=0)


class CommandStyleGateTests(unittest.TestCase):
    def test_zero_gate_preserves_original_rewards_and_scores(self):
        commands = torch.tensor([[0., 0., 0.], [0.1, -0.2, 0.3]])
        torch.testing.assert_close(command_style_gate(commands, 0.), torch.ones(2), rtol=0, atol=0)
        class Prior:
            def __init__(self):
                self.style, self.score = torch.tensor([0.1, 0.2]), torch.tensor([-0.2, 0.8])
                self.calls = 0
            def reward(self, f0, f1):
                self.calls += 1
                return self.style, self.score
        trainer = AMPTrainer.__new__(AMPTrainer)
        trainer.cfg = AMPTrainCfg(style_command_gate=0.)
        trainer.prior = Prior()
        actual, score = trainer.style_reward(torch.zeros(2, 1), torch.zeros(2, 1), commands)
        self.assertIs(actual, trainer.prior.style)
        self.assertIs(score, trainer.prior.score)
        self.assertEqual(trainer.prior.calls, 1)

    def test_gate_uses_command_planar_norm_and_half_yaw_then_clamps(self):
        commands = torch.tensor([[0., 0., 0.], [0.1, 0., 0.], [0., 0., 0.4], [0.3, 0.4, 0.], [0., 0., 2.]])
        torch.testing.assert_close(command_style_gate(commands, 0.5), torch.tensor([0., 0.2, 0.4, 1., 1.]))

    def test_gate_retains_finite_command_gradients_below_saturation(self):
        command = torch.tensor([[0.1, 0.1, 0.2]], requires_grad=True)
        command_style_gate(command, 0.5).sum().backward()
        self.assertTrue(bool(torch.isfinite(command.grad).all()))
        self.assertTrue(bool((command.grad > 0).all()))

    def test_enabled_trainer_gate_changes_style_but_preserves_discriminator_scores(self):
        raw_style = torch.tensor([0.1, 0.2])
        raw_score = torch.tensor([-0.2, 0.8])
        trainer = AMPTrainer.__new__(AMPTrainer)
        trainer.cfg = AMPTrainCfg(style_command_gate=0.5)
        trainer.prior = SimpleNamespace(reward=lambda first, second: (raw_style, raw_score))
        actual, scores = trainer.style_reward(torch.zeros(2, 1), torch.zeros(2, 1), torch.tensor([[0., 0., 0.], [0.25, 0., 0.]]))
        torch.testing.assert_close(actual, torch.tensor([0., 0.1]))
        self.assertIs(scores, raw_score)
        torch.testing.assert_close(raw_style, torch.tensor([0.1, 0.2]), rtol=0, atol=0)


class AugmentationConfigTests(unittest.TestCase):
    fields = ("expert_time_scales", "expert_augmented_fraction", "expert_motion_threshold", "style_command_gate")

    def setUp(self):
        world = patch.dict(os.environ, {"WORLD_SIZE": "1"})
        world.start(); self.addCleanup(world.stop)
        self.cfg = AMPTrainCfg()
        self.saved = dataclasses.asdict(self.cfg)

    def test_legacy_configs_restore_disabled_defaults_and_identical_reward_config(self):
        legacy = {key: value for key, value in self.saved.items() if key not in self.fields}
        restored = AMPTrainCfg(**legacy)
        self.assertEqual(restored.expert_time_scales, "1.0")
        self.assertEqual(restored.expert_augmented_fraction, 0.)
        self.assertEqual(restored.expert_motion_threshold, 0.15)
        self.assertEqual(restored.style_command_gate, 0.)
        self.assertEqual(reward_config(legacy), reward_config(restored))
        self.assertEqual(validate_resume_config(legacy, restored), {})

    def test_every_augmentation_reward_field_is_guarded_on_resume(self):
        values = ("1.0,0.5", 0.25, 0.2, 0.5)
        for key, value in zip(self.fields, values):
            with self.subTest(field=key):
                candidate = dataclasses.replace(self.cfg, **{key: value})
                with self.assertRaises(ValueError):
                    validate_resume_config(self.saved, candidate)
                candidate.allow_reward_change = True
                self.assertEqual(validate_resume_config(self.saved, candidate), {
                    key: {"old": reward_config(self.saved)[key], "new": reward_config(candidate)[key]},
                })

    def test_semantically_identical_scale_spellings_do_not_change_the_reward(self):
        saved = dataclasses.asdict(dataclasses.replace(self.cfg, expert_time_scales="1.0,0.5,0.25"))
        candidate = dataclasses.replace(self.cfg, expert_time_scales="0.5,1,0.25,0.5")
        self.assertEqual(validate_resume_config(saved, candidate), {})

    def test_validation_accepts_curriculum_and_rejects_invalid_fields(self):
        validate_cfg(dataclasses.replace(self.cfg, expert_time_scales="1.0,0.5,0.25", expert_augmented_fraction=0.25,
                                         expert_motion_threshold=0.2, style_command_gate=0.5))
        bad = {
            "expert_time_scales": ("", "nan", "0,1.0", "1.2"),
            "expert_augmented_fraction": (-0.1, 1.1, float("nan"), float("inf")),
            "expert_motion_threshold": (-0.1, float("nan"), float("inf")),
            "style_command_gate": (-0.1, float("nan"), float("inf")),
        }
        for key, values in bad.items():
            for value in values:
                with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                    validate_cfg(dataclasses.replace(self.cfg, **{key: value}))


if __name__ == "__main__":
    unittest.main()
