"""CPU checks that per-copy gait validation excludes poisoned reset episodes."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_amp_gait_robustness import evaluate_robustness, summarize_trials
from test_amp_eval import FakeEnv


class PhysicalFakeEnv(FakeEnv):
    def __init__(self):
        super().__init__([[.3, 0., 0.], [.3, 0., 0.]], [0., 0.], [2, None])
        self.joint_names = [str(i) for i in range(29)]
        self.task = SimpleNamespace(
            kin=SimpleNamespace(body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            body_kin=lambda s: (s["pos"][:, None].repeat(1, 2, 1), s["rot"][:, None].repeat(1, 2, 1, 1),
                                s["v"][:, None].repeat(1, 2, 1), s["w"][:, None].repeat(1, 2, 1)),
        )

    def step(self, action):
        _, info = super().step(action)
        state = {"pos": torch.full((2, 3), .8), "rot": torch.eye(3)[None].repeat(2, 1, 1),
                 "q": torch.zeros(2, 29), "qd": torch.zeros(2, 29),
                 "v": info["velocity_heading"], "w": torch.zeros(2, 3)}
        return state, info


class GaitRobustnessTests(unittest.TestCase):
    def test_trace_prefix_excludes_reset_episode_nans(self):
        with tempfile.TemporaryDirectory() as directory:
            report = evaluate_robustness(lambda obs: torch.zeros(2, 29), PhysicalFakeEnv(), directory,
                                         seconds=.08, perturbation=0.)
            for trial, length in zip(report["trials"], (2, 4)):
                with np.load(Path(directory) / trial["trace"]) as trace:
                    self.assertEqual(len(trace["pos"]), length)
                    self.assertTrue(np.isfinite(trace["v"]).all())
            self.assertEqual(report["falls"], 1)
            self.assertEqual(report["survived"], 1)
            self.assertEqual(report["style_observations"]["copies"], 2)
            json.dumps(report, allow_nan=False)

    def test_style_fraction_counts_all_copies_and_requires_late_persistence(self):
        observed = {"alternating_steps_observed": True, "flags": []}
        flat = {"alternating_steps_observed": False, "flags": [{"code": "shallow_high_frequency_shuffle"}]}
        trials = [
            {"survived_full_horizon": True, "gait_after_2s": observed, "gait_final_half": observed},
            {"survived_full_horizon": False, "gait_after_2s": observed, "gait_final_half": observed},
            {"survived_full_horizon": True, "gait_after_2s": observed, "gait_final_half": flat},
        ]
        result = summarize_trials(trials)
        self.assertEqual(result["alternating_steps_observed_after_2s_count"], 3)
        self.assertEqual(result["surviving_alternating_unflagged_in_both_windows_count"], 1)
        self.assertAlmostEqual(result["surviving_alternating_unflagged_in_both_windows_fraction"], 1 / 3)
        self.assertFalse(result["meets_observation_target"])


if __name__ == "__main__":
    unittest.main()
