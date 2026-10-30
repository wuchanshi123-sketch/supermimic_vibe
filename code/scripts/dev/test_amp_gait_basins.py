"""CPU checks for phase diversity and the basin probe's gait measurements."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_amp_gait_basins import diverse_indices, gait_metrics, reset_scaled_reference, rollout


class BasinProbeTests(unittest.TestCase):
    def test_reference_reset_preserves_pose_and_scales_only_velocities(self):
        frames = torch.tensor([4, 19])
        reference = {key: torch.ones(2, width) for key, width in
                     (("pos", 3), ("quat", 4), ("q", 29), ("v", 3), ("w", 3), ("qd", 29))}
        written, calls = [], []
        env = SimpleNamespace(num_envs=2, device="cpu", rsi_probability=.5,
                              lib=SimpleNamespace(frame=lambda selected: reference),
                              write_state=lambda *args: written.append(args),
                              _finish_resets=lambda ids, force: calls.append((ids, force)))

        def reset(ids, frames):
            self.assertEqual(env.rsi_probability, 1.)
            torch.testing.assert_close(frames, torch.tensor([4, 19]))
            calls.append("normal reset bookkeeping")

        env.reset_envs = reset
        reset_scaled_reference(env, frames, .4)
        self.assertEqual(env.rsi_probability, .5)
        self.assertEqual(calls[0], "normal reset bookkeeping")
        for actual, key in zip(written[0][1:], ("pos", "quat", "q", "v", "w", "qd")):
            torch.testing.assert_close(actual, reference[key] * (.4 if key in ("v", "w", "qd") else 1))

    def test_trace_stops_at_first_fall_and_excludes_poisoned_reset_states(self):
        state = {key: torch.zeros(2, width) for key, width in
                 (("pos", 3), ("q", 29), ("v", 3), ("w", 3), ("qd", 29))}
        state["rot"] = torch.eye(3)[None].repeat(2, 1, 1)
        state["pos"][:, 2] = .8
        state["v"][:, 0] = .3
        names = ["left_ankle_roll_link", "right_ankle_roll_link"]

        def kin(s):
            return (s["pos"][:, None].repeat(1, 2, 1), s["rot"][:, None].repeat(1, 2, 1, 1),
                    s["v"][:, None].repeat(1, 2, 1), s["w"][:, None].repeat(1, 2, 1))

        env = SimpleNamespace(num_envs=2, device="cpu", tcfg=SimpleNamespace(dt=.02), state=state,
                              task=SimpleNamespace(body_kin=kin, kin=SimpleNamespace(body_names=names)),
                              fixed_command=torch.tensor([.3, 0., 0.]), joint_names=[str(i) for i in range(29)],
                              policy_inputs=lambda: torch.zeros(2, 384))
        step_number = 0

        def step(action):
            nonlocal step_number
            step_number += 1
            successor = {key: value.clone() for key, value in state.items()}
            if step_number > 2:
                for value in successor.values():
                    value[0] = float("nan")
            done = torch.tensor([step_number == 2, False])
            return successor, {"velocity_heading": successor["v"], "yaw_rate": successor["w"][:, 2],
                               "terminated": done, "done": done, "truncated": torch.zeros(2, dtype=torch.bool)}

        env.step = step
        with tempfile.TemporaryDirectory() as directory:
            report = rollout(lambda obs: torch.zeros(2, 29), env, .1, Path(directory), "fake", ["a", "b"])
            first = np.load(Path(directory) / report["copies"][0]["trace"])
            second = np.load(Path(directory) / report["copies"][1]["trace"])
            self.assertEqual(len(first["pos"]), 2)
            self.assertEqual(len(second["pos"]), 5)
            self.assertTrue(np.isfinite(first["v"]).all())
            self.assertTrue(np.isfinite(second["v"]).all())
            self.assertEqual(report["falls"], 1)
            self.assertEqual(report["survived"], 1)

    def test_diverse_selection_returns_distinct_motion_phases(self):
        angle = torch.arange(100) * (2 * torch.pi / 100)
        features = torch.stack((angle.sin(), angle.cos()), -1)
        selected = diverse_indices(features, 4)
        self.assertEqual(len(set(selected.tolist())), 4)
        points = features[selected]
        self.assertGreater(float(torch.pdist(points).min()), 1.3)

    def test_height_statistics_distinguish_steps_from_shallow_shuffle(self):
        dt = .02
        time = np.arange(500) * dt
        for amplitude, frequency in ((.06, 1.), (.004, 4.5)):
            height = .1 + amplitude * np.stack((np.sin(2 * np.pi * frequency * time),
                                               -np.sin(2 * np.pi * frequency * time)), -1)
            position = np.zeros((500, 2, 3))
            position[:, :, 2] = height
            report = gait_metrics(position, np.zeros_like(position), np.tile([.3, 0., 0.], (500, 1)),
                                  np.zeros(500), dt, np.array([.3, 0., 0.]))
            self.assertAlmostEqual(report["height_dominant_hz"], frequency, places=6)
            self.assertAlmostEqual(abs(report["height_left_right_phase_degrees"]), 180., places=5)
            self.assertAlmostEqual(report["planar_rmse_mps"], 0.)
            self.assertGreater(report["height_p95_minus_p05_m"][0], amplitude * 1.9)


if __name__ == "__main__":
    unittest.main()
