"""CPU checks for fixed reference distances and first-episode gait metrics."""

from types import SimpleNamespace
import unittest

import torch

from supermimic.amp_diagnostics import FixedExpertStyle, GaitDiagnostics
from supermimic.amp_eval import evaluate_command, perturb_standing_reset
from supermimic.amp_prior import AMP_BODY_NAMES


class FixedExpertTests(unittest.TestCase):
    def test_reference_pairs_do_not_cross_clips_and_distances_are_fixed(self):
        features = torch.arange(20, dtype=torch.float32)[:, None].repeat(1, 3)
        metric = FixedExpertStyle(features, torch.tensor([0, 10]), torch.tensor([10, 10]))
        self.assertEqual(metric.metadata["reference_bank_pairs"], 14)
        self.assertEqual(metric.metadata["calibration_pairs"], 4)
        torch.testing.assert_close(metric.distance(features[[0, 10]], features[[1, 11]]), torch.zeros(2))
        # A policy sample must not update the expert-only normalization or bank.
        old = (metric.mean.clone(), metric.std.clone(), metric.bank.clone())
        far = metric.distance(features[[0]] + 100, features[[1]] + 100)
        self.assertGreater(float(far[0]), 10)
        for actual, expected in zip((metric.mean, metric.std, metric.bank), old):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_reference_subsampling_is_deterministic(self):
        features = torch.randn(100, 5, generator=torch.Generator().manual_seed(2))
        a = FixedExpertStyle(features, torch.tensor([0]), torch.tensor([100]), bank_size=12, calibration_size=8)
        b = FixedExpertStyle(features, torch.tensor([0]), torch.tensor([100]), bank_size=12, calibration_size=8)
        self.assertEqual(a.metadata, b.metadata)
        torch.testing.assert_close(a.bank, b.bank, rtol=0, atol=0)


class FakeStateEnv:
    def __init__(self):
        self.num_envs, self.device = 2, "cpu"
        self.tcfg = SimpleNamespace(dt=0.02)
        self.task = SimpleNamespace(
            kin=SimpleNamespace(body_names=list(AMP_BODY_NAMES) + ["torso_link"]),
            body_kin=self.body_kin,
            qmin=torch.full((29,), -1.), qmax=torch.full((29,), 1.),
        )

    @staticmethod
    def body_kin(state):
        n = state["pos"].shape[0]
        return (
            state["pos"][:, None].expand(n, 14, 3),
            state["rot"][:, None].expand(n, 14, 3, 3),
            state["v"][:, None].expand(n, 14, 3),
            state["w"][:, None].expand(n, 14, 3),
        )

    def reset(self):
        self.t = 0
        self.prev_action = torch.zeros(2, 29)
        self.state = {
            "pos": torch.tensor([[0., 0., 0.76]]).repeat(2, 1),
            "rot": torch.eye(3)[None].repeat(2, 1, 1),
            "q": torch.zeros(2, 29), "qd": torch.zeros(2, 29),
            "v": torch.zeros(2, 3), "w": torch.zeros(2, 3),
        }

    def policy_inputs(self):
        return torch.zeros(2, 384)

    def write_state(self, ids, pos, quat, q, v, w, qd):
        from supermimic.rotations import quat_to_mat
        self.state = {"pos": pos, "rot": quat_to_mat(quat), "q": q, "v": v, "w": w, "qd": qd}

    def _finish_resets(self, ids):
        pass

    def step(self, action):
        self.t += 1
        state = {key: value.clone() for key, value in self.state.items()}
        state["pos"][:, 0] += 0.01
        state["qd"][:] = 2.0
        if self.t > 1:
            # Copy zero has already fallen. Neither diagnostics nor errors may
            # consume these subsequent reset-episode samples.
            for value in state.values():
                value[0] = float("nan")
        self.state = state
        self.prev_action = action.clone()
        done = torch.tensor([self.t == 1, False])
        return state, {
            "velocity_heading": torch.zeros(2, 3), "yaw_rate": torch.zeros(2),
            "task_reward": torch.zeros(2), "terminated": done,
            "truncated": torch.zeros(2, dtype=torch.bool), "done": done,
        }


class GaitTests(unittest.TestCase):
    def test_slip_proxies_use_foot_height_and_preserve_sides(self):
        env = FakeStateEnv()
        env.reset()
        original = env.task.body_kin
        feet = [env.task.kin.body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]

        def kinematics(state):
            p, r, v, w = original(state)
            p, v = p.clone(), v.clone()
            p[:, feet, 2] = torch.tensor([[0.04, 0.08], [0.03, 0.04]])
            v[:, feet, 0] = torch.tensor([[0.2, 9.0], [0.4, 0.6]])
            return p, r, v, w

        env.task.body_kin = kinematics
        measures = GaitDiagnostics(env)
        measures.observe(env.state, env.state, env.prev_action, env.prev_action,
                         torch.ones(2, dtype=torch.bool))
        result = measures.report()
        self.assertAlmostEqual(result["low_ankle_planar_speed_mps"]["mean"], 0.4, places=6)
        self.assertEqual(result["low_ankle_planar_speed_mps_samples"], 3)
        self.assertAlmostEqual(result["left_low_ankle_planar_speed_mps"]["mean"], 0.3, places=6)
        self.assertAlmostEqual(result["right_low_ankle_planar_speed_mps"]["mean"], 0.6, places=6)
        self.assertAlmostEqual(result["both_ankles_low_fraction"]["mean"], 0.5)
        self.assertAlmostEqual(result["both_ankles_high_fraction"]["mean"], 0.0)
        self.assertAlmostEqual(result["right_ankle_planar_speed_mps"]["mean"], 4.8, places=6)

    def test_perturbed_resets_are_seeded_bounded_and_distinct(self):
        env = FakeStateEnv()
        env.reset()
        torch.manual_seed(17)
        perturb_standing_reset(env, 1.0)
        expected = {key: value.clone() for key, value in env.state.items()}
        self.assertTrue(bool((env.state["q"].abs() <= 0.02).all()))
        self.assertTrue(bool((env.state["v"].abs() <= 0.05).all()))
        torch.testing.assert_close(env.state["pos"][:, 2], torch.full((2,), 0.76))
        self.assertFalse(torch.equal(env.state["q"][0], env.state["q"][1]))
        env.reset()
        torch.manual_seed(17)
        perturb_standing_reset(env, 1.0)
        for key, value in env.state.items():
            torch.testing.assert_close(value, expected[key], atol=0, rtol=0)

    def test_gait_metrics_exclude_reset_episodes(self):
        env = FakeStateEnv()
        diagnostics = GaitDiagnostics(env)
        result = evaluate_command(lambda obs: torch.full((2, 29), 0.5), env,
                                  (0., 0., 0.), 3, diagnostics=diagnostics)
        gait = result["gait_diagnostics_alive"]
        self.assertAlmostEqual(gait["root_height_m"]["mean"], 0.76, places=6)
        self.assertAlmostEqual(gait["joint_speed_abs_radps"]["mean"], 2.0)
        self.assertAlmostEqual(gait["root_tilt_degrees"]["max"], 0.0)
        self.assertAlmostEqual(gait["net_planar_displacement_m"]["mean"], 0.02, places=6)
        self.assertIsNone(gait["low_ankle_planar_speed_mps"])
        self.assertEqual(gait["low_ankle_planar_speed_mps_samples"], 0)
        self.assertAlmostEqual(gait["both_ankles_high_fraction"]["mean"], 1.0)
        self.assertEqual(result["falls"], 1)
        self.assertEqual(result["survived"], 1)


if __name__ == "__main__":
    unittest.main()
