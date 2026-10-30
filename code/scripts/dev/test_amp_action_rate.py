"""CPU checks for configurable action variation and historical AMP compatibility."""

import dataclasses
from pathlib import Path
import tempfile
import unittest

import torch

from supermimic.amp_networks import CommandActor
from supermimic.amp_task import AMPTask, AMPTaskCfg
from supermimic.amp_train import (
    AMPTrainCfg, CHECKPOINT_TYPE, load_amp_actor, reward_config,
    task_config, validate_cfg, validate_resume_config,
)
from test_amp_curriculum import FakeBase, kinematics, state


class ActionRateTests(unittest.TestCase):
    def test_fivefold_weight_changes_only_action_term(self):
        cfg = AMPTrainCfg()
        old = AMPTask(FakeBase(), task_config(cfg))
        new = AMPTask(FakeBase(), task_config(dataclasses.replace(cfg, action_rate_weight=-.05)))
        action = torch.linspace(-.5, .5, 58).reshape(2, 29)
        previous = torch.zeros_like(action)
        args = (state(), action, previous, torch.zeros(2, 3))
        before, a = old.reward(*args, kin=kinematics(bodies=2))
        after, b = new.reward(*args, kin=kinematics(bodies=2))
        torch.testing.assert_close(b["action_rate"], 5 * a["action_rate"])
        for key in a:
            if key != "action_rate":
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        torch.testing.assert_close(after - before, b["action_rate"] - a["action_rate"], atol=1e-8, rtol=1e-5)

    def test_default_configuration_and_reward_are_exactly_historical(self):
        cfg = AMPTrainCfg()
        explicit = AMPTaskCfg(history=cfg.history, linear_std=cfg.linear_std, angular_std=cfg.angular_std,
                              w_foot_slip=cfg.foot_slip_weight,
                              foot_slip_height_threshold=cfg.foot_slip_height_threshold,
                              foot_slip_width=cfg.foot_slip_width)
        self.assertEqual(dataclasses.asdict(task_config(cfg)), dataclasses.asdict(explicit))
        a = torch.full((2, 29), .17)
        previous = torch.full_like(a, -.1)
        args = (state(), a, previous, torch.zeros(2, 3))
        before, _ = AMPTask(FakeBase(), explicit).reward(*args, kin=kinematics(bodies=2))
        after, _ = AMPTask(FakeBase(), task_config(cfg)).reward(*args, kin=kinematics(bodies=2))
        torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_state_and_action_gradients_remain_finite(self):
        task = AMPTask(FakeBase(), task_config(AMPTrainCfg(action_rate_weight=-.05)))
        s = state()
        s["q"] = torch.full((2, 29), 1.1, requires_grad=True)
        s["v"] = torch.full((2, 3), .2, requires_grad=True)
        action = torch.full((2, 29), .3, requires_grad=True)
        p, r, _, w = kinematics(bodies=2)
        kin = (p, r, s["v"][:, None].expand(-1, 2, -1), w)
        reward, _ = task.reward(s, action, torch.zeros_like(action), torch.zeros(2, 3), kin=kin)
        reward.sum().backward()
        for value in (s["q"], s["v"], action):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_legacy_resume_defaults_guard_and_reward_provenance(self):
        legacy = dataclasses.asdict(AMPTrainCfg())
        legacy.pop("action_rate_weight")
        current = AMPTrainCfg(**legacy)
        self.assertEqual(current.action_rate_weight, -.01)
        self.assertEqual(validate_resume_config(legacy, current), {})
        changed = dataclasses.replace(current, action_rate_weight=-.05)
        with self.assertRaisesRegex(ValueError, "action_rate_weight"):
            validate_resume_config(legacy, changed)
        changed.allow_reward_change = True
        self.assertEqual(validate_resume_config(legacy, changed), {
            "action_rate_weight": {"old": -.01, "new": -.05},
        })
        self.assertEqual(reward_config(legacy)["action_rate_weight"], -.01)
        self.assertEqual(reward_config(changed)["action_rate_weight"], -.05)

    def test_invalid_coefficients_are_rejected_and_zero_is_valid(self):
        for value in (.01, float("inf"), -float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "action_rate_weight"):
                validate_cfg(AMPTrainCfg(action_rate_weight=value))
        validate_cfg(AMPTrainCfg(action_rate_weight=0.))

    def test_old_actor_checkpoint_loads_with_default_and_identical_outputs(self):
        torch.manual_seed(51)
        source = CommandActor().eval()
        obs = torch.randn(3, 384)
        source.obs_norm.update(torch.randn(10, 384))
        legacy = dataclasses.asdict(AMPTrainCfg())
        legacy.pop("action_rate_weight")
        checkpoint = {"checkpoint_type": CHECKPOINT_TYPE, "cfg": legacy, "actor": source.state_dict()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save(checkpoint, path)
            restored, cfg, _ = load_amp_actor(path, device="cpu")
        self.assertEqual(cfg.action_rate_weight, -.01)
        torch.testing.assert_close(restored(obs), source(obs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
