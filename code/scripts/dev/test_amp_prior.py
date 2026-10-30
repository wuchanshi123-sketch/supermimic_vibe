"""CPU regression checks for AMP coordinates, pair boundaries and gradient isolation."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import torch

from supermimic.amp_prior import AMP_ANCHOR_NAME, AMP_BODY_NAMES, AMPFeatureExtractor, AMPPrior, ExpertFeaturePairs


def yaw_matrix(angle):
    c, s = angle.cos(), angle.sin()
    z, o = torch.zeros_like(angle), torch.ones_like(angle)
    return torch.stack((c, -s, z, s, c, z, z, z, o), -1).reshape(*angle.shape, 3, 3)


class DummyTask:
    def __init__(self):
        self.kin = SimpleNamespace(body_names=list(AMP_BODY_NAMES) + [AMP_ANCHOR_NAME])

    def body_kin(self, state):
        return state["kin"]


class AMPFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.extract = AMPFeatureExtractor(DummyTask())
        self.positions = torch.randn(3, 14, 3, dtype=torch.float64)
        self.rotations = yaw_matrix(torch.randn(3, 14, dtype=torch.float64))
        self.velocities = torch.randn(3, 14, 3, dtype=torch.float64)
        self.angular = torch.randn(3, 14, 3, dtype=torch.float64)

    def test_global_translation_and_heading_invariance(self):
        kin = (self.positions, self.rotations, self.velocities, self.angular)
        baseline = self.extract({"kin": kin})
        self.assertEqual(baseline.shape, (3, 195))
        heading = yaw_matrix(torch.tensor(0.73, dtype=torch.float64))
        transform = lambda x: (heading @ x.unsqueeze(-1)).squeeze(-1)
        moved = (
            transform(self.positions) + torch.tensor([3.0, -2.0, 1.0]),
            heading @ self.rotations,
            transform(self.velocities),
            transform(self.angular),
        )
        torch.testing.assert_close(self.extract({}, kin=moved), baseline, atol=1e-12, rtol=1e-12)

    def test_body_features_preserve_input_gradients(self):
        angles = torch.randn(3, 14, dtype=torch.float64, requires_grad=True)
        p = self.positions.clone().requires_grad_()
        v = self.velocities.clone().requires_grad_()
        w = self.angular.clone().requires_grad_()
        feature = self.extract({}, kin=(p, yaw_matrix(angles), v, w))
        weights = torch.arange(1, 196, dtype=torch.float64)
        (feature * weights).sum().backward()
        for tensor in (p, angles, v, w):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0)


class ExpertPairTests(unittest.TestCase):
    def test_pairs_never_cross_clips_and_clips_are_uniform(self):
        torch.manual_seed(8)
        features = torch.arange(12, dtype=torch.float32).unsqueeze(-1).repeat(1, 2)
        pairs = ExpertFeaturePairs(features, [0, 5], [5, 7])
        indices = pairs.sample_indices(10000)
        self.assertTrue(((indices < 4) | ((indices >= 5) & (indices < 11))).all())
        fraction_first_clip = float((indices < 4).float().mean())
        self.assertTrue(0.45 < fraction_first_clip < 0.55)
        current, following = pairs.sample(256)
        torch.testing.assert_close(following - current, torch.ones_like(current))
        with self.assertRaises(ValueError):
            ExpertFeaturePairs(features, [0], [1])


class AMPTrainingTests(unittest.TestCase):
    def test_frozen_prior_keeps_reward_input_gradient_and_statistics(self):
        torch.manual_seed(9)
        prior = AMPPrior(hidden=(32, 16), device="cpu")
        prior.normalizer.update(torch.randn(64, 195))
        prior.requires_grad_(False)
        before = {k: v.clone() for k, v in prior.state_dict().items()}
        f0 = (0.1 * torch.randn(8, 195)).requires_grad_()
        f1 = (0.1 * torch.randn(8, 195)).requires_grad_()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            style, score = prior.reward(f0, f1)
        self.assertEqual(style.dtype, torch.float32)
        self.assertTrue(((style >= 0) & (style <= 0.1)).all())
        style.mean().backward()
        for value in (f0, f1):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in prior.parameters()))
        for key, value in prior.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)
        self.assertTrue(torch.isfinite(score).all())

    def test_update_trains_discriminator_without_input_gradients(self):
        torch.manual_seed(10)
        prior = AMPPrior(hidden=(32, 16), device="cpu")
        prior.requires_grad_(False)
        before = [p.detach().clone() for p in prior.discriminator.parameters()]
        samples = [torch.randn(16, 195).add(i * 0.5).requires_grad_() for i in range(4)]
        expected_mean = torch.cat([x.detach() for x in samples]).mean(0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            metrics = prior.update(*samples)
        self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.assertTrue(all(x.grad is None for x in samples))
        self.assertTrue(all(p.grad is None for p in prior.discriminator.parameters()))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, prior.discriminator.parameters())))
        torch.testing.assert_close(prior.normalizer.mean, expected_mean)
        clone = AMPPrior(hidden=(32, 16), device="cpu")
        clone.load_state_dict(prior.state_dict())
        torch.testing.assert_close(clone.reward(*samples[:2])[0], prior.reward(*samples[:2])[0])

    def test_gradient_penalty_is_zero_centered(self):
        prior = AMPPrior(feature_dim=2, hidden=(), device="cpu", gp_weight=10.0)
        with torch.no_grad():
            prior.discriminator[-1].weight.fill_(0.25)
            prior.discriminator[-1].bias.zero_()
        samples = [torch.randn(8, 2) for _ in range(4)]
        metrics = prior.update(*samples)
        self.assertAlmostEqual(metrics["gradient_penalty"], 10 * 4 * 0.25**2, places=5)


if __name__ == "__main__":
    unittest.main()
