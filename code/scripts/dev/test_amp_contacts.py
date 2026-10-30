"""Actual-contact metric checks, independent of GPU environment rollout."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_amp_contacts import ContactRecorder, point_velocity, tangent_speed


class ContactTests(unittest.TestCase):
    def test_point_velocity_includes_rotation_around_reference(self):
        cvel = torch.tensor([[0., 0., 2., 1., 0., 0.]])
        point = torch.tensor([[3., 4., 0.]])
        origin = torch.tensor([[2., 4., 0.]])
        torch.testing.assert_close(point_velocity(cvel, point, origin), torch.tensor([[1., 2., 0.]]))

    def test_tangent_projection_excludes_normal_motion(self):
        torch.testing.assert_close(tangent_speed(torch.tensor([[3., 4., 9.]]), torch.tensor([[0., 0., 1.]])), torch.tensor([5.]))

    def test_common_rigid_motion_has_zero_relative_point_slip(self):
        p = torch.tensor([[1., 2., 0.]])
        a = point_velocity(torch.tensor([[0., 0., 1., 0., 0., 0.]]), p, torch.zeros_like(p))
        b = point_velocity(torch.tensor([[0., 0., 1., -1., 0., 0.]]), p, torch.tensor([[0., 1., 0.]]))
        torch.testing.assert_close(a, b)

    def test_aggregate_excludes_reset_samples_and_keeps_force_weighting(self):
        r = ContactRecorder.__new__(ContactRecorder)
        r.n, r.dt, r.point_threshold, r.foot_threshold = 2, .005, 1., 5.
        r.records = []
        for step in range(3):
            row = np.zeros((2, 2, 8), np.float32)
            for world, force, slip in ((0, 100., .1), (1, 200., .3)):
                row[world, 0, :6] = [force, 0., force, force * slip, slip, 1.]
            active = np.array([step == 0, True])
            torque = np.array([2. if step == 0 else 9999., 3.])
            r.records.append((row, active, torque))
        r.point_samples = [np.array([[0, 0, 0., 100., .1, .001],
            [1, 0, 0., 200., .3, .002], [1, 0, .005, 200., .3, .002],
            [1, 0, .01, 200., .3, .002]], np.float32)]
        with tempfile.TemporaryDirectory() as folder:
            out = r.finish(Path(folder) / 'trace.npz', skip_seconds=.005)
        full = out['full_episode']
        self.assertAlmostEqual(full['force_weighted_slip_mps'], 190 / 700, places=6)
        self.assertAlmostEqual(full['normal_impulse_per_copy_Ns'], 1.75)
        self.assertEqual(full['max_abs_joint_actuator_torque_nm']['max'], 3.)
        self.assertAlmostEqual(full['trials'][0]['feet'][0]['observed_seconds'], .005)
        self.assertIsNone(out['after_transient']['trials'][0]['feet'][0]['stance_fraction'])
        self.assertEqual(out['after_transient']['loaded_contact_point_samples'], 2)


if __name__ == '__main__':
    unittest.main()
