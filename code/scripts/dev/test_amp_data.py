"""CPU data-contract checks for AMP expert conversion; no simulator or checkpoint."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from supermimic.amp_data import prepare_amp_library


def clip(path, frames=3, fps=50, offset=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    pos = np.zeros((frames, 30, 3), dtype=np.float32)
    pos[:, 0, 0] = offset + np.arange(frames)
    pos[:, 0, 2] = 0.75
    quat = np.zeros((frames, 30, 4), dtype=np.float32)
    quat[:, :, 0] = 1
    lin = np.zeros_like(pos)
    angular = np.zeros_like(pos)
    lin[:, 0, 0] = 0.5
    angular[:, 0, 2] = 0.25
    joint = np.arange(frames * 29, dtype=np.float32).reshape(frames, 29) * 0.001
    np.savez(path, fps=np.array([fps]), joint_pos=joint, joint_vel=joint + 1,
             body_pos_w=pos, body_quat_w=quat, body_lin_vel_w=lin, body_ang_vel_w=angular)


class AMPDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "WalkandRun"
        self.output = self.root / "prepared" / "amp.npz"

    def test_single_clip_preserves_state_and_world_angular_velocity(self):
        path = self.source / "walk_one.npz"
        clip(path, offset=7)
        manifest = prepare_amp_library(self.source, self.output)
        self.assertEqual(manifest["clips"], 1)
        with np.load(self.output, allow_pickle=False) as data, np.load(path) as source:
            self.assertEqual(data["qpos"].shape, (3, 36))
            self.assertEqual(data["qvel"].shape, (3, 35))
            self.assertEqual(data["qpos"].dtype, np.float32)
            np.testing.assert_allclose(data["qpos"][:, :3], source["body_pos_w"][:, 0])
            np.testing.assert_allclose(data["qpos"][:, 3:7], source["body_quat_w"][:, 0])
            np.testing.assert_allclose(data["qpos"][:, 7:], source["joint_pos"])
            np.testing.assert_allclose(data["qvel"][:, 3:6], source["body_ang_vel_w"][:, 0])
        saved = json.loads(self.output.with_suffix(".json").read_text())
        self.assertEqual(saved["source_paths"], [str(path.resolve())])
        self.assertEqual(saved["qvel_shape"], [3, 35])

    def test_multiple_clips_keep_independent_boundaries(self):
        clip(self.source / "walk_b.npz", frames=4, offset=100)
        clip(self.source / "walk_a.npz", frames=2, offset=0)
        prepare_amp_library(self.source, self.output)
        with np.load(self.output) as data:
            self.assertEqual(data["names"].tolist(), ["walk_a", "walk_b"])
            self.assertEqual(data["starts"].tolist(), [0, 2])
            self.assertEqual(data["lengths"].tolist(), [2, 4])
            self.assertEqual(data["qpos"][2, 0], 100)

    def test_walking_filter_excludes_jog_and_recovery(self):
        clip(self.source / "walk_one.npz")
        clip(self.source / "idle_turn_one.npz")
        clip(self.source / "jog_one.npz")
        clip(self.source / "Recovery" / "walk_recovery.npz")
        manifest = prepare_amp_library(self.source, self.output)
        self.assertEqual(manifest["names"], ["idle_turn_one", "walk_one"])
        manifest = prepare_amp_library(self.source, self.output, "all", overwrite=True)
        self.assertEqual(manifest["names"], ["idle_turn_one", "jog_one", "walk_one"])

    def test_wrong_fps_rejected_without_writing_output(self):
        clip(self.source / "walk_bad.npz", fps=30)
        with self.assertRaisesRegex(ValueError, "expected 50 Hz"):
            prepare_amp_library(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_invalid_quaternion_and_nonfinite_joint_rejected(self):
        path = self.source / "walk_bad.npz"
        clip(path)
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["body_quat_w"][0, 0] = 0
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "unit wxyz"):
            prepare_amp_library(self.source, self.output)
        clip(path)
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        arrays["joint_pos"][0, 0] = np.nan
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "finite real"):
            prepare_amp_library(self.source, self.output)

    def test_original_expert_cannot_be_output_even_with_overwrite(self):
        path = self.source / "walk_original.npz"
        clip(path)
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, "original expert clip"):
            prepare_amp_library(self.source, path, overwrite=True)
        self.assertEqual(path.read_bytes(), before)

    def test_output_protected_unless_explicit_overwrite(self):
        clip(self.source / "walk_one.npz")
        prepare_amp_library(self.source, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_amp_library(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), before)
        prepare_amp_library(self.source, self.output, overwrite=True)
        self.assertTrue(self.output.with_suffix(".json").exists())


if __name__ == "__main__":
    unittest.main()
