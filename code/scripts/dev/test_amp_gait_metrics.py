"""CPU synthetic trace checks for AMP gait observations, not policy quality."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "amp_gait_metrics.py"
SPEC = importlib.util.spec_from_file_location("amp_gait_metrics", SCRIPT)
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def trace(frequency=1.0, amplitude=0.04, baseline=0.075, seconds=12):
    dt = 0.02
    time = (np.arange(round(seconds / dt)) + 1) * dt
    wave = np.sin(2 * np.pi * frequency * time)
    positions = np.zeros((len(time), 2, 3))
    positions[:, 0, 2] = baseline + amplitude * wave
    positions[:, 1, 2] = baseline - amplitude * wave
    return {"ankle_pos_w": positions, "time_s": time, "dt": np.asarray(dt),
            "command": np.asarray([0.3, 0, 0]), "ankle_names": np.asarray(["left", "right"])}


class GaitMetricsTests(unittest.TestCase):
    def test_opposite_phase_one_hz_observes_alternation(self):
        measured = METRICS.inspect_arrays(trace())
        self.assertEqual(measured["status"], "measured")
        self.assertTrue(measured["alternating_steps_observed"])
        self.assertFalse(measured["quality_certified"])
        self.assertAlmostEqual(measured["shared_dominant_height_frequency_hz"], 1.0, delta=0.11)
        self.assertGreater(abs(measured["left_right_height_phase_degrees"]), 170)
        self.assertGreater(measured["feet"]["left"]["height_span_p95_p05_m"], .07)
        self.assertFalse(measured["flags"])

    def test_slow_point_three_hz_alternation_is_supported(self):
        measured = METRICS.inspect_arrays(trace(frequency=.3, seconds=16))
        self.assertTrue(measured["alternating_steps_observed"])
        self.assertAlmostEqual(measured["shared_dominant_height_frequency_hz"], .3, delta=.08)

    def test_five_hz_five_mm_shuffle_is_flagged(self):
        measured = METRICS.inspect_arrays(trace(frequency=5, amplitude=.005, baseline=.04))
        self.assertFalse(measured["alternating_steps_observed"])
        self.assertIn("shallow_high_frequency_shuffle", [flag["code"] for flag in measured["flags"]])
        self.assertAlmostEqual(measured["shared_dominant_height_frequency_hz"], 5.0, delta=.11)
        self.assertLess(measured["feet"]["right"]["height_span_p95_p05_m"], .02)

    def test_static_trace_has_no_valid_frequency_or_alternation(self):
        measured = METRICS.inspect_arrays(trace(amplitude=0, baseline=.035))
        self.assertEqual(measured["status"], "measured")
        self.assertIsNone(measured["shared_dominant_height_frequency_hz"])
        self.assertFalse(measured["feet"]["left"]["height_periodicity_valid"])
        self.assertFalse(measured["alternating_steps_observed"])
        self.assertEqual(measured["both_low_ankle_fraction"], 1)
        json.dumps(measured, allow_nan=False)

    def test_nonfinite_missing_and_short_fields_are_safe(self):
        value = trace()
        value["ankle_pos_w"][150, 0, 2] = np.nan
        measured = METRICS.inspect_arrays(value)
        self.assertEqual(measured["status"], "invalid_data")
        self.assertFalse(measured["alternating_steps_observed"])
        json.dumps(measured, allow_nan=False)
        self.assertEqual(METRICS.inspect_arrays({})["status"], "missing_required_fields")
        self.assertEqual(METRICS.inspect_arrays(trace(seconds=2.5))["status"], "insufficient_post_transient_samples")

    def test_optional_velocity_and_action_spectra(self):
        value = trace()
        time = value["time_s"]
        value["action"] = np.sin(2 * np.pi * 7 * time)[:, None]
        value["ankle_vel_w"] = np.zeros_like(value["ankle_pos_w"])
        value["ankle_vel_w"][:, :, 0] = .2
        measured = METRICS.inspect_arrays(value)
        self.assertGreater(measured["high_frequency_energy"]["action_fraction"], .99)
        self.assertEqual(measured["ankle_velocity_source"], "recorded ankle_vel_w")
        self.assertAlmostEqual(measured["feet"]["left"]["mean_low_ankle_planar_speed_mps"], .2)

    def test_transient_is_excluded_and_directory_report_is_written(self):
        value = trace()
        value["ankle_pos_w"][:100] = np.nan
        self.assertTrue(METRICS.inspect_arrays(value)["alternating_steps_observed"])
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            np.savez_compressed(directory / "command_00_trace.npz", **value)
            (directory / "summary.json").write_text(json.dumps({"iteration": 26000, "checkpoint": "example.pt"}))
            report = METRICS.analyze_video_directory(directory)
            self.assertEqual(report["iteration"], 26000)
            saved = json.loads((directory / "gait_summary.json").read_text())
            self.assertTrue(saved["commands"][0]["alternating_steps_observed"])
            json.dumps(METRICS.compact_summary(report), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
