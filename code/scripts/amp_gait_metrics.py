"""Inspect recorded AMP ankle trajectories without certifying walking quality.

Examples:
  .venv/bin/python code/scripts/amp_gait_metrics.py runs/amp_supervision/video_NAME
  .venv/bin/python code/scripts/amp_gait_metrics.py 'runs/amp_supervision/video_*'

All quantities refer to ankle-roll body origins, not sole clearance or measured
ground contact. The first two seconds are excluded by default. A periodic ankle
signal can support an alternating-step observation; visual review, tracking and
robustness evaluation are still required before declaring a useful walking gait.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path

import numpy as np


VERSION = 1
HEIGHT_BAND = (0.25, 8.0)
MIN_PEAK_ENERGY = 0.35


def _detrend(value):
    value = np.asarray(value, dtype=np.float64)
    centered = value - value.mean(axis=0)
    axis = np.arange(len(value), dtype=np.float64)
    axis -= axis.mean()
    denominator = np.dot(axis, axis)
    if denominator:
        centered -= axis.reshape((-1,) + (1,) * (value.ndim - 1)) * np.tensordot(axis, centered, axes=(0, 0)) / denominator
    return centered


def _spectrum(value, dt):
    centered = _detrend(value)
    window = np.hanning(len(value)).reshape((-1,) + (1,) * (centered.ndim - 1))
    transform = np.fft.rfft(centered * window, axis=0)
    frequencies = np.fft.rfftfreq(len(value), dt)
    return frequencies, transform, np.abs(transform) ** 2


def _height_spectrum(height, dt):
    frequencies, transform, power = _spectrum(height, dt)
    band = (frequencies >= HEIGHT_BAND[0]) & (frequencies <= HEIGHT_BAND[1])
    ids = np.flatnonzero(band)
    if not len(ids) or np.max(np.abs(_detrend(height))) < 1e-6:
        return {"frequency_hz": None, "peak_energy_fraction": 0.0, "valid": False}, transform, power
    peak = ids[np.argmax(power[ids])]
    neighborhood = ids[np.abs(ids - peak) <= 1]
    total = power[ids].sum()
    fraction = float(power[neighborhood].sum() / total) if total > 1e-18 else 0.0
    span = float(np.percentile(height, 95) - np.percentile(height, 5))
    return {
        "frequency_hz": float(frequencies[peak]),
        "peak_energy_fraction": fraction,
        "valid": bool(span > 0.002 and fraction >= MIN_PEAK_ENERGY
                      and len(height) * dt * frequencies[peak] >= 1.8),
    }, transform, power


def _high_frequency_fraction(value, dt, cutoff=5.0):
    if value is None or len(value) < 16 or not np.isfinite(value).all():
        return None
    frequencies, _, power = _spectrum(value, dt)
    denominator = power[frequencies > 0].sum()
    if denominator < 1e-18:
        return None
    return float(power[frequencies > cutoff].sum() / denominator)


def _finite_command(data):
    command = np.asarray(data.get("command", []), dtype=np.float64)
    return command.tolist() if command.shape == (3,) and np.isfinite(command).all() else None


def inspect_arrays(data, skip_seconds=2.0):
    """Measure one trace; invalid/missing fields produce JSON-safe diagnostics."""
    result = {
        "command": _finite_command(data), "status": "unavailable", "flags": [],
        "issues": [], "transient_excluded_seconds": float(skip_seconds),
        "alternating_steps_observed": False, "quality_certified": False,
    }
    if "ankle_pos_w" not in data:
        result.update(status="missing_required_fields", issues=["ankle_pos_w is required"])
        return result
    try:
        positions = np.asarray(data["ankle_pos_w"], dtype=np.float64)
        if positions.ndim != 3 or positions.shape[1:] != (2, 3):
            raise ValueError("ankle_pos_w must have shape (T, 2, 3)")
        total = len(positions)
        if "time_s" in data:
            times = np.asarray(data["time_s"], dtype=np.float64)
            if times.shape != (total,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("time_s must be a finite strictly increasing (T,) array")
            dt = float(np.median(np.diff(times))) if total > 1 else float(np.asarray(data.get("dt", 0.02)))
            if total > 1 and not np.allclose(np.diff(times), dt, rtol=0.01, atol=1e-7):
                raise ValueError("spectral inspection requires uniformly sampled time_s")
        elif "dt" in data:
            dt = float(np.asarray(data["dt"]))
            times = (np.arange(total) + 1) * dt
        else:
            result.update(status="missing_required_fields", issues=["time_s or dt is required"])
            return result
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("sampling dt must be finite and positive")
    except (TypeError, ValueError) as exc:
        result.update(status="invalid_data", issues=[str(exc)])
        return result
    selected = times > skip_seconds
    result.update(total_samples=total, analyzed_samples=int(selected.sum()), dt_seconds=dt,
                  analyzed_seconds=float(selected.sum() * dt))
    if selected.sum() < max(16, int(np.ceil(2.0 / dt))):
        result.update(status="insufficient_post_transient_samples", issues=["at least 2 s of post-transient data is required"])
        return result
    positions = positions[selected]
    invalid = ~np.isfinite(positions).all(axis=(1, 2))
    if invalid.any():
        result.update(status="invalid_data", nonfinite_position_samples=int(invalid.sum()),
                      issues=["non-finite ankle positions; no interpolation or gait classification performed"])
        return result
    names = np.asarray(data.get("ankle_names", ["left", "right"])).tolist()
    if names == ["right", "left"]:
        positions = positions[:, ::-1]
        order = [1, 0]
    else:
        order = [0, 1]
        if names != ["left", "right"]:
            result["issues"].append("unrecognized ankle_names; assumed columns are left/right")
    velocities = None
    if "ankle_vel_w" in data:
        candidate = np.asarray(data["ankle_vel_w"], dtype=np.float64)
        if candidate.shape == (total, 2, 3) and np.isfinite(candidate[selected]).all():
            velocities = candidate[selected][:, order]
        else:
            result["issues"].append("invalid ankle_vel_w; low-ankle speeds derived from position differences")
    if velocities is None:
        velocities = np.gradient(positions, dt, axis=0)
        velocity_source = "finite difference of ankle_pos_w"
    else:
        velocity_source = "recorded ankle_vel_w"
    height = positions[:, :, 2]
    low = height < 0.05
    planar_speed = np.linalg.norm(velocities[:, :, :2], axis=-1)
    feet = {}
    transforms, powers = [], []
    for index, name in enumerate(("left", "right")):
        spectral, transform, power = _height_spectrum(height[:, index], dt)
        transforms.append(transform)
        powers.append(power)
        feet[name] = {
            "height_p05_m": float(np.percentile(height[:, index], 5)),
            "height_p95_m": float(np.percentile(height[:, index], 95)),
            "height_span_p95_p05_m": float(np.percentile(height[:, index], 95) - np.percentile(height[:, index], 5)),
            "low_ankle_fraction": float(low[:, index].mean()),
            "mean_low_ankle_planar_speed_mps": float(planar_speed[low[:, index], index].mean()) if low[:, index].any() else None,
            "dominant_height_frequency_hz": spectral["frequency_hz"],
            "height_peak_energy_fraction": spectral["peak_energy_fraction"],
            "height_periodicity_valid": spectral["valid"],
        }
    frequencies = np.fft.rfftfreq(len(height), dt)
    band = (frequencies >= HEIGHT_BAND[0]) & (frequencies <= HEIGHT_BAND[1])
    ids = np.flatnonzero(band)
    shared_frequency, phase, shared_energy = None, None, []
    if len(ids) and all(foot["height_periodicity_valid"] for foot in feet.values()):
        total_power = powers[0] + powers[1]
        peak = ids[np.argmax(total_power[ids])]
        shared_frequency = float(frequencies[peak])
        phase = float(np.degrees(np.angle(transforms[0][peak] * np.conj(transforms[1][peak]))))
        neighborhood = ids[np.abs(ids - peak) <= 1]
        shared_energy = [float(power[neighborhood].sum() / power[ids].sum()) for power in powers]
    spans = [feet[name]["height_span_p95_p05_m"] for name in ("left", "right")]
    both_low = float(low.all(axis=1).mean())
    lack_clearance = all(foot["height_p95_m"] < 0.05 for foot in feet.values())
    shallow_fast = all(span < 0.02 for span in spans) and shared_frequency is not None and shared_frequency > 2.5
    persistently_low = both_low > 0.9 and lack_clearance
    if shallow_fast or persistently_low:
        result["flags"].append({
            "code": "shallow_high_frequency_shuffle",
            "reasons": (["both ankle height spans < 0.02 m and shared frequency > 2.5 Hz"] if shallow_fast else [])
                       + (["both ankles below 0.05 m for >90% of samples with no p95 ankle clearance above 0.05 m"] if persistently_low else []),
        })
    alternating = bool(all(span > 0.03 for span in spans)
                       and shared_frequency is not None and 0.25 <= shared_frequency <= 2.0
                       and phase is not None and abs(phase) > 120
                       and all(fraction >= MIN_PEAK_ENERGY for fraction in shared_energy))
    actions = None
    if "action" in data:
        candidate = np.asarray(data["action"], dtype=np.float64)
        if candidate.ndim == 2 and len(candidate) == total and np.isfinite(candidate[selected]).all():
            actions = candidate[selected]
        else:
            result["issues"].append("invalid action field; action spectrum omitted")
    result.update(
        status="measured", feet=feet, both_low_ankle_fraction=both_low,
        low_ankle_height_threshold_m=0.05, ankle_velocity_source=velocity_source,
        shared_dominant_height_frequency_hz=shared_frequency,
        left_right_height_phase_degrees=phase,
        shared_peak_energy_fractions=shared_energy,
        alternating_steps_observed=alternating,
        high_frequency_energy={
            "cutoff_hz": 5.0,
            "ankle_height_fraction": _high_frequency_fraction(height, dt),
            "ankle_position_fraction": _high_frequency_fraction(positions, dt),
            "ankle_velocity_fraction": _high_frequency_fraction(velocities, dt),
            "action_fraction": _high_frequency_fraction(actions, dt),
        },
    )
    return result


def inspect_trace(path, skip_seconds=2.0):
    try:
        with np.load(path, allow_pickle=False) as data:
            result = inspect_arrays(data, skip_seconds)
    except (OSError, TypeError, ValueError) as exc:
        result = {"status": "trace_read_failed", "issues": [str(exc)], "flags": [],
                  "alternating_steps_observed": False, "quality_certified": False}
    result["trace"] = Path(path).name
    return result


def analyze_video_directory(directory, skip_seconds=2.0):
    directory = Path(directory)
    traces = sorted(directory.glob("command_*_trace.npz"))
    results = [inspect_trace(path, skip_seconds) for path in traces]
    report = {
        "schema": "amp_gait_trace_metrics_v1", "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory.resolve()), "transient_excluded_seconds": skip_seconds,
        "protocol": {
            "measurement": "ankle-roll body origins, not sole clearance or measured contact",
            "height_frequency_band_hz": list(HEIGHT_BAND),
            "minimum_peak_energy_fraction": MIN_PEAK_ENERGY,
            "alternation_rule": "both p95-p05 height spans >0.03 m; shared frequency 0.25–2 Hz; absolute L/R phase >120 degrees; valid periodic energy in both ankles",
            "interpretation": "Ankle height frequencies describe per-foot oscillations, not automatically steps per second. Flags and alternation observations do not certify converged gait or user quality; inspect videos and tracking/robustness results.",
        },
        "commands": results, "quality_certified": False,
        "requires_visual_review": True,
    }
    summary = directory / "summary.json"
    if summary.exists():
        try:
            metadata = json.loads(summary.read_text())
            report.update(checkpoint=metadata.get("checkpoint"), iteration=metadata.get("iteration"))
        except (OSError, json.JSONDecodeError):
            pass
    if not traces:
        report["issues"] = ["no command_*_trace.npz files found"]
    temporary = directory / "gait_summary.tmp.json"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "gait_summary.json")
    return report


def compact_summary(report):
    commands = []
    for result in report["commands"]:
        feet = result.get("feet", {})
        commands.append({
            "command": result.get("command"), "status": result["status"],
            "height_spans_m": [feet.get(side, {}).get("height_span_p95_p05_m") for side in ("left", "right")],
            "height_frequencies_hz": [feet.get(side, {}).get("dominant_height_frequency_hz") for side in ("left", "right")],
            "low_ankle_planar_speeds_mps": [feet.get(side, {}).get("mean_low_ankle_planar_speed_mps") for side in ("left", "right")],
            "shared_height_frequency_hz": result.get("shared_dominant_height_frequency_hz"),
            "left_right_phase_degrees": result.get("left_right_height_phase_degrees"),
            "both_low_fraction": result.get("both_low_ankle_fraction"),
            "high_frequency_energy": result.get("high_frequency_energy"),
            "alternating_steps_observed": result["alternating_steps_observed"],
            "flags": [flag["code"] for flag in result["flags"]],
            "issues": result.get("issues", []),
        })
    return {"directory": report["directory"], "iteration": report.get("iteration"),
            "commands": commands, "quality_certified": False, "requires_visual_review": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="video directories, traces, or quoted glob patterns")
    parser.add_argument("--skip-seconds", type=float, default=2.0)
    parser.add_argument("--recursive", action="store_true", help="find all nested trace directories")
    args = parser.parse_args()
    if not np.isfinite(args.skip_seconds) or args.skip_seconds < 0:
        parser.error("skip-seconds must be finite and nonnegative")
    directories = set()
    for item in args.inputs:
        for name in glob.glob(item):
            path = Path(name)
            if path.is_file() and path.name.endswith("_trace.npz"):
                directories.add(path.parent)
            elif path.is_dir():
                if list(path.glob("command_*_trace.npz")):
                    directories.add(path)
                elif args.recursive:
                    directories.update(trace.parent for trace in path.rglob("command_*_trace.npz"))
    if not directories:
        parser.error("no command_*_trace.npz files found in inputs")
    for directory in sorted(directories):
        report = analyze_video_directory(directory, args.skip_seconds)
        print(json.dumps(compact_summary(report), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
