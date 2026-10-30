"""Read-only AMP training summaries and curves; never controls a learner process.

Example:
  .venv/bin/python code/scripts/monitor_amp_training.py \
    runs/amp_walk_24gb_train --out runs/amp_supervision/monitor

Window sizes count log records, not iterations. Flags are investigation prompts,
not convergence decisions: changing discriminator rewards and imagined returns
cannot establish walking quality without fixed-command evaluation and video.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


METRICS = (
    "episode_seconds", "episode_fall_fraction", "episode_return",
    "completed_episodes", "linear_error", "yaw_error", "sim_task_reward",
    "sim_style_reward", "sim_reward", "wm_err1", "wm_errk", "wm_loss",
    "imagined_survival", "imagined_return", "imagined_task_reward",
    "imagined_style_reward", "critic_loss", "critic_value", "actor_grad",
    "wm_grad", "discriminator_loss", "policy_score", "expert_score",
    "gradient_penalty", "mem_gb", "iter_s", "lr",
)
EPISODE_METRICS = {"episode_seconds", "episode_fall_fraction", "episode_return"}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_records(path):
    """Keep the last occurrence of each iteration, including resumed snapshots.

    A partially written final line is ignored and reported. Malformed complete
    lines are also reported instead of hiding all subsequent valid measurements.
    """
    text = path.read_text()
    records = {}
    issues = []
    duplicates = 0
    lines = text.splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or not finite_number(record.get("it")):
                raise ValueError("missing numeric iteration")
            iteration = int(record["it"])
            if iteration != record["it"] or iteration < 0:
                raise ValueError("iteration is not a nonnegative integer")
        except (json.JSONDecodeError, ValueError) as exc:
            issues.append({"line": number, "message": str(exc),
                           "possibly_incomplete": number == len(lines) and not text.endswith("\n")})
            continue
        if iteration in records:
            duplicates += 1
        records[iteration] = record
    return sorted(records.values(), key=lambda row: row["it"]), issues, duplicates


def means(rows):
    result = {}
    for key in METRICS:
        values = []
        weights = []
        for row in rows:
            value = row.get(key)
            if not finite_number(value):
                continue
            # Episode means must be weighted by the number of completed
            # episodes. Logs with zero completions carry no episode evidence.
            weight = row.get("completed_episodes", 1) if key in EPISODE_METRICS else 1
            if not finite_number(weight) or weight <= 0:
                continue
            values.append(value)
            weights.append(weight)
        if weights:
            result[key] = sum(v * w for v, w in zip(values, weights)) / sum(weights)
    return result


def comparison(rows, width):
    current = rows[-width:]
    previous = rows[-2 * width:-width]
    now, before = means(current), means(previous)
    changes = {}
    for key in now.keys() & before.keys():
        delta = now[key] - before[key]
        changes[key] = {"absolute": delta,
                        "relative": delta / abs(before[key]) if abs(before[key]) > 1e-12 else None}
    return {
        "requested_records": width, "current_records": len(current),
        "previous_records": len(previous),
        "full_comparison": len(current) == len(previous) == width,
        "current_iteration_range": [current[0]["it"], current[-1]["it"]] if current else None,
        "previous_iteration_range": [previous[0]["it"], previous[-1]["it"]] if previous else None,
        "current_mean": now, "previous_mean": before, "change": changes,
        "actor_updates_increasing": bool(previous and current
                                         and current[-1].get("actor_updates", 0) > previous[-1].get("actor_updates", 0)),
    }


def summarize(path, windows, stale_seconds):
    rows, parse_issues, duplicates = read_records(path)
    age = max(0.0, time.time() - path.stat().st_mtime)
    report = {
        "path": str(path.resolve()), "name": path.parent.name,
        "records": len(rows), "duplicate_iterations_replaced": duplicates,
        "file_age_seconds": age, "parse_issues": parse_issues,
        "latest": None, "windows": {}, "flags": [], "status": "insufficient_data",
    }
    if not rows:
        report["flags"].append({"severity": "error", "code": "no_records", "message": "No valid log records."})
        return report, rows
    latest = rows[-1]
    # JSON reports remain standards compliant even if the source contains NaN.
    report["latest"] = {key: value for key, value in latest.items()
                        if not isinstance(value, (int, float)) or math.isfinite(value)}
    bad = [{"it": row["it"], "keys": [key for key, value in row.items()
                                      if isinstance(value, (int, float)) and not math.isfinite(value)]}
           for row in rows]
    bad = [entry for entry in bad if entry["keys"]]
    if bad:
        report["flags"].append({"severity": "error" if bad[-1]["it"] == latest["it"] else "warning",
                                "code": "nonfinite_values", "records": bad,
                                "message": "Non-finite metrics require investigation; excluded from averages."})
    if parse_issues:
        report["flags"].append({"severity": "warning", "code": "unreadable_lines",
                                "message": f"Skipped {len(parse_issues)} unreadable lines; final line may still be in flight."})
    if age > stale_seconds:
        report["flags"].append({"severity": "info", "code": "stale_log",
                                "message": f"No log write for {age:.0f}s. Learner may be stopped, evaluating, or compiling."})
    for width in windows:
        report["windows"][str(width)] = comparison(rows, width)

    # Actor-active rows avoid classifying the deliberate warm-up as a plateau.
    active = [row for row in rows if row.get("actor_updates", 0) > 0]
    stable_window = max(windows)
    if len(active) >= 2 * stable_window:
        long = comparison(active, stable_window)
        now, before, changes = long["current_mean"], long["previous_mean"], long["change"]
        survival = changes.get("episode_seconds", {}).get("relative")
        tracking = changes.get("linear_error", {}).get("relative")
        task = changes.get("sim_task_reward", {}).get("relative")
        if (long["actor_updates_increasing"] and survival is not None
                and tracking is not None and task is not None
                and survival < 0.1 and tracking > -0.05 and task < 0.1
                and now.get("episode_fall_fraction", 1) > 0.2):
            report["flags"].append({"severity": "warning", "code": "possible_plateau",
                                    "message": f"Across two {stable_window}-record actor-active windows, survival, tracking and task reward show little progress. Check fixed-command evaluations before intervening."})

    short = report["windows"][str(min(windows))]
    if short["full_comparison"]:
        changes = short["change"]
        def relative(key):
            return changes.get(key, {}).get("relative")
        regression = []
        for key, threshold, direction in (("episode_seconds", -0.4, "below"),
                                          ("linear_error", 0.4, "above"),
                                          ("wm_errk", 2.0, "above")):
            value = relative(key)
            if value is not None and ((direction == "below" and value < threshold)
                                      or (direction == "above" and value > threshold)):
                regression.append(key)
        if len(regression) >= 2 and short["actor_updates_increasing"]:
            report["flags"].append({"severity": "warning", "code": "suspected_regression",
                                    "metrics": regression,
                                    "message": "Several short-window metrics worsened sharply. Confirm with simulation evaluation; policy changes can increase WM error."})
        survival, tracking = relative("episode_seconds"), relative("linear_error")
        report["status"] = ("improving" if survival is not None and survival > 0.1
                            and (tracking is None or tracking < 0.1) else "mixed")
    codes = {flag["code"] for flag in report["flags"]}
    for code in ("possible_plateau", "suspected_regression", "nonfinite_values"):
        if code in codes:
            report["status"] = code
    return report, rows


def finite_or_none(value):
    return float(value) if finite_number(value) else None


def evaluation_command_summary(result):
    """Summarize a fixed-command result without relying on its learned critic/D.

    A norm of each trial's *time-mean* velocity detects stationary nontracking;
    it is not an instantaneous speed or path-length measurement. Gait/visual
    checks remain necessary to distinguish smooth walking from oscillation.
    """
    command = result.get("command", [])
    if len(command) != 3 or not all(finite_number(value) for value in command):
        raise ValueError("evaluation command must contain three finite values")
    weighted_velocities = []
    weighted_speeds = []
    for trial in result.get("trials", []):
        velocity = trial.get("mean_velocity_heading_mps_alive", [])
        weight = trial.get("observed_steps", 1)
        if (len(velocity) < 2 or not all(finite_number(value) for value in velocity[:2])
                or not finite_number(weight) or weight <= 0):
            continue
        weighted_velocities.append((velocity[:2], weight))
        weighted_speeds.append((math.hypot(*velocity[:2]), weight))
    if weighted_velocities:
        denominator = sum(weight for _, weight in weighted_velocities)
        velocity = [sum(vector[j] * weight for vector, weight in weighted_velocities) / denominator
                    for j in range(2)]
        speed = sum(value * weight for value, weight in weighted_speeds) / denominator
    else:
        direct = result.get("mean_velocity_heading_mps_alive", [])
        velocity = ([float(value) for value in direct[:2]]
                    if len(direct) >= 2 and all(finite_number(value) for value in direct[:2]) else None)
        speed = math.hypot(*velocity) if velocity is not None else None
    gait = result.get("gait_diagnostics_alive", {})
    gait_means = {key: finite_or_none(gait.get(key, {}).get("mean")) for key in (
        "root_height_m", "torso_tilt_degrees", "ankle_vertical_speed_mps",
        "ankle_planar_speed_mps", "joint_speed_abs_radps", "net_planar_displacement_m",
    )}
    survival_rate = finite_or_none(result.get("survival_rate"))
    moving = math.hypot(*command[:2]) > 0.2
    stationary = bool(moving and speed is not None and speed < 0.1
                      and survival_rate is not None and survival_rate >= 0.8)
    return {
        "command": list(command), "copies": result.get("copies"),
        "survival_rate": survival_rate,
        "mean_survival_seconds": finite_or_none(result.get("mean_survival_seconds")),
        "planar_rmse_mps_alive": finite_or_none(result.get("planar_rmse_mps_alive")),
        "mean_abs_yaw_error_radps_alive": finite_or_none(result.get("mean_abs_yaw_error_radps_alive")),
        "mean_velocity_heading_xy_mps_alive": velocity,
        "mean_planar_speed_from_trial_means_mps": speed,
        "moving_planar_command": moving,
        "stationary_nontracking_candidate": stationary,
        "fixed_expert_pair_distance_mean": finite_or_none(gait.get("fixed_expert_pair_distance", {}).get("mean")),
        "gait_means_alive": gait_means,
    }


def evaluation_protocol(data):
    """Only compare identical scopes; individual commands are matched separately."""
    reference = data.get("fixed_expert_style_reference", {})
    return {
        "checkpoint_run": str(Path(data["checkpoint"]).parent),
        "initialization": data.get("initialization"),
        "initial_state_perturbations": data.get("initial_state_perturbations"),
        "horizon_seconds": data.get("horizon_seconds", data.get("requested_seconds")),
        "num_envs_per_command": data.get("num_envs_per_command"),
        "friction": data.get("friction"), "observation_noise": data.get("observation_noise"),
        "reset_noise": data.get("reset_noise"), "randomization": data.get("randomization"),
        "deterministic_policy": data.get("deterministic_policy"),
        "same_initial_state_copies": data.get("same_initial_state_copies"),
        "expert_library_sha256": data.get("expert_library_sha256"),
        "style_reference": {key: reference.get(key) for key in (
            "version", "feature_dimension", "reference_bank_pairs", "calibration_pairs",
            "normalization_std_floor", "split", "metric",
        )},
    }


def independent_evaluations(directory):
    """Read evaluation snapshots, excluding open-loop dynamics diagnostic files."""
    snapshots = {}
    skipped_dynamics = 0
    issues = []
    files = sorted(directory.glob("eval_*.json"), key=lambda path: (path.stat().st_mtime, path.name))
    for path in files:
        if "dynamics" in path.stem:
            skipped_dynamics += 1
            continue
        try:
            data = json.loads(path.read_text())
            iteration = data.get("checkpoint_iteration")
            if not finite_number(iteration) or not data.get("commands"):
                raise ValueError("not a checkpoint command evaluation")
            if any("horizons" in result or "survival_rate" not in result for result in data["commands"]):
                skipped_dynamics += 1
                continue
            protocol = evaluation_protocol(data)
            protocol_key = json.dumps(protocol, sort_keys=True, allow_nan=False)
            protocol_id = hashlib.sha256(protocol_key.encode()).hexdigest()[:12]
            commands = [evaluation_command_summary(result) for result in data["commands"]]
            # Deduplicate commands so repeated copies of one command cannot
            # satisfy the two-distinct-command safeguard.
            commands = list({tuple(result["command"]): result for result in commands}.values())
            stationary = [result["command"] for result in commands if result["stationary_nontracking_candidate"]]
            reference = data.get("fixed_expert_style_reference", {})
            flags = []
            if len(stationary) >= 2:
                flags.append({
                    "severity": "warning", "code": "stationary_nontracking",
                    "commands": stationary,
                    "message": "At least two moving commands survive but have mean planar speed below 0.1 m/s. Stable balance does not establish command-following walking; inspect gait and video.",
                })
            snapshot = {
                "path": str(path.resolve()), "checkpoint": data["checkpoint"],
                "checkpoint_iteration": int(iteration), "protocol_id": protocol_id,
                "protocol": protocol, "commands": commands,
                "fixed_expert_calibration_distance_mean": finite_or_none(reference.get("calibration_distance", {}).get("mean")),
                "conclusion": "stationary_nontracking" if flags else "measurement_requires_gait_and_visual_validation",
                "flags": flags, "visual_validation_required": True,
            }
            # A rewritten report for the same checkpoint and exact protocol
            # supersedes the earlier one, without combining different scopes.
            snapshots[(protocol_id, int(iteration))] = snapshot
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            issues.append({"path": str(path), "message": str(exc)})
    ordered = sorted(snapshots.values(), key=lambda snapshot: (snapshot["checkpoint_iteration"], snapshot["protocol_id"]))
    latest = {}
    for snapshot in ordered:
        latest[snapshot["protocol_id"]] = snapshot
    return {
        "directory": str(directory.resolve()), "files_scanned": len(files),
        "dynamics_reports_excluded": skipped_dynamics, "issues": issues,
        "comparison_scope": "Only identical protocol_id and command are comparable; command sets can differ. No automatic convergence certification.",
        "velocity_measurement": "Observed-step-weighted planar velocity from each trial's alive time mean; speed is the weighted mean of the per-trial vector norms, not instantaneous path speed.",
        "stationary_nontracking_rule": "At least two distinct commands with planar norm > 0.2 m/s, survival >= 0.8, and mean planar speed from trial means < 0.1 m/s.",
        "snapshots": ordered, "latest_by_protocol": list(latest.values()),
        "conclusion": ("stationary_nontracking_detected_in_latest_protocol_measurements"
                       if any(snapshot["flags"] for snapshot in latest.values())
                       else "fixed_measurements_require_gait_and_visual_validation" if ordered else "no_command_evaluations"),
    }


def plot_curves(series, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = (
        ("Completed episode duration", ("episode_seconds",), "seconds", False),
        ("Completed episode fall fraction", ("episode_fall_fraction",), "fraction", False),
        ("Command tracking errors", ("linear_error", "yaw_error"), "m/s or rad/s", False),
        ("Real task and style rewards", ("sim_task_reward", "sim_style_reward"), "reward / step", False),
        ("World model error", ("wm_err1", "wm_errk"), "normalized MSE", True),
        ("Imagined survival", ("imagined_survival",), "fraction over horizon", False),
        ("Imagined return and critic", ("imagined_return", "critic_value"), "value", False),
        ("Critic fitting loss", ("critic_loss",), "MSE", True),
        ("Gradient norms", ("actor_grad", "wm_grad"), "norm before clipping", True),
    )
    fig, axes = plt.subplots(3, 3, figsize=(16, 11), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for axis, (title, keys, ylabel, logscale) in zip(axes.flat, panels):
        for run_index, (name, rows) in enumerate(series):
            for key_index, key in enumerate(keys):
                points = [(row["it"], row[key]) for row in rows if finite_number(row.get(key))
                          and (not logscale or row[key] > 0)]
                if not points:
                    continue
                x, y = zip(*points)
                label = f"{name}: {key}" if len(series) > 1 else key
                axis.plot(x, y, label=label, color=colors[run_index % len(colors)],
                          linestyle=("-", "--", ":")[key_index % 3], linewidth=1.2, alpha=0.9)
        axis.set_title(title)
        axis.set_xlabel("iteration")
        axis.set_ylabel(ylabel)
        if logscale:
            axis.set_yscale("log")
        if keys in (("episode_fall_fraction",), ("imagined_survival",)):
            axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.2)
        if axis.lines:
            axis.legend(fontsize=7)
    fig.suptitle("AMP training measurements — convergence requires fixed-command evaluation", fontsize=14)
    temporary = output.with_suffix(".tmp.png")
    fig.savefig(temporary, dpi=130)
    plt.close(fig)
    temporary.replace(output)


def render_status(report):
    print(f"AMP monitor {report['generated_at']}", flush=True)
    for run in report["runs"]:
        latest = run["latest"]
        if latest is None:
            print(f"{run['name']}: no valid records", flush=True)
            continue
        def value(key, digits=3):
            v = latest.get(key)
            return f"{v:.{digits}f}" if finite_number(v) else "n/a"
        print(f"{run['name']}: it={latest['it']} actor={latest.get('actor_updates', 0)} "
              f"len={value('episode_seconds', 2)}s fall={value('episode_fall_fraction')} "
              f"vel={value('linear_error')} yaw={value('yaw_error')} "
              f"task={value('sim_task_reward', 4)} style={value('sim_style_reward', 4)} "
              f"wmk={value('wm_errk', 4)} status={run['status']}", flush=True)
        for width, window in run["windows"].items():
            change = window["change"]
            def delta(key):
                v = change.get(key, {}).get("relative")
                return f"{v:+.0%}" if v is not None else "n/a"
            print(f"  window {width}: {window['current_records']}/{width} vs "
                  f"{window['previous_records']}/{width} records; "
                  f"len {delta('episode_seconds')}, vel {delta('linear_error')}, "
                  f"task {delta('sim_task_reward')}, style {delta('sim_style_reward')}, "
                  f"wmk {delta('wm_errk')}", flush=True)
        for flag in run["flags"]:
            print(f"  {flag['severity']}: {flag['code']}: {flag['message']}", flush=True)
    evaluations = report.get("independent_evaluations")
    if evaluations:
        print(f"Independent evaluations: {evaluations['conclusion']}", flush=True)
        for snapshot in evaluations["latest_by_protocol"]:
            protocol = snapshot["protocol"]
            print(f"  {Path(protocol['checkpoint_run']).name} it={snapshot['checkpoint_iteration']} "
                  f"protocol={snapshot['protocol_id']} horizon={protocol['horizon_seconds']}s "
                  f"copies={protocol['num_envs_per_command']} conclusion={snapshot['conclusion']}", flush=True)
            for result in snapshot["commands"]:
                def number(key):
                    value = result.get(key)
                    return f"{value:.3f}" if finite_number(value) else "n/a"
                print(f"    command={result['command']} survival={number('survival_rate')} "
                      f"planar_RMSE={number('planar_rmse_mps_alive')} "
                      f"actual_mean_xy={result['mean_velocity_heading_xy_mps_alive']} "
                      f"style_distance={number('fixed_expert_pair_distance_mean')}", flush=True)
            for flag in snapshot["flags"]:
                print(f"    warning: {flag['code']}: {flag['message']}", flush=True)
        for issue in evaluations["issues"]:
            print(f"  evaluation read issue: {issue['path']}: {issue['message']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="log.jsonl files or run directories")
    parser.add_argument("--out", type=Path, default=Path("runs/amp_supervision/monitor"))
    parser.add_argument("--windows", nargs="+", type=int, default=[10, 50])
    parser.add_argument("--stale-seconds", type=float, default=900)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--evaluations", type=Path, help="directory of fixed-command eval_*.json reports; dynamics reports excluded")
    args = parser.parse_args()
    if any(width < 1 for width in args.windows) or args.interval <= 0 or args.stale_seconds <= 0:
        parser.error("windows, interval and stale-seconds must be positive")
    paths = [path / "log.jsonl" if path.is_dir() else path for path in args.logs]
    for path in paths:
        if not path.is_file():
            parser.error(f"Log does not exist: {path}")
    if args.evaluations is not None and not args.evaluations.is_dir():
        parser.error(f"Evaluation directory does not exist: {args.evaluations}")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            summaries = [summarize(path, sorted(set(args.windows)), args.stale_seconds) for path in paths]
            report = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "scope": "Training-only diagnostics; no automatic learner control or convergence certification.",
                "episode_mean_weighting": "completed_episodes; other window metrics average logged means",
                "runs": [summary for summary, _ in summaries],
            }
            if args.evaluations is not None:
                report["independent_evaluations"] = independent_evaluations(args.evaluations)
            temporary = args.out / "status.tmp.json"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            temporary.replace(args.out / "status.json")
            if not args.no_plot:
                plot_curves([(summary["name"], rows) for summary, rows in summaries], args.out / "curves.png")
            render_status(report)
            if not args.watch:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped monitoring; learner processes were not touched.", flush=True)


if __name__ == "__main__":
    main()
