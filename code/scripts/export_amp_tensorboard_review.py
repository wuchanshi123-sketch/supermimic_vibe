#!/usr/bin/env python3
"""Import existing AMP evidence into a separate, provenance-labelled TB review.

No learner, checkpoint, source event, or evaluation is modified.  An existing
output containing regular files is refused; use --out for another fresh export.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
LINEAGE = (
    ("amp_walk_24gb_train", 0, 5000),
    ("amp_walk_h16", 5000, 24000),
    ("amp_walk_h16_precision", 24000, 28000),
    ("amp_walk_h16_slow_prior", 28000, 34000),
    ("amp_walk_h16_smooth", 34000, 35000),
    ("amp_walk_h16_smooth10", 35000, 36000),
    ("amp_walk_h32_smooth10", 36000, 37000),
    ("amp_walk_forward_h32", 37000, 40000),
    ("amp_walk_forward_expert_h32", 40000, 42000),
)
EPISODE_TAGS = {"episode_seconds", "episode_fall_fraction", "episode_return"}
PERFORMANCE_TAGS = {"sim_task_reward", "sim_reward", "linear_error", "yaw_error"}
AMP_TAGS = {
    "sim_style_reward", "imagined_style_reward", "policy_score", "expert_score",
    "gradient_penalty", "disc_updates",
}
README = """# AMP 训练与独立验收 TensorBoard

本目录是**从已有证据派生的只读评审视图**，不是新的训练或评估。原始事件和报告未修改。

- `selected_lineage`：仅拼接最终 42000 检查点的真实祖先区间，共 840 个训练记录；保留原始 iteration 和 wall time。分组为 performance、survival、amp、dynamics、optimizer。
- `eval_20s16_seed17/<run>`：导入同一协议的独立评估：20 秒、每命令 16 个样本、seed=17、初态扰动 scale=1。每个 run/iteration 只保留最新报告。所有点均为已有 JSON 导入，wall time 是来源文件修改时间。
- `acceptance_30s64`：导入所选 42000 检查点的最终独立验收：每速度 64 个样本、30 秒；连续命令切换为 40 秒。此组含步态周期图片。

先看独立评估的 survival_rate、planar_rmse_mps、yaw_mae_radps，再看 gait 的抬脚幅度、左右相位与交替观察。style_nn_distance 使用固定专家特征参考，数值越小越接近该参考；它不是通过率，参考校准集也未从策略训练中隔离。

训练中的 linear_error 是平面速度误差均值（m/s），yaw_error 是偏航绝对误差均值（rad/s）。它们不是固定命令验收的 RMSE。训练含随机命令、动作噪声和参考状态初始化。

**completed_episodes=0 的窗口不输出 episode 长度、摔倒比例和回报。** 原始训练日志在这些窗口写入的 0 不代表零摔倒率。观察摔倒比例时同时看样本数。

课程在 24000、28000、34000、35000、37000、40000 改变过奖励、专家采样或命令分布；规划窗口也从 H=16 改为 H=32。跨阶段奖励不可直接比较。判别器训练目标为 expert=+1、policy=-1；低判别器损失不能单独证明模仿成功。imagined_* 来自学习到的动力学模型。

视频步态标量是匹配检查点的单条视频轨迹观察，**不是 16 样本评估的均值**。抬脚高度测量的是踝关节刚体原点，并非足底离地间隙。交替观察、风格距离和验收覆盖均不等于通用收敛或真实机器人质量认证。最终报告保留 quality_certified=false；请结合视频。

选择的模型是 42000，最后训练迭代是 43000。最终验收仅覆盖平地仿真与报告内的初态扰动及前进/命令切换条件。完整来源、筛选规则和计数见 manifest.json。
"""


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def source_info(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    try:
        name = str(path.relative_to(root))
    except ValueError:
        name = str(path)
    return {"path": name, "mtime": stat.st_mtime, "bytes": stat.st_size}


def ensure_fresh_output(out: Path) -> None:
    if out.is_symlink():
        raise ValueError(f"Output must be a real directory, not a symlink: {out}")
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output is not a directory: {out}")
    # The serving process may have prepared raw event symlinks. Do not follow
    # them, and never delete them or any previous derived export.
    def check(directory: Path) -> None:
        for child in directory.iterdir():
            if child.is_symlink():
                continue
            if child.is_dir():
                check(child)
            else:
                raise ValueError(
                    f"Refusing nonempty derived output ({child}); use --out NEW_DIRECTORY"
                )
    if out.exists():
        check(out)


def group_for(tag: str) -> str:
    if tag in PERFORMANCE_TAGS:
        return "performance"
    if tag in EPISODE_TAGS or tag == "completed_episodes":
        return "survival"
    if tag in AMP_TAGS or tag.startswith("discriminator_"):
        return "amp"
    if tag.startswith(("wm_", "imagined_")):
        return "dynamics"
    return "optimizer"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def command_tag(command: list[float]) -> str:
    def number(value: float) -> str:
        return format(float(value), ".6g")
    return "command/" + "_".join(
        f"{axis}_{number(value)}" for axis, value in zip(("vx", "vy", "wz"), command)
    )


class ReviewWriter:
    """Reject duplicate scalar tag/step pairs instead of hiding evidence."""

    def __init__(self, directory: Path):
        if directory.is_symlink():
            raise ValueError(f"Derived run cannot be a symlink: {directory}")
        self.writer = SummaryWriter(str(directory))
        self.seen: set[tuple[str, int]] = set()
        self.counts: Counter[str] = Counter()

    def scalar(self, tag: str, value: Any, step: int, walltime: float) -> None:
        if not finite_number(value):
            return
        key = (tag, step)
        if key in self.seen:
            raise ValueError(f"Duplicate scalar point: {tag} at {step}")
        self.seen.add(key)
        self.writer.add_scalar(tag, float(value), step, walltime=walltime)
        self.counts[tag] += 1

    def text(self, tag: str, value: str, step: int = 0, walltime: float | None = None) -> None:
        self.writer.add_text(tag, value, step, walltime=walltime)

    def close(self) -> None:
        self.writer.close()


def export_lineage(root: Path, out: Path) -> dict[str, Any]:
    writer = ReviewWriter(out / "selected_lineage")
    stages = []
    all_steps: set[int] = set()
    skipped_episode_points = 0
    try:
        writer.text("meta/README", README)
        for stage_index, (name, low, high) in enumerate(LINEAGE):
            directory = root / "runs" / name / "tb"
            events = EventAccumulator(
                str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False
            ).Reload()
            tags = events.Tags()["scalars"]
            scalars = {}
            deduplicated = 0
            for tag in tags:
                by_step = {}
                for event in sorted(events.Scalars(tag), key=lambda item: item.wall_time):
                    if low < event.step <= high:
                        deduplicated += int(event.step in by_step)
                        by_step[event.step] = event
                scalars[tag] = by_step
            steps = sorted(scalars["it"])
            expected = list(range(low + 50, high + 1, 50))
            if steps != expected:
                raise ValueError(f"Unexpected selected-lineage steps in {name}: {steps}")
            if all_steps.intersection(steps):
                raise ValueError(f"Overlapping lineage stage: {name}")
            all_steps.update(steps)
            if set(scalars["completed_episodes"]) != set(steps):
                raise ValueError(f"Missing completed-episode counts in {name}")
            stage_skipped = 0
            for tag, points in scalars.items():
                for step, event in sorted(points.items()):
                    if tag in EPISODE_TAGS and scalars["completed_episodes"][step].value <= 0:
                        stage_skipped += 1
                        continue
                    writer.scalar(f"{group_for(tag)}/{tag}", event.value, step, event.wall_time)
            writer.text(
                "meta/source_stage",
                f"派生且过滤的获胜训练谱系，第 {stage_index + 1} 段：`runs/{name}/tb`；"
                f"仅保留 {low} < iteration <= {high}，原始 wall time。"
                "奖励/专家采样/命令分布可能与前段不同，跨阶段奖励不可直接比较。",
                steps[0], scalars["it"][steps[0]].wall_time,
            )
            skipped_episode_points += stage_skipped
            stages.append({
                "run": name, "filter": {"iteration_gt": low, "iteration_lte": high},
                "records": len(steps), "first_step": steps[0], "last_step": steps[-1],
                "duplicate_source_scalar_points_replaced_by_latest_walltime": deduplicated,
                "episode_scalar_points_omitted_for_zero_completions": stage_skipped,
                "sources": [source_info(path, root) for path in sorted(directory.glob("events.*"))],
            })
    finally:
        writer.close()
    if sorted(all_steps) != list(range(50, 42001, 50)):
        raise ValueError("Selected lineage must contain exactly 840 contiguous 50-iteration records")
    return {
        "records": len(all_steps), "first_step": min(all_steps), "last_step": max(all_steps),
        "scalar_points": sum(writer.counts.values()), "scalar_counts": dict(writer.counts),
        "episode_scalar_points_omitted_for_zero_completions": skipped_episode_points,
        "walltime": "Original event wall_time, unchanged",
        "stages": stages,
    }


def eval_protocol_errors(report: dict[str, Any]) -> list[str]:
    expected = {
        "requested_seconds": 20, "horizon_seconds": 20,
        "num_envs_per_command": 16, "deterministic_policy": True,
        "friction": 1.0, "observation_noise": False,
        "reset_noise": False, "randomization": False,
    }
    errors = [f"{key} != {value!r}" for key, value in expected.items() if report.get(key) != value]
    perturb = report.get("initial_state_perturbations", {})
    for key, value in {"seed": 17, "scale": 1.0}.items():
        if perturb.get(key) != value:
            errors.append(f"initial_state_perturbations.{key} != {value}")
    commands = report.get("commands", [])
    if not commands:
        errors.append("no command evaluations")
    if any(command.get("copies") != 16 for command in commands):
        errors.append("per-command copies != 16")
    if not report.get("checkpoint") or not isinstance(report.get("checkpoint_iteration"), int):
        errors.append("missing checkpoint identity")
    return errors


def write_command_metrics(
    writer: ReviewWriter, report: dict[str, Any], step: int, walltime: float,
) -> None:
    mapping = {
        "survival_rate": "survival_rate", "survived": "survived_count", "copies": "trial_count",
        "fall_rate": "fall_rate", "mean_survival_seconds": "mean_survival_seconds",
        "mean_coverage": "mean_coverage", "planar_rmse_mps_alive": "planar_rmse_mps",
        "mean_planar_error_mps_alive": "planar_mean_error_mps",
        "mean_abs_yaw_error_radps_alive": "yaw_mae_radps",
    }
    for command in report["commands"]:
        prefix = command_tag(command["command"])
        for source, name in mapping.items():
            writer.scalar(f"{prefix}/{name}", command.get(source), step, walltime)
        distance = command.get("gait_diagnostics_alive", {}).get("fixed_expert_pair_distance", {})
        for name in ("mean", "p05", "p95"):
            writer.scalar(f"{prefix}/style_nn_distance_{name}", distance.get(name), step, walltime)


def write_gait_metrics(writer: ReviewWriter, gait: dict[str, Any], step: int, walltime: float) -> None:
    for command in gait.get("commands", []):
        prefix = command_tag(command["command"]) + "/gait_video_observation"
        for side in ("left", "right"):
            foot = command.get("feet", {}).get(side, {})
            writer.scalar(f"{prefix}/{side}_height_span_m", foot.get("height_span_p95_p05_m"), step, walltime)
        writer.scalar(f"{prefix}/shared_frequency_hz", command.get("shared_dominant_height_frequency_hz"), step, walltime)
        phase = command.get("left_right_height_phase_degrees")
        writer.scalar(f"{prefix}/absolute_phase_degrees", abs(phase) if finite_number(phase) else None, step, walltime)
        writer.scalar(f"{prefix}/both_low_ankle_fraction", command.get("both_low_ankle_fraction"), step, walltime)
        writer.scalar(f"{prefix}/alternating_steps_observed", command.get("alternating_steps_observed"), step, walltime)


def export_comparable_evals(root: Path, out: Path) -> dict[str, Any]:
    directory = root / "runs" / "amp_supervision"
    accepted = {}
    excluded = []
    duplicates = []
    candidates = sorted(directory.glob("eval_*_perturbed.json"), key=lambda path: (path.stat().st_mtime, path.name))
    for path in candidates:
        if "dynamics" in path.name.lower():
            excluded.append({"source": source_info(path, root), "reasons": ["dynamics evaluation excluded"]})
            continue
        try:
            report = read_json(path)
            errors = eval_protocol_errors(report)
        except (ValueError, KeyError, TypeError) as error:
            excluded.append({"source": source_info(path, root), "reasons": [str(error)]})
            continue
        if errors:
            excluded.append({"source": source_info(path, root), "reasons": errors})
            continue
        run = Path(report["checkpoint"]).parent.name
        step = report["checkpoint_iteration"]
        key = (run, step)
        if key in accepted:
            previous, _ = accepted[key]
            duplicates.append({"run": run, "step": step, "replaced": str(previous.relative_to(root)), "kept": str(path.relative_to(root))})
        accepted[key] = (path, report)
    by_run = defaultdict(list)
    for (run, step), (path, report) in sorted(accepted.items()):
        by_run[run].append((step, path, report))
    run_manifest = []
    for run, entries in sorted(by_run.items()):
        writer = ReviewWriter(out / "eval_20s16_seed17" / run)
        sources = []
        gait_sources = []
        try:
            writer.text("meta/README", "导入已有独立评估 JSON，不是原始训练事件或新测量。"
                        "协议：20 s、每命令 N=16、seed=17、初态扰动 scale=1、确定性策略、摩擦1、无噪声/随机化。"
                        "wall time 使用来源文件修改时间。视频步态标量是单条视频轨迹观察，不是 N=16 均值。")
            for step, path, report in entries:
                walltime = path.stat().st_mtime
                write_command_metrics(writer, report, step, walltime)
                writer.text("meta/source_report", f"来源：`{path.relative_to(root)}`；checkpoint=`{report['checkpoint']}`。", step, walltime)
                sources.append(source_info(path, root))
                gait_path = directory / f"video_{run}_{step:07d}" / "gait_summary.json"
                if gait_path.exists():
                    gait = read_json(gait_path)
                    if gait.get("iteration") != step or Path(gait.get("checkpoint", "")).parent.name != run:
                        raise ValueError(f"Video gait checkpoint identity differs from eval: {gait_path}")
                    write_gait_metrics(writer, gait, step, gait_path.stat().st_mtime)
                    gait_sources.append(source_info(gait_path, root))
        finally:
            writer.close()
        run_manifest.append({
            "run": run, "records": len(entries), "first_step": entries[0][0], "last_step": entries[-1][0],
            "scalar_points": sum(writer.counts.values()), "scalar_counts": dict(writer.counts),
            "sources": sources, "video_gait_observation_sources": gait_sources,
        })
    return {
        "filter": {
            "glob": "runs/amp_supervision/eval_*_perturbed.json", "exclude_name_contains": "dynamics",
            "seconds": 20, "copies_per_command": 16, "seed": 17, "perturbation_scale": 1,
            "deterministic": True, "friction": 1, "noise_and_randomization": False,
            "duplicate_policy": "latest file mtime for each checkpoint run/iteration; path name breaks ties",
        },
        "candidate_files": len(candidates), "records": len(accepted), "runs": run_manifest,
        "duplicate_reports": duplicates, "excluded_reports": excluded,
        "walltime": "Imported source JSON file mtime, not original training time",
    }


def export_acceptance(root: Path, out: Path) -> dict[str, Any]:
    best = root / "runs" / "amp_walk_best"
    fixed_path = best / "validation" / "fixed_commands.json"
    sequence_path = best / "validation" / "command_sequence.json"
    gait_path = best / "validation" / "gait_robustness" / "summary.json"
    selection_path = best / "selection.json"
    fixed, sequence, gait, selection = (read_json(path) for path in (fixed_path, sequence_path, gait_path, selection_path))
    step = 42000
    if any(report.get("checkpoint_iteration") != step for report in (fixed, sequence, gait)):
        raise ValueError("Acceptance reports must identify checkpoint 42000")
    if fixed.get("requested_seconds") != 30 or fixed.get("num_envs_per_command") != 64:
        raise ValueError("Fixed-command acceptance protocol must be 30 seconds / 64 copies")
    if selection.get("selected_iteration") != step:
        raise ValueError("Selected checkpoint is not iteration 42000")
    for selected, evaluated in zip(selection["fixed_command_validation"], fixed["commands"], strict=True):
        if (selected["command"] != evaluated["command"]
                or selected["survived"] != evaluated["survived"]
                or selected["planar_rmse_mps"] != evaluated["planar_rmse_mps_alive"]
                or selected["yaw_mae_radps"] != evaluated["mean_abs_yaw_error_radps_alive"]):
            raise ValueError("Selection metrics differ from independent fixed-command evaluation")
    writer = ReviewWriter(out / "acceptance_30s64")
    image_path = best / "gait_cycles.png"
    try:
        writer.text("meta/README", README, step)
        writer.text("meta/protocol", "由已有独立验收 JSON 导入，不是原始训练指标。"
                    "固定速度：30 s、每命令64样本、seed59、初态扰动scale1；"
                    "命令序列：40 s、64样本、seed101；确定性策略，平地仿真、无噪声/域随机化。"
                    "步态报告 quality_certified=false，需结合视频。", step)
        write_command_metrics(writer, fixed, step, fixed_path.stat().st_mtime)
        seq_time = sequence_path.stat().st_mtime
        for key in ("copies", "horizon_seconds", "survived_full_sequence", "falls", "truncations"):
            writer.scalar(f"sequence_40s/{key}", sequence.get(key), step, seq_time)
        writer.scalar("sequence_40s/survival_rate", sequence["survived_full_sequence"] / sequence["copies"], step, seq_time)
        gait_time = gait_path.stat().st_mtime
        for key in ("survived", "copies", "survival_rate", "planar_rmse_mps_alive", "mean_abs_yaw_error_radps_alive"):
            writer.scalar(f"gait_robustness_0.3mps/{key}", gait.get(key), step, gait_time)
        for key, value in gait["style_observations"].items():
            writer.scalar(f"gait_robustness_0.3mps/observations/{key}", value, step, gait_time)
        with Image.open(image_path) as image:
            writer.writer.add_image("gait/gait_cycles", np.asarray(image.convert("RGB")), step,
                                    walltime=image_path.stat().st_mtime, dataformats="HWC")
    finally:
        writer.close()
    return {
        "records": 1, "step": step, "scalar_points": sum(writer.counts.values()),
        "scalar_counts": dict(writer.counts), "selection_metrics_equal_independent_eval": True,
        "sources": [source_info(path, root) for path in (fixed_path, sequence_path, gait_path, selection_path, image_path)],
        "walltime": "Imported source JSON/image mtime; metadata text uses export time",
        "protocol": {"fixed_commands_seconds": 30, "fixed_commands_copies": 64,
                     "fixed_commands_seed": 59, "sequence_seconds": 40, "sequence_copies": 64, "sequence_seed": 101},
    }


def verify_export(out: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    lineage = EventAccumulator(str(out / "selected_lineage"), size_guidance={"scalars": 0}).Reload()
    steps = [point.step for point in lineage.Scalars("performance/linear_error")]
    if steps != list(range(50, 42001, 50)):
        raise ValueError("Derived event verification failed: lineage is not exactly 840 records")
    survival = {point.step for point in lineage.Scalars("survival/episode_fall_fraction")}
    counts = {point.step: point.value for point in lineage.Scalars("survival/completed_episodes")}
    if survival != {step for step, value in counts.items() if value > 0}:
        raise ValueError("Derived episode metrics include unsampled windows or omit sampled windows")
    verified_runs = 0
    for run in [out / "selected_lineage", out / "acceptance_30s64"] + [
        out / "eval_20s16_seed17" / entry["run"] for entry in manifest["eval_20s16_seed17"]["runs"]
    ]:
        events = EventAccumulator(str(run), size_guidance={"scalars": 0}).Reload()
        for tag in events.Tags()["scalars"]:
            point_steps = [point.step for point in events.Scalars(tag)]
            if len(point_steps) != len(set(point_steps)):
                raise ValueError(f"Duplicate derived scalar points in {run}: {tag}")
        verified_runs += 1
    accepted = EventAccumulator(str(out / "acceptance_30s64"), size_guidance={"scalars": 0}).Reload()
    expected_rmse = (0.0839414894580841, 0.08844834566116333, 0.10324735939502716)
    for speed, expected in zip((0.3, 0.5, 0.8), expected_rmse, strict=True):
        prefix = command_tag([speed, 0, 0])
        rmse = accepted.Scalars(f"{prefix}/planar_rmse_mps")
        survival = accepted.Scalars(f"{prefix}/survival_rate")
        if len(rmse) != 1 or rmse[0].step != 42000 or not math.isclose(rmse[0].value, expected, rel_tol=1e-6):
            raise ValueError(f"Acceptance RMSE failed event round trip at {speed} m/s")
        if len(survival) != 1 or survival[0].value != 1:
            raise ValueError(f"Acceptance survival failed event round trip at {speed} m/s")
    if not accepted.Tags()["images"]:
        raise ValueError("Acceptance gait-cycle image missing")
    return {"lineage_records": len(steps), "derived_runs_checked_for_duplicate_scalar_steps": verified_runs,
            "episode_zero_sample_mask_verified": True, "acceptance_scalar_round_trip_verified": True,
            "acceptance_image_verified": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Repository root")
    parser.add_argument("--out", type=Path, default=None, help="Fresh output directory; default ROOT/runs/amp_tensorboard_review")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.absolute() if args.out else root / "runs" / "amp_tensorboard_review"
    ensure_fresh_output(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "amp_tensorboard_review_v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "output": str(out),
        "interpretation": "Derived imports of existing evidence, not new training/evaluation; original sources unchanged",
        "selected_lineage": export_lineage(root, out),
        "eval_20s16_seed17": export_comparable_evals(root, out),
        "acceptance_30s64": export_acceptance(root, out),
    }
    manifest["verification"] = verify_export(out, manifest)
    (out / "README.md").write_text(README, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "lineage_records": manifest["selected_lineage"]["records"],
                      "comparable_eval_records": manifest["eval_20s16_seed17"]["records"],
                      "comparable_eval_runs": len(manifest["eval_20s16_seed17"]["runs"]),
                      "verification": manifest["verification"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
