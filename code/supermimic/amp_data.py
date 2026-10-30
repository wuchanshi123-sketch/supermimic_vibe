"""Prepare the self-contained AMP walking library from existing 50 Hz G1 NPZ clips.

Only root/joint states are copied. Expert body features must be computed with the
target simulator's FK rather than copied from the source robot asset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

WALKING_PREFIXES = ("walk_", "arc_walk_", "idle_turn_", "step_rotate_")
BODY_FIELDS = {
    "body_pos_w": 3,
    "body_quat_w": 4,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}
CONTROL_FPS = 50.0


def _selected(path: Path, preset: str) -> bool:
    if any(part.lower() == "recovery" for part in path.parts):
        return False
    return preset == "all" or path.stem.startswith(WALKING_PREFIXES)


def _numeric(array, name: str, source: Path):
    array = np.asarray(array)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{source}: {name} must contain finite real numbers")
    return array


def _read_clip(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"fps", "joint_pos", "joint_vel", *BODY_FIELDS}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        fps_values = _numeric(data["fps"], "fps", path).reshape(-1)
        if fps_values.size != 1 or abs(float(fps_values[0]) - CONTROL_FPS) > 1e-6:
            raise ValueError(f"{path}: expected 50 Hz expert data, got {fps_values.tolist()}")
        joint_pos = _numeric(data["joint_pos"], "joint_pos", path)
        joint_vel = _numeric(data["joint_vel"], "joint_vel", path)
        if joint_pos.ndim != 2 or joint_pos.shape[1] != 29 or len(joint_pos) < 2:
            raise ValueError(f"{path}: joint_pos must have shape (T>=2, 29)")
        frames = len(joint_pos)
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path}: joint_vel must match joint_pos shape")
        bodies = {}
        body_count = None
        for key, width in BODY_FIELDS.items():
            value = _numeric(data[key], key, path)
            if value.ndim != 3 or value.shape[0] != frames or value.shape[2] != width or value.shape[1] < 1:
                raise ValueError(f"{path}: {key} must have shape (T={frames}, B>=1, {width})")
            if body_count is not None and value.shape[1] != body_count:
                raise ValueError(f"{path}: inconsistent body counts")
            body_count = value.shape[1]
            bodies[key] = value
        norms = np.linalg.norm(bodies["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3, rtol=0):
            raise ValueError(f"{path}: body_quat_w must contain unit wxyz quaternions")
        qpos = np.concatenate(
            (bodies["body_pos_w"][:, 0], bodies["body_quat_w"][:, 0], joint_pos), axis=-1
        ).astype(np.float32)
        qvel = np.concatenate(
            (bodies["body_lin_vel_w"][:, 0], bodies["body_ang_vel_w"][:, 0], joint_vel), axis=-1
        ).astype(np.float32)
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError(f"{path}: states overflow float32")
    return qpos, qvel, int(body_count)


def _write_outputs(output: Path, arrays: dict, manifest_path: Path, manifest: dict, overwrite: bool):
    """Stage both files; no-overwrite commits use exclusive hard-link creation."""
    temporary_paths = []
    committed = []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".amp_data_", suffix=".npz", delete=False) as handle:
            staged_npz = Path(handle.name)
            temporary_paths.append(staged_npz)
            np.savez(handle, **arrays)
        with tempfile.NamedTemporaryFile(
            dir=output.parent, prefix=".amp_data_", suffix=".json",
            mode="w", encoding="utf-8", delete=False,
        ) as handle:
            staged_json = Path(handle.name)
            temporary_paths.append(staged_json)
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        for staged, target in ((staged_npz, output), (staged_json, manifest_path)):
            if overwrite:
                os.replace(staged, target)
            else:
                os.link(staged, target)
                committed.append(target)
    except BaseException:
        # Remove only files exclusively created by this call, never pre-existing data.
        if not overwrite:
            for target in committed:
                target.unlink(missing_ok=True)
        raise
    finally:
        for staged in temporary_paths:
            staged.unlink(missing_ok=True)


def prepare_amp_library(input_dir, output, preset="walking", *, overwrite=False) -> dict:
    """Merge reference clips into vibe's packed qpos/qvel format.

    Walking selects walking and idle/turn clips, excluding jog and Recovery.
    All includes all NPZ clips in the input tree except Recovery directories.
    No checkpoints, teachers, or precomputed body features are needed.
    """
    if preset not in ("walking", "all"):
        raise ValueError("preset must be 'walking' or 'all'")
    source = Path(input_dir).expanduser().resolve()
    requested_output = Path(output).expanduser()
    # Resolve the directory, not an existing file symlink, so overwrite replaces
    # an output link rather than modifying the link's original target.
    output = requested_output.parent.resolve() / requested_output.name
    if not source.is_dir():
        raise NotADirectoryError(f"Expert directory not found: {source}")
    if output.suffix != ".npz":
        raise ValueError("output must end in .npz")
    manifest_path = output.with_suffix(".json")
    for target in (output, manifest_path):
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {target}; use --overwrite explicitly")
    all_paths = sorted(source.rglob("*.npz"))
    if output in all_paths:
        with np.load(output, allow_pickle=False) as existing:
            if "joint_pos" in existing and "body_pos_w" in existing:
                raise ValueError("output must not overwrite an original expert clip")
    paths = [p for p in all_paths if _selected(p.relative_to(source), preset)
             and source.name.lower() != "recovery"
             and p.resolve() != output]
    if not paths:
        raise ValueError(f"No expert clips selected under {source} with preset {preset!r}")
    names = [p.stem for p in paths]
    if len(set(names)) != len(names):
        raise ValueError("Selected expert clips have duplicate names")
    qposes, qvels, lengths, body_counts = [], [], [], []
    for path in paths:
        qpos, qvel, body_count = _read_clip(path)
        qposes.append(qpos)
        qvels.append(qvel)
        lengths.append(len(qpos))
        body_counts.append(body_count)
    lengths_array = np.asarray(lengths, dtype=np.int64)
    starts = np.r_[np.int64(0), np.cumsum(lengths_array[:-1])].astype(np.int64)
    total_frames = int(lengths_array.sum())
    manifest = {
        "format": "supermimic_vibe_amp_library_v1",
        "preset": preset,
        "input_dir": str(source),
        "output": str(output),
        "manifest": str(manifest_path),
        "source_paths": [str(p.resolve()) for p in paths],
        "names": names,
        "clips": len(names),
        "lengths": lengths,
        "starts": starts.tolist(),
        "frames": total_frames,
        "fps": CONTROL_FPS,
        "seconds": total_frames / CONTROL_FPS,
        "source_body_counts": body_counts,
        "qpos_shape": [total_frames, 36],
        "qvel_shape": [total_frames, 35],
        "dtype": "float32",
        "qpos_layout": "root world position[0:3], root quaternion wxyz[3:7], SDK-order joint position[7:36]",
        "qvel_layout": "root world linear velocity[0:3], root world angular velocity[3:6], joint velocity[6:35]",
        "source_root_body_index": 0,
        "body_features": "Not copied. Recompute expert AMP features using the target robot FK.",
        "clip_boundaries": "starts/lengths identify independent clips; transition pairs must not cross boundaries",
        "finite": True,
        "quaternion_norm_atol": 1e-3,
    }
    arrays = {
        "qpos": np.concatenate(qposes),
        "qvel": np.concatenate(qvels),
        "starts": starts,
        "lengths": lengths_array,
        "names": np.asarray(names),
        "fps": np.asarray(CONTROL_FPS, dtype=np.float32),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_outputs(output, arrays, manifest_path, manifest, overwrite)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, dest="input_dir")
    parser.add_argument("--output", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--preset", choices=("walking", "all"), default="walking")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        manifest = prepare_amp_library(args.input_dir, args.output, args.preset, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"AMP data preparation failed: {exc}\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
