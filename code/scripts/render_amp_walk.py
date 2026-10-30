"""Render deterministic AMP command trials as MP4 plus a contact-sheet PNG.

The physics and first-termination protocol match supermimic.amp_eval. Rendering
mirrors pre-reset simulator poses into a CPU MuJoCo scene; it does not resimulate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

from supermimic import networks
from supermimic.amp_diagnostics import FixedExpertStyle, GaitDiagnostics
from supermimic.amp_eval import DEFAULT_COMMANDS, evaluate_command, make_amp_eval_env, parse_commands
from supermimic.amp_train import load_amp_actor, task_config
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import mat_to_quat


class CommandRenderer:
    def __init__(self, joint_names, width=640, height=480, azimuth=125):
        model = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                               entities={"robot": get_g1_robot_cfg()}), device="cpu").compile()
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot/floating_base_joint")
        if root < 0:
            pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
            root = model.body_jntadr[pelvis]
        root_adr = model.jnt_qposadr[root]
        hinges = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")]
                  for name in joint_names]
        self.addresses = np.r_[np.arange(root_adr, root_adr + 7), hinges]
        self.model, self.data = model, mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 3.0, -15.0, azimuth
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 17)
        except OSError:
            self.font = ImageFont.load_default()

    def frame(self, pose, lines):
        self.data.qpos[self.addresses] = pose
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = (pose[0], pose[1], 0.65)
        self.renderer.update_scene(self.data, camera=self.camera)
        image = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(image)
        for row, line in enumerate(lines):
            draw.text((10, 10 + 24 * row), line, font=self.font,
                      fill="white", stroke_width=2, stroke_fill="black")
        return np.asarray(image)

    def close(self):
        self.renderer.close()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--commands", default=DEFAULT_COMMANDS)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--hold", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--azimuth", type=float, default=125)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0 or not math.isfinite(args.hold) or args.hold < 0:
        parser.error("--seconds must be positive and --hold nonnegative")
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        except ImportError:
            parser.error("ffmpeg or imageio-ffmpeg is required")
    commands = parse_commands(args.commands)
    networks.AMP["enabled"] = False
    torch.manual_seed(0)
    actor, cfg, checkpoint = load_amp_actor(args.ckpt, args.device)
    lib = MotionLibrary(str(args.motions), args.device)
    dt = task_config(cfg).dt
    steps = max(1, round(args.seconds / dt))
    env = make_amp_eval_env(lib, cfg, 1, args.device, steps * dt)
    expert = FixedExpertStyle.from_library(env.task, lib) if args.diagnostics else None
    renderer = CommandRenderer(env.joint_names, azimuth=args.azimuth)
    args.out.mkdir(parents=True, exist_ok=True)
    reports, sheets = [], []
    try:
        for index, command in enumerate(commands):
            trace = []
            physical = {key: [] for key in ("pos", "rot", "q", "v", "w", "qd", "action",
                                             "ankle_pos_w", "ankle_vel_w")}
            last_action = None

            def recorded_actor(observation):
                nonlocal last_action
                last_action = actor(observation)
                return last_action

            def observe(step, state, info, active):
                pose = torch.cat((state["pos"], mat_to_quat(state["rot"]), state["q"]), -1)[0]
                velocity = info["velocity_heading"][0]
                trace.append((pose.cpu().numpy(), velocity.cpu().numpy(),
                              float(info["yaw_rate"][0]), bool(info["terminated"][0])))
                for key in ("pos", "rot", "q", "v", "w", "qd"):
                    physical[key].append(state[key][0].cpu().numpy().copy())
                physical["action"].append(last_action[0].cpu().numpy().copy())
                positions, _, velocities, _ = env.task.body_kin(state)
                feet = [env.task.kin.body_names.index(f"{side}_ankle_roll_link")
                        for side in ("left", "right")]
                physical["ankle_pos_w"].append(positions[0, feet].cpu().numpy().copy())
                physical["ankle_vel_w"].append(velocities[0, feet].cpu().numpy().copy())

            diagnostics = GaitDiagnostics(env, expert) if expert is not None else None
            result = evaluate_command(recorded_actor, env, command, steps, diagnostics=diagnostics, observer=observe)
            trace_name = f"command_{index:02d}_trace.npz"
            np.savez_compressed(
                args.out / trace_name, **{key: np.stack(value) for key, value in physical.items()},
                time_s=(np.arange(len(trace)) + 1) * dt, dt=np.float64(dt),
                command=np.asarray(command), joint_names=np.asarray(env.joint_names),
                ankle_names=np.asarray(("left", "right")),
                velocity_heading=np.stack([row[1] for row in trace]),
                yaw_rate=np.asarray([row[2] for row in trace]),
                terminated=np.asarray([row[3] for row in trace]),
            )
            result["physical_trace"] = trace_name
            frames = []
            for t in range(len(trace) + round(args.hold / dt)):
                j = min(t, len(trace) - 1)
                pose, velocity, yaw, failed = trace[j]
                lines = [
                    f"AMP iteration {checkpoint['it']} | command {command}",
                    f"t={(j + 1) * dt:.2f}s | measured vx/vy/yaw={velocity[0]:+.2f}/{velocity[1]:+.2f}/{yaw:+.2f}",
                    f"{'FALL / terminal pose held' if failed else 'deterministic standing-start trial'}",
                ]
                frames.append(renderer.frame(pose, lines))
            name = f"command_{index:02d}.mp4"
            mediapy.write_video(str(args.out / name), frames, fps=round(1 / dt), codec="h264", crf=23)
            result["video"] = name
            reports.append(result)
            selected = np.linspace(0, len(trace) - 1, 6).round().astype(int)
            thumbnails = [np.asarray(Image.fromarray(frames[t]).resize((480, 360))) for t in selected]
            sheet = np.concatenate([np.concatenate(thumbnails[:3], axis=1),
                                    np.concatenate(thumbnails[3:], axis=1)], axis=0)
            Image.fromarray(sheet).save(args.out / f"command_{index:02d}_frames.png")
            sheets.append(sheet)
            print(f"Wrote {args.out / name}; survival={result['mean_survival_seconds']:.2f}s", flush=True)
    finally:
        renderer.close()
    Image.fromarray(np.concatenate(sheets, axis=0)).save(args.out / "contact_sheet.png")
    report = {"checkpoint": str(args.ckpt), "iteration": int(checkpoint["it"]),
              "requested_seconds": args.seconds, "initialization": "nominal standing; deterministic; no noise/randomization",
              "terminal_behavior": "first terminal pose held; no reset episode enters video",
              "physical_trace_protocol": "50 Hz pre-reset successor states and their executed clean actions; no video hold frames",
              "commands": reports}
    if expert is not None:
        report["fixed_expert_style_reference"] = expert.metadata
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
