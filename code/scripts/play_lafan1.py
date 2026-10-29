"""Live checkpoint tracking in MuJoCo's native window; no recorded trajectories."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import time
import threading

# local_runtime.sh normally selects EGL for offline rendering.
os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "glx"

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

from supermimic import networks
from supermimic.evaluate import load_supermimic_policy
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import mat_to_quat
from supermimic.sim_env import EnvCfg, TrackingSimEnv
from supermimic.task import TaskCfg


def short_name(name):
    return Path(str(name)).name.removesuffix(".csv")


@dataclass
class Controls:
    paused: bool = False
    speed: float = 1.0
    follow: bool = True
    reference_visible: bool = True

    def key(self, code):
        if code == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif code in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.speed = min(4.0, self.speed * 2)
        elif code in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.speed = max(0.125, self.speed / 2)
        elif code == glfw.KEY_F:
            self.follow = not self.follow
        elif code == glfw.KEY_G:
            self.reference_visible = not self.reference_visible
        return {
            glfw.KEY_R: "restart",
            glfw.KEY_N: "next",
            glfw.KEY_P: "previous",
            glfw.KEY_L: "reload",
            glfw.KEY_Q: "quit",
            glfw.KEY_ESCAPE: "quit",
        }.get(code)


class LiveTracking:
    """Policy and physics retain the training environment's state/action contract."""

    def __init__(self, checkpoint, motions, device, clip_ids=None, initial_index=0):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.lib = MotionLibrary(str(motions), device, clip_ids=clip_ids)
        self.names = [short_name(n) for n in self.lib.names]
        self.reload_policy()
        self.env = TrackingSimEnv(
            self.lib,
            EnvCfg(
                num_envs=1, episode_length_s=1e9, obs_noise=False,
                reset_noise=False, domain_rand=True, compile=False,
                nconmax=256, njmax=500,
            ),
            TaskCfg(), device,
        )
        self.env.eval_mode = True
        self.env.reset()
        self.ids = torch.zeros(1, device=device, dtype=torch.long)
        self.total_steps = 0
        self.restarts = 0
        self.reset(initial_index)

    def reload_policy(self):
        # Loading the entire training snapshot on CPU avoids allocating optimizers on GPU.
        policy, ck = load_supermimic_policy(str(self.checkpoint), "cpu")
        self.policy = policy.to(self.device).eval()
        self.checkpoint_iteration = int(ck["it"])
        print(f"Loaded {self.checkpoint} (iteration {self.checkpoint_iteration})", flush=True)

    @staticmethod
    def pose(state):
        return torch.cat(
            (state["pos"], mat_to_quat(state["rot"]), state["q"]), -1
        )[0].detach().cpu().numpy().copy()

    @torch.no_grad()
    def reset(self, index):
        self.index = index % self.lib.num_clips
        frame = self.lib.starts[self.index:self.index + 1]
        self.env.reset_envs(self.ids, frames=frame)
        self.env._finish_resets(self.ids, force=True)
        self.robot_pose = self.pose(self.env.state)
        self.reference_pose = self.pose(self.lib.frame(frame))
        self.steps = 0
        self.reward = 0.0
        self.done = False
        self.failed = False
        self.done_since = None
        self.restarts += 1
        print(f"Playing [{self.index + 1}/{self.lib.num_clips}] {self.name}", flush=True)

    @property
    def name(self):
        return self.names[self.index]

    @property
    def length(self):
        return int(self.lib.lengths[self.index])

    @property
    def norm_return(self):
        return self.reward / (self.env.tcfg.max_step_reward * max(self.length - 1, 1))

    @property
    def root_error(self):
        return float(np.linalg.norm(self.robot_pose[:3] - self.reference_pose[:3]))

    @torch.no_grad()
    def step(self):
        if self.done:
            return
        # step() auto-resets its internal episode. Keep the pre-reset successor for display.
        next_frame = self.env.frame.clone() + 1
        obs, goal = self.env.policy_inputs()
        state, info = self.env.step(self.policy(obs, goal))
        self.robot_pose = self.pose(state)
        self.reference_pose = self.pose(self.lib.frame(next_frame))
        self.steps += 1
        self.total_steps += 1
        self.reward += float(info["reward"][0])
        self.done = bool(info["done"][0])
        self.failed = bool(info["terminated"][0])
        if self.done:
            self.done_since = time.monotonic()
            print(
                f"{'FAILED' if self.failed else 'COMPLETED'} {self.name}: "
                f"t={self.steps * self.lib.dt:.2f}s, return={self.norm_return:.4f}",
                flush=True,
            )


class DisplayScene:
    """CPU pose mirror of the live GPU simulation, plus a non-colliding reference."""

    def __init__(self, joint_names, reference_offset):
        self.model = Scene(
            SceneCfg(
                terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                entities={"robot": get_g1_robot_cfg(), "ghost": get_g1_robot_cfg()},
            ), device="cpu",
        ).compile()
        self.data = mujoco.MjData(self.model)
        self.reference_offset = reference_offset
        self.addresses = {}
        for entity in ("robot", "ghost"):
            body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{entity}/pelvis")
            root = self.model.jnt_qposadr[self.model.body_jntadr[body]]
            joints = [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{entity}/{name}")
                ] for name in joint_names
            ]
            self.addresses[entity] = np.r_[np.arange(root, root + 7), joints].astype(int)
        self.ghost_geoms = [
            g for g in range(self.model.ngeom)
            if (mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]
            ) or "").startswith("ghost/")
        ]
        self.model.geom_contype[self.ghost_geoms] = 0
        self.model.geom_conaffinity[self.ghost_geoms] = 0

    def update(self, tracking, visible=True):
        reference = tracking.reference_pose.copy()
        reference[1] += self.reference_offset
        self.data.qpos[self.addresses["robot"]] = tracking.robot_pose
        self.data.qpos[self.addresses["ghost"]] = reference
        self.model.geom_rgba[self.ghost_geoms] = (1.0, 0.55, 0.1, 0.45 if visible else 0.0)
        self.data.time = tracking.steps * tracking.lib.dt
        mujoco.mj_forward(self.model, self.data)
        center = (0.5 * (tracking.robot_pose[:3] + reference[:3])
                  if visible else tracking.robot_pose[:3].copy())
        center[2] = 0.65
        return center


def display_preflight():
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("No DISPLAY. Run in a terminal on your Linux desktop.")
    if not glfw.init():
        raise RuntimeError("GLFW cannot initialize the desktop display.")
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(160, 120, "SuperMimic display check", None, None)
        if not window:
            raise RuntimeError("GLFW cannot create an OpenGL window.")
        glfw.destroy_window(window)
        glfw.default_window_hints()
    finally:
        glfw.terminate()


def run(args, tracking, controls, display):
    keys = queue.SimpleQueue()
    handle = None
    viewer_threads = []
    if not args.headless:
        previous_threads = set(threading.enumerate())
        handle = mujoco.viewer.launch_passive(
            display.model, display.data, key_callback=keys.put,
            show_left_ui=False, show_right_ui=False,
        )
        created_threads = [t for t in threading.enumerate() if t not in previous_threads]
        viewer_threads = [t for t in created_threads if "_launch_internal" in t.name] or created_threads
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            handle.cam.distance = 3.5
            handle.cam.elevation = -15
            handle.cam.azimuth = 125
            handle.cam.lookat[:] = display.update(tracking)
        print("MuJoCo window opened.", flush=True)
    start = time.monotonic()
    try:
        while handle is None or handle.is_running():
            tick = time.monotonic()
            if args.duration and tick - start >= args.duration:
                break
            quit_requested = False
            while not keys.empty():
                action = controls.key(keys.get())
                if action == "quit":
                    quit_requested = True
                elif action == "restart":
                    tracking.reset(tracking.index)
                elif action == "next":
                    tracking.reset(tracking.index + 1)
                elif action == "previous":
                    tracking.reset(tracking.index - 1)
                elif action == "reload":
                    try:
                        tracking.reload_policy()
                        tracking.reset(tracking.index)
                    except Exception as exc:
                        print(f"Reload failed; keeping current playback: {exc}", flush=True)
            if quit_requested:
                break
            if not controls.paused:
                if not tracking.done:
                    tracking.step()
                elif args.loop and tick - tracking.done_since >= args.hold:
                    tracking.reset(tracking.index + int(args.auto_next))
                elif args.headless and not args.loop:
                    break
            if handle is not None:
                status = "FAILED" if tracking.failed else ("COMPLETED" if tracking.done else "PLAYING")
                if controls.paused:
                    status += " / PAUSED"
                with handle.lock():
                    center = display.update(tracking, controls.reference_visible)
                    if controls.follow:
                        handle.cam.lookat[:] = center
                handle.set_texts([
                    (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "SuperMimic\nClip\nCheckpoint\nTime\nStatus\nTarget speed\nRoot error\nReturn so far / full-clip max",
                     f"solid: policy / orange: reference (+{args.reference_offset:g}m Y)\n"
                     f"{tracking.name} [{tracking.index + 1}/{tracking.lib.num_clips}]\n"
                     f"{tracking.checkpoint_iteration}\n"
                     f"{tracking.steps * tracking.lib.dt:.2f} / {(tracking.length - 1) * tracking.lib.dt:.2f}s\n"
                     f"{status}\n{controls.speed:g}x\n{tracking.root_error:.3f}m\n{tracking.norm_return:.4f}"),
                    (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "Space: pause | R: restart | N/P: next/previous | L: reload checkpoint\n"
                     "+/-: speed | F: follow camera | G: reference | Q/Esc: quit", ""),
                ])
                handle.sync()
            if args.max_steps and tracking.total_steps >= args.max_steps:
                break
            period = tracking.lib.dt / controls.speed if not controls.paused and not tracking.done else 1 / 30
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        if handle is not None:
            handle.close()
            # close() requests exit asynchronously. Finish GLFW teardown before CUDA finalization.
            for thread in viewer_threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print("Waiting for the MuJoCo window to finish closing...", flush=True)
                    while thread.is_alive():
                        thread.join(timeout=1.0)
            glfw.terminate()
            atexit.unregister(glfw.terminate)
    report = {
        "gui_opened": handle is not None,
        "viewer_threads_joined": all(not t.is_alive() for t in viewer_threads),
        "checkpoint": str(tracking.checkpoint),
        "checkpoint_iteration": tracking.checkpoint_iteration, "clip": tracking.name,
        "steps": tracking.total_steps, "restarts": tracking.restarts,
        "failed": tracking.failed, "done": tracking.done,
        "partial_full_clip_return": tracking.norm_return, "root_error_m": tracking.root_error,
        "wall_seconds": time.monotonic() - start, "reference_offset_m": args.reference_offset,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("PLAYBACK_REPORT " + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("runs/lafan1_24gb_train/last.pt"))
    parser.add_argument("--motions", type=Path, default=Path("data/processed/lafan1_50hz.npz"))
    parser.add_argument("--clip", default=None, help="Initial clip; defaults to walk1_subject1 when available.")
    parser.add_argument("--clips", default="all", help="Comma-separated selectable clips, or all.")
    parser.add_argument("--list-clips", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--reference-offset", type=float, default=1.0, help="Display-only Y offset; 0 overlays the reference.")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--auto-next", action="store_true")
    parser.add_argument("--hold", type=float, default=2.0, help="Hold the end/failure pose before looping.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action="store_true", help="Integration checks without a window.")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many control steps; 0 is unlimited.")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many playback wall-seconds.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.speed <= 0 or not np.isfinite(args.speed):
        parser.error("--speed must be finite and positive")
    if not np.isfinite(args.hold) or not np.isfinite(args.duration):
        parser.error("Time limits must be finite")
    if args.hold < 0 or args.duration < 0 or args.max_steps < 0:
        parser.error("Time and step limits must be nonnegative")
    if not np.isfinite(args.reference_offset):
        parser.error("--reference-offset must be finite")
    if not args.motions.is_file():
        parser.error(f"Motion library not found: {args.motions}")
    with np.load(args.motions, allow_pickle=False) as data:
        names = [short_name(n) for n in data["names"]]
    if args.list_clips:
        print("\n".join(names))
        return
    if not args.ckpt.is_file():
        parser.error(f"Checkpoint not found: {args.ckpt}")
    chosen = names if args.clips == "all" else [n.strip() for n in args.clips.split(",") if n.strip()]
    if not chosen or len(set(chosen)) != len(chosen):
        parser.error("--clips must contain distinct clip names")
    missing = [n for n in chosen if n not in names]
    if missing:
        parser.error(f"Unknown clips: {missing}; use --list-clips")
    if args.clip is not None and args.clip not in chosen:
        parser.error(f"Initial clip {args.clip!r} is not in the selected clips; use --list-clips")
    if args.headless and not (args.max_steps or args.duration):
        parser.error("--headless requires --max-steps or --duration")
    if args.headless and args.paused and not args.duration:
        parser.error("Paused headless checks require --duration")
    if not args.headless:
        display_preflight()
    networks.AMP["enabled"] = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    initial = args.clip or ("walk1_subject1" if "walk1_subject1" in chosen else chosen[0])
    tracking = LiveTracking(args.ckpt, args.motions, args.device,
                            [names.index(n) for n in chosen], initial_index=chosen.index(initial))
    display = DisplayScene(tracking.env.joint_names, args.reference_offset)
    controls = Controls(paused=args.paused, speed=args.speed)
    print("Controls: Space pause; R restart; N/P next/previous; L reload; +/- speed; F follow; G reference; Q quit.", flush=True)
    run(args, tracking, controls, display)


if __name__ == "__main__":
    main()
