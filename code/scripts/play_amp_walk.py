"""Native live velocity-command playback of a from-scratch AMP actor."""

from __future__ import annotations

import argparse
import atexit
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time

os.environ["MUJOCO_GL"] = "glfw"
os.environ["PYOPENGL_PLATFORM"] = "glx"

import glfw
import mujoco
import mujoco.viewer
import numpy as np
import torch

from supermimic import networks
from supermimic.amp_env import AMPEnv
from supermimic.amp_eval import parse_commands
from supermimic.amp_gamepad import F710Reader, list_gamepads, map_velocity
from supermimic.amp_task import COMMAND_HI, COMMAND_LO
from supermimic.amp_train import load_amp_actor, task_config
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import mat_to_quat
from supermimic.sim_env import EnvCfg

# Reuse the already validated native pose-mirror and GLX preflight.
script_directory = str(Path(__file__).resolve().parent)
if script_directory not in sys.path:
    sys.path.insert(0, script_directory)
from play_lafan1 import DisplayScene, display_preflight


@dataclass
class AMPControls:
    command: np.ndarray = field(default_factory=lambda: np.array((0.5, 0.0, 0.0), dtype=np.float32))
    paused: bool = False
    follow: bool = True
    speed: float = 1.0

    def key(self, code):
        change = {
            glfw.KEY_W: (0, 0.1), glfw.KEY_S: (0, -0.1),
            glfw.KEY_A: (1, 0.1), glfw.KEY_D: (1, -0.1),
            glfw.KEY_Z: (2, 0.2), glfw.KEY_C: (2, -0.2),
        }.get(code)
        if change:
            index, amount = change
            self.command[index] = np.clip(
                self.command[index] + amount, COMMAND_LO[index], COMMAND_HI[index],
            )
            return "command"
        if code == glfw.KEY_X:
            self.command[:] = 0
            return "command"
        if code == glfw.KEY_SPACE:
            self.paused = not self.paused
        elif code == glfw.KEY_F:
            self.follow = not self.follow
        elif code in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self.speed = min(4.0, self.speed * 2)
        elif code in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self.speed = max(0.125, self.speed / 2)
        return {
            glfw.KEY_R: "restart", glfw.KEY_L: "reload",
            glfw.KEY_G: "gamepad",
            glfw.KEY_Q: "quit", glfw.KEY_ESCAPE: "quit",
        }.get(code)


class AMPGamepadInput:
    """Own command input until a keyboard command takes over or G toggles it.

    Connect/reconnect and A/B require centered sticks before motion resumes.
    Continuous input changes never restart an episode that is being held after
    a fall; only the explicit reset button or the existing loop timer does so.
    """

    def __init__(self, reader, deadzone=0.15, forward=0.8, backward=0.5,
                 lateral=0.5, yaw=1.0):
        self.reader = reader
        self.deadzone = deadzone
        self.limits = dict(forward=forward, backward=backward, lateral=lateral, yaw=yaw)
        map_velocity((0., 0., 0.), deadzone, **self.limits)
        self.enabled = True
        self.awaiting_neutral = True
        self.connected = self.ready = False
        self.path = self.name = None
        self.status = "searching for F710"
        self.command_updates = self.disconnects = 0
        self._notice = None

    def keyboard_takeover(self):
        self.enabled = False
        self.awaiting_neutral = True

    def toggle(self, controls):
        self.enabled = not self.enabled
        self.awaiting_neutral = True
        self._set_command(controls, (0., 0., 0.))

    def _set_command(self, controls, command):
        command = np.asarray(command, dtype=np.float32)
        # Centering must produce exact zero even below the change tolerance.
        changed = (not np.allclose(command, controls.command, atol=1e-3, rtol=0)
                   or (not np.any(command) and bool(np.any(controls.command))))
        if changed:
            controls.command[:] = command
            self.command_updates += 1
        return changed

    def poll(self, controls):
        sample = self.reader.poll()
        connected_now = sample.connected and (not self.connected or sample.path != self.path)
        disconnected_now = self.connected and not sample.connected
        self.connected, self.ready = sample.connected, sample.ready
        self.name, self.path, self.status = sample.name, sample.path, sample.status
        changed = False
        if disconnected_now:
            self.disconnects += 1
            self.awaiting_neutral = True
            if self.enabled:
                changed = self._set_command(controls, (0., 0., 0.))
        if not self.enabled or not sample.connected:
            return ["command"] if changed else []
        if connected_now or not sample.ready:
            self.awaiting_neutral = True
            changed |= self._set_command(controls, (0., 0., 0.))
        if not sample.ready:
            return ["command"] if changed else []
        events = []
        if "start" in sample.pressed:
            controls.paused = not controls.paused
            events.append("pause")
        if sample.pressed & {"a", "b"}:
            self.awaiting_neutral = True
            self._set_command(controls, (0., 0., 0.))
            events.append("command")
            if "a" in sample.pressed:
                events.append("restart")
            return events
        command = map_velocity(sample.axes, self.deadzone, **self.limits)
        if self.awaiting_neutral:
            if not any(command):
                self.awaiting_neutral = False
            command = (0., 0., 0.)
        changed |= self._set_command(controls, command)
        if changed:
            events.append("command")
        return events

    @property
    def label(self):
        if not self.enabled:
            return "Keyboard (G: enable F710)"
        if not self.connected:
            return "Keyboard / F710 disconnected"
        if not self.ready or self.awaiting_neutral:
            return "F710: center sticks to enable"
        return "F710"

    def announce_status(self):
        notice = (self.label, self.status, self.path)
        if notice != self._notice:
            print(f"Input: {self.label}; {self.name or 'F710'} "
                  f"{self.path or ''} ({self.status})", flush=True)
            self._notice = notice

    def report(self):
        return dict(enabled=self.enabled, connected=self.connected, ready=self.ready,
                    awaiting_neutral=self.awaiting_neutral, device=self.path, name=self.name,
                    status=self.status, deadzone=self.deadzone, limits=self.limits,
                    command_updates=self.command_updates, disconnects=self.disconnects)

    def close(self):
        self.reader.close()


class AMPPlayback:
    def __init__(self, checkpoint, motions, device, command):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.command = np.asarray(command, dtype=np.float32).copy()
        self.reload_policy()
        self.lib = MotionLibrary(str(motions), device)
        if abs(self.lib.dt - task_config(self.cfg).dt) > 1e-8:
            raise ValueError("Expert library/control intervals differ")
        self.env = AMPEnv(
            self.lib,
            EnvCfg(num_envs=1, episode_length_s=1e9, obs_noise=False,
                   reset_noise=False, domain_rand=True, compile=False,
                   nconmax=128, njmax=500),
            device, task_config(self.cfg), gamma=self.cfg.gamma,
            rsi_probability=0.0, stand_height=self.cfg.stand_height,
        )
        self.env.eval_mode = True
        self.total_steps = 0
        self.restarts = 0
        self.reset()

    def reload_policy(self):
        actor, cfg, ck = load_amp_actor(self.checkpoint, "cpu")
        if hasattr(self, "cfg"):
            if (asdict(task_config(cfg)) != asdict(task_config(self.cfg))
                    or cfg.stand_height != self.cfg.stand_height):
                raise ValueError("Checkpoint task settings changed; restart the viewer")
        self.actor = actor.to(self.device).eval()
        self.cfg = cfg
        self.checkpoint_iteration = int(ck["it"])
        self.actor_updates = int(ck.get("actor_updates", 0))
        print(f"Loaded AMP {self.checkpoint} (iteration {self.checkpoint_iteration}, "
              f"actor updates {self.actor_updates})", flush=True)

    @staticmethod
    def pose(state):
        return torch.cat((state["pos"], mat_to_quat(state["rot"]), state["q"]), -1
                         )[0].detach().cpu().numpy().copy()

    @torch.no_grad()
    def reset(self):
        self.env.fixed_command = torch.as_tensor(self.command, device=self.device).clone()
        self.env.reset()
        self.robot_pose = self.pose(self.env.state)
        # Hidden by DisplayScene.update(..., visible=False); no expert phase is played.
        self.reference_pose = self.robot_pose.copy()
        self.steps = 0
        self.done = self.failed = False
        self.done_since = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.yaw_rate = 0.0
        self.restarts += 1
        print(f"Standing reset; command={self.command.tolist()}", flush=True)

    @torch.no_grad()
    def set_command(self, command, *, announce=True, reset_done=True):
        self.command = np.asarray(command, dtype=np.float32).copy()
        self.env.fixed_command = torch.as_tensor(self.command, device=self.device).clone()
        if self.done:
            if reset_done:
                self.reset()
            return
        self.env.command[:] = self.env.fixed_command
        self.env.command_age.zero_()
        # Update only the latest historical observation. Older commands remain
        # in their historical frames, just as they do in collected AMP replay.
        self.env.obs_hist[:, -1] = self.env.amp_task.observation(
            self.env.state, self.env.prev_action, self.env.command, self.env.xi,
        )
        if announce:
            print(f"Command={self.command.tolist()}", flush=True)

    @torch.no_grad()
    def step(self):
        if self.done:
            return
        state, info = self.env.step(self.actor(self.env.policy_inputs()))
        # Preserve the pre-reset physical successor at a failure.
        self.robot_pose = self.pose(state)
        self.reference_pose = self.robot_pose.copy()
        self.velocity = info["velocity_heading"][0].cpu().numpy().copy()
        self.yaw_rate = float(info["yaw_rate"][0])
        self.steps += 1
        self.total_steps += 1
        self.done = bool(info["done"][0])
        self.failed = bool(info["terminated"][0])
        if self.done:
            self.done_since = time.monotonic()
            print(f"{'FAILED' if self.failed else 'ENDED'} at "
                  f"{self.steps * self.lib.dt:.2f}s", flush=True)


def play(args, playback, controls, display, gamepad=None):
    keys = queue.SimpleQueue()
    handle = None
    viewer_threads = []
    if not args.headless:
        previous_threads = set(threading.enumerate())
        handle = mujoco.viewer.launch_passive(
            display.model, display.data, key_callback=keys.put,
            show_left_ui=False, show_right_ui=False,
        )
        created = [t for t in threading.enumerate() if t not in previous_threads]
        viewer_threads = [t for t in created if "_launch_internal" in t.name] or created
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            handle.cam.distance = 3.0
            handle.cam.elevation = -15
            handle.cam.azimuth = 125
            handle.cam.lookat[:] = display.update(playback, visible=False)
        print("AMP MuJoCo window opened.", flush=True)
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
                elif action == "command":
                    if gamepad is not None:
                        gamepad.keyboard_takeover()
                    playback.set_command(controls.command)
                elif action == "gamepad":
                    if gamepad is not None:
                        gamepad.toggle(controls)
                        playback.set_command(controls.command, reset_done=False)
                elif action == "restart":
                    playback.reset()
                elif action == "reload":
                    try:
                        playback.reload_policy()
                        playback.reset()
                    except Exception as exc:
                        print(f"Reload failed; keeping current actor: {exc}", flush=True)
            if quit_requested:
                break
            if gamepad is not None:
                for action in gamepad.poll(controls):
                    if action == "command":
                        playback.set_command(controls.command, announce=False, reset_done=False)
                    elif action == "restart":
                        playback.reset()
                gamepad.announce_status()
            if not controls.paused:
                if not playback.done:
                    playback.step()
                elif args.loop and tick - playback.done_since >= args.hold:
                    playback.reset()
                elif args.headless and not args.loop:
                    break
            if handle is not None:
                status = "FAILED" if playback.failed else ("ENDED" if playback.done else "PLAYING")
                if controls.paused:
                    status += " / PAUSED"
                with handle.lock():
                    center = display.update(playback, visible=False)
                    if controls.follow:
                        handle.cam.lookat[:] = center
                handle.set_texts([
                    (None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                     "AMP walking\nCheckpoint\nActor updates\nCommand vx / vy / yaw\nMeasured vx / vy / yaw\nTime\nStatus\nTarget speed\nInput",
                     f"commanded policy, no reference phase\n{playback.checkpoint_iteration}\n"
                     f"{playback.actor_updates}\n"
                     f"{controls.command[0]:+.2f} / {controls.command[1]:+.2f} / {controls.command[2]:+.2f}\n"
                     f"{playback.velocity[0]:+.2f} / {playback.velocity[1]:+.2f} / {playback.yaw_rate:+.2f}\n"
                     f"{playback.steps * playback.lib.dt:.2f}s\n{status}\n{controls.speed:g}x\n"
                     f"{gamepad.label if gamepad is not None else 'Keyboard'}"),
                    (None, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                     "W/S: forward/back | A/D: left/right | Z/C: yaw | X: zero command\n"
                     "F710 L: vx/vy | R: yaw | A: reset | B: zero | Start: pause | G: input toggle\n"
                     "Space: pause | R: stand reset | L: reload | F: follow | +/-: speed | Q/Esc: quit", ""),
                ])
                handle.sync()
            if args.max_steps and playback.total_steps >= args.max_steps:
                break
            period = playback.lib.dt / controls.speed if not controls.paused and not playback.done else 1 / 30
            time.sleep(max(0.0, period - (time.monotonic() - tick)))
    finally:
        if gamepad is not None:
            gamepad.close()
        if handle is not None:
            handle.close()
            for thread in viewer_threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    print("Waiting for the MuJoCo window to finish closing...", flush=True)
                    while thread.is_alive():
                        thread.join(timeout=1.0)
            glfw.terminate()
            atexit.unregister(glfw.terminate)
    report = {
        "checkpoint": str(playback.checkpoint),
        "checkpoint_iteration": playback.checkpoint_iteration,
        "actor_updates": playback.actor_updates,
        "gui_opened": handle is not None,
        "viewer_threads_joined": all(not t.is_alive() for t in viewer_threads),
        "steps": playback.total_steps, "restarts": playback.restarts,
        "command": playback.command.tolist(),
        "measured_velocity_heading_mps": playback.velocity.tolist(),
        "measured_yaw_rate_radps": playback.yaw_rate,
        "failed": playback.failed, "wall_seconds": time.monotonic() - start,
        "scope": "Live checkpoint playback; not an assertion of trained walking",
    }
    if gamepad is not None:
        report["gamepad"] = gamepad.report()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("AMP_PLAYBACK_REPORT " + json.dumps(report), flush=True)


def default_amp_checkpoint():
    best = Path("runs/amp_walk_best/best.pt")
    return best if best.is_file() else Path("runs/amp_walk_24gb_train/last.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=default_amp_checkpoint(),
                        help="Defaults to the validated best.pt when present, otherwise the training last.pt")
    parser.add_argument("--motions", type=Path, default=Path("data/processed/amp_walk_50hz.npz"))
    parser.add_argument("--command", default="0.5,0,0", help="vx,vy [m/s], yaw [rad/s]")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hold", type=float, default=2.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gamepad", choices=("auto", "on", "off"), default="auto",
                        help="auto: detect F710, on: wait with zero command, off: keyboard only")
    parser.add_argument("--gamepad-device", type=Path, help="Optional F710 joydev path, e.g. /dev/input/js1")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.15)
    parser.add_argument("--gamepad-forward-speed", type=float, default=0.8, help="Forward stick maximum [m/s]")
    parser.add_argument("--gamepad-backward-speed", type=float, default=0.5, help="Backward stick magnitude [m/s]")
    parser.add_argument("--gamepad-lateral-speed", type=float, default=0.5, help="Lateral stick magnitude [m/s]")
    parser.add_argument("--gamepad-yaw-speed", type=float, default=1.0, help="Yaw stick magnitude [rad/s]")
    parser.add_argument("--list-gamepads", action="store_true", help="List input devices and exit without simulation")
    parser.add_argument("--test-gamepad", action="store_true", help="Print gamepad commands without simulation")
    args = parser.parse_args()
    if args.list_gamepads:
        print(json.dumps(list_gamepads(), indent=2, ensure_ascii=False), flush=True)
        return
    if not args.test_gamepad and (not args.ckpt.is_file() or not args.motions.is_file()):
        parser.error("An AMP checkpoint and its motion library are required")
    try:
        commands = parse_commands(args.command)
        if len(commands) != 1:
            raise ValueError("--command must contain one vx,vy,yaw triple")
        command = np.asarray(commands[0], dtype=np.float32)
        if not np.isfinite(command).all():
            raise ValueError("Command exceeds the FP32 observation range")
    except ValueError as exc:
        parser.error(str(exc))
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not all(math.isfinite(v) and v >= 0 for v in (args.hold, args.duration)) or args.max_steps < 0:
        parser.error("Time and step limits must be finite and nonnegative")
    scales = dict(forward=args.gamepad_forward_speed, backward=args.gamepad_backward_speed,
                  lateral=args.gamepad_lateral_speed, yaw=args.gamepad_yaw_speed)
    try:
        map_velocity((0., 0., 0.), args.gamepad_deadzone, **scales)
    except ValueError as exc:
        parser.error(str(exc))
    if (scales["forward"] > COMMAND_HI[0] or scales["backward"] > -COMMAND_LO[0]
            or scales["lateral"] > min(COMMAND_HI[1], -COMMAND_LO[1])
            or scales["yaw"] > min(COMMAND_HI[2], -COMMAND_LO[2])):
        parser.error("Gamepad limits exceed command ranges: vx[-0.5,1], vy[-0.5,0.5], yaw[-1,1]")
    if args.gamepad == "off" and (args.gamepad_device or args.test_gamepad):
        parser.error("--gamepad off cannot be combined with device selection or --test-gamepad")
    if args.test_gamepad:
        gamepad = AMPGamepadInput(F710Reader(str(args.gamepad_device) if args.gamepad_device else None),
                                  args.gamepad_deadzone, **scales)
        controls = AMPControls(command=np.zeros(3, dtype=np.float32))
        deadline = time.monotonic() + (args.duration or 10.)
        try:
            while time.monotonic() < deadline:
                events = gamepad.poll(controls)
                gamepad.announce_status()
                if events:
                    print(f"Gamepad events={events}; command={controls.command.tolist()}; "
                          f"paused={controls.paused}", flush=True)
                time.sleep(0.02)
        finally:
            gamepad.close()
        report = dict(gamepad=gamepad.report(), command=controls.command.tolist(), paused=controls.paused)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("AMP_GAMEPAD_REPORT " + json.dumps(report), flush=True)
        return
    if args.headless and not (args.max_steps or args.duration):
        parser.error("--headless requires --max-steps or --duration")
    if args.headless and args.paused and not args.duration:
        parser.error("Paused headless checks require --duration")
    if not args.headless:
        display_preflight()
    networks.AMP["enabled"] = False
    torch.manual_seed(0)
    if args.gamepad == "on":
        command[:] = 0
    controls = AMPControls(command=command.copy(), paused=args.paused, speed=args.speed)
    gamepad = None
    try:
        if args.gamepad != "off":
            gamepad = AMPGamepadInput(F710Reader(str(args.gamepad_device) if args.gamepad_device else None),
                                      args.gamepad_deadzone, **scales)
            # Sample before constructing the simulation so an already connected
            # centered controller starts the robot with a zero velocity target.
            gamepad.poll(controls)
            gamepad.announce_status()
        playback = AMPPlayback(args.ckpt, args.motions, args.device, controls.command)
        display = DisplayScene(playback.env.joint_names, reference_offset=0.0)
        print("Keys: W/S vx, A/D vy, Z/C yaw, X zero, Space pause, R reset, L reload, F follow, G gamepad, Q quit.", flush=True)
        play(args, playback, controls, display, gamepad=gamepad)
    finally:
        if gamepad is not None:
            gamepad.close()


if __name__ == "__main__":
    main()
