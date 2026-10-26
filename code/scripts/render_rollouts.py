"""Render benchmark rollouts of a SuperMimic checkpoint to MP4.

Each selected clip plays once from its first frame under the benchmark protocol (App. F:
friction 1.0, no domain randomization or noise, deterministic policy). The simulated robot is
drawn solid and the reference motion as a translucent orange ghost at its own world position,
so tracking error shows as the gap between the two. Writes one MP4 per clip, a tiled montage
and summary.json (per-clip normalized return, failure, rendered window).

usage: render_rollouts.py --ckpt runs/x/ckpt_N.pt --motions data.npz --clips a,b,c --out dir
                          [--window 15] [--cols 3] [--title "..."]
Clip names are matched against the motion file's names without directory or ".csv".
Clips longer than --window seconds are cut to the window with the fastest reference motion
(the whole clip is still simulated from frame 0).
"""

import argparse
import json
import os
import shutil

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import mediapy
import mujoco
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

from supermimic import networks
from supermimic.evaluate import load_supermimic_policy, make_eval_env
from supermimic.motion_lib import MotionLibrary
from supermimic.rotations import mat_to_quat

FPS = 50
W, H = 640, 480
GHOST_RGBA = (1.0, 0.55, 0.1, 0.4)


def short(name):
  return os.path.basename(name).removesuffix(".csv")


@torch.no_grad()
def rollout(pol, lib, env):
  """Robot and reference qpos (T, n, 36) per frame, valid frame count, failure step, return."""
  dev = env.device
  n = lib.num_clips
  env.reset()
  ids = torch.arange(n, device=dev)
  env.reset_envs(ids, frames=lib.starts)
  env._finish_resets(ids, force=True)
  T = int(lib.lengths.max())
  robot = torch.zeros(T, n, 36, device=dev)
  s = env.read_state()
  robot[0] = torch.cat([s["pos"], mat_to_quat(s["rot"]), s["q"]], -1)
  alive = torch.ones(n, dtype=torch.bool, device=dev)
  nframes = torch.ones(n, dtype=torch.long, device=dev)
  fail_step = torch.full((n,), -1, dtype=torch.long, device=dev)
  ret = torch.zeros(n, device=dev)
  t = 0
  while alive.any():
    t += 1
    obs, goal = env.policy_inputs()
    s, info = env.step(pol(obs, goal))
    q = torch.cat([s["pos"], mat_to_quat(s["rot"]), s["q"]], -1)
    robot[t, alive] = q[alive]
    nframes[alive] += 1
    ret += info["reward"] * alive
    term = info["terminated"] & alive
    fail_step[term] = t
    alive &= ~info["done"]
  ref = torch.zeros(T, n, 36, device=dev)
  for i in range(n):
    a, L = int(lib.starts[i]), int(lib.lengths[i])
    ref[:L, i] = torch.cat([lib.root_pos[a : a + L], lib.root_quat[a : a + L], lib.q[a : a + L]], -1)
  norm = (ret / (env.tcfg.max_step_reward * (lib.lengths - 1).float())).clamp(max=1.0)
  return robot.cpu().numpy(), ref.cpu().numpy(), nframes.cpu().numpy(), fail_step.cpu().numpy(), norm.cpu().numpy()


class Renderer:
  """Two G1s in one MuJoCo model: "robot" (simulated state) and "ghost" (reference)."""

  def __init__(self, joint_names):
    cfg = SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      entities={"robot": get_g1_robot_cfg(), "ghost": get_g1_robot_cfg()},
    )
    m = Scene(cfg, device="cpu").compile()
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    self.adr = {}
    for ent in ("robot", "ghost"):
      pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{ent}/pelvis")
      root = m.jnt_qposadr[m.body_jntadr[pelvis]]
      hinges = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{ent}/{nm}")] for nm in joint_names]
      self.adr[ent] = np.concatenate([np.arange(root, root + 7), np.array(hinges)])
    for g in range(m.ngeom):
      body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
      if body.startswith("ghost/"):
        m.geom_rgba[g] = GHOST_RGBA
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
    self.m, self.d = m, mujoco.MjData(m)
    self.r = mujoco.Renderer(m, height=H, width=W)
    self.cam = mujoco.MjvCamera()
    self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.cam.distance, self.cam.elevation = 3.0, -12.0
    self.opt = mujoco.MjvOption()

  def frame(self, q_robot, q_ref, lookat, azimuth):
    self.d.qpos[self.adr["robot"]] = q_robot
    self.d.qpos[self.adr["ghost"]] = q_ref
    mujoco.mj_forward(self.m, self.d)
    self.cam.lookat[:] = lookat
    self.cam.azimuth = azimuth
    self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
    return self.r.render()


def yaw_deg(quat):
  w, x, y, z = quat
  return np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def pick_window(ref_q, L, win):
  """Start frame of the `win`-frame window with the largest mean joint speed."""
  if L <= win:
    return 0
  speed = np.abs(np.diff(ref_q[:L, 7:], axis=0)).sum(-1)
  c = np.concatenate([[0], np.cumsum(speed)])
  sums = c[win - 1 :] - c[: len(c) - win + 1]
  return int(np.argmax(sums))


def font(size):
  try:
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)
  except Exception:
    return ImageFont.load_default()


def annotate(img, lines, color_last=None):
  im = Image.fromarray(img)
  dr = ImageDraw.Draw(im)
  f, fs = font(17), font(14)
  y = 8
  for k, (txt, big) in enumerate(lines):
    fill = color_last if (color_last and k == len(lines) - 1) else (255, 255, 255)
    ff = f if big else fs
    dr.text((10, y), txt, font=ff, fill=fill, stroke_width=2, stroke_fill=(0, 0, 0))
    y += 22 if big else 18
  return np.asarray(im)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--motions", required=True)
  ap.add_argument("--clips", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--window", type=float, default=15.0)
  ap.add_argument("--cols", type=int, default=3)
  ap.add_argument("--title", default="SuperMimic")
  args = ap.parse_args()
  assert shutil.which("ffmpeg"), "ffmpeg not on PATH"
  dev = "cuda"
  networks.AMP["enabled"] = False
  torch.backends.cuda.matmul.allow_tf32 = True

  want = [c.strip() for c in args.clips.split(",") if c.strip()]
  names = [short(n) for n in np.load(args.motions)["names"]]
  ids = [names.index(c) for c in want]
  lib = MotionLibrary(args.motions, dev, clip_ids=ids)
  env = make_eval_env(lib, lib.num_clips, dev)
  pol, ck = load_supermimic_policy(args.ckpt, dev)
  robot, ref, nframes, fail_step, norm = rollout(pol, lib, env)
  print(f"checkpoint {args.ckpt} (it {ck['it']}): " + ", ".join(
    f"{c} {r:.3f}{' FAILED' if f >= 0 else ''}" for c, r, f in zip(want, norm, fail_step)), flush=True)

  os.makedirs(args.out, exist_ok=True)
  rend = Renderer(env.joint_names)
  win = int(round(args.window * FPS))
  videos, summary = [], []
  for i, c in enumerate(want):
    L = int(lib.lengths[i])
    a = pick_window(ref[:, i], L, win)
    b = min(L, a + win)
    az = yaw_deg(ref[a, i, 3:7]) + 125.0
    look = robot[a, i, :3].copy()
    frames = []
    for t in range(a, b):
      tr = min(t, int(nframes[i]) - 1)  # after a termination the robot freezes at its last state
      target = robot[tr, i, :3].copy()
      target[2] = 0.65
      look = look + 0.08 * (target - look)
      img = rend.frame(robot[tr, i], ref[t, i], look, az)
      failed_now = fail_step[i] >= 0 and t >= fail_step[i]
      status = (f"terminated at {fail_step[i] / FPS:.1f} s" if failed_now
                else f"clip return {norm[i]:.3f}" + (" (fails later)" if fail_step[i] >= 0 else ""))
      img = annotate(img, [(c, True), (f"t = {t / FPS:4.1f} s   solid: simulated robot   orange: reference", False), (status, False)],
                     color_last=(255, 90, 90) if failed_now else (150, 255, 150))
      frames.append(img)
    path = os.path.join(args.out, f"{c}.mp4")
    mediapy.write_video(path, frames, fps=FPS, codec="h264", crf=23)
    videos.append(frames)
    summary.append({"clip": c, "norm_return": float(norm[i]), "failed": bool(fail_step[i] >= 0),
                    "fail_time_s": float(fail_step[i] / FPS) if fail_step[i] >= 0 else None,
                    "clip_seconds": L / FPS, "window_s": [a / FPS, b / FPS], "video": os.path.basename(path)})
    print(f"  wrote {path} ({len(frames)} frames)", flush=True)

  # Montage: tiles at half resolution; shorter clips hold their last frame.
  cols = min(args.cols, len(videos))
  rows = (len(videos) + cols - 1) // cols
  tw, th = W // 2 * 3 // 2, H // 2 * 3 // 2  # 480x360
  T = max(len(v) for v in videos)
  mont = []
  for t in range(T):
    canvas = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, v in enumerate(videos):
      im = np.asarray(Image.fromarray(v[min(t, len(v) - 1)]).resize((tw, th), Image.LANCZOS))
      r, cc = divmod(k, cols)
      canvas[r * th : (r + 1) * th, cc * tw : (cc + 1) * tw] = im
    mont.append(canvas)
  mpath = os.path.join(args.out, "montage.mp4")
  mediapy.write_video(mpath, mont, fps=FPS, codec="h264", crf=25)
  json.dump({"ckpt": args.ckpt, "it": int(ck["it"]), "hours": ck["train_time"] / 3600, "title": args.title, "clips": summary},
            open(os.path.join(args.out, "summary.json"), "w"), indent=1)
  print("wrote", mpath, flush=True)


if __name__ == "__main__":
  main()
