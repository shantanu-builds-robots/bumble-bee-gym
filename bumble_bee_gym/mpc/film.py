"""Film two race flights one above the other (vertical 1080x1920, 50 fps, real time): NMPC on top, PPO below.

    python -m bumble_bee_gym.mpc.film outputs/mpc/cmp_mpc_nominal.npz outputs/mpc/cmp_ppo_nominal.npz \
        [--labels "NMPC" "PPO"] [--out outputs/mpc/race_mpc_vs_ppo.mp4]

Runs in the playground image (needs an OpenGL renderer and ffmpeg via mediapy). Each half uses the same chase camera
as play_gates.py, with the props spinning, and a caption with the controller, the race clock and the gates passed
(from the run's gate times, in the .json written by compare.py / race.py when given, else counted from the path).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .. import course as course_mod, hoop, params as params_mod
from .judge import Judge, quat_to_mat as quat_to_mat_np

FONT = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"


def gate_times(path: np.ndarray, c, p) -> list[float]:
    """Times the judge counts a gate, replayed from a saved path (qpos[7] + time per row)."""
    j, out = Judge(c, p), []
    for a, b in zip(path, path[1:]):
        before = j.passed
        if j.step(a[:3], b[:3], quat_to_mat_np(b[3:7])):
            break
        if j.passed > before:
            out.append(b[7])
    return out


def chase_frames(m, d, spin, r, cam, qpos: np.ndarray, n_frames: int) -> list[np.ndarray]:
    frames, look, az = [], qpos[0, :3].copy(), None
    end = len(qpos) - 1
    for f in range(n_frames):
        t = min(2 * f, end)                                      # 100 Hz path -> 50 fps
        d.qpos[:7] = qpos[t, :7]
        spin()
        mujoco.mj_forward(m, d)
        vel = qpos[min(t + 2, end), :3] - qpos[max(t - 2, 0), :3]
        heading = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-3 else (az or 0.0)
        az = heading if az is None else az + 0.1 * ((heading - az + 180) % 360 - 180)
        look = 0.8 * look + 0.2 * d.qpos[:3]
        cam.lookat[:], cam.azimuth = look, az
        r.update_scene(d, cam)
        frames.append(r.render())
    return frames


def caption(img: np.ndarray, title: str, sub: str, font_big, font_small) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rounded_rectangle([24, 24, 24 + 640, 24 + 132], 18, fill=(0, 0, 0, 150))
    dr.text((48, 34), title, font=font_big, fill=(255, 255, 255, 255))
    dr.text((48, 100), sub, font=font_small, fill=(255, 220, 120, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("top")
    ap.add_argument("bottom")
    ap.add_argument("--labels", nargs=2, default=["NMPC  (model-based)", "PPO  (learned)"])
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--out", default="outputs/mpc/race_mpc_vs_ppo.mp4")
    a = ap.parse_args()
    c, p = course_mod.load("race_gates"), params_mod.load()
    m = mujoco.MjModel.from_xml_string(hoop.scene_xml(c))
    m.vis.global_.offwidth, m.vis.global_.offheight = 1080, 960
    font_big, font_small = ImageFont.truetype(FONT, 46), ImageFont.truetype(FONT, 34)
    n = int(a.seconds * 50)
    halves = []
    r = mujoco.Renderer(m, 960, 1080)          # one renderer for both: a second EGL context renders garbage
    for path_file, label in ((a.top, a.labels[0]), (a.bottom, a.labels[1])):
        path = np.load(path_file)["path"]
        gt = gate_times(path, c, p)
        d = mujoco.MjData(m)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation = 3.0, -15
        frames = chase_frames(m, d, hoop.PropSpinner(m, p), r, cam, path, n)
        out = []
        for f, img in enumerate(frames):
            t = f / 50
            k = sum(1 for g in gt if g <= t + 1e-9)
            out.append(caption(img, label, f"{t:5.2f} s     gates {k}", font_big, font_small))
        halves.append(out)
        print(f"{label}: {len(gt)} gates" + (f", 33rd at {gt[32]:.2f} s" if len(gt) >= 33 else ""), flush=True)
    video = [np.vstack([a_, b_]) for a_, b_ in zip(*halves)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    mediapy.write_video(a.out, video, fps=50)
    print(f"{len(video)} frames -> {a.out}")


if __name__ == "__main__":
    main()
