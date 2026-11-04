#!/usr/bin/env python3
"""Render still images of a course's MuJoCo scene (no window needed; uses the GPU through EGL).

    scripts/render_course.py race_gates                 # -> docs/media/race_gates_<view>.png
    scripts/render_course.py delivery_city -o /tmp/x    # custom output folder
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from bumble_bee_gym import course, worlds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def course_bounds(c: course.Course) -> tuple[np.ndarray, np.ndarray]:
    pts = [c.spawn_xyz] + list(c.pads) + [p for _, p in c.parcels] + [g.xyz for g in c.gates]
    for b in c.buildings:
        r = np.linalg.norm(b.size[:2]) / 2
        pts += [b.xyz + [r, r, b.size[2]], b.xyz - [r, r, 0]]
    pts = np.array(pts)
    return pts.min(axis=0), pts.max(axis=0)


def views(c: course.Course) -> dict[str, dict]:
    lo, hi = course_bounds(c)
    centre, span = (lo + hi) / 2, float(np.max(hi[:2] - lo[:2]))
    out = {
        "overview": dict(lookat=centre, distance=0.85 * span + 2, azimuth=-55, elevation=-28),
        "top": dict(lookat=centre, distance=1.1 * span + 2, azimuth=-90, elevation=-89),
        "drone": dict(lookat=c.spawn_xyz + [0, 0, 0.1], distance=1.0, azimuth=-130, elevation=-20),
    }
    if c.gates:  # look through the narrowest gate, from behind it along the flight direction
        g = min(c.gates, key=lambda g: min(g.opening))
        out["needle"] = dict(lookat=g.xyz, distance=2.5, azimuth=g.yaw + 180 - 25, elevation=-12)
        first = c.gates[0]  # what the drone sees at the start line
        out["start"] = dict(lookat=first.xyz, distance=float(np.linalg.norm(first.xyz - c.spawn_xyz)) + 1.0,
                            azimuth=first.yaw + 180, elevation=-12)
    return out


def render(name: str, out_dir: Path, width: int, height: int) -> list[Path]:
    c = course.load(name)
    _, xml = worlds.write(c, ROOT / "generated")
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0  # hide collision boxes (gate detector sites are in group 4, hidden by default)
    written = []
    with mujoco.Renderer(m, height, width) as r:
        for view, cam in views(c).items():
            free = mujoco.MjvCamera()
            free.type = mujoco.mjtCamera.mjCAMERA_FREE
            free.lookat[:] = cam["lookat"]
            free.distance, free.azimuth, free.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
            r.update_scene(d, camera=free, scene_option=opt)
            path = out_dir / f"{c.name}_{view}.png"
            Image.fromarray(r.render()).save(path)
            written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("courses", nargs="*")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "docs" / "media")
    ap.add_argument("--size", default="1280x720", help="WIDTHxHEIGHT")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    for name in args.courses or sorted(p.stem for p in course.COURSE_DIR.glob("*.yaml")):
        for p in render(name, args.out, w, h):
            print(p)


if __name__ == "__main__":
    main()
