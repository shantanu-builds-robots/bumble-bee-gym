"""Path planning for a race course: a smooth, collision-free 3-D minimum-snap path through every gate.

    python -m bumble_bee_gym.mpc.path --course long_track [--v-max 10] [--a-max 10] [--out outputs/mpc/path_long.npz]

What the PID racer (pid.py) follows, and the initial guess the time-optimal planner (plan.py) starts from.

1. Route: the start point, then for each gate a point `approach` metres before it on its axis, its centre, and a
   point the same distance after it, so the path goes through every gate square to it.
2. Detours: each leg of the route that passes through an obstacle (a building, inflated by the drone's reach plus a
   clearance) gets a waypoint around it. An obstacle lower than hop_height is flown over; a taller one is passed on
   the side the leg is already nearer to.
3. Minimum snap (trajectory.py) through the waypoints, with segment times allocated against the speed and
   acceleration limits.
4. Clearance check: wherever the curve comes closer to an obstacle than allowed, the offending segment gets its
   midpoint (on the collision-free route) as an extra waypoint, and the curve is refitted.

Saved as samples every 10 ms: time, position, velocity, acceleration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .. import course as course_mod, hoop, trajectory, worlds

OUT = Path(__file__).resolve().parents[2] / "outputs" / "mpc"
REACH = 0.19            # [m] farthest a collision sphere reaches from the drone's centre (prop hub 0.12 + 0.07)


class Boxes:
    """The course's obstacles as boxes standing on the ground, with distances from points to them."""

    def __init__(self, c: course_mod.Course):
        b = c.buildings
        self.c = np.array([[o.xyz[0], o.xyz[1], o.size[2] / 2] for o in b]).reshape(-1, 3)
        self.h = np.array([np.asarray(o.size, float) / 2 for o in b]).reshape(-1, 3)
        self.r = np.array([worlds._rot_z(o.yaw) for o in b]).reshape(-1, 3, 3)
        self.height = 2 * self.h[:, 2]

    def __len__(self):
        return len(self.c)

    def distance(self, pts: np.ndarray) -> np.ndarray:
        """(n, O) distance from each point to each box (0 inside)."""
        if not len(self):
            return np.full((len(pts), 0), np.inf)
        local = np.einsum("noj,ojk->nok", pts[:, None] - self.c[None], self.r)
        return np.linalg.norm(np.maximum(np.abs(local) - self.h[None], 0.0), axis=-1)


def route(c: course_mod.Course, start: np.ndarray, approach: float) -> np.ndarray:
    pts = [start]
    for g in c.gates:
        centre, rot = hoop.hoop_frame(g)
        n = rot[:, 0]
        pts += [centre - approach * n, centre, centre + approach * n]
    return np.array(pts)


def detour(a: np.ndarray, b: np.ndarray, boxes: Boxes, clear: float, hop_height: float) -> list[np.ndarray]:
    """Waypoints to insert between a and b so the straight leg misses every box by `clear`."""
    s = np.linspace(0, 1, 200)
    p = a + (b - a) * s[:, None]
    d = boxes.distance(p)
    if not d.size or d.min() >= clear:
        return []
    o = int(np.argmin(d.min(axis=0)))
    hit = p[d[:, o] < clear]
    mid = hit.mean(axis=0)
    if boxes.height[o] < hop_height:                      # over the top
        over = mid.copy()
        over[2] = boxes.height[o] + clear + 0.3
        return [over]
    # around the side: from the box centre, perpendicular to the leg, on the side the leg passes
    along = (b - a)[:2] / max(np.linalg.norm((b - a)[:2]), 1e-9)
    perp = np.array([-along[1], along[0]])
    side = 1.0 if np.dot(mid[:2] - boxes.c[o, :2], perp) >= 0 else -1.0
    reach = np.linalg.norm(boxes.h[o, :2]) + clear + 0.2
    point = np.array([*(boxes.c[o, :2] + side * reach * perp), mid[2]])
    return [point]


def plan(c: course_mod.Course, start: np.ndarray, v_max: float = 10.0, a_max: float = 10.0, approach: float = 0.8,
         clear: float = REACH + 0.25, hop_height: float = 4.0, max_rounds: int = 30) -> trajectory.Trajectory:
    boxes = Boxes(c)
    w = route(c, start, approach)
    pts = [w[0]]
    for a, b in zip(w, w[1:]):
        pts += detour(a, b, boxes, clear, hop_height)
        pts.append(b)
    w = trajectory.subdivide(np.array(pts), 3.0)
    for _ in range(max_rounds):
        tr = trajectory._allocate(w, v_max, a_max, passes=4)
        t, p, _, _ = tr.sample()
        d = boxes.distance(p).min(axis=1) if len(boxes) else np.full(len(p), np.inf)
        if d.min() >= clear - 0.05:
            return tr
        starts = np.concatenate([[0.0], np.cumsum(tr.durations)])
        bad = sorted({int(np.clip(np.searchsorted(starts, ti, side="right") - 1, 0, len(tr.durations) - 1))
                      for ti in t[d < clear - 0.05]})
        new = [w[0]]
        for i in range(len(w) - 1):
            if i in bad:
                new.append((w[i] + w[i + 1]) / 2)
            new.append(w[i + 1])
        w = np.array(new)
    raise ValueError("path still too close to an obstacle")


def start_point(c: course_mod.Course) -> tuple[np.ndarray, float]:
    """Where a race starts: the spawn point on a point-to-point course, else the loop's start line; and the heading
    to gate 0."""
    g0 = c.gates[0].xyz.astype(float)
    if not c.extra.get("loop", True):
        s = c.spawn_xyz.astype(float)
    else:
        last = c.gates[-1].xyz.astype(float)
        s = last + 0.3 * (g0 - last)
    return s, float(np.arctan2(*(g0 - s)[[1, 0]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", default="long_track")
    ap.add_argument("--v-max", type=float, default=10.0)
    ap.add_argument("--a-max", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = course_mod.load(a.course)
    start, _ = start_point(c)
    tr = plan(c, start, a.v_max, a.a_max)
    t, p, v, acc = tr.sample(0.01)
    out = a.out or str(OUT / f"path_{a.course}.npz")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, t=t, p=p, v=v, a=acc, waypoints=tr.waypoints)
    d = Boxes(c).distance(p)
    print(f"{len(tr.waypoints)} waypoints, {tr.total:.2f} s, top speed {np.linalg.norm(v, axis=1).max():.1f} m/s, "
          f"closest to an obstacle {d.min() if d.size else float('inf'):.2f} m -> {out}")


if __name__ == "__main__":
    main()
