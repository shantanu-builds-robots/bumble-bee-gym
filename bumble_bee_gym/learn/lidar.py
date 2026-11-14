"""A simulated lidar for the racing drone: beams fixed to the body, measuring the distance to the course's obstacles,
gate frames and the ground.

Beams: ``azimuths`` directions around the drone at each of ``elevations`` (degrees, in the body frame: 0 = along the
props' plane, negative = down), so with the defaults 12 x 3 = 36 beams that see walls below and above as well as
pillars to the side. Each returns the distance to the first thing it hits, up to ``range`` metres, scaled to [0, 1]
(1 = nothing within range).

Everything the drone can hit is a box (obstacles stand on the ground; each gate frame is four bars), so a beam is an
exact ray-box intersection (the slab method) against every box, plus the ground plane. Same geometry in JAX (for
training, vmapped over beams and boxes) and in numpy (for flying a policy on the CPU), from one `Scene`.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bumble_bee_gym import hoop, worlds

DEFAULTS = dict(enabled=False, range=8.0, azimuths=12, elevations=[-25.0, 0.0, 25.0])


def default_config():
    from ml_collections import config_dict      # training only; flying on the CPU uses settings()
    return config_dict.create(**DEFAULTS)


def settings(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{**DEFAULTS, **overrides})


class Scene:
    """Beam directions (body frame) and every box the beams can hit (world frame)."""

    def __init__(self, course, cfg):
        az = np.radians(np.arange(cfg.azimuths) * 360.0 / cfg.azimuths)
        el = np.radians(np.asarray(cfg.elevations, float))
        self.dirs = np.array([[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)] for e in el for a in az])
        self.range = float(cfg.range)
        cs, hs, rs = [], [], []
        for b in course.buildings:
            cs.append([b.xyz[0], b.xyz[1], b.size[2] / 2])
            hs.append(np.asarray(b.size, float) / 2)
            rs.append(worlds._rot_z(b.yaw))
        for g in course.gates:
            centre, rot = hoop.hoop_frame(g)
            for bc, bh in g.bars():
                cs.append(centre + rot @ bc)
                hs.append(bh)
                rs.append(rot)
        self.c, self.h, self.r = np.array(cs, float), np.array(hs, float), np.array(rs, float)

    @property
    def n(self) -> int:
        return len(self.dirs)

    def numpy(self, pos: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Scaled distances (n,) for a drone at pos with attitude R (body to world)."""
        d_w = self.dirs @ R.T                                                   # (n, 3)
        o = np.einsum("bj,bjk->bk", pos[None] - self.c, self.r)                 # (B, 3) box frames
        d = np.einsum("nj,bjk->bnk", d_w, self.r)                               # (B, n, 3)
        inv = 1.0 / np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (-self.h[:, None] - o[:, None]) * inv, (self.h[:, None] - o[:, None]) * inv
        tmin, tmax = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
        hit = tmax >= np.maximum(tmin, 0.0)
        dist = np.where(hit, np.maximum(tmin, 0.0), np.inf).min(0)
        ground = np.where(d_w[:, 2] < -1e-6, -pos[2] / np.where(d_w[:, 2] < -1e-6, d_w[:, 2], -1.0), np.inf)
        return np.minimum(np.minimum(dist, ground), self.range) / self.range

    def jax_fn(self):
        import jax.numpy as jp
        dirs, c, h, r, rng = (jp.array(self.dirs), jp.array(self.c), jp.array(self.h), jp.array(self.r),
                              self.range)

        def scan(pos, R):
            d_w = dirs @ R.T
            o = jp.einsum("bj,bjk->bk", pos[None] - c, r)
            d = jp.einsum("nj,bjk->bnk", d_w, r)
            inv = 1.0 / jp.where(jp.abs(d) < 1e-9, 1e-9, d)
            t1, t2 = (-h[:, None] - o[:, None]) * inv, (h[:, None] - o[:, None]) * inv
            tmin, tmax = jp.minimum(t1, t2).max(-1), jp.maximum(t1, t2).min(-1)
            hit = tmax >= jp.maximum(tmin, 0.0)
            dist = jp.where(hit, jp.maximum(tmin, 0.0), jp.inf).min(0)
            down = d_w[:, 2] < -1e-6
            ground = jp.where(down, -pos[2] / jp.where(down, d_w[:, 2], -1.0), jp.inf)
            return jp.minimum(jp.minimum(dist, ground), rng) / rng

        return scan
