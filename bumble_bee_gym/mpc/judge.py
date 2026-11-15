"""The race's rules in numpy: GatesEnv's gate-pass and failure checks, for flights flown outside JAX (NMPC, PPO on the
CPU, replays). No CasADi, so it also runs in the GPU image that films the flights."""

from __future__ import annotations

import numpy as np

from .. import course as course_mod, hoop, worlds


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Judge:
    """GatesEnv's pass / fail rules, in numpy. On a point-to-point course (loop: false) going through the last gate
    sets ``finished``; touching an obstacle (the course's buildings) is a failure like touching a frame."""

    def __init__(self, c: course_mod.Course, p, ground=0.1, max_dist=12.0, miss_radius=2.0):
        frames = [hoop.hoop_frame(g) for g in c.gates]
        self.C = np.stack([f[0] for f in frames])
        self.R = np.stack([f[1] for f in frames])
        self.half = np.array([np.array(g.opening) / 2 for g in c.gates])
        self.bar_c = np.array([[b[0] for b in g.bars()] for g in c.gates])     # (G, 4, 3)
        self.bar_h = np.array([[b[1] for b in g.bars()] for g in c.gates])
        self.sph_c, self.sph_r = hoop.collision_spheres(p)
        self.n = len(c.gates)
        self.ground, self.max_dist, self.miss_radius = ground, max_dist, miss_radius
        self.k, self.passed, self.finished = 0, 0, False
        self.loop = bool(c.extra.get("loop", True))
        self.obs_c = np.array([[b.xyz[0], b.xyz[1], b.size[2] / 2] for b in c.buildings]).reshape(-1, 3)
        self.obs_h = np.array([np.asarray(b.size, float) / 2 for b in c.buildings]).reshape(-1, 3)
        self.obs_r = np.array([worlds._rot_z(b.yaw) for b in c.buildings]).reshape(-1, 3, 3)

    def touches(self, pos, R) -> bool:
        centres = pos + self.sph_c @ R.T                                          # (5, 3)
        local = np.einsum("gsj,gjk->gsk", centres[None] - self.C[:, None], self.R)  # (G, 5, 3), gate frames
        gap = np.abs(local[:, :, None] - self.bar_c[:, None]) - self.bar_h[:, None]   # (G, 5, 4, 3)
        d = np.linalg.norm(np.maximum(gap, 0.0), axis=-1)
        return bool(np.any(d < self.sph_r[None, :, None]))

    def touches_obstacle(self, pos, R) -> bool:
        if not len(self.obs_c):
            return False
        centres = pos + self.sph_c @ R.T
        local = np.einsum("osj,ojk->osk", centres[None] - self.obs_c[:, None], self.obs_r)   # (O, 5, 3)
        d = np.linalg.norm(np.maximum(np.abs(local) - self.obs_h[:, None], 0.0), axis=-1)
        return bool(np.any(d < self.sph_r[None]))

    def step(self, pos0, pos1, R1) -> str | None:
        """Update with one control step's motion; returns the failure, if any ('frame', 'missed', ...)."""
        k = self.k
        p0, p1 = self.R[k].T @ (pos0 - self.C[k]), self.R[k].T @ (pos1 - self.C[k])
        hit = self.touches(pos1, R1)
        if p0[0] < 0.0 <= p1[0]:
            yz = (p0 + (-p0[0] / max(p1[0] - p0[0], 1e-6)) * (p1 - p0))[1:]
            inside = np.all(np.abs(yz) < self.half[k])
            if inside and not hit:
                self.passed += 1
                if not self.loop and k == self.n - 1:
                    self.finished = True
                    return None
                self.k = (k + 1) % self.n
            elif not inside and np.all(np.abs(yz) < self.miss_radius) and not hit:
                return "missed"
        if hit:
            return "frame"
        if self.touches_obstacle(pos1, R1):
            return "obstacle"
        if R1[2, 2] < 0.0:
            return "flipped"
        if pos1[2] < self.ground:
            return "ground"
        if np.linalg.norm(p1) > self.max_dist:
            return "away"
        return None
