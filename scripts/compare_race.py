"""Fly the same exported policy from the same start in MuJoCo (CPU, numpy policy) and overlay it on a Gazebo flight.

    python scripts/compare_race.py   # (playground image: needs mujoco + matplotlib)
        outputs/gates_02/policy_000024576000.npz outputs/gates_02/gazebo_race.csv

Prints gates passed in each and how far apart the two paths are, and saves a top-down + side plot next to the CSV.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from bumble_bee_gym import course as course_mod, hoop, params as params_mod
from bumble_bee_gym.policy_np import Policy



def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


policy_path, gz_csv = sys.argv[1], Path(sys.argv[2])
gz = np.genfromtxt(gz_csv, delimiter=",", names=True)
T = float(gz["t"][-1])
c = course_mod.load("race_gates")
p = params_mod.load()
m = mujoco.MjModel.from_xml_string(hoop.scene_xml(c, p, train=True))
m.opt.timestep = 0.002
d = mujoco.MjData(m)
pol = Policy(policy_path)
frames = [hoop.hoop_frame(g) for g in c.gates]
C, Rg = np.stack([f[0] for f in frames]), np.stack([f[1] for f in frames])
half = np.array([np.array(g.opening) / 2 for g in c.gates])
n = len(c.gates)
hover, wmax = float(p.hover_speed()), float(p.max_rotor_speed)
c2, c1, c0 = p.thrust_map

start = C[-1] + 0.3 * (C[0] - C[-1])
yaw = np.arctan2(*(C[0] - start)[[1, 0]])
d.qpos[:3], d.qpos[3:7] = start, [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
mujoco.mj_forward(m, d)
k, passed, last, prev, path = 0, 0, np.zeros(4), None, []
for step in range(int(round(T / 0.01))):
    pos, R = d.qpos[:3].copy(), quat_to_mat(*d.qpos[3:7])
    p1 = Rg[k].T @ (pos - C[k])
    if prev is not None and prev[0] < 0 <= p1[0]:
        yz = (prev + (-prev[0] / max(p1[0] - prev[0], 1e-6)) * (p1 - prev))[1:]
        if np.all(np.abs(yz) < half[k]):
            passed, k = passed + 1, (k + 1) % n
    prev = Rg[k].T @ (pos - C[k])
    nx = (k + 1) % n
    o = np.hstack([Rg[k].T @ (pos - C[k]), Rg[k].T @ d.qvel[:3], (Rg[k].T @ R).ravel(), d.qvel[3:6], last,
                   Rg[k].T @ (C[nx] - C[k]), Rg[k].T @ Rg[nx][:, 0], np.eye(n)[k]])
    last = pol(o)
    w = np.clip(hover + hoop.SPEED_SPAN * last, 0, wmax)
    d.ctrl[:] = np.maximum(0, c2 * w * w + c1 * w + c0)
    mujoco.mj_step(m, d, 5)
    path.append(pos)
mj = np.array(path)
g = np.stack([gz["x"], gz["y"], gz["z"]], 1)
L = min(len(mj), len(g))
gap = np.linalg.norm(mj[:L] - g[:L], axis=1)
print(f"MuJoCo: {passed} gates   Gazebo: {int(gz['gates'][-1])} gates   path gap: median {np.median(gap)*100:.1f} cm, "
      f"max {gap.max()*100:.1f} cm (first lap max {gap[:350].max()*100:.1f} cm)")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.4, 1]})
for gt, cc in zip(c.gates, C):
    nrm = Rg[list(C[:, 0]).index(cc[0])][:, 1] * gt.opening[0] / 2
    ax[0].plot([cc[0] - nrm[0], cc[0] + nrm[0]], [cc[1] - nrm[1], cc[1] + nrm[1]], color="#d35400", lw=4)
ax[0].plot(mj[:, 0], mj[:, 1], lw=1.2, label="MuJoCo (trained here)")
ax[0].plot(g[:, 0], g[:, 1], "--", lw=1.2, label="Gazebo (bumble-bee plugin)")
ax[0].set_aspect("equal"); ax[0].set_title("Top view, 15 s (4 laps)"); ax[0].legend(); ax[0].set_xlabel("x [m]")
ax[0].set_ylabel("y [m]")
t = np.arange(L) * 0.01
ax[1].plot(t, gap * 100); ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("distance between the two paths [cm]")
ax[1].set_title("MuJoCo vs Gazebo, same policy, same start")
fig.tight_layout()
outp = gz_csv.with_name("gazebo_vs_mujoco.png")
fig.savefig(outp, dpi=130)
print(outp)
