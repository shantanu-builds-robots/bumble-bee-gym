"""Gazebo vs MuJoCo parity: one open-loop rotor-speed profile, flown from the same mid-air start in both.

Gazebo side: ``scripts/gazebo_parity_probe.py`` (a ROS 2 node) records ``gazebo_parity.csv`` and the sim
times at which each command was actually sent. MuJoCo side: :func:`simulate_mujoco` replays those
switch times. :func:`compare` reports the largest differences.

This module only needs numpy so the probe can import it inside the ROS container.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HOVER = 738.4          # rad/s, from the plugin's thrust map for 0.73 kg at g = 9.8
START_Z = 2.0          # m
DURATION = 2.0         # s

# (start time [s], speeds of props 1..4 [rad/s, magnitudes]). Props 1,2 are on +y, 1,4 on -x;
# props 1,3 spin positive and 2,4 negative.
PROFILE = [
    (0.0, (HOVER, HOVER, HOVER, HOVER)),                        # hover
    (0.5, (HOVER + 30, HOVER + 30, HOVER + 30, HOVER + 30)),    # collective step: climb
    (1.0, (HOVER + 15, HOVER + 15, HOVER - 15, HOVER - 15)),    # roll step: +y side pushes harder
    (1.2, (HOVER, HOVER, HOVER, HOVER)),
    (1.5, (HOVER + 40, HOVER - 40, HOVER + 40, HOVER - 40)),    # yaw step: props 1,3 faster
    (1.8, (HOVER, HOVER, HOVER, HOVER)),
]
SPIN_SIGN = np.array([1.0, -1.0, 1.0, -1.0])


def command_at(t: float, profile=PROFILE) -> np.ndarray:
    """Rotor speed magnitudes active at time t."""
    current = profile[0][1]
    for start, speeds in profile:
        if t >= start:
            current = speeds
    return np.array(current, dtype=float)


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def simulate_mujoco(xml_path: Path, switch_times: list[float] | None = None,
                    start: dict | None = None) -> dict[str, np.ndarray]:
    """Fly the profile in MuJoCo. switch_times: the sim times Gazebo actually applied each segment."""
    import mujoco

    from . import params as params_mod

    p = params_mod.load()
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    if start is not None:  # start exactly where Gazebo started
        d.qpos[0:3] = start["pos"]
        d.qpos[3:7] = start["quat_wxyz"]
    mujoco.mj_forward(m, d)
    times = switch_times or [s for s, _ in PROFILE]
    profile = [(t, speeds) for t, (_, speeds) in zip(times, PROFILE)]
    n = int(round(DURATION / m.opt.timestep))
    out = {k: np.zeros(n + 1) for k in ("t", "x", "y", "z", "qw", "qx", "qy", "qz", "wx", "wy", "wz")}

    def record(i):
        out["t"][i] = d.time
        out["x"][i], out["y"][i], out["z"][i] = d.qpos[0:3]
        out["qw"][i], out["qx"][i], out["qy"][i], out["qz"][i] = d.qpos[3:7]
        out["wx"][i], out["wy"][i], out["wz"][i] = d.qvel[3:6]  # body-frame angular velocity

    record(0)
    for i in range(1, n + 1):
        d.ctrl[:] = p.thrust(command_at(d.time, profile))
        mujoco.mj_step(m, d)
        record(i)
    return out


def euler(qw, qx, qy, qz):
    roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def compare(gz: dict, mj: dict) -> dict[str, float]:
    """Largest absolute differences over the run, Gazebo samples interpolated onto MuJoCo's times."""
    t = mj["t"]
    res = {}
    for k in ("x", "y", "z"):
        res[f"max_d{k}_m"] = float(np.max(np.abs(np.interp(t, gz["t"], gz[k]) - mj[k])))
    g_e = euler(*(np.interp(t, gz["t"], gz[k]) for k in ("qw", "qx", "qy", "qz")))
    m_e = euler(mj["qw"], mj["qx"], mj["qy"], mj["qz"])
    for name, a, b in zip(("roll", "pitch", "yaw"), g_e, m_e):
        res[f"max_d{name}_deg"] = float(np.degrees(np.max(np.abs(np.angle(np.exp(1j * (a - b)))))))
    return res
