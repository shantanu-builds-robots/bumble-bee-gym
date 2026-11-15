"""The classical racer: bumble-bee's cascaded PID flight controller following a planned path (path.py).

A port of bumble-bee's fly node (drone_delivery/src/fly.cpp), same structure and same motor mixer:

  position (x, y):  error, reference velocity and the tilt that gives the reference acceleration
                    (a = g tan(tilt))  ->  desired pitch (x) and roll (-y), clamped to max_tilt
  height:           hover thrust for the reference vertical acceleration, divided by cos(roll) cos(pitch) so
                    tilting does not lose height, + PID on the height error  ->  common rotor speed
  attitude:         roll, pitch, yaw PIDs (yaw held at 0)  ->  rotor speed differences
  mixer:            prop_1 = base + h + r + p + y,  prop_2 = base + h + r - p - y,
                    prop_3 = base + h - r - p + y,  prop_4 = base + h - r + p - y

What differs from fly.cpp is the tuning, for racing instead of parcel delivery. fly.cpp's gains come from its plant
model (near hover: z'' = 0.0318 u, roll/pitch acceleration = 0.5 u per rad/s of rotor speed difference, x'' = g tilt)
with slow, gentle targets (position 0.8 rad/s, tilt within 0.35 rad). Here the same plant numbers give gains for
position 4 rad/s, attitude 20 rad/s and height 5 rad/s, all with damping 0.8 to 0.9 (on long_track, 3 / 15 / 4 lag
far enough to clip a 0.7 m gate, and 6 / 30 / 6 oscillates into the ground), the height loop gets
feedforward, and tilt may reach 0.8 rad. Velocities come from the simulator (fly.cpp differentiates the pose, which
at 100 Hz is the same up to noise). Rotor speeds are limited as for the other racers (hover + hoop.SPEED_SPAN).
"""

from __future__ import annotations

import numpy as np

from .. import hoop
from ..params import DroneParams
from ..worlds import GRAVITY


def _gains(wn: float, zeta: float, plant: float, ki_ratio: float = 0.0):
    """PID gains for a double integrator x'' = plant * u: kp = wn^2 / plant, kd = 2 zeta wn / plant."""
    kp, kd = wn * wn / plant, 2 * zeta * wn / plant
    return kp, ki_ratio * kp, kd


class PathPID:
    def __init__(self, p: DroneParams, path_npz, max_tilt: float = 0.8, g: float = GRAVITY):
        z = np.load(path_npz)
        self.t, self.p, self.v, self.a = z["t"], z["p"], z["v"], z["a"]
        self.total = float(self.t[-1])
        self.params, self.g, self.max_tilt = p, g, max_tilt
        self.hover = float(p.hover_speed())
        self.w_max = self.hover + hoop.SPEED_SPAN
        self.c2, self.c1, self.c0 = p.thrust_map
        self.mass = p.mass
        self.pos_g = _gains(4.0, 0.9, g, 0.1)          # x'' = g * tilt  -> tilt [rad] per m
        self.z_g = _gains(5.0, 0.8, 0.0318, 0.3)       # z'' = 0.0318 * rotor speed change
        self.att_g = _gains(20.0, 0.8, 0.5, 0.3)       # roll/pitch acceleration = 0.5 * speed difference
        self.yaw_g = _gains(3.0, 0.8, 0.0606, 0.1)
        self.reset()

    def reset(self):
        self.i_xy = np.zeros(2)
        self.i_z = 0.0
        self.i_att = np.zeros(3)

    def ref(self, t: float):
        i = int(np.clip(round(t / 0.01), 0, len(self.t) - 1))
        if t >= self.total:                              # past the end: hold the last point
            return self.p[-1], np.zeros(3), np.zeros(3)
        return self.p[i], self.v[i], self.a[i]

    def speed_for(self, f: float) -> float:
        f = max(f, 0.0)
        return (-self.c1 + np.sqrt(self.c1 ** 2 - 4 * self.c2 * (self.c0 - f))) / (2 * self.c2)

    def __call__(self, pos, vel, quat, rates_body, t: float, dt: float = 0.01) -> np.ndarray:
        """Rotor thrusts [N] (after the speed limit and the thrust map, as the others return) for this state."""
        pr, vr, ar = self.ref(t)
        roll, pitch, yaw = euler(quat)

        # Position -> desired tilt (fly.cpp getRollPitchYaw), with integral and reference feedforward.
        e = pr[:2] - pos[:2]
        self.i_xy = np.clip(self.i_xy + e * dt, -1.0, 1.0)
        kp, ki, kd = self.pos_g
        sig = kp * e + ki * self.i_xy + kd * (vr[:2] - vel[:2]) + np.arctan(ar[:2] / self.g)
        sig = np.clip(sig, -self.max_tilt, self.max_tilt)
        roll_d, pitch_d, yaw_d = -sig[1], sig[0], 0.0

        # Height -> common rotor speed (fly.cpp PID_height_control), on top of the speed that carries the weight
        # plus the reference vertical acceleration at the current tilt.
        ez = pr[2] - pos[2]
        self.i_z = float(np.clip(self.i_z + ez * dt, -1.0, 1.0))
        kp, ki, kd = self.z_g
        tilt_c = max(np.cos(roll) * np.cos(pitch), 0.3)
        base = self.speed_for(self.mass * (self.g + ar[2]) / (4 * tilt_c))
        h = kp * ez + ki * self.i_z + kd * (vr[2] - vel[2])

        # Attitude -> rotor speed differences (fly.cpp PID_orientation_control), derivative on the measured rates.
        err = np.array([roll_d - roll, pitch_d - pitch, np.remainder(yaw_d - yaw + np.pi, 2 * np.pi) - np.pi])
        self.i_att = np.clip(self.i_att + err * dt, -0.5, 0.5)
        out = np.empty(3)
        for k, gains in enumerate((self.att_g, self.att_g, self.yaw_g)):
            kp, ki, kd = gains
            out[k] = kp * err[k] + ki * self.i_att[k] - kd * rates_body[k]
        r, p_, y = out

        w = base + h + np.array([r + p_ + y, r - p_ - y, -r - p_ + y, -r + p_ - y])
        w = np.clip(w, 0.0, self.w_max)
        return np.maximum(0.0, self.c2 * w * w + self.c1 * w + self.c0)


def euler(q) -> tuple[float, float, float]:
    """Roll, pitch, yaw (ZYX, as tf2's getRPY) of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw
