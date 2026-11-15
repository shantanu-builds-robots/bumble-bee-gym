"""The drone as a rigid body driven by four rotor thrusts, in CasADi, for planning and model predictive control.

Built from the same parameters as the MuJoCo twin and the Gazebo model (bumble-bee's URDF, via params.load()), with
nothing fitted:

  state x (13):   position of the centre of mass (world), its velocity (world), attitude quaternion (w, x, y, z,
                  body to world), body rates (body frame)
  control u (4):  thrust of each rotor [N], along its axis

  p' = v
  v' = R(q) sum(f_i a_i) / m - g e_z
  q' = q (x) (0, w) / 2
  w' = J^-1 (tau - w x J w),   tau = sum (r_i - com) x f_i a_i + s_i k f_i a_i

with a_i the rotor axis, r_i its hub, s_i its spin sign and k the drag torque per newton (the yaw reaction torque the
Gazebo plugin and the MuJoCo actuators apply). There is no aerodynamic drag: neither simulator models any.

MuJoCo's free joint reports the body origin (base_link), not the centre of mass, with its angular velocity in the
body frame; :func:`state_from_mujoco` converts.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .. import hoop
from ..params import DroneParams
from ..worlds import GRAVITY

NX, NU = 13, 4


class Model:
    def __init__(self, p: DroneParams, g: float = GRAVITY, f_max: float | None = None):
        self.p, self.g = p, g
        self.mass = p.mass
        self.com = np.asarray(p.com, float)
        self.J = np.asarray(p.inertia, float)
        self.hover_thrust = p.mass * g / 4
        # Per-rotor limit: by default what the PPO racer can command (hover speed + hoop.SPEED_SPAN), so the two
        # controllers race on the same motors.
        self.f_max = float(f_max if f_max is not None else p.thrust(p.hover_speed() + hoop.SPEED_SPAN))
        arms = np.stack([r.position - self.com for r in p.rotors])           # (4, 3) from the CoM
        axes = np.stack([r.axis for r in p.rotors])                           # (4, 3)
        signs = np.array([r.spin_sign for r in p.rotors])
        self.force_map = axes.T                                               # body force = force_map @ f
        self.torque_map = (np.cross(arms, axes) + p.torque_to_thrust * signs[:, None] * axes).T   # (3, 4)

        x, u = ca.SX.sym("x", NX), ca.SX.sym("u", NU)
        self.f = ca.Function("f", [x, u], [self._xdot(x, u)], ["x", "u"], ["xdot"])

    def _xdot(self, x, u):
        v, q, w = x[3:6], x[6:10], x[10:13]
        R = quat_to_mat(q)
        acc = R @ (ca.DM(self.force_map) @ u) / self.mass - ca.vertcat(0, 0, self.g)
        qdot = 0.5 * quat_mul(q, ca.vertcat(0, w))
        J = ca.DM(self.J)
        wdot = ca.solve(J, ca.DM(self.torque_map) @ u - ca.cross(w, J @ w))
        return ca.vertcat(v, acc, qdot, wdot)

    def rk4(self, dt, steps: int = 1) -> ca.Function:
        """x_next = F(x, u) over dt (fixed float or a symbol passed as the third input), `steps` RK4 steps."""
        x, u, h = ca.SX.sym("x", NX), ca.SX.sym("u", NU), ca.SX.sym("dt")
        y, s = x, h / steps
        for _ in range(steps):
            k1 = self.f(y, u)
            k2 = self.f(y + s / 2 * k1, u)
            k3 = self.f(y + s / 2 * k2, u)
            k4 = self.f(y + s * k3, u)
            y = y + s / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        y = ca.vertcat(y[:6], y[6:10] / ca.norm_2(y[6:10]), y[10:])
        F = ca.Function("F", [x, u, h], [y], ["x", "u", "dt"], ["x_next"])
        if dt is None:
            return F
        return ca.Function("F", [x, u], [F(x, u, dt)], ["x", "u"], ["x_next"])

    # ------------------------------------------------------------------ MuJoCo <-> model state

    def state_from_mujoco(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        q = np.asarray(qpos[3:7], float)
        R = quat_to_mat_np(q)
        w = np.asarray(qvel[3:6], float)                                    # body frame
        p = qpos[0:3] + R @ self.com
        v = qvel[0:3] + R @ np.cross(w, self.com)
        return np.concatenate([p, v, q, w])

    def body_origin(self, x: np.ndarray) -> np.ndarray:
        """base_link origin (what MuJoCo and Gazebo report) for model states (..., 13)."""
        x = np.atleast_2d(x)
        return np.stack([xi[:3] - quat_to_mat_np(xi[6:10]) @ self.com for xi in x])


def quat_to_mat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        ca.horzcat(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        ca.horzcat(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def quat_to_mat_np(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    aw, ax, ay, az = a[0], a[1], a[2], a[3]
    bw, bx, by, bz = b[0], b[1], b[2], b[3]
    return ca.vertcat(aw * bw - ax * bx - ay * by - az * bz,
                      aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx,
                      aw * bz + ax * by - ay * bx + az * bw)


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
