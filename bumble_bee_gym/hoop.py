"""The single-hoop task's fixed pieces: the scene, the hoop's frame, and how actions become rotor speeds.

Kept free of JAX so it can be checked on the host; the GPU environment (``learn/hoop_env.py``) builds on it.

Actions are what bumble-bee's ``fly`` node sends: four rotor speeds. The policy outputs a in [-1, 1] per rotor and
the speed is ``hover + SPEED_SPAN * a`` (clipped to [0, max]), so a = 0 is hover. Thrust then follows the drone
plugin's own thrust map, which gives zero below about 400 rad/s and 3.8x the drone's weight at a = 1.
"""

from __future__ import annotations

import re

import numpy as np

from . import params as params_mod
from . import worlds
from .course import Course, Gate

SPEED_SPAN = 750.0     # [rad/s] rotor speed change at |a| = 1


def rotor_speeds(a: np.ndarray, p: params_mod.DroneParams) -> np.ndarray:
    """Policy action in [-1, 1] -> commanded rotor speeds [rad/s]."""
    return np.clip(p.hover_speed() + SPEED_SPAN * np.clip(a, -1.0, 1.0), 0.0, p.max_rotor_speed)


def hoop_frame(g: Gate) -> tuple[np.ndarray, np.ndarray]:
    """(centre, R) with R's columns = the hoop's axes in the world: x through the hoop, y across, z up."""
    return g.xyz.astype(float), worlds._rot_z(g.yaw)


PROP_RADIUS = 0.07     # [m] prop sphere around each rotor hub (half-prop 0.064 m)
BODY_RADIUS = 0.06     # [m] sphere at the centre; the prop spheres cover the body's corners


def collision_spheres(p: params_mod.DroneParams) -> tuple[np.ndarray, np.ndarray]:
    """(centres in the body frame (5, 3), radii (5,)): one sphere per prop plus one for the body.

    What touches the hoop's frame is checked with these, so a level drone (6 cm tall) fits through much more of the
    opening than its 0.37 m span would suggest, and a tilted or turned one fits through less."""
    centres = np.vstack([np.zeros(3)] + [r.position for r in p.rotors])
    return centres, np.array([BODY_RADIUS] + [PROP_RADIUS] * len(p.rotors))


def scene_xml(course: Course, p: params_mod.DroneParams | None = None, train: bool = False) -> str:
    """MuJoCo scene of the course with the drone.

    ``train=True`` gives the physics-only version the GPU trains on: no meshes, and contacts turned off (crashes
    into the ground or the frame are detected geometrically by the task, which is cheaper than collision
    detection). The drone's mass, inertia and actuators are identical in both."""
    xml = worlds.to_mjcf(course, p)
    if not train:
        return xml
    xml = re.sub(r'\s*<mesh name="[^"]+" file=[^>]+/>', "", xml)
    xml = re.sub(r'\s*<geom name="[^"]+_visual" type="mesh"[^>]*/>', "", xml)
    xml = re.sub(r'\s*<texture [^>]*/>|\s*<material [^>]*/>', "", xml)
    xml = xml.replace(' material="grid"', "")
    return xml.replace('integrator="Euler"/>', 'integrator="Euler">\n    <flag contact="disable"/>\n  </option>')


class PropSpinner:
    """Turns the propeller meshes of a rendering model so the props visibly spin in replays.

    Replays set only the drone's free joint, and the props are fixed geoms on the body, so without this they never
    move. Each call advances every prop by ``step`` radians about its rotor axis, in its real spin direction. Real
    props turn ~800 rad/s, far past what a 50 fps video can show (it would alias into a random flicker), so the
    default 0.6 rad per frame is a readable stand-in, not the true speed."""

    def __init__(self, m, p: params_mod.DroneParams, step: float = 0.6):
        import mujoco
        self._mujoco, self._m, self._step = mujoco, m, step
        self._props = []
        for r in p.rotors:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{r.name}_visual")
            if gid >= 0:
                self._props.append((gid, m.geom_quat[gid].copy(), np.asarray(r.axis, float), r.spin_sign))
        self._angle = 0.0

    def __call__(self) -> None:
        self._angle += self._step
        for gid, q0, axis, sign in self._props:
            q = np.zeros(4)
            self._mujoco.mju_axisAngle2Quat(q, axis, sign * self._angle)
            out = np.zeros(4)
            self._mujoco.mju_mulQuat(out, q, q0)
            self._m.geom_quat[gid] = out
