"""Drone parameters, read from the Bumble Bee URDF so the MuJoCo twin can never drift from Gazebo.

The single source of truth is ``drone_delivery/urdf/drone.urdf.xacro`` in the bumble-bee repo.
Point ``BUMBLE_BEE_ROOT`` at a checkout, or keep it next to this repo (``../bumble-bee``).
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

URDF_RELPATH = Path("drone_delivery/urdf/drone.urdf.xacro")
MESH_RELPATH = Path("drone_delivery/meshes")
PROP_LINKS = ("prop_1", "prop_2", "prop_3", "prop_4")
# Spin direction of each rotor as commanded by fly.cpp ({p1, -p2, p3, -p4}); the plugin's yaw
# reaction torque on the body takes the sign of the commanded speed.
PROP_SPIN_SIGN = (1.0, -1.0, 1.0, -1.0)


def bumble_bee_root() -> Path:
    """Locate the bumble-bee checkout."""
    env = os.environ.get("BUMBLE_BEE_ROOT")
    root = Path(env) if env else Path(__file__).resolve().parents[2] / "bumble-bee"
    if not (root / URDF_RELPATH).is_file():
        raise FileNotFoundError(
            f"bumble-bee URDF not found under {root}; set BUMBLE_BEE_ROOT to your bumble-bee checkout")
    return root


@dataclass(frozen=True)
class Rotor:
    name: str
    position: np.ndarray  # rotor hub in the base_link frame [m]
    axis: np.ndarray      # thrust direction in the base_link frame (unit)
    spin_sign: float      # +1 or -1, sign of the yaw reaction torque on the body


@dataclass(frozen=True)
class DroneParams:
    mass: float                       # whole drone [kg]
    com: np.ndarray                   # centre of mass in the base_link frame [m]
    inertia: np.ndarray               # 3x3 about the CoM, base_link axes [kg m^2]
    rotors: tuple[Rotor, ...]
    thrust_map: tuple[float, float, float]  # thrust [N] = c2 w^2 + c1 w + c0, never negative
    torque_to_thrust: float           # yaw reaction torque per newton of thrust [m]
    max_rotor_speed: float            # [rad/s]
    mesh_dir: Path
    link_meshes: dict = field(default_factory=dict)  # link name -> (mesh file, pose 4x4 in base_link)

    def thrust(self, w: np.ndarray | float) -> np.ndarray:
        """Per-rotor thrust [N] for rotor speeds [rad/s], exactly as drone_plugin computes it."""
        c2, c1, c0 = self.thrust_map
        s = np.minimum(np.abs(np.asarray(w, dtype=float)), self.max_rotor_speed)
        return np.maximum(0.0, c2 * s * s + c1 * s + c0)

    def hover_speed(self, extra_mass: float = 0.0, g: float = 9.8) -> float:
        """Rotor speed [rad/s] at which four rotors carry the drone (plus extra_mass)."""
        c2, c1, c0 = self.thrust_map
        t = (self.mass + extra_mass) * g / 4.0
        return (-c1 + np.sqrt(c1 * c1 - 4.0 * c2 * (c0 - t))) / (2.0 * c2)


def _resolve_xacro(text: str) -> str:
    """Substitute simple ${...} expressions using the file's xacro:property values."""
    props = {m.group(1): m.group(2) for m in re.finditer(
        r'<xacro:property\s+name="([^"]+)"\s+value="([^"]+)"\s*/>', text)}
    env = {}
    for k, v in props.items():
        try:
            env[k] = float(eval(v, {"__builtins__": {}}, env))  # noqa: S307 - trusted local file
        except Exception:  # non-numeric property, leave it out
            pass
    text = re.sub(r"\$\(find [^)]+\)", "", text)
    return re.sub(r"\$\{([^}]*)\}", lambda m: repr(eval(m.group(1), {"__builtins__": {}}, env)), text)  # noqa: S307


def _rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _origin(elem) -> np.ndarray:
    """4x4 transform from a URDF <origin> (identity if missing)."""
    t = np.eye(4)
    if elem is None:
        return t
    t[:3, :3] = _rpy_matrix([float(v) for v in elem.get("rpy", "0 0 0").split()])
    t[:3, 3] = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    return t


def _link_poses(robot) -> dict[str, np.ndarray]:
    """Pose of every link in the base_link frame, following the joint tree from base_link."""
    poses = {"base_link": np.eye(4)}
    joints = [(j.find("parent").get("link"), j.find("child").get("link"), _origin(j.find("origin")))
              for j in robot.findall("joint")]
    changed = True
    while changed:
        changed = False
        for parent, child, t in joints:
            if parent in poses and child not in poses:
                poses[child] = poses[parent] @ t
                changed = True
    return poses


def load(root: Path | None = None) -> DroneParams:
    """Read the drone's parameters from the bumble-bee URDF."""
    root = root or bumble_bee_root()
    text = _resolve_xacro((root / URDF_RELPATH).read_text())
    robot = ET.fromstring(text)
    poses = _link_poses(robot)

    # Composite mass, CoM and inertia over every link that has an inertial (the props spin, but
    # their inertia about the spin axis is tiny, so treating them as rigid is what Gazebo's
    # fixed-joint merge plus held prop joints amounts to as well).
    parts = []
    for link in robot.findall("link"):
        inertial = link.find("inertial")
        name = link.get("name")
        if inertial is None or name not in poses:
            continue
        m = float(inertial.find("mass").get("value"))
        frame = poses[name] @ _origin(inertial.find("origin"))
        i = inertial.find("inertia")
        ii = {k: float(i.get(k)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        local = np.array([[ii["ixx"], ii["ixy"], ii["ixz"]],
                          [ii["ixy"], ii["iyy"], ii["iyz"]],
                          [ii["ixz"], ii["iyz"], ii["izz"]]])
        rot = frame[:3, :3]
        parts.append((m, frame[:3, 3], rot @ local @ rot.T))
    mass = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / mass
    inertia = np.zeros((3, 3))
    for m, c, i_c in parts:
        d = c - com
        inertia += i_c + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    rotors = []
    for name, sign in zip(PROP_LINKS, PROP_SPIN_SIGN):
        pose = poses[name]
        rotors.append(Rotor(name, pose[:3, 3].copy(), pose[:3, :3] @ np.array([0.0, 1.0, 0.0]), sign))

    plugin = next(p for p in robot.iter("plugin") if p.get("filename") == "libdrone.so")
    thrust_map = tuple(float(v) for v in plugin.findtext("thrustMap").split())
    meshes = {}
    for link in robot.findall("link"):
        vis = link.find("visual")
        mesh = vis.find("geometry/mesh") if vis is not None else None
        if mesh is not None and link.get("name") in poses:
            meshes[link.get("name")] = (Path(mesh.get("filename")).name,
                                        poses[link.get("name")] @ _origin(vis.find("origin")))
    return DroneParams(
        mass=mass, com=com, inertia=inertia, rotors=tuple(rotors), thrust_map=thrust_map,
        torque_to_thrust=float(plugin.findtext("torqueToThrust")),
        max_rotor_speed=float(plugin.findtext("maxRotorSpeed")),
        mesh_dir=root / MESH_RELPATH, link_meshes=meshes)
