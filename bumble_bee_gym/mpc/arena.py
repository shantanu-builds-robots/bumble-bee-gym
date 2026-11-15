"""The race course restyled for films: a night arena with neon gates, hazard-striped obstacles and checkered flags.

Only the look changes (colours, materials, lights, decorations with no collision); the geometry the racers flew is
the course's own, from worlds.to_mjcf. Gates glow in a colour that climbs with the course's sections (``sections:`` in
the YAML), from cool cyan on the warm-up to red on the needle, so the track visibly gets harder; each opening carries
a faint fill of its colour. ``flags:`` entries become checkered flags on poles, and each consecutive pair (start,
finish) also gets a checkered banner across the top and a checkered line on the ground.
"""

from __future__ import annotations

import re

import numpy as np

from .. import hoop
from ..course import Course
from ..worlds import _f, _rot_z, _yaw_quat

# Section colours, easiest to hardest.
NEON = [(0.0, 0.95, 1.0), (0.2, 1.0, 0.35), (1.0, 0.92, 0.1), (1.0, 0.55, 0.05), (1.0, 0.3, 0.1),
        (1.0, 0.1, 0.75), (1.0, 0.05, 0.25), (1.0, 1.0, 1.0)]


def gate_sections(c: Course) -> list[int]:
    """Section index of every gate."""
    secs = sorted((s["gate"], i) for i, s in enumerate(c.extra.get("sections", [{"gate": 0, "name": ""}])))
    out = []
    for g in range(len(c.gates)):
        out.append(max(i for first, i in secs if first <= g))
    return out


def gate_colour(c: Course, g: int) -> tuple[float, float, float]:
    secs = gate_sections(c)
    n = len(c.extra.get("sections", [None]))
    s = secs[g]
    if s == n - 1 and n > 1:                     # the last section (finish) is white
        return NEON[-1]
    return NEON[min(int(round(s * (len(NEON) - 2) / max(n - 2, 1))), len(NEON) - 2)]


def arena_xml(c: Course, p=None) -> str:
    xml = hoop.scene_xml(c, p)
    # Night: dark sky and haze, a dark floor with faint grid lines, dimmer key light, cool headlight.
    xml = re.sub(r'<texture name="sky"[^>]*/>',
                 '<texture name="sky" type="skybox" builtin="gradient" rgb1="0.05 0.07 0.16" rgb2="0.0 0.0 0.02" '
                 'width="512" height="3072"/>', xml)
    xml = re.sub(r'<texture name="grid"[^>]*/>',
                 '<texture name="grid" type="2d" builtin="checker" rgb1="0.06 0.065 0.09" rgb2="0.05 0.055 0.075" '
                 'mark="edge" markrgb="0.12 0.2 0.38" width="512" height="512"/>', xml)
    xml = xml.replace('<material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>',
                      '<material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.15"/>')
    xml = re.sub(r'<headlight [^>]*/>', '<headlight ambient="0.18 0.2 0.26" diffuse="0.4 0.42 0.5" specular="0.1 0.1 0.1"/>', xml)
    xml = re.sub(r'<map zfar="[^"]*" haze="[^"]*"/>', '<map zfar="250" haze="0.3"/>', xml)
    xml = re.sub(r'<rgba haze="[^"]*"/>', '<rgba haze="0.03 0.04 0.1 1"/>', xml)
    xml = re.sub(r'<light pos="0 0 20"[^>]*/>',
                 '<light pos="0 0 30" dir="-0.3 0.4 -1" directional="true" diffuse="0.45 0.45 0.55" '
                 'castshadow="true"/>', xml)

    assets = [
        '    <texture name="hazard" type="2d" builtin="checker" rgb1="0.95 0.72 0.05" rgb2="0.06 0.06 0.06" '
        'width="256" height="256"/>',
        '    <material name="hazard" texture="hazard" texrepeat="3 3" texuniform="true" specular="0.3"/>',
        '    <texture name="checker" type="2d" builtin="checker" rgb1="0.97 0.97 0.97" rgb2="0.03 0.03 0.03" '
        'width="256" height="256"/>',
        # Checks sized per object (texrepeat counts texture tiles of 2 x 2 checks across the face), so thin flags
        # and banners show squares rather than stripes.
        '    <material name="checker_flag" texture="checker" texrepeat="3 2" emission="0.25"/>',
        '    <material name="pole" rgba="0.75 0.77 0.8 1" specular="0.8" shininess="0.8"/>',
    ]
    for g in range(len(c.gates)):
        r, gg, b = gate_colour(c, g)
        assets.append(f'    <material name="neon_{g}" rgba="{r} {gg} {b} 0.85" emission="0.9" specular="0.5"/>')
        assets.append(f'    <material name="fill_{g}" rgba="{r} {gg} {b} 0.1" emission="0.6"/>')

    # Neon gates, with a faint fill across each opening.
    xml = re.sub(r'(<geom name="gate_(\d+)_bar_\d+"[^>]*?) rgba="[^"]*"/>',
                 lambda m: f'{m.group(1)} material="neon_{m.group(2)}"/>', xml)
    # Obstacles in hazard stripes.
    xml = re.sub(r'(<geom name="building_\d+"[^>]*?) rgba="[^"]*"/>', r'\1 material="hazard"/>', xml)

    deco, extra_assets = [], []
    for i, g in enumerate(c.gates):
        w, h = g.opening
        deco.append(f'    <geom name="gate_{i}_fill" type="box" pos="{_f(g.xyz)}" quat="{_f(_yaw_quat(g.yaw))}" '
                    f'size="0.004 {w / 2} {h / 2}" material="fill_{i}" contype="0" conaffinity="0"/>')
    flags = c.extra.get("flags", [])
    for i, fl in enumerate(flags):
        x, y = fl["xyz"][:2]
        yaw = float(fl.get("yaw", 0))
        rot = _rot_z(yaw + 90)
        deco.append(f'    <geom name="flag_pole_{i}" type="cylinder" pos="{x} {y} 1.6" size="0.04 1.6" material="pole" '
                    'contype="0" conaffinity="0"/>')
        fx = np.array([x, y, 2.85]) + rot @ np.array([0.38, 0, 0])
        deco.append(f'    <geom name="flag_{i}" type="box" pos="{_f(fx)}" quat="{_f(_yaw_quat(yaw + 90))}" '
                    'size="0.36 0.01 0.26" material="checker_flag" contype="0" conaffinity="0"/>')
    for i in range(0, len(flags) - 1, 2):
        a, b = np.array(flags[i]["xyz"][:2], float), np.array(flags[i + 1]["xyz"][:2], float)
        mid, span = (a + b) / 2, np.linalg.norm(b - a)
        yaw = np.degrees(np.arctan2(*(b - a)[[1, 0]]))
        k = i // 2
        extra_assets.append(f'    <material name="checker_banner_{k}" texture="checker" '
                            f'texrepeat="{max(1, round(span / 0.36))} 1" emission="0.25"/>')
        extra_assets.append(f'    <material name="checker_line_{k}" texture="checker" '
                            f'texrepeat="{max(1, round(span / 0.5))} 1" emission="0.1"/>')
        deco.append(f'    <geom name="banner_{k}" type="box" pos="{mid[0]} {mid[1]} 3.25" '
                    f'quat="{_f(_yaw_quat(yaw))}" size="{span / 2} 0.02 0.18" material="checker_banner_{k}" '
                    'contype="0" conaffinity="0"/>')
        deco.append(f'    <geom name="line_{k}" type="box" pos="{mid[0]} {mid[1]} 0.003" '
                    f'quat="{_f(_yaw_quat(yaw))}" size="{span / 2} 0.25 0.002" material="checker_line_{k}" '
                    'contype="0" conaffinity="0"/>')
    xml = xml.replace("  </asset>", "\n".join(assets + extra_assets) + "\n  </asset>")
    return xml.replace("  </worldbody>", "\n".join(deco) + "\n  </worldbody>")
