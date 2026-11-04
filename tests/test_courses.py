"""Every course renders into a valid Gazebo world and a MuJoCo scene that agree on what is where."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from bumble_bee_gym import course, worlds

COURSES = sorted(p.stem for p in course.COURSE_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", COURSES)
def test_course_renders_into_both_simulators(name, tmp_path):
    c = course.load(name)
    sdf, xml = worlds.write(c, tmp_path)

    world = ET.parse(sdf).getroot().find("world")
    assert world.findtext("gravity").split() == ["0", "0", f"-{worlds.GRAVITY:g}"]
    models = [m.get("name") for m in world.findall("model")]
    includes = [i.findtext("name") for i in world.findall("include")]
    assert sum(n.startswith("building_") for n in models) == len(c.buildings)
    assert sum(n.startswith("gate_") for n in models) == len(c.gates)
    assert sum(bool(n) and n.startswith("landing_pad_") for n in includes) == len(c.pads)

    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body("drone").id] == pytest.approx(c.spawn_xyz)
    gate_geoms = [m.geom(i).name for i in range(m.ngeom) if m.geom(i).name.startswith("gate_")]
    assert len(gate_geoms) == 4 * len(c.gates)


def test_gate_opening_is_clear():
    """The frame bars leave exactly the declared opening around the gate centre."""
    g = course.Gate(np.zeros(3), 0.0, (0.8, 0.6))
    top, bottom, left, right = g.bars()
    assert top[0][2] - top[1][2] - (bottom[0][2] + bottom[1][2]) == pytest.approx(0.6)
    assert left[0][1] - left[1][1] - (right[0][1] + right[1][1]) == pytest.approx(0.8)


def test_race_has_a_needle():
    """At least one race gate is less than twice the drone's width."""
    race = course.load("race_gates")
    assert min(min(g.opening) for g in race.gates) <= 0.6
