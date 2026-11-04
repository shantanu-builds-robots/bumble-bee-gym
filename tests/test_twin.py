"""The MuJoCo twin must match the drone Gazebo flies: same parameters, same hover, same trajectory."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from bumble_bee_gym import course, params, parity, worlds

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def p():
    return params.load()


@pytest.fixture(scope="module")
def empty_xml(tmp_path_factory):
    _, xml = worlds.write(course.load("empty"), tmp_path_factory.mktemp("worlds"))
    return xml


def test_params_match_plugin_model(p):
    # The published simulator model these numbers come from: 0.73 kg, its thrust map, kappa 0.016.
    assert p.mass == pytest.approx(0.73, abs=1e-9)
    assert p.thrust_map == pytest.approx((1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325))
    assert p.torque_to_thrust == pytest.approx(0.016)
    assert p.hover_speed() == pytest.approx(738.4, abs=0.05)
    assert p.hover_speed(extra_mass=course.PARCEL_MASS) == pytest.approx(780.2, abs=0.05)


def test_mjcf_carries_the_same_body(p, empty_xml):
    m = mujoco.MjModel.from_xml_path(str(empty_xml))
    body = m.body("drone").id
    assert m.body_subtreemass[body] == pytest.approx(p.mass, rel=1e-6)
    assert m.opt.timestep == worlds.TIMESTEP
    assert m.opt.gravity[2] == -worlds.GRAVITY
    # MuJoCo stores principal moments; compare the eigenvalues of the URDF's inertia tensor.
    assert np.sort(m.body_inertia[body]) == pytest.approx(np.sort(np.linalg.eigvalsh(p.inertia)), rel=1e-5)
    assert m.nu == 4


def test_mujoco_hovers_at_plugin_hover_speed(empty_xml):
    out = parity.simulate_mujoco(empty_xml, switch_times=[0.0] + [99.0] * 5,
                                 start={"pos": [0, 0, 2.0], "quat_wxyz": [1, 0, 0, 0]})
    assert abs(out["z"][-1] - 2.0) < 0.002          # metres drifted after 2 s
    # The CAD's centre of mass sits a few microns off the rotor centre, so even a perfect hover tips
    # very slowly; Gazebo shows the same drift.
    assert np.abs(out["x"]).max() < 1e-4 and np.abs(out["y"]).max() < 1e-4


def test_matches_recorded_gazebo_flight(empty_xml):
    """Replay the open-loop profile Gazebo flew (tests/data, recorded by scripts/run_gazebo_parity.sh)."""
    gz = parity.read_csv(DATA / "gazebo_parity.csv")
    switch = [float(s) for s in (DATA / "gazebo_parity.switch.txt").read_text().split()]
    mj = parity.simulate_mujoco(empty_xml, switch, {"pos": [gz["x"][0], gz["y"][0], gz["z"][0]],
                                                    "quat_wxyz": [gz["qw"][0], gz["qx"][0], gz["qy"][0], gz["qz"][0]]})
    d = parity.compare(gz, mj)
    # By 2 s the drone has climbed 0.35 m, slid 1.6 m sideways and rolled past 75 degrees.
    assert max(d["max_dx_m"], d["max_dy_m"], d["max_dz_m"]) < 0.01, d
    assert max(d["max_droll_deg"], d["max_dpitch_deg"], d["max_dyaw_deg"]) < 1.0, d
