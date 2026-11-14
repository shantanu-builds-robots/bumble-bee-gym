import mujoco
import numpy as np

from bumble_bee_gym import course, hoop, params


def test_zero_action_is_hover_and_full_action_lifts_well_over_weight():
    p = params.load()
    assert np.allclose(hoop.rotor_speeds(np.zeros(4), p), p.hover_speed())
    lift = p.thrust(hoop.rotor_speeds(np.ones(4), p)).sum() / (p.mass * 9.8)
    assert 3.0 < lift < 5.0


def test_training_scene_matches_visual_drone_without_meshes_or_contacts():
    c, p = course.load("one_hoop"), params.load()
    vis = mujoco.MjModel.from_xml_string(hoop.scene_xml(c, p))
    train = mujoco.MjModel.from_xml_string(hoop.scene_xml(c, p, train=True))
    assert train.nmesh == 0 and train.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT
    b = "drone"
    assert np.isclose(vis.body(b).mass, train.body(b).mass)
    assert np.allclose(vis.body(b).inertia, train.body(b).inertia)
    assert np.allclose(vis.actuator_gear, train.actuator_gear)


def test_level_drone_fits_the_opening_with_its_real_shape():
    p, g = params.load(), course.load("one_hoop").gates[0]
    centres, radii = hoop.collision_spheres(p)
    half_height = np.max(centres[:, 2] + radii)
    assert half_height < g.opening[1] / 2 - 0.25          # much less than a 0.19 m sphere would need
