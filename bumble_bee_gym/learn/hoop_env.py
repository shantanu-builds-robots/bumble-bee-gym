"""Fly through one hoop: the first learning task, on the GPU (MuJoCo's MJX API, trained with Brax PPO).

The drone is the MuJoCo twin of Bumble Bee (mass, inertia, rotors and thrust map read from bumble-bee's URDF). Each
episode it starts 2 to 5 m in front of the hoop at a random height, offset, attitude and velocity, and has 5 s to
fly through it.

Policy, 100 Hz:
  observation (22): position and velocity relative to the hoop (hoop frame), attitude as a rotation matrix (hoop
                    frame), body rates, and the previous action.
  action (4):       rotor speeds, a in [-1, 1] -> hover + 750 a rad/s (see bumble_bee_gym.hoop).

An episode ends when the drone goes through (success), touches the frame or the ground, flips over, flies past the
hoop's plane outside it, or strays more than 8 m away. Contacts are off in the physics; touching is checked with a
sphere around each prop and one around the body, at the drone's actual attitude, against the frame's four bars.
"""
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from bumble_bee_gym import course as course_mod
from bumble_bee_gym import hoop, params as params_mod


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.002,
        episode_length=500,
        impl="jax",
        course="one_hoop",
        start=config_dict.create(
            dist=[2.0, 5.0],        # [m] in front of the hoop
            lateral=1.5,            # [m] +- sideways
            height=[0.7, 2.5],      # [m] above the ground
            yaw=0.6,                # [rad] +- heading relative to the hoop
            tilt=0.2,               # [rad] +- roll and pitch
            vel=0.5,                # [m/s] +- each axis
            rate=0.5,               # [rad/s] +- each axis
        ),
        ground=0.1,                 # [m] centre height that counts as hitting the ground
        max_dist=8.0,               # [m] from the hoop
        reward_config=config_dict.create(
            scales=config_dict.create(
                progress=5.0,       # per metre closer to the hoop's centre
                aim=5.0,            # per metre closer to the line through the hoop's centre
                through=20.0,       # once, on going through
                centred=10.0,       # once, scaled by how close to the centre it went through
                crash=-20.0,        # once, on any failure
                rates=-0.002,       # per step, |body rates|^2
                action_rate=-0.02,  # per step, |a - previous a|^2
            ),
        ),
    )


def quat_to_mat(q: jax.Array) -> jax.Array:
    w, x, y, z = q
    return jp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_quat(roll, pitch, yaw) -> jax.Array:
    cr, sr = jp.cos(roll / 2), jp.sin(roll / 2)
    cp, sp = jp.cos(pitch / 2), jp.sin(pitch / 2)
    cy, sy = jp.cos(yaw / 2), jp.sin(yaw / 2)
    return jp.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class HoopEnv(mjx_env.MjxEnv):

    def __init__(self, config: Optional[config_dict.ConfigDict] = None,
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config or default_config(), config_overrides)
        c = self._config
        self._course = course_mod.load(c.course)
        self._p = p = params_mod.load()
        self._mj_model = mujoco.MjModel.from_xml_string(hoop.scene_xml(self._course, p, train=True))
        self._mj_model.opt.timestep = c.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=c.impl)

        gate = self._course.gates[0]
        centre, rot = hoop.hoop_frame(gate)
        self._centre, self._rot = jp.array(centre), jp.array(rot)
        self._hoop_yaw = float(np.radians(gate.yaw))
        self._half = jp.array(gate.opening) / 2
        sc, sr = hoop.collision_spheres(p)
        self._sph_c, self._sph_r = jp.array(sc), jp.array(sr)
        bars = gate.bars()
        self._bar_c = jp.array([b[0] for b in bars])          # in the hoop frame
        self._bar_h = jp.array([b[1] for b in bars])
        self._thrust_map = jp.array(p.thrust_map)
        self._hover, self._wmax = float(p.hover_speed()), float(p.max_rotor_speed)

    # ------------------------------------------------------------------ helpers

    def thrust(self, a: jax.Array) -> jax.Array:
        """Action -> rotor speeds -> per-rotor thrust [N], exactly as the Gazebo plugin maps speed to thrust."""
        w = jp.clip(self._hover + hoop.SPEED_SPAN * jp.clip(a, -1.0, 1.0), 0.0, self._wmax)
        c2, c1, c0 = self._thrust_map
        return jp.maximum(0.0, c2 * w * w + c1 * w + c0)

    def to_hoop(self, data) -> Dict[str, jax.Array]:
        """Drone state in the hoop frame (x through the hoop, origin at its centre)."""
        pos, quat = data.qpos[0:3], data.qpos[3:7]
        return {"pos": self._rot.T @ (pos - self._centre), "vel": self._rot.T @ data.qvel[0:3],
                "att": self._rot.T @ quat_to_mat(quat), "rates": data.qvel[3:6], "z": pos[2]}

    def touches_frame(self, ph: jax.Array, att: jax.Array) -> jax.Array:
        """Does any of the drone's spheres (centre ph, attitude att, both in the hoop frame) touch a bar?"""
        centres = ph + self._sph_c @ att.T                                       # (5, 3)
        gap = jp.abs(centres[:, None] - self._bar_c[None]) - self._bar_h[None]   # (5 spheres, 4 bars, 3)
        d = jp.linalg.norm(jp.maximum(gap, 0.0), axis=-1)
        return jp.any(d < self._sph_r[:, None])

    def _obs(self, s, last_act) -> jax.Array:
        return jp.hstack([s["pos"], s["vel"], s["att"].ravel(), s["rates"], last_act])

    # ------------------------------------------------------------------ env API

    def reset(self, rng: jax.Array) -> mjx_env.State:
        st = self._config.start
        rng, *k = jax.random.split(rng, 8)
        u = lambda key, lo, hi, n=(): jax.random.uniform(key, n, minval=lo, maxval=hi)  # noqa: E731
        local = jp.array([-u(k[0], *st.dist), u(k[1], -st.lateral, st.lateral), 0.0])
        pos = (self._centre + self._rot @ local).at[2].set(u(k[2], *st.height))
        rp = u(k[3], -st.tilt, st.tilt, (2,))
        quat = euler_to_quat(rp[0], rp[1], self._hoop_yaw + u(k[4], -st.yaw, st.yaw))
        qvel = jp.hstack([u(k[5], -st.vel, st.vel, (3,)), u(k[6], -st.rate, st.rate, (3,))])
        data = mjx_env.make_data(self._mj_model, qpos=jp.hstack([pos, quat]), qvel=qvel,
                                 ctrl=self.thrust(jp.zeros(4)), impl=self._mjx_model.impl.value)
        data = mjx.forward(self._mjx_model, data)
        s = self.to_hoop(data)
        info = {"rng": rng}
        metrics = {f"reward/{n}": jp.zeros(()) for n in self._config.reward_config.scales.keys()}
        metrics.update({n: jp.zeros(()) for n in ("through", "crashed", "speed_through", "miss_offset", "fail/frame",
                                                  "fail/missed", "fail/flipped", "fail/ground", "fail/away")})
        return mjx_env.State(data, self._obs(s, jp.zeros(4)), jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        c = self._config
        # The previous position and action come from state.data and state.obs, not state.info: on auto-reset Brax
        # restores data and obs to the episode's start but carries info over from the episode that just ended.
        p0, last_act = self.to_hoop(state.data)["pos"], state.obs[-4:]
        data = mjx_env.step(self._mjx_model, state.data, self.thrust(action), self.n_substeps)
        s = self.to_hoop(data)
        p1 = s["pos"]

        # Crossing the hoop's plane this step, and where (linear between the two samples).
        crossed = (p0[0] < 0.0) & (p1[0] >= 0.0)
        frac = jp.clip(-p0[0] / jp.maximum(p1[0] - p0[0], 1e-6), 0.0, 1.0)
        yz = (p0 + frac * (p1 - p0))[1:]
        inside = jp.all(jp.abs(yz) < self._half)
        hit_frame = self.touches_frame(p1, s["att"]) | (crossed & self.touches_frame(p0 + frac * (p1 - p0), s["att"]))
        through = crossed & inside & ~hit_frame
        missed = crossed & ~inside
        upside_down = s["att"][2, 2] < 0.0
        ground, away = s["z"] < c.ground, jp.linalg.norm(p1) > c.max_dist
        fail = hit_frame | missed | upside_down | ground | away
        done = through | fail

        offset = jp.max(jp.abs(yz) / self._half)            # 0 at the centre, 1 at the opening's edge
        terms = {
            "progress": jp.linalg.norm(p0) - jp.linalg.norm(p1),
            "aim": jp.linalg.norm(p0[1:]) - jp.linalg.norm(p1[1:]),
            "through": through.astype(jp.float32),
            "centred": through * jp.clip(1.0 - offset, 0.0, 1.0),
            "crash": fail.astype(jp.float32),
            "rates": jp.sum(jp.square(s["rates"])),
            "action_rate": jp.sum(jp.square(action - last_act)),
        }
        terms = {k: v * c.reward_config.scales[k] for k, v in terms.items()}
        reward = sum(terms.values())

        for k, v in terms.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["through"] = through.astype(jp.float32)
        state.metrics["crashed"] = fail.astype(jp.float32)
        for k, v in (("frame", hit_frame), ("missed", missed & ~hit_frame), ("flipped", upside_down),
                     ("ground", ground), ("away", away)):
            state.metrics[f"fail/{k}"] = v.astype(jp.float32)
        state.metrics["speed_through"] = through * jp.linalg.norm(s["vel"])
        state.metrics["miss_offset"] = through * offset
        obs = self._obs(s, action)
        return state.replace(data=data, obs=obs, reward=reward.astype(jp.float32), done=done.astype(jp.float32))

    # ------------------------------------------------------------------ accessors

    @property
    def xml_path(self) -> str:
        return str(course_mod.COURSE_DIR / f"{self._config.course}.yaml")

    @property
    def action_size(self) -> int:
        return 4

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def course(self) -> course_mod.Course:
        return self._course
