"""Race the gates: fly a course's gates in order, lap after lap (the race_gates course by default).

Built on the single-hoop task (hoop_env.py): the same drone, actions, rewards and frame-touch check, applied to the
gate the drone is currently flying at. Passing it makes the next gate (in course order, wrapping round) the target.

Each episode starts on a random leg of the course: somewhere between the previous gate and the target, a little off
the line, heading roughly at the target, with a random speed, attitude and body rates. So every gate, including the
0.5 m needle, is practised from the direction it is really approached.

Policy, 100 Hz:
  observation (36): the hoop task's 22 (position, velocity and attitude in the target gate's frame, body rates,
                    previous action), the next gate's centre and through-axis in the target gate's frame (6 per gate,
                    ``lookahead`` gates), the lidar if enabled, and which gate is the target (one-hot, 8). The target is read back from the observation each step:
                    on auto-reset Brax restores obs and data to the episode's start but carries info over.
  action (4):       rotor speeds, as in the hoop task.

An episode ends on a failure: touching any gate's frame or any obstacle (the course's buildings: pillars, walls),
crossing the target gate's plane near it but outside the opening, the ground, flipping over, or straying more than
max_dist from the target. Otherwise it runs its full length (15 s, a bit over one lap at racing speed).

Courses with ``loop: false`` (long_track) are point to point instead: gate 0 is approached from the spawn point, the
last gate's "next gate" is a point 5 m beyond it, and going through the last gate finishes the episode. On those,
``course_line`` measures progress and aim along the planned path (mpc/path.py: around the pillars, over the wall)
instead of straight at the target's centre, so the reward does not fight the detours an obstacle forces. Only the
stretch of path between the previous gate and the target counts, so a nearby later part of the track cannot be
"progress".
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
from bumble_bee_gym.learn import lidar as lidar_mod, realism
from bumble_bee_gym.learn.hoop_env import euler_to_quat, quat_to_mat


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.002,
        episode_length=1500,
        impl="jax",
        course="race_gates",
        start=config_dict.create(
            leg=[0.15, 0.75],       # share of the way from the previous gate to the target
            offset=0.6,             # [m] +- off the line, each axis
            yaw=0.5,                # [rad] +- heading relative to the target
            tilt=0.2,               # [rad] +- roll and pitch
            speed=[0.0, 3.0],       # [m/s] towards the target
            focus=[],               # gates whose legs get extra practice (e.g. the ones with an obstacle in the way)
            focus_prob=0.0,         # share of episodes that start on a focus leg
            fast_prob=0.0,          # share of episodes that start just past the previous gate at racing speed:
            fast_leg=[0.02, 0.2],   #   this share of the way along the leg,
            fast_speed=[4.0, 10.0],  #  this fast [m/s] towards the target (practice for linking gates)
            lift_prob=0.0,          # share of episodes that start higher, by up to lift [m]
            lift=2.5,               #   (above an obstacle in the way, e.g. over the wall)
            vel=0.5,                # [m/s] +- extra, each axis
            rate=0.5,               # [rad/s] +- each axis
        ),
        ground=0.1,                 # [m] centre height that counts as hitting the ground
        max_dist=12.0,              # [m] from the target gate
        real=realism.default_config(),   # delay, motor lag, drag, thrust spread, sensor noise (see realism.py)
        lidar=lidar_mod.default_config(),  # body-fixed range beams to obstacles, gate frames, ground (lidar.py)
        near_dist=0.8,              # [m] closer than this to an obstacle costs the near_obstacle reward
        miss_radius=2.0,            # [m] crossing the target's plane further out than this is not a miss
        lookahead=1,                # gates after the target shown in the observation (centre + axis each)
        course_line=False,          # progress/aim measured along the planned path (mpc/path.py) instead of
                                    #   straight to the target's centre; point-to-point courses only
        reward_config=config_dict.create(
            scales=config_dict.create(
                progress=5.0,       # per metre closer to the target's centre
                aim=5.0,            # per metre closer to the line through the target's centre
                through=20.0,       # once per gate
                centred=10.0,       # once per gate, scaled by how close to the centre it went through
                crash=-20.0,        # once, on any failure
                rates=-0.002,       # per step, |body rates|^2
                action_rate=-0.02,  # per step, |a - previous a|^2
                near_obstacle=0.0,  # per step, (1 - distance / near_dist)^2 within near_dist of an obstacle
                obstacle_crash=0.0,  # once, extra on touching an obstacle (on top of crash)
            ),
        ),
    )


class GatesEnv(mjx_env.MjxEnv):

    def __init__(self, config: Optional[config_dict.ConfigDict] = None,
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config or default_config(), config_overrides)
        c = self._config
        self._course = course_mod.load(c.course)
        self._p = p = params_mod.load()
        self._mj_model = mujoco.MjModel.from_xml_string(realism.add_userdata(hoop.scene_xml(self._course, p, train=True)))
        self._mj_model.opt.timestep = c.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=c.impl)
        self._real = realism.Realism(c.real, self._mj_model, p)

        gates = self._course.gates
        self._n = len(gates)
        frames = [hoop.hoop_frame(g) for g in gates]
        self._centres = jp.array(np.stack([f[0] for f in frames]))           # (G, 3)
        self._rots = jp.array(np.stack([f[1] for f in frames]))              # (G, 3, 3), columns = gate axes
        self._yaws = jp.array([np.radians(g.yaw) for g in gates])
        self._halves = jp.array([np.array(g.opening) / 2 for g in gates])    # (G, 2)
        self._bar_c = jp.array([[b[0] for b in g.bars()] for g in gates])   # (G, 4, 3), gate frame
        self._bar_h = jp.array([[b[1] for b in g.bars()] for g in gates])
        sc, sr = hoop.collision_spheres(p)
        self._sph_c, self._sph_r = jp.array(sc), jp.array(sr)
        self._thrust_map = jp.array(p.thrust_map)
        self._hover, self._wmax = float(p.hover_speed()), float(p.max_rotor_speed)
        self._loop = bool(self._course.extra.get("loop", True))
        self._lidar = lidar_mod.Scene(self._course, c.lidar).jax_fn() if c.lidar.enabled else None
        self._focus = jp.array(list(c.start.focus), jp.int32) if len(c.start.focus) else None
        self._spawn = jp.array(self._course.spawn_xyz)
        b = self._course.buildings
        self._n_obs = len(b)
        if b:   # boxes standing on the ground: centre, half sizes, rotation (columns = box axes)
            self._obs_c = jp.array([[o.xyz[0], o.xyz[1], o.size[2] / 2] for o in b])
            self._obs_h = jp.array([np.asarray(o.size, float) / 2 for o in b])
            self._obs_r = jp.array(np.stack([hoop.worlds._rot_z(o.yaw) for o in b]))
        self._line = self._course_line() if c.course_line else None

    def _course_line(self):
        """The planned path as a polyline (segment starts, directions, lengths, arc length at each start) and, per
        target gate, the index range of segments on the leg leading to it (previous gate, or the start, to it)."""
        if self._loop:
            raise ValueError("course_line needs a point-to-point course (loop: false)")
        from bumble_bee_gym.mpc import path as path_mod
        start, _ = path_mod.start_point(self._course)
        _, pts, _, _ = path_mod.plan(self._course, start).sample()
        pts = pts[::5]                                            # ~5 cm to 0.5 m apart, plenty for a polyline
        a, d = pts[:-1], np.diff(pts, axis=0)
        ln = np.linalg.norm(d, axis=1)
        keep = ln > 1e-6
        a, d, ln = a[keep], d[keep], ln[keep]
        s = np.concatenate([[0.0], np.cumsum(ln)[:-1]])
        ends, lo = [], 0
        for centre in np.asarray(self._centres):                 # the path passes each gate's centre, in order
            i = lo + int(np.argmin(np.linalg.norm(a[lo:] - centre, axis=1)))
            ends.append(i)
            lo = i
        los = [0] + ends[:-1]
        gate_s = []                                               # arc length at each gate's centre
        for centre, i in zip(np.asarray(self._centres), ends):
            j = np.arange(max(i - 3, 0), min(i + 4, len(a)))
            t = np.clip(np.einsum("mj,mj->m", centre - a[j], d[j]) / ln[j] ** 2, 0.0, 1.0)
            m = int(np.argmin(np.linalg.norm(a[j] + t[:, None] * d[j] - centre, axis=1)))
            gate_s.append(s[j[m]] + t[m] * ln[j[m]])
        return dict(a=jp.array(a), d=jp.array(d), len=jp.array(ln), s=jp.array(s), gate_s=jp.array(gate_s),
                    lo=jp.array(los, jp.int32), hi=jp.array(ends, jp.int32))

    def along_line(self, pos: jax.Array, k) -> tuple[jax.Array, jax.Array]:
        """Arc length along the planned path of the nearest point on the leg to gate k, and the distance to it."""
        L = self._line
        t = jp.clip(jp.einsum("mj,mj->m", pos[None] - L["a"], L["d"]) / jp.square(L["len"]), 0.0, 1.0)
        dist = jp.linalg.norm(L["a"] + t[:, None] * L["d"] - pos[None], axis=1)
        idx = jp.arange(len(t))
        dist = jp.where((idx >= L["lo"][k]) & (idx <= L["hi"][k]), dist, jp.inf)
        i = jp.argmin(dist)
        return L["s"][i] + t[i] * L["len"][i], dist[i]

    # ------------------------------------------------------------------ helpers

    def thrust(self, a: jax.Array) -> jax.Array:
        w = jp.clip(self._hover + hoop.SPEED_SPAN * jp.clip(a, -1.0, 1.0), 0.0, self._wmax)
        c2, c1, c0 = self._thrust_map
        return jp.maximum(0.0, c2 * w * w + c1 * w + c0)

    def to_gate(self, data, k) -> Dict[str, jax.Array]:
        """Drone state in gate k's frame (x through the gate, origin at its centre)."""
        rot = self._rots[k]
        pos, quat = data.qpos[0:3], data.qpos[3:7]
        return {"pos": rot.T @ (pos - self._centres[k]), "vel": rot.T @ data.qvel[0:3],
                "att": rot.T @ quat_to_mat(quat), "rates": data.qvel[3:6], "z": pos[2]}

    def touches_any_frame(self, pos: jax.Array, att_world: jax.Array) -> jax.Array:
        """Does any of the drone's spheres touch any bar of any gate?"""
        centres_w = pos + self._sph_c @ att_world.T                                   # (5, 3)

        def one(c, rot, bc, bh):
            local = (centres_w - c) @ rot                                              # (5, 3) gate frame
            gap = jp.abs(local[:, None] - bc[None]) - bh[None]
            d = jp.linalg.norm(jp.maximum(gap, 0.0), axis=-1)
            return jp.any(d < self._sph_r[:, None])

        return jp.any(jax.vmap(one)(self._centres, self._rots, self._bar_c, self._bar_h))

    def touches_obstacle(self, pos: jax.Array, att_world: jax.Array) -> jax.Array:
        """Does any of the drone's spheres touch any obstacle box?"""
        if self._n_obs == 0:
            return jp.zeros((), bool)
        centres_w = pos + self._sph_c @ att_world.T                                   # (5, 3)
        local = jp.einsum("osj,ojk->osk", centres_w[None] - self._obs_c[:, None], self._obs_r)   # (O, 5, 3)
        d = jp.linalg.norm(jp.maximum(jp.abs(local) - self._obs_h[:, None], 0.0), axis=-1)
        return jp.any(d < self._sph_r[None])

    def next_gate(self, k, j=1):
        """Centre and through-axis of the j-th gate after gate k (point to point: past the last gate, points 5 m apart
        beyond it along its axis)."""
        if self._loop:
            nxt = (k + j) % self._n
            return self._centres[nxt], self._rots[nxt][:, 0]
        last = self._n - 1
        past = k + j - last                                       # > 0: that many beyond the last gate
        nxt = jp.minimum(k + j, last)
        axis = self._rots[nxt][:, 0]
        beyond = self._centres[last] + 5.0 * jp.maximum(past, 0) * self._rots[last][:, 0]
        return jp.where(past > 0, beyond, self._centres[nxt]), axis

    def obstacle_distance(self, pos: jax.Array) -> jax.Array:
        """Distance from the drone's centre to the nearest obstacle box (inf without obstacles)."""
        if self._n_obs == 0:
            return jp.array(jp.inf)
        local = jp.einsum("oj,ojk->ok", pos[None] - self._obs_c, self._obs_r)
        return jp.min(jp.linalg.norm(jp.maximum(jp.abs(local) - self._obs_h, 0.0), axis=-1))

    def _obs(self, s, last_act, k) -> jax.Array:
        rot = self._rots[k]
        parts = [s["pos"], s["vel"], s["att"].ravel(), s["rates"], last_act]
        for j in range(1, self._config.lookahead + 1):
            nc, na = self.next_gate(k, j)
            parts += [rot.T @ (nc - self._centres[k]), rot.T @ na]
        if self._lidar is not None:     # before the one-hot, which target() reads back from the end
            parts.append(self._lidar(rot @ s["pos"] + self._centres[k], rot @ s["att"]))
        return jp.hstack(parts + [jax.nn.one_hot(k, self._n)])

    def target(self, obs: jax.Array) -> jax.Array:
        return jp.argmax(obs[-self._n:])

    # ------------------------------------------------------------------ env API

    def reset(self, rng: jax.Array) -> mjx_env.State:
        st = self._config.start
        rng, *k = jax.random.split(rng, 11)
        u = lambda key, lo, hi, n=(): jax.random.uniform(key, n, minval=lo, maxval=hi)  # noqa: E731
        tgt = jax.random.randint(k[0], (), 0, self._n)
        if self._focus is not None:     # some episodes start on a leg that needs practice
            kf1, kf2 = jax.random.split(k[0])
            pick = self._focus[jax.random.randint(kf1, (), 0, len(self._focus))]
            tgt = jp.where(jax.random.uniform(kf2) < st.focus_prob, pick, tgt)
        prev = (tgt - 1) % self._n
        a, b = self._centres[prev], self._centres[tgt]
        if not self._loop:
            a = jp.where(tgt == 0, self._spawn, a)
        kx = jax.random.split(k[1], 5)
        fast = jax.random.uniform(kx[0]) < st.fast_prob
        frac = jp.where(fast, u(kx[1], *st.fast_leg), u(k[1], *st.leg))
        pos = a + frac * (b - a) + u(k[2], -st.offset, st.offset, (3,))
        pos = pos.at[2].add(jp.where(jax.random.uniform(kx[2]) < st.lift_prob, u(kx[3], 0.0, st.lift), 0.0))
        pos = pos.at[2].set(jp.maximum(pos[2], 0.5))
        if self._n_obs:   # a start inside an obstacle's footprint and below its top goes on top of it
            local = jp.einsum("oj,ojk->ok", pos[None] - self._obs_c, self._obs_r)
            over = jp.all(jp.abs(local[:, :2]) < self._obs_h[:, :2] + 0.3, axis=1) & (pos[2] < 2 * self._obs_h[:, 2] + 0.4)
            pos = pos.at[2].set(jp.maximum(pos[2], jp.max(jp.where(over, 2 * self._obs_h[:, 2] + 0.6, 0.0))))
        to_b = b - pos
        heading = jp.arctan2(to_b[1], to_b[0])
        rp = u(k[3], -st.tilt, st.tilt, (2,))
        quat = euler_to_quat(rp[0], rp[1], heading + u(k[4], -st.yaw, st.yaw))
        speed = jp.where(fast, u(kx[4], *st.fast_speed), u(k[5], *st.speed))
        vel = speed * to_b / jp.maximum(jp.linalg.norm(to_b), 1e-6) + u(k[6], -st.vel, st.vel, (3,))
        qvel = jp.hstack([vel, u(k[7], -st.rate, st.rate, (3,))])
        data = mjx_env.make_data(self._mj_model, qpos=jp.hstack([pos, quat]), qvel=qvel,
                                 ctrl=self.thrust(jp.zeros(4)), impl=self._mjx_model.impl.value)
        data = mjx.forward(self._mjx_model, data.replace(userdata=self._real.sample(k[8])))
        info = {"rng": rng}
        metrics = {f"reward/{n}": jp.zeros(()) for n in self._config.reward_config.scales.keys()}
        metrics.update({n: jp.zeros(()) for n in ("gates", "crashed", "finished", "speed_through", "miss_offset",
                                                  "fail/frame", "fail/obstacle", "fail/missed", "fail/flipped",
                                                  "fail/ground", "fail/away")})
        obs = self._real.noisy(self._obs(self.to_gate(data, tgt), jp.zeros(4), tgt), k[9])
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        c = self._config
        k = self.target(state.obs)
        p0, last_act = self.to_gate(state.data, k)["pos"], state.obs[22 - 4:22]
        rng, k_step, k_obs = jax.random.split(state.info["rng"], 3)
        data = self._real.step(self._mjx_model, state.data, action, self.n_substeps, k_step)
        s = self.to_gate(data, k)
        p1 = s["pos"]

        crossed = (p0[0] < 0.0) & (p1[0] >= 0.0)
        frac = jp.clip(-p0[0] / jp.maximum(p1[0] - p0[0], 1e-6), 0.0, 1.0)
        yz = (p0 + frac * (p1 - p0))[1:]
        half = self._halves[k]
        inside = jp.all(jp.abs(yz) < half)
        near = jp.all(jp.abs(yz) < c.miss_radius)
        att_w = quat_to_mat(data.qpos[3:7])
        hit_frame = self.touches_any_frame(data.qpos[0:3], att_w)
        hit_obstacle = self.touches_obstacle(data.qpos[0:3], att_w)
        through = crossed & inside & ~hit_frame
        missed = crossed & near & ~inside
        upside_down = att_w[2, 2] < 0.0
        ground, away = s["z"] < c.ground, jp.linalg.norm(p1) > c.max_dist
        fail = hit_frame | hit_obstacle | missed | upside_down | ground | away
        finished = through & (k == self._n - 1) if not self._loop else jp.zeros((), bool)

        offset = jp.max(jp.abs(yz) / half)
        if self._line is not None:      # along the planned path, which swerves round the pillars and over the wall
            # Distance to go, unrolled along the path: arc length left to the gate and distance off the path, combined
            # like the straight-line distance to the centre, so being off the path matters most close to the gate.
            s0, d0 = self.along_line(state.data.qpos[0:3], k)
            s1, d1 = self.along_line(data.qpos[0:3], k)
            gs = self._line["gate_s"][k]
            progress = jp.hypot(gs - s0, d0) - jp.hypot(gs - s1, d1)
            aim = d0 - d1
        else:
            progress = jp.linalg.norm(p0) - jp.linalg.norm(p1)
            aim = jp.linalg.norm(p0[1:]) - jp.linalg.norm(p1[1:])
        terms = {
            "progress": progress,
            "aim": aim,
            "through": through.astype(jp.float32),
            "centred": through * jp.clip(1.0 - offset, 0.0, 1.0),
            "crash": fail.astype(jp.float32),
            "rates": jp.sum(jp.square(s["rates"])),
            "action_rate": jp.sum(jp.square(action - last_act)),
            "near_obstacle": jp.square(jp.clip(1.0 - self.obstacle_distance(data.qpos[0:3]) / c.near_dist, 0.0, 1.0)),
            "obstacle_crash": hit_obstacle.astype(jp.float32),
        }
        terms = {kk: v * c.reward_config.scales[kk] for kk, v in terms.items()}
        reward = sum(terms.values())

        new_k = jp.where(through, (k + 1) % self._n, k)
        for kk, v in terms.items():
            state.metrics[f"reward/{kk}"] = v
        state.metrics["gates"] = through.astype(jp.float32)
        state.metrics["crashed"] = fail.astype(jp.float32)
        state.metrics["finished"] = finished.astype(jp.float32)
        for kk, v in (("frame", hit_frame), ("obstacle", hit_obstacle), ("missed", missed & ~hit_frame),
                      ("flipped", upside_down), ("ground", ground), ("away", away)):
            state.metrics[f"fail/{kk}"] = v.astype(jp.float32)
        state.metrics["speed_through"] = through * jp.linalg.norm(s["vel"])
        state.metrics["miss_offset"] = through * offset
        obs = self._real.noisy(self._obs(self.to_gate(data, new_k), action, new_k), k_obs)
        state.info["rng"] = rng
        return state.replace(data=data, obs=obs, reward=reward.astype(jp.float32),
                             done=(fail | finished).astype(jp.float32))

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
