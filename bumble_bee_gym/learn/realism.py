"""The real-world effects a trained policy meets outside the training simulator, added to training.

Measured in Gazebo at full real time (scripts/run_gazebo_race.sh, pilot on a 100 Hz wall-clock loop): a command
reaches the motors about 5 to 15 ms after the pose it was computed from (median about 7 ms), with occasional extra
jitter when a control tick lands late. A policy trained with none of that overcorrects. So every episode draws:

  delay         [ms]   pose-to-motor delay, plus up to ``jitter_ms`` more each step
  motor_tau     [s]    first-order rotor-speed lag (0 = instant, as bumble-bee's Gazebo plugin is)
  drag          [N s^2/m^2]  quadratic air drag on the body, F = -drag |v| v (the plugin has none)
  thrust_scale  [-]    every rotor's thrust scaled by this (motor and prop spread)

and observations get small Gaussian noise. Each range includes Gazebo's own value, so the policy that copes with
all of them copes with Gazebo too.

The per-episode draw, the two previous commands still in flight, and the actual rotor speeds live in MuJoCo's
``userdata`` array, so Brax's auto-reset restores them with the rest of ``data`` at each episode start.
"""
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from bumble_bee_gym import hoop

# userdata layout
DELAY, TAU, DRAG, SCALE = 0, 1, 2, 3
PREV1, PREV2, ROTOR = slice(4, 8), slice(8, 12), slice(12, 16)
N_USERDATA = 16
MAX_DELAY_MS = 20.0          # two control steps of commands are kept in flight


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        enabled=True,
        delay_ms=[2.0, 12.0],        # per episode
        jitter_ms=6.0,               # extra per step, uniform [0, jitter]
        motor_tau=[0.0, 0.015],      # [s]
        drag=[0.0, 0.003],           # [N s^2/m^2]; 0.003 is about 0.5 N at 13 m/s
        thrust_scale=[0.97, 1.03],
        noise=config_dict.create(pos=0.005, vel=0.03, att=0.005, rates=0.03),  # std, observation units
    )


def add_userdata(xml: str) -> str:
    """Give the MuJoCo model room for the per-episode state."""
    j = xml.index(">", xml.index("<mujoco"))
    return xml[:j + 1] + f'\n  <size nuserdata="{N_USERDATA}"/>' + xml[j + 1:]


class Realism:
    def __init__(self, cfg: config_dict.ConfigDict, mj_model, p):
        self.cfg = cfg
        self.hover, self.wmax = float(p.hover_speed()), float(p.max_rotor_speed)
        self.thrust_map = jp.array(p.thrust_map)
        self.body = int(mj_model.jnt_bodyid[0])          # the drone's free-joint body
        self.dt = float(mj_model.opt.timestep)

    def speeds(self, a):
        return jp.clip(self.hover + hoop.SPEED_SPAN * jp.clip(a, -1.0, 1.0), 0.0, self.wmax)

    def thrust_of(self, w, scale=1.0):
        c2, c1, c0 = self.thrust_map
        return scale * jp.maximum(0.0, c2 * w * w + c1 * w + c0)

    def sample(self, rng: jax.Array) -> jax.Array:
        """A fresh userdata vector for one episode."""
        c = self.cfg
        k = jax.random.split(rng, 4)
        u = lambda key, lohi: jax.random.uniform(key, (), minval=lohi[0], maxval=lohi[1])  # noqa: E731
        on = 1.0 if c.enabled else 0.0
        ud = jp.zeros(N_USERDATA)
        ud = ud.at[DELAY].set(on * u(k[0], c.delay_ms))
        ud = ud.at[TAU].set(on * u(k[1], c.motor_tau))
        ud = ud.at[DRAG].set(on * u(k[2], c.drag))
        ud = ud.at[SCALE].set(u(k[3], c.thrust_scale) if c.enabled else 1.0)
        return ud.at[ROTOR].set(self.hover)

    def step(self, model: mjx.Model, data: mjx.Data, action: jax.Array, n_substeps: int, rng: jax.Array):
        """Advance one control step: the command that is actually at the motors in each physics substep (delayed),
        rotor speeds lagging behind it, thrust from the speeds, and drag on the body."""
        ud = data.userdata
        jitter = jax.random.uniform(rng, (), minval=0.0, maxval=self.cfg.jitter_ms) if self.cfg.enabled else 0.0
        delay = jp.clip(ud[DELAY] + jitter, 0.0, MAX_DELAY_MS) * 1e-3
        ctrl_dt = n_substeps * self.dt
        tau = ud[TAU]
        alpha = jp.where(tau > 1e-4, 1.0 - jp.exp(-self.dt / jp.maximum(tau, 1e-4)), 1.0)
        w = ud[ROTOR]
        for j in range(n_substeps):                       # unrolled: n_substeps is small (5)
            since = j * self.dt - delay                   # time since this step's command could have arrived
            cmd = jp.where(since >= 0.0, action, jp.where(since >= -ctrl_dt, ud[PREV1], ud[PREV2]))
            w = w + alpha * (self.speeds(cmd) - w)
            v = data.qvel[0:3]
            f = -ud[DRAG] * jp.linalg.norm(v) * v
            xfrc = data.xfrc_applied.at[self.body, 0:3].set(f)
            data = data.replace(ctrl=self.thrust_of(w, ud[SCALE]), xfrc_applied=xfrc)
            data = mjx.step(model, data)
        ud = ud.at[PREV2].set(ud[PREV1]).at[PREV1].set(action).at[ROTOR].set(w)
        return data.replace(userdata=ud)

    def noisy(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        """Noise on the first 18 observation entries: position, velocity, attitude (9) and body rates."""
        if not self.cfg.enabled:
            return obs
        n = self.cfg.noise
        std = jp.hstack([jp.full(3, n.pos), jp.full(3, n.vel), jp.full(9, n.att), jp.full(3, n.rates)])
        return obs.at[:18].add(std * jax.random.normal(rng, (18,)))
