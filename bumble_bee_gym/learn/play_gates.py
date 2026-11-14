"""Score a race-gates policy and film it flying the course.

    python -m bumble_bee_gym.learn.play_gates --name gates_01 [--episodes 1000] [--seconds 15] [--checkpoint STEP]
                                              [--set real.enabled=false ...] [--no_film]

Runs --episodes random starts (as in training) for the full episode and reports gates passed, how many complete a
lap (8 gates in a row), and how the rest end. Then films one flight from the start line (approaching gate 0 from the
last gate) with a chase camera, 1080x1920, real time, into outputs/<name>/race.mp4.
"""
import argparse
import json
import os

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np

from bumble_bee_gym.learn import jax_compat  # noqa: F401
from bumble_bee_gym import hoop
from bumble_bee_gym.learn.gates_env import GatesEnv
from bumble_bee_gym.learn.hoop_env import euler_to_quat
from bumble_bee_gym.learn.play import load_policy
from bumble_bee_gym.learn.train import OUT_ROOT, parse_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checkpoint", default=None, help="a checkpoints/<step> name instead of the final params.pkl")
    ap.add_argument("--set", nargs="*", help="environment overrides on top of the run's own, e.g. real.enabled=false")
    ap.add_argument("--no_film", action="store_true")
    args = ap.parse_args()
    out = os.path.join(OUT_ROOT, args.name)

    if args.checkpoint:
        # Brax's checkpoint.load_policy cannot rebuild the network from its saved config, so build it from the
        # run's own settings (config.json) and load only the parameters.
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import checkpoint, networks as ppo_networks
        with open(os.path.join(out, "config.json")) as f:
            cfg = json.load(f)
        env = GatesEnv(config_overrides={**(cfg.get("env_overrides") or {}), **parse_set(args.set)})
        net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                             preprocess_observations_fn=running_statistics.normalize,
                                             **cfg["ppo"]["network_factory"])
        params = checkpoint.load(os.path.abspath(os.path.join(out, "checkpoints", args.checkpoint)))
        policy = ppo_networks.make_inference_fn(net)(params, deterministic=True)
    else:
        policy, saved = load_policy(os.path.join(out, "params.pkl"), GatesEnv())
        env = GatesEnv(config_overrides={**saved["env_overrides"], **parse_set(args.set)})
    T = int(args.seconds / env.dt)

    def rollout(s0, key):
        def body(s, k):
            a, _ = policy(s.obs, k)
            s2 = env.step(s, a)
            s2 = jax.tree_util.tree_map(lambda x, y: jp.where(s.done > 0, x, y), s, s2)   # freeze after a crash
            return s2, (s2.data.qpos, s2.done, s2.metrics["gates"] * (1 - s.done), s2.metrics["speed_through"] * (1 - s.done))
        _, trace = jax.lax.scan(body, s0, jax.random.split(key, T))
        return trace

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    starts = jax.jit(jax.vmap(env.reset))(keys)
    _, done, gates, speed = jax.jit(jax.vmap(rollout))(starts, keys)
    done, gates, speed = np.asarray(done), np.asarray(gates), np.asarray(speed)
    n = gates.sum(axis=1)
    crashed = done.any(axis=1)
    print(f"{args.episodes} random starts, {args.seconds:.0f} s each: {n.mean():.2f} gates on average, "
          f"{(n >= 8).mean():.1%} complete a lap (8 gates), {crashed.mean():.1%} crash, "
          f"through speed {speed.sum() / max(gates.sum(), 1):.1f} m/s")
    print("  gates passed: " + "  ".join(f"{k}:{(n == k).mean():.0%}" for k in range(int(n.max()) + 1)))

    # One flight from the start line: on the leg from the last gate to gate 0, level, at rest.
    s0 = env.reset(jax.random.PRNGKey(0))
    c = np.asarray(env._centres)
    pos = c[-1] + 0.3 * (c[0] - c[-1])
    yaw = float(np.arctan2(*(c[0] - pos)[[1, 0]]))
    q = jp.hstack([jp.array(pos), euler_to_quat(0.0, 0.0, yaw)])
    data = s0.data.replace(qpos=q, qvel=jp.zeros(6))
    s0 = s0.replace(data=data, obs=env._obs(env.to_gate(data, 0), jp.zeros(4), 0))
    qpos, done1, gates1, _ = jax.jit(rollout)(s0, jax.random.PRNGKey(7))
    qpos, done1 = np.asarray(qpos), np.asarray(done1)
    end = int(done1.argmax()) if done1.any() else T - 1
    print(f"start-line flight: {int(np.asarray(gates1).sum())} gates in {(end + 1) * env.dt:.1f} s, "
          f"{'crashed' if done1.any() else 'no crash'}")
    if args.no_film:
        return

    m = mujoco.MjModel.from_xml_string(hoop.scene_xml(env.course))
    d = mujoco.MjData(m)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1080, 1920
    r = mujoco.Renderer(m, 1920, 1080)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 3.0, -15
    spin = hoop.PropSpinner(m, env._p)
    frames, look, az = [], qpos[0, :3].copy(), None
    for t in range(0, min(end + 30, T), 2):                      # 100 Hz -> 50 fps, real time
        d.qpos[:] = qpos[min(t, end)]
        spin()
        mujoco.mj_forward(m, d)
        vel = qpos[min(t + 2, end), :3] - qpos[max(t - 2, 0), :3]
        heading = np.degrees(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel[:2]) > 1e-3 else (az or 0.0)
        az = heading if az is None else az + 0.1 * ((heading - az + 180) % 360 - 180)   # smooth chase
        look = 0.8 * look + 0.2 * d.qpos[:3]
        cam.lookat[:], cam.azimuth = look, az
        r.update_scene(d, cam)
        frames.append(r.render())
    path = os.path.join(out, f"race_{args.checkpoint}.mp4" if args.checkpoint else "race.mp4")
    mediapy.write_video(path, frames, fps=50)
    print(f"{len(frames)} frames -> {path}")


if __name__ == "__main__":
    main()
