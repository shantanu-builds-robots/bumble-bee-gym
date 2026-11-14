"""Score a trained hoop policy and film it.

    python -m bumble_bee_gym.learn.play --name hoop_01 [--episodes 1000] [--video 6]

Runs --episodes random starts in parallel and prints the success rate, then films --video of them one after the
other (the full visual scene, 1080x1920, 2x slow motion) into outputs/<name>/hoop.mp4.
"""
import argparse
import os
import pickle

import jax
import jax.numpy as jp
import mediapy
import mujoco
import numpy as np

from bumble_bee_gym.learn import jax_compat  # noqa: F401
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from bumble_bee_gym import hoop
from bumble_bee_gym.learn.hoop_env import HoopEnv
from bumble_bee_gym.learn.train import OUT_ROOT


def load_policy(path, env):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    nets = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                          preprocess_observations_fn=running_statistics.normalize,
                                          **saved["network"])
    return ppo_networks.make_inference_fn(nets)(saved["params"], deterministic=True), saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--video", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = os.path.join(OUT_ROOT, args.name)

    probe = HoopEnv()
    policy, saved = load_policy(os.path.join(out, "params.pkl"), probe)
    env = HoopEnv(config_overrides=saved["env_overrides"])
    T = env._config.episode_length

    def rollout(key):
        s0 = env.reset(key)

        def body(s, k):
            a, _ = policy(s.obs, k)
            s2 = env.step(s, a)
            s2 = jax.tree_util.tree_map(lambda x, y: jp.where(s.done > 0, x, y), s, s2)   # freeze after the end
            return s2, (s2.data.qpos, s2.done, s2.metrics["through"], s2.metrics["speed_through"])

        _, trace = jax.lax.scan(body, s0, jax.random.split(key, T))
        return s0.data.qpos, trace

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    q0, (qpos, done, through, speed) = jax.jit(jax.vmap(rollout))(keys)
    through, done, speed = np.asarray(through), np.asarray(done), np.asarray(speed)
    ok = through.max(axis=1) > 0
    ends = np.where(done.any(axis=1), done.argmax(axis=1), T - 1)
    print(f"{args.episodes} random starts: through {ok.mean():.1%}, crashed or missed "
          f"{(done.any(axis=1) & ~ok).mean():.1%}, timed out {(~done.any(axis=1)).mean():.1%}")
    if ok.any():
        print(f"  successful: {np.mean((ends[ok] + 1) * env.dt):.2f} s to go through, "
              f"{np.mean(speed.max(axis=1)[ok]):.2f} m/s through the hoop")

    if args.video <= 0:
        return
    m = mujoco.MjModel.from_xml_string(hoop.scene_xml(env.course))
    d = mujoco.MjData(m)
    m.vis.global_.offwidth, m.vis.global_.offheight = 1080, 1920
    r = mujoco.Renderer(m, 1920, 1080)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = env.course.gates[0].xyz + np.array([-1.5, 0.0, -0.1])   # looking back through the hoop
    cam.distance, cam.azimuth, cam.elevation = 7.0, 160 + env.course.gates[0].yaw, -10
    frames, qpos = [], np.asarray(qpos)
    for e in range(args.video):
        for t in range(0, ends[e] + 25):                    # every 100 Hz step at 50 fps: 2x slow motion
            d.qpos[:] = qpos[e, min(t, ends[e])]
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            frames.append(r.render())
    path = os.path.join(out, "hoop.mp4")
    mediapy.write_video(path, frames, fps=50)
    print(f"{len(frames)} frames -> {path}  (episodes 0-{args.video - 1}: "
          f"{', '.join('through' if o else 'failed' for o in ok[:args.video])})")


if __name__ == "__main__":
    main()
