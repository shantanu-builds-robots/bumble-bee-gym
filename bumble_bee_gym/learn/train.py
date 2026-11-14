"""Train a flight policy with Brax PPO on the GPU: the single hoop (default) or the race gates.

    python -m bumble_bee_gym.learn.train --name hoop_01 [--env hoop|gates] [--timesteps 30e6] [--num_envs 4096] [--seed 0]
                                         [--set reward_config.scales.progress=3 ...] [--ppo learning_rate=1e-4 ...]

Writes outputs/<name>/: config.json, progress.jsonl (one line per evaluation), checkpoints/, params.pkl (final policy).
Run it in the playground image through learn/run.sh.
"""
import argparse
import functools
import json
import os
import pickle
import time

import jax

from bumble_bee_gym.learn import jax_compat  # noqa: F401  (before brax.training)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco_playground import wrapper

from bumble_bee_gym.learn.gates_env import GatesEnv
from bumble_bee_gym.learn.hoop_env import HoopEnv

ENVS = {"hoop": HoopEnv, "gates": GatesEnv}

OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "outputs")


def ppo_config(episode_length: int) -> config_dict.ConfigDict:
    return config_dict.create(
        num_timesteps=30_000_000,
        num_evals=15,
        reward_scaling=1.0,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=4,
        discounting=0.99,
        learning_rate=3e-4,
        entropy_cost=1e-3,
        num_envs=4096,
        batch_size=256,
        max_grad_norm=1.0,
        num_resets_per_eval=10,
        network_factory=config_dict.create(
            policy_hidden_layer_sizes=(128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256),
        ),
    )


def parse_set(items):
    """--set a.b=1.5 c=[1,2] -> {"a.b": 1.5, "c": [1, 2]} (values parsed as JSON when possible)."""
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--env", default="hoop", choices=sorted(ENVS))
    ap.add_argument("--timesteps", type=float, default=None)
    ap.add_argument("--num_envs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", help="environment config overrides, dotted keys")
    ap.add_argument("--ppo", nargs="*", help="PPO config overrides, e.g. learning_rate=1e-4")
    ap.add_argument("--restore", default=None, help="checkpoint directory to warm-start from")
    args = ap.parse_args()

    out = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out, exist_ok=True)
    overrides = parse_set(args.set)
    Env = ENVS[args.env]
    env, eval_env = Env(config_overrides=overrides), Env(config_overrides=overrides)
    cfg = ppo_config(env._config.episode_length)
    cfg.update(parse_set(args.ppo))
    if args.timesteps:
        cfg.num_timesteps = int(args.timesteps)
    if args.num_envs:
        cfg.num_envs = args.num_envs
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"env": args.env, "seed": args.seed, "env_overrides": overrides, "restore": args.restore,
                   "env_config": env._config.to_dict(), "ppo": cfg.to_dict()}, f, indent=2, default=str)

    params = cfg.to_dict()
    net = params.pop("network_factory")
    t0 = time.time()
    log = open(os.path.join(out, "progress.jsonl"), "a")

    def progress(step, metrics):
        if "eval/episode_reward" not in metrics:
            return
        row = {"step": int(step), "minutes": round((time.time() - t0) / 60, 2)}
        row.update({k: round(float(v), 4) for k, v in metrics.items() if k.startswith("eval/") or k == "training/sps"})
        log.write(json.dumps(row) + "\n"); log.flush()
        ok = row.get("eval/episode_through", row.get("eval/episode_gates", 0.0))
        speed = row.get("eval/episode_speed_through", 0.0) / max(ok, 1e-6)
        off = row.get("eval/episode_miss_offset", 0.0) / max(ok, 1e-6)
        passed = f"through {ok:5.1%}" if args.env == "hoop" else f"gates/episode {ok:5.2f}"
        print(f"[{row['minutes']:6.1f} min] step {step:>11,}  reward {row['eval/episode_reward']:8.2f}  "
              f"{passed}  crashed {row.get('eval/episode_crashed', 0):5.1%}  "
              f"ep_len {row.get('eval/avg_episode_length', float('nan')):6.1f}  speed {speed:4.2f} m/s  "
              f"offset {off:4.2f}  fails " + " ".join(f"{k}:{row.get(f'eval/episode_fail/{k}', 0):.0%}"
                                                 for k in ("frame", "missed", "flipped", "ground", "away")), flush=True)

    train_fn = functools.partial(
        ppo.train, **params, network_factory=functools.partial(ppo_networks.make_ppo_networks, **net),
        seed=args.seed, progress_fn=progress, save_checkpoint_path=os.path.join(out, "checkpoints"),
        restore_checkpoint_path=args.restore)
    _, policy_params, _ = train_fn(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training)
    with open(os.path.join(out, "params.pkl"), "wb") as f:
        pickle.dump({"env": args.env, "env_overrides": overrides, "network": net, "params": jax.device_get(policy_params)}, f)
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
