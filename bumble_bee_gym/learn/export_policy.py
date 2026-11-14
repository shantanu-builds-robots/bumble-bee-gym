"""Export a trained race-gates policy to a plain numpy file, so it can fly without JAX (e.g. in the Gazebo container).

    python -m bumble_bee_gym.learn.export_policy --name gates_02 [--checkpoint STEP]

Writes outputs/<name>/policy[_<STEP>].npz with the observation normaliser (mean, std) and the policy MLP's weights
(W0, b0, W1, b1, ...), then checks that bumble_bee_gym.policy_np gives the same actions as Brax on 4096 random
observations.
"""
import argparse
import json
import os

import jax
import jax.numpy as jp
import numpy as np

from bumble_bee_gym.learn import jax_compat  # noqa: F401
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks as ppo_networks

from bumble_bee_gym import policy_np
from bumble_bee_gym.learn.gates_env import GatesEnv
from bumble_bee_gym.learn.train import OUT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()
    out = os.path.join(OUT_ROOT, args.name)
    with open(os.path.join(out, "config.json")) as f:
        cfg = json.load(f)
    env = GatesEnv(config_overrides=cfg.get("env_overrides") or {})
    net = ppo_networks.make_ppo_networks(env.observation_size, env.action_size,
                                         preprocess_observations_fn=running_statistics.normalize,
                                         **cfg["ppo"]["network_factory"])
    if args.checkpoint:
        params = checkpoint.load(os.path.abspath(os.path.join(out, "checkpoints", args.checkpoint)))
    else:
        import pickle
        with open(os.path.join(out, "params.pkl"), "rb") as f:
            params = pickle.load(f)["params"]
    norm, pol = params[0], params[1]

    layers = pol["params"]
    names = sorted(layers, key=lambda n: int(n.split("_")[-1]))
    arrays = {"mean": np.asarray(norm.mean), "std": np.asarray(norm.std)}
    for i, n in enumerate(names):
        arrays[f"W{i}"], arrays[f"b{i}"] = np.asarray(layers[n]["kernel"]), np.asarray(layers[n]["bias"])
    path = os.path.join(out, f"policy_{args.checkpoint}.npz" if args.checkpoint else "policy.npz")
    np.savez(path, **arrays)

    # Same answer as Brax?
    brax_policy = jax.jit(ppo_networks.make_inference_fn(net)(params, deterministic=True))
    obs = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (4096, env.observation_size))) * 2.0
    want = np.asarray(brax_policy(jp.array(obs), jax.random.PRNGKey(1))[0])
    got = policy_np.Policy(path)(obs)
    print(f"{len(names)} layers -> {path}; max |numpy - brax| over 4096 observations: {np.abs(got - want).max():.2e}")


if __name__ == "__main__":
    main()
