"""Compatibility shim: Brax 0.14.2's PPO calls jax.device_put_replicated, which JAX 0.10 removed.

Import this before brax.training. It restores the old behaviour: stack one copy per device along a new leading
axis and place each slice on its device (what pmap expects for a replicated argument).
"""
import jax
import jax.numpy as jp


def _device_put_replicated(x, devices):
    devices = list(devices)

    def rep(a):
        shards = [jax.device_put(jp.asarray(a)[None], d) for d in devices]
        if len(devices) == 1:
            return shards[0]
        mesh = jax.sharding.Mesh(devices, ("d",))
        spec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))
        return jax.make_array_from_single_device_arrays((len(devices),) + jp.shape(a), spec, shards)

    return jax.tree_util.tree_map(rep, x)


if "device_put_replicated" not in jax.__dict__:
    jax.device_put_replicated = _device_put_replicated
