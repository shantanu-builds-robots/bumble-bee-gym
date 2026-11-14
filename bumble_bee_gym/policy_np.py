"""A trained flight policy evaluated with numpy only (no JAX), for flying it in Gazebo.

Loads the .npz written by ``learn/export_policy.py``: Brax PPO's policy is observation normalisation, an MLP with
swish activations, and a last layer giving the mean and scale of a tanh-squashed Gaussian. Flying uses the mean,
so the action is tanh(mean).
"""
import numpy as np


class Policy:
    def __init__(self, path):
        self.path = str(path)
        z = np.load(path)
        self.mean, self.std = z["mean"], z["std"]
        n = len([k for k in z.files if k.startswith("W")])
        self.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (np.asarray(obs, np.float64) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            x = x @ W + b
            if i < len(self.layers) - 1:
                x = x / (1.0 + np.exp(-x))          # swish
        loc = x[..., : x.shape[-1] // 2]
        return np.tanh(loc)
