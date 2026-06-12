"""Configurable noise models for sensor simulation."""

import numpy as np
from collections import deque


class GaussianNoise:
    """Additive zero-mean Gaussian noise."""

    def __init__(self, sigma, bias=None, rng: np.random.Generator = None):
        self.sigma = np.asarray(sigma, dtype=float)
        self.bias = np.zeros_like(self.sigma) if bias is None else np.asarray(bias, float)
        self.rng = rng or np.random.default_rng()

    def sample(self, x: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sigma, size=x.shape)
        return x + noise + self.bias


class DelayedNoise:
    """Gaussian noise + fixed step delay (FIFO buffer)."""

    def __init__(self, sigma, delay_steps: int, bias=None,
                 rng: np.random.Generator = None):
        self.noise = GaussianNoise(sigma, bias, rng)
        self._buf: deque = deque(maxlen=max(delay_steps, 1))
        self._delay = delay_steps
        self._first = True

    def sample(self, x: np.ndarray) -> np.ndarray:
        noisy = self.noise.sample(x)
        if self._first:
            # fill buffer with initial value
            for _ in range(self._delay):
                self._buf.append(noisy.copy())
            self._first = False
        self._buf.append(noisy)
        return self._buf[0].copy()


def orientation_noise(R: np.ndarray, sigma_rad: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Perturb rotation matrix R by a random small rotation.

    Samples a random axis-angle vector with magnitude ~ N(0, sigma_rad) and
    applies it as a left-multiplicative perturbation: R_noisy = exp(skew(v)) @ R.
    """
    v = rng.normal(0.0, sigma_rad, size=3)
    angle = np.linalg.norm(v)
    if angle < 1e-9:
        return R.copy()
    axis = v / angle
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    dR = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return dR @ R
