"""Basic signal functions."""

import numpy as np


def time(nsamp, fs=1):
    """Create time array based on signal length and sample rate."""
    return 1 / fs * np.linspace(0, nsamp - 1, nsamp)


def sin(t, amp=0.5, offset=0.5, freq=1e3, ph0=0):
    """Generate a sine wave."""
    return offset + amp * np.sin(ph0 + 2 * np.pi * freq * t)


def noise(t, mean=0, std=0.1):
    """Generate random noise."""
    return np.random.normal(mean, std, size=len(t))
