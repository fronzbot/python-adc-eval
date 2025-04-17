"""Basic signal functions."""

import numpy as np


def time(nlen, fs=1):
    """
    Create time array based on signal length and sample rate.

    Paraneters
    ----------
    nlen : int
        Desired length of time array.
    fs : float, optional
        Sample frequency of data in Hz. Default is 1 Hz.

    Returns
    -------
    ndarray
        Time list stored in ndarray type.

    """
    return 1 / fs * np.linspace(0, nlen - 1, nlen)


def sin(t, amp=0.5, offset=0.5, freq=1e3, ph0=0):
    """
    Generate a sine wave.

    Parameters
    ----------
    t : ndarray
        Time array list for sine wave.
    amp : float, optional
        Amplitude of desired sine wave. Default is 0.5.
    offset : float, optional
        DC offset of the desired sine wave. Default is 0.5.
    freq : float, optional
        Desired frequency of the sine wave in Hz. Default is 1kHz.
    ph0 : float, optional
        Desired phase shift of the sine wave in radians. Default is 0.

    Returns
    -------
    ndarray
        Sine wave stored in ndarray type.

    """
    return offset + amp * np.sin(ph0 + 2 * np.pi * freq * t)


def noise(nlen, mean=0, std=0.1):
    """
    Generate random noise.

    Parameters
    ----------
    nlen : int
        Desired length of noise array.
    mean : float, optional
        Desired average of noise array. Default is 0.
    std : float, optional
        Desired standard deviation of noise array. Default is 0.1.

    Returns
    -------
    ndarray
        Gaussian distributed noise array.
    """
    return np.random.normal(mean, std, size=nlen)


def impulse(nlen, mag=1):
    """
    Generate an impulse input.

    Parameters
    ----------
    nlen : int
        Desired length of noise array.
    mag : float, optional
        Desired magnitude of impulse. Default is 1.

    Returns
    -------
    ndarray
        Impulse waveform in ndarray type.
    """
    data = np.zeros(nlen)
    data[0] = mag
    return data
