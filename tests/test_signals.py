"""Test the signals module."""

import pytest
import numpy as np
from scipy import stats
from adc_eval import signals


RTOL = 0.01


@pytest.mark.parametrize("nlen", np.random.randint(4, 2**16, 3))
@pytest.mark.parametrize("fs", np.random.uniform(1, 1e9, 3))
def test_time(nlen, fs):
    """Test time with random data."""
    value = signals.time(nlen, fs=fs)
    assert value.size == nlen
    assert value[0] == 0
    assert np.isclose(value[nlen - 1], (nlen - 1) / fs, rtol=RTOL)


@pytest.mark.parametrize("nlen", np.random.randint(2**10, 2**16, 3))
@pytest.mark.parametrize("offset", np.random.uniform(-10, 10, 3))
@pytest.mark.parametrize("amp", np.random.uniform(0, 10, 3))
def test_sin(nlen, offset, amp):
    """Test sine generation with random data."""
    fs = np.random.uniform(1, 1e9)
    fin = np.random.uniform(fs / 10, fs / 3)

    t = signals.time(nlen, fs=fs)
    value = signals.sin(t, amp=amp, offset=offset, freq=fin)

    exp_peaks = [offset - amp, amp + offset]

    assert value.size == nlen
    assert np.isclose(max(value), exp_peaks[1], rtol=RTOL)
    assert np.isclose(min(value), exp_peaks[0], rtol=RTOL)
    assert value[0] == offset


@pytest.mark.parametrize("nlen", np.random.randint(1, 2**16, 5))
def test_noise_length(nlen):
    """Test noise generation with random data."""
    t = signals.time(nlen, fs=1)
    value = signals.noise(t, mean=0, std=1)

    # Just check correct size
    assert value.size == nlen


@pytest.mark.parametrize("std", np.random.uniform(0, 1, 4))
def test_noise_length(std):
    """Test noise is gaussian with random data."""
    nlen = 2**12
    t = signals.time(nlen, fs=1)
    noise = signals.noise(t, mean=0, std=std)
    autocorr = np.correlate(noise, noise, mode="full")
    autocorr /= max(autocorr)
    asize = autocorr.size

    midlag = autocorr.size // 2
    acorr_nopeak = np.concatenate([autocorr[0 : midlag - 1], autocorr[midlag + 1 :]])

    shapiro = stats.shapiro(acorr_nopeak)

    # Check that middle lag is 1
    assert autocorr[midlag] == 1

    # Now check that noise is gaussian
    assert shapiro.pvalue < 0.01
