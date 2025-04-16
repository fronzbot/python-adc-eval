"""Test the spectrum module."""

import pytest
import numpy as np
from unittest import mock
from adc_eval.eval import spectrum


RTOL = 0.05
NLEN = 2**18
NFFT = 2**8
DATA_SINE = [
    {
        "f1": np.random.randint(1, NFFT / 4 - 1),
        "f2": np.random.randint(NFFT / 4, NFFT / 2 - 1),
        "a1": np.random.uniform(low=0.5, high=0.8),
        "a2": np.random.uniform(low=0.1, high=0.4),
    }
    for _ in range(10)
]

@mock.patch("adc_eval.eval.spectrum.calc_psd")
def test_get_spectrum(mock_calc_psd):
    """Test that the get_spectrum method returns power spectrum."""
    fs = 4
    nfft = 3
    data = np.array([1])
    exp_spectrum = np.array([fs / nfft])

    mock_calc_psd.return_value = (None, data, None, 2*data)

    assert (None, exp_spectrum) == spectrum.get_spectrum(None, fs=fs, nfft=nfft, single_sided=True)


@mock.patch("adc_eval.eval.spectrum.calc_psd")
def test_get_spectrum_dual(mock_calc_psd):
    """Test that the get_spectrum method returns dual-sided power spectrum."""
    fs = 4
    nfft = 3
    data = np.array([1])
    exp_spectrum = np.array([fs / nfft])

    mock_calc_psd.return_value = (None, data, None, 2*data)

    assert (None, 2*exp_spectrum) == spectrum.get_spectrum(None, fs=fs, nfft=nfft, single_sided=False)


@pytest.mark.parametrize("data", [np.random.randn(NLEN) for _ in range(10)])
def test_calc_psd_randomized_dual(data):
    """Test calc_psd with random data."""
    (_, _, freq, psd) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 1, rtol=RTOL)


@pytest.mark.parametrize("data", [np.random.randn(NLEN) for _ in range(10)])
def test_calc_psd_randomized_single(data):
    """Test calc_psd with random data and single-sided."""
    (freq, psd, _, _) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 2, rtol=RTOL)


def test_calc_psd_zeros_dual():
    """Test calc_psd with zeros."""
    data = np.zeros(NLEN)
    (_, _, freq, psd) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 0, rtol=RTOL)


def test_calc_psd_zeros_single():
    """Test calc_psd with zeros and single-sided.."""
    data = np.zeros(NLEN)
    (freq, psd, _, _) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 0, rtol=RTOL)


def test_calc_psd_ones_dual():
    """Test calc_psd with ones."""
    data = np.ones(NLEN)
    (_, _, freq, psd) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 1, rtol=RTOL)


def test_calc_psd_ones_single():
    """Test calc_psd with ones and single-sided."""
    data = np.ones(NLEN)
    (freq, psd, _, _) = spectrum.calc_psd(data, 1, nfft=NFFT)
    mean_val = np.mean(psd)
    assert np.isclose(mean_val, 2, rtol=RTOL)


@pytest.mark.parametrize("data", DATA_SINE)
def test_calc_psd_two_sine_dual(data):
    """Test calc_psd with two sine waves."""
    fs = 1
    fbin = fs / NFFT
    f1 = data["f1"] * fbin
    f2 = data["f2"] * fbin
    a1 = data["a1"]
    a2 = data["a2"]

    t = 1 / fs * np.linspace(0, NLEN - 1, NLEN)
    pin = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)

    (_, _, freq, psd) = spectrum.calc_psd(pin, fs, nfft=NFFT)

    exp_peaks = [
        round(a1**2 / 4 * NFFT, 3),
        round(a2**2 / 4 * NFFT, 3),
    ]

    exp_f1 = [round(-f1, 2), round(f1, 2)]
    exp_f2 = [round(-f2, 2), round(f2, 2)]

    peak1 = max(psd)
    ipeaks = np.where(psd >= peak1 * (1 - RTOL))[0]
    fpeaks = [round(freq[ipeaks[0]], 2), round(freq[ipeaks[1]], 2)]

    assertmsg = f"f1={f1} | f2={f2} | a1={a1} | a2={a2}"

    assert np.allclose(peak1, exp_peaks[0], rtol=RTOL), assertmsg
    assert np.allclose(fpeaks, exp_f1, rtol=RTOL), assertmsg

    psd[ipeaks[0]] = 0
    psd[ipeaks[1]] = 0

    peak2 = max(psd)
    ipeaks = np.where(psd >= peak2 * (1 - RTOL))[0]
    fpeaks = [round(freq[ipeaks[0]], 2), round(freq[ipeaks[1]], 2)]

    assert np.allclose(peak2, exp_peaks[1], rtol=RTOL), assertmsg
    assert np.allclose(fpeaks, exp_f2), assertmsg


@pytest.mark.parametrize("data", DATA_SINE)
def test_calc_psd_two_sine_single(data):
    """Test calc_psd with two sine waves, single-eided."""
    fs = 1
    fbin = fs / NFFT
    f1 = data["f1"] * fbin
    f2 = data["f2"] * fbin
    a1 = data["a1"]
    a2 = data["a2"]

    t = 1 / fs * np.linspace(0, NLEN - 1, NLEN)
    pin = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)

    (freq, psd, _, _) = spectrum.calc_psd(pin, fs, nfft=NFFT)

    exp_peaks = [
        round(a1**2 / 2 * NFFT, 3),
        round(a2**2 / 2 * NFFT, 3),
    ]
    exp_f1 = round(f1, 2)
    exp_f2 = round(f2, 2)

    peak1 = max(psd)
    ipeak = np.where(psd == peak1)[0][0]
    fpeak = round(freq[ipeak], 2)

    assertmsg = f"f1={f1} | f2={f2} | a1={a1} | a2={a2}"

    assert np.allclose(peak1, exp_peaks[0], rtol=RTOL), assertmsg
    assert np.allclose(fpeak, exp_f1), assertmsg

    psd[ipeak] = 0

    peak2 = max(psd)
    ipeak = np.where(psd == peak2)[0][0]
    fpeak = round(freq[ipeak], 2)

    assert np.allclose(peak2, exp_peaks[1], rtol=RTOL), assertmsg
    assert np.allclose(fpeak, exp_f2), assertmsg
