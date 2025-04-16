"""Test the spectrum plotting functions."""

import pytest
import numpy as np
from unittest import mock
from adc_eval.eval import spectrum


RTOL = 0.05
NLEN = 2**16
AMPLITUDE = 0.5 / np.sqrt(2)

RAND_HARMS = 4
RAND_NFFT = 3
RAND_LEAK = 3


def gen_spectrum(sig_bin, harmonics, nfft):
    """Generate a wave with arbitrary harmonics."""
    t = np.linspace(0, NLEN - 1, NLEN)
    vin = np.zeros(len(t))
    fin = sig_bin / nfft
    for i in range(1, harmonics + 1):
        vin += np.sqrt(2) * AMPLITUDE / i * np.sin(2 * np.pi * i * fin * t)

    return spectrum.get_spectrum(vin, fs=1, nfft=nfft)


@pytest.mark.parametrize("harms", np.random.randint(1, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(10, 16, RAND_NFFT)))
def test_find_harmonics(harms, nfft):
    """Test the find harmonics method."""
    nbin = np.random.randint(1, int(nfft / (2 * harms)) - 1)
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = spectrum.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=0, fscale=1e-6
    )

    for n in range(2, harms + 1):
        msg_txt = f"nfft={nfft}, nbin={nbin}, harm={harms}, index={n}"
        exp_bin = n * nbin
        exp_power = (AMPLITUDE / n) ** 2
        exp_freq = freq[exp_bin] * 1e6
        assert stats["harm"][n]["bin"] == exp_bin, msg_txt
        assert np.allclose(stats["harm"][n]["freq"], exp_freq, rtol=RTOL), msg_txt
        assert np.allclose(stats["harm"][n]["power"], exp_power, rtol=RTOL), msg_txt


@pytest.mark.parametrize("harms", np.random.randint(1, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(12, 16, RAND_NFFT)))
@pytest.mark.parametrize("leak", np.random.randint(1, 10, RAND_LEAK))
def test_find_harmonics_with_leakage(harms, nfft, leak):
    """Test the find harmonics method with spectral leakage."""
    nbin = np.random.randint(2 * leak, int(nfft / (2 * harms)) - 1)
    nbin = nbin + round(
        np.random.uniform(-0.5, 0.5), 2
    )  # Ensures we're not coherently sampled
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = spectrum.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=leak
    )

    for n in range(2, harms + 1):
        msg_txt = f"nfft={nfft}, nbin={nbin}, harm={harms}, leak={leak}, index={n}"
        bin_low = n * nbin - leak
        bin_high = n * nbin + leak
        assert bin_low <= stats["harm"][n]["bin"] <= bin_high, msg_txt


@pytest.mark.parametrize("harms", np.random.randint(2, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(12, 16, RAND_NFFT)))
@pytest.mark.parametrize("leak", np.random.randint(1, 10, RAND_LEAK))
def test_find_harmonics_with_leakage_outside_bounds(harms, nfft, leak):
    """Test find harmonics with leakage bins exceeding array bounds."""
    nbin = nfft / 4 - 0.5
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)
    stats = spectrum.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=leak
    )
    # Only check second harmonic which is guaranteed to be at edge of FFT
    msg_txt = f"nfft={nfft}, nbin={nbin}, harm={harms}, leak={leak}"
    assert nfft / 2 - leak <= stats["harm"][2]["bin"] <= nfft / 2 - 1


@pytest.mark.parametrize("harms", np.random.randint(6, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(8, 16, RAND_NFFT)))
def test_find_harmonics_on_fft_bound(harms, nfft):
    """Test find harmonics with harmonics landing at nfft/2."""
    nbin = nfft / 8
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = spectrum.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=0
    )

    exp_bin = {
        2: 2 * nbin,
        3: 3 * nbin,
        4: 0,
        5: nfft - 5 * nbin,
    }

    for n, exp_val in exp_bin.items():
        msg_txt = f"nfft={nfft}, nbin={nbin}, harm={harms}, index={n}"
        assert stats["harm"][n]["bin"] == exp_val, msg_txt


@pytest.mark.parametrize("harms", np.random.randint(2, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(8, 16, RAND_NFFT)))
def test_plot_string(harms, nfft):
    """Test proper return of plotting string."""
    nbin = np.random.randint(2, int(nfft / (2 * harms)) - 1)
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = spectrum.sndr_sfdr(pwr, freq, 1, nfft, leak=0, full_scale=0)
    hstats = spectrum.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=0, fscale=1
    )

    all_stats = {**stats, **hstats}

    plt_str = spectrum.get_plot_string(
        all_stats, 0, 1, nfft, window="rectangular", xscale=1, fscale="Hz"
    )

    # Check for important information, not everything
    msg_txt = f"{all_stats}\n{plt_str}"

    assert f"NFFT = {nfft}" in plt_str, msg_txt
    assert f"ENOB = {all_stats['enob']['bits']} bits" in plt_str, msg_txt
    assert f"SNDR = {all_stats['sndr']['dBFS']} dBFS" in plt_str, msg_txt
    assert f"SFDR = {all_stats['sfdr']['dBFS']} dBFS" in plt_str, msg_txt
    assert f"Noise Floor = {all_stats['noise']['dBHz']} dBFS" in plt_str, msg_txt

    for n in range(2, harms + 1):
        harm_power = round(all_stats["harm"][n]["dB"], 1)
        harm_freq = all_stats["harm"][n]["freq"]
        assert f"HD{n} = {harm_power} dBFS @ {harm_freq} Hz" in plt_str, msg_txt
