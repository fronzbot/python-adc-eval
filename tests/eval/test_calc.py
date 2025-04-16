"""Test the eval.calc functions."""

import pytest
import numpy as np
from unittest import mock
from adc_eval.eval import calc
from adc_eval.eval import spectrum


RTOL = 0.05
NLEN = 2**16
AMPLITUDE = 0.5 / np.sqrt(2)
RAND_HARMS = 4
RAND_NFFT = 3
RAND_LEAK = 3
TEST_SNDR = [
    np.random.uniform(low=0.1, high=100, size=np.random.randint(4, 31))
    for _ in range(10)
]
TEST_VALS = [np.random.uniform(low=0.1, high=50) for _ in range(3)]
PLACES = [i for i in range(6)]


def gen_spectrum(sig_bin, harmonics, nfft):
    """Generate a wave with arbitrary harmonics."""
    t = np.linspace(0, NLEN - 1, NLEN)
    vin = np.zeros(len(t))
    fin = sig_bin / nfft
    for i in range(1, harmonics + 1):
        vin += np.sqrt(2) * AMPLITUDE / i * np.sin(2 * np.pi * i * fin * t)

    return spectrum.get_spectrum(vin, fs=1, nfft=nfft)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_db_to_pow_places(data, places):
    """Test the db_to_pow conversion with multiple places."""
    exp_val = round(10 ** (data / 10), places)
    assert exp_val == calc.db_to_pow(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_db_to_pow_ndarray(data, places):
    """Test db_to_pow with ndarray input."""
    data = np.array(data)
    exp_val = np.array(round(10 ** (data / 10), places))
    assert exp_val == calc.db_to_pow(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_dbW(data, places):
    """Test the dbW conversion with normal inputs."""
    exp_val = round(10 * np.log10(data), places)
    assert exp_val == calc.dBW(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_dbW_ndarray(data, places):
    """Test dbW with ndarray input."""
    data = np.array(data)
    exp_val = np.array(round(10 * np.log10(data), places))
    assert exp_val == calc.dBW(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_enob(data, places):
    """Test enob with muliple places."""
    exp_val = round(1 / 6.02 * (data - 1.76), places)
    assert exp_val == calc.enob(data, places=places)


@pytest.mark.parametrize("harms", np.random.randint(1, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(10, 16, RAND_NFFT)))
def test_find_harmonics(harms, nfft):
    """Test the find harmonics method."""
    nbin = np.random.randint(1, int(nfft / (2 * harms)) - 1)
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = calc.find_harmonics(
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

    stats = calc.find_harmonics(
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
    stats = calc.find_harmonics(
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

    stats = calc.find_harmonics(
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


@pytest.mark.parametrize("data", TEST_SNDR)
def test_sndr_sfdr_outputs(data):
    """Test the sndr_sfdr method outputs."""
    freq = np.linspace(0, 1000, np.size(data))
    full_scale = -3
    nfft = 2**8
    fs = 1

    psd_test = data.copy()
    psd_exp = data.copy()

    result = calc.sndr_sfdr(psd_test, freq, fs, nfft, 0, full_scale=full_scale)

    data[0] = 0
    psd_exp[0] = 0
    data_string = f"F = {freq}\nD = {data}"

    indices = np.argsort(psd_exp)
    sbin = indices[-1]
    spurbin = indices[-2]
    sfreq = freq[sbin]
    spwr = psd_exp[sbin]

    psd_exp[sbin] = 0
    spurfreq = freq[spurbin]
    spurpwr = psd_exp[spurbin]

    noise_pwr = np.sum(psd_exp[1:])

    exp_return = {
        "sig": {
            "freq": sfreq,
            "bin": sbin,
            "power": spwr,
            "dB": round(10 * np.log10(spwr), 1),
            "dBFS": round(10 * np.log10(spwr) - full_scale, 1),
        },
        "spur": {
            "freq": spurfreq,
            "bin": spurbin,
            "power": spurpwr,
            "dB": 10 * np.log10(spurpwr),
            "dBFS": round(10 * np.log10(spurpwr) - full_scale, 1),
        },
        "noise": {
            "floor": 2 * noise_pwr / nfft,
            "power": noise_pwr,
            "rms": np.sqrt(noise_pwr),
            "dBHz": round(10 * np.log10(2 * noise_pwr / nfft) - full_scale, 1),
            "NSD": round(
                10 * np.log10(2 * noise_pwr / nfft)
                - full_scale
                - 2 * 10 * np.log10(fs / nfft),
                1,
            ),
        },
        "sndr": {
            "dBc": round(10 * np.log10(spwr / noise_pwr), 1),
            "dBFS": round(full_scale - 10 * np.log10(noise_pwr), 1),
        },
        "sfdr": {
            "dBc": round(10 * np.log10(spwr / spurpwr), 1),
            "dBFS": round(full_scale - 10 * np.log10(spurpwr), 1),
        },
        "enob": {
            "bits": round((full_scale - 10 * np.log10(noise_pwr) - 1.76) / 6.02, 1),
        },
    }

    for key, val in exp_return.items():
        for measure, measure_val in val.items():
            msg = f"{data_string}\n{key} -> {measure} | Expected {measure_val} | Got {result[key][measure]}"
            assert np.allclose(measure_val, result[key][measure], rtol=RTOL), msg


@pytest.mark.parametrize("harms", np.random.randint(2, 21, RAND_HARMS))
@pytest.mark.parametrize("nfft", 2 ** (np.random.randint(8, 16, RAND_NFFT)))
def test_plot_string(harms, nfft):
    """Test proper return of plotting string."""
    nbin = np.random.randint(2, int(nfft / (2 * harms)) - 1)
    (freq, pwr) = gen_spectrum(nbin, harms, nfft)

    stats = calc.sndr_sfdr(pwr, freq, 1, nfft, leak=0, full_scale=0)
    hstats = calc.find_harmonics(
        pwr, freq, nfft, nbin, AMPLITUDE, harms=harms, leak=0, fscale=1
    )

    all_stats = {**stats, **hstats}

    plt_str = calc.get_plot_string(
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