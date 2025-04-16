"""Test the spectrum module."""

import pytest
import numpy as np
from unittest import mock
from adc_eval.eval import spectrum


RTOL = 0.05
TEST_SNDR = [
    np.random.uniform(low=0.1, high=100, size=np.random.randint(4, 31))
    for _ in range(10)
]
TEST_VALS = [np.random.uniform(low=0.1, high=50) for _ in range(3)]
PLACES = [i for i in range(6)]


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_db_to_pow_places(data, places):
    """Test the db_to_pow conversion with multiple places."""
    exp_val = round(10 ** (data / 10), places)
    assert exp_val == spectrum.db_to_pow(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_db_to_pow_ndarray(data, places):
    """Test db_to_pow with ndarray input."""
    data = np.array(data)
    exp_val = np.array(round(10 ** (data / 10), places))
    assert exp_val == spectrum.db_to_pow(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_dbW(data, places):
    """Test the dbW conversion with normal inputs."""
    exp_val = round(10 * np.log10(data), places)
    assert exp_val == spectrum.dBW(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_dbW_ndarray(data, places):
    """Test dbW with ndarray input."""
    data = np.array(data)
    exp_val = np.array(round(10 * np.log10(data), places))
    assert exp_val == spectrum.dBW(data, places=places)


@pytest.mark.parametrize("data", TEST_VALS)
@pytest.mark.parametrize("places", PLACES)
def test_enob(data, places):
    """Test enob with muliple places."""
    exp_val = round(1 / 6.02 * (data - 1.76), places)
    assert exp_val == spectrum.enob(data, places=places)


@mock.patch("adc_eval.eval.spectrum.calc_psd")
def test_get_spectrum(mock_calc_psd):
    """Test that the get_spectrum method returns power spectrum."""
    fs = 4
    nfft = 3
    data = np.array([1])
    exp_spectrum = np.array([fs / nfft])

    mock_calc_psd.return_value = (None, data, None, data)

    assert (None, exp_spectrum) == spectrum.get_spectrum(None, fs=fs, nfft=nfft)


@pytest.mark.parametrize("data", TEST_SNDR)
def test_sndr_sfdr_outputs(data):
    """Test the sndr_sfdr method outputs."""
    freq = np.linspace(0, 1000, np.size(data))
    full_scale = -3
    nfft = 2**8
    fs = 1

    psd_test = data.copy()
    psd_exp = data.copy()

    result = spectrum.sndr_sfdr(psd_test, freq, fs, nfft, 0, full_scale=full_scale)

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
