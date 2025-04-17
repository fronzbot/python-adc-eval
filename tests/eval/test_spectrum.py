"""Test the spectrum module."""

import pytest
import numpy as np
from unittest import mock
from adc_eval.eval import spectrum


RTOL = 0.05
NLEN = 2**18
NFFT = 2**10
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


def test_window_data_as_list():
    """Tests the window_data function when given a list instead of numpy array."""
    data = np.random.rand(NLEN).tolist()
    wdata = spectrum.window_data(data, window="rectangular")
    
    assert type(data) == type(list())
    assert type(wdata) == type(np.ndarray([]))


def test_window_data_bad_window_type(capfd):
    """Tests the window_data function with an incorrect window selection."""
    data = np.random.rand(NLEN)
    wdata = spectrum.window_data(data, window="foobar")
    captured = capfd.readouterr()
    
    assert data.size == wdata.size
    assert data.all() == wdata.all()
    assert "WARNING" in captured.out


@mock.patch("adc_eval.eval.spectrum.plot_spectrum")
def test_analyze_bad_input_scalar(mock_plot_spectrum, capfd):
    """Tests bad input scalar keys."""
    mock_plot_spectrum.return_value = (None, None, None)
    mock_plot_spectrum.side_effect = lambda *args, **kwargs: (kwargs, None, None)
    (kwargs, _, _) = spectrum.analyze([0], 1, fscale="foobar")
    captured = capfd.readouterr()

    assert "WARNING" in captured.out
    assert kwargs.get("fscale") == ("MHz", 1e6)


@mock.patch("adc_eval.eval.spectrum.plot_spectrum")
def test_analyze_valid_input_scalar(mock_plot_spectrum):
    """Tests the valid input scalar keys."""
    mock_plot_spectrum.return_value = (None, None, None)
    mock_plot_spectrum.side_effect = lambda *args, **kwargs: (kwargs, None, None)
    
    test_vals = {
        "Hz": 1,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
    }
    for key, val in test_vals.items():
        (kwargs, _, _) = spectrum.analyze([0], 1, fscale=key)
        assert kwargs.get("fscale") == (key, val)


@mock.patch("adc_eval.eval.calc.sndr_sfdr")
@mock.patch("adc_eval.eval.calc.find_harmonics")
def test_analyze_no_plot(mock_sndr_sfdr, mock_find_harmonics):
    """Tests the psd output of the analyze function with no plotting."""
    data = np.random.rand(NLEN)
    data_sndr = {
        "sig": {"bin": 1, "power": 2},
    }
    data_harms = {"harmonics": 3}
    exp_stats = {**data_sndr, **data_harms}
    
    mock_sndr_sfdr.return_value = data_sndr
    mock_find_harmonics = data_harms
    
    (freq, psd, stats) = spectrum.analyze(
        data,
        fs=1,
        nfft=NFFT,
        dr=1,
        harmonics=0,
        leak=0,
        window="rectangular",
        no_plot=True,
        yaxis="power",
        single_sided=True,
        fscale="Hz",
    )
    
    assert freq.all() == np.linspace(0, 1, int(NFFT/2)).all()
    assert psd.size == int(NFFT/2)
    
    for key, value in stats.items():
        assert value == exp_stats[key]


@mock.patch("adc_eval.eval.calc.sndr_sfdr")
@mock.patch("adc_eval.eval.calc.find_harmonics")
def test_analyze_no_plot_dual(mock_sndr_sfdr, mock_find_harmonics):
    """Tests the psd output of the analyze function with no plotting."""
    data = np.random.rand(NLEN)
    data_sndr = {
        "sig": {"bin": 1, "power": 2},
    }
    data_harms = {"harmonics": 3}
    exp_stats = {**data_sndr, **data_harms}
    
    mock_sndr_sfdr.return_value = data_sndr
    mock_find_harmonics = data_harms
    
    (freq, psd, stats) = spectrum.analyze(
        data,
        fs=1,
        nfft=NFFT,
        dr=1,
        harmonics=0,
        leak=0,
        window="rectangular",
        no_plot=True,
        yaxis="power",
        single_sided=False,
        fscale="Hz",
    )
    
    assert freq.all() == np.linspace(-0.5, 0.5, NFFT-1).all()
    assert psd.size == NFFT
    for key, value in stats.items():
        assert value == exp_stats[key]


@mock.patch("adc_eval.eval.calc.sndr_sfdr")
@mock.patch("adc_eval.eval.calc.find_harmonics")
def test_analyze_no_plot_magnitude(mock_sndr_sfdr, mock_find_harmonics):
    """Tests the psd output of the analyze function with no plotting."""
    data = np.random.rand(NLEN)
    data_sndr = {
        "sig": {"bin": 1, "power": 2},
    }
    data_harms = {"harmonics": 3}
    exp_stats = {**data_sndr, **data_harms}
    
    mock_sndr_sfdr.return_value = data_sndr
    mock_find_harmonics = data_harms
    
    (freq, psd, stats) = spectrum.analyze(
        data,
        fs=1,
        nfft=NFFT,
        dr=1,
        harmonics=0,
        leak=0,
        window="rectangular",
        no_plot=True,
        yaxis="magnitude",
        single_sided=True,
        fscale="Hz",
    )
    
    assert freq.all() == np.linspace(0, 1, int(NFFT/2)).all()
    assert psd.size == int(NFFT/2)
    for key, value in stats.items():
        assert value == exp_stats[key]