"""Test the filter module."""

import pytest
import numpy as np
from unittest import mock
from adc_eval import filt
from adc_eval import signals


@pytest.mark.parametrize("dec", np.random.randint(1, 20, 4))
def test_cic_decimate_set_dec_updates_gain(dec):
    """Tests that changing decimation factor updates gain."""
    cicfilt = filt.CICDecimate(dec=1, order=2)
    assert cicfilt.gain == 1

    cicfilt.dec = dec
    assert cicfilt.gain == (dec**2)


@pytest.mark.parametrize("order", np.random.randint(1, 20, 4))
def test_cic_decimate_set_order_updates_gain(order):
    """Tests that changing filter order updates gain."""
    cicfilt = filt.CICDecimate(dec=2, order=1)
    assert cicfilt.gain == 2

    cicfilt.order = order
    assert cicfilt.gain == (2**order)


def test_cic_decimate_returns_ndarray():
    """Tests the CICDecimate output data conversion."""
    cicfilt = filt.CICDecimate()
    data = np.random.randn(100).tolist()
    cicfilt._xout = data

    assert type(cicfilt.out) == type(np.array(list()))
    assert cicfilt.out.all() == np.array(data).all()


@pytest.mark.parametrize("dec", np.random.randint(1, 20, 4))
def test_cic_decimate_function(dec):
    """Tests the CICDecimate decimate function."""
    cicfilt = filt.CICDecimate(dec=dec)
    data = np.random.randn(100)
    cicfilt.decimate(data)

    exp_result = data[::dec]

    assert cicfilt.out.size == exp_result.size
    assert cicfilt.out.all() == exp_result.all()


def test_cic_decimate_function_none_input():
    """Tests the CICDecimate decimate function with no input arg."""
    cicfilt = filt.CICDecimate(dec=1)
    data = np.random.randn(100)
    cicfilt._xfilt = data
    cicfilt.decimate()

    exp_result = data

    assert cicfilt.out.size == exp_result.size
    assert cicfilt.out.all() == exp_result.all()


@pytest.mark.parametrize("nlen", np.random.randint(8, 2**10, 4))
def test_cic_decimate_all_ones(nlen):
    """Test the CICDecimate filtering with all ones."""
    cicfilt = filt.CICDecimate(dec=1, order=1)
    data = np.ones(nlen)
    cicfilt.run(data)

    exp_data = data.copy()
    exp_data[0] = 0

    assert cicfilt.out.all() == exp_data.all()


@pytest.mark.parametrize("nlen", np.random.randint(8, 2**10, 4))
def test_cic_decimate_all_zeros(nlen):
    """Test the CICDecimate filtering with all zeros."""
    cicfilt = filt.CICDecimate(dec=1, order=1)
    data = np.zeros(nlen)
    cicfilt.run(data)

    exp_data = data.copy()

    assert cicfilt.out.all() == exp_data.all()


@pytest.mark.parametrize("nlen", np.random.randint(8, 2**10, 4))
def test_cic_decimate_impulse(nlen):
    """Test the CICDecimate filtering with impulse."""
    cicfilt = filt.CICDecimate(dec=1, order=1)
    data = signals.impulse(nlen)
    cicfilt.run(data)

    exp_data = np.concatenate([[0], data[0:-1]])

    assert cicfilt.out.all() == exp_data.all()


def test_fir_lowpass_returns_ndarray():
    """Tests the FIRLowPass output data conversion."""
    fir = filt.FIRLowPass()
    data = np.random.randn(100).tolist()
    fir._out = data

    assert type(fir.out) == type(np.array(list()))
    assert fir.out.all() == np.array(data).all()


@pytest.mark.parametrize("dec", np.random.randint(1, 20, 4))
def test_fir_decimate_function(dec):
    """Tests the FIRLowPass decimate function."""
    fir = filt.FIRLowPass(dec=dec)
    data = np.random.randn(100)
    fir.decimate(data)

    exp_result = data[::dec]

    assert fir.out.size == exp_result.size
    assert fir.out.all() == exp_result.all()


def test_fir_decimate_function_none_input():
    """Tests the FIRLowPass decimate function with no input arg."""
    fir = filt.FIRLowPass(dec=1)
    data = np.random.randn(100)
    fir.yfilt = data
    fir.decimate()

    exp_result = data

    assert fir.out.size == exp_result.size
    assert fir.out.all() == exp_result.all()


@mock.patch("adc_eval.filt.remez")
def test_fir_lowpass_tap_generation(mock_remez, capfd):
    """Tests the FIRLowPass decimate function."""
    fir = filt.FIRLowPass()
    fir.ntaps = 3
    fir.bit_depth = 12
    mock_remez.return_value = np.ones(3)

    (taps, coeffs) = fir.generate_taps(0.1)

    captured = capfd.readouterr()
    exp_coeffs = [2**12, 2**12, 2**12]

    assert "WARNING" in captured.out
    assert taps == 3
    assert coeffs == exp_coeffs


@pytest.mark.parametrize("ntaps", np.random.randint(3, 511, 5))
def test_fir_lowpass_run(ntaps):
    """Tests the FIRLowPass run function."""
    fir = filt.FIRLowPass()
    fir.ntaps = ntaps
    fir.bit_depth = 10
    fir.coeffs = 2**10 * np.ones(ntaps)
    data = signals.impulse(2**12)
    fir.run(data)

    exp_sum = np.ceil((ntaps + 1) / 2)
    out_sum = sum(fir.out)

    tap_val = int(exp_sum)

    assert fir.out.size == data.size
    assert max(fir.out) == 1
    assert min(fir.out) == 0
    assert fir.out[0:tap_val].all() == 1
    assert fir.out[tap_val + 1 :].all() == 0
    assert out_sum == exp_sum
