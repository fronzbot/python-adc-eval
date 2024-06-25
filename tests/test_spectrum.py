"""Test the spectrum module."""

import unittest
import numpy as np
from unittest import mock
from adc_eval import spectrum


class TestSpectrum(unittest.TestCase):
    """Test the spectrum module."""

    def setUp(self):
        """Initialize tests."""
        pass

    def test_db_to_pow_places(self):
        """Test the db_to_pow conversion with multiple places."""
        test_val = 29.9460497
        exp_val = [988, 987.7, 987.65, 987.654, 987.6543]

        for i in range(0, len(exp_val)):
            self.assertEqual(spectrum.db_to_pow(test_val, places=i), exp_val[i])

    def test_db_to_pow_ndarray(self):
        """Test db_to_pow with ndarray input."""
        test_val = np.array([30.0])
        self.assertEqual(spectrum.db_to_pow(test_val), np.array([1000.0]))

    def test_dbW(self):
        """Test the dbW conversion with normal inputs."""
        test_val = 9.7197255
        exp_val = [10, 9.9, 9.88, 9.877, 9.8765]

        for i in range(0, len(exp_val)):
            self.assertEqual(spectrum.dBW(test_val, places=i), exp_val[i])

    def test_dbW_ndarray(self):
        """Test dbW with ndarray input."""
        test_val = np.array([100.0])
        self.assertEqual(spectrum.dBW(test_val), np.array([20.0]))

    def test_enob(self):
        """Test enob with muliple places."""
        test_val = 60.123456
        exp_val = [10, 9.7, 9.69, 9.695, 9.6949]

        for i in range(0, len(exp_val)):
            self.assertEqual(spectrum.enob(test_val, places=i), exp_val[i])

    @mock.patch("adc_eval.spectrum.calc_psd")
    def test_get_spectrum(self, mock_calc_psd):
        """Test that the get_spectrum method returns power spectrum."""
        fs = 4
        nfft = 3
        data = np.array([1])
        exp_spectrum = np.array([fs / nfft])

        mock_calc_psd.return_value = (None, data)

        self.assertEqual(
            spectrum.get_spectrum(None, fs=fs, nfft=nfft), (None, exp_spectrum)
        )

    def test_sndr_sfdr_outputs(self):
        """Test the sndr_sfdr method outputs."""
        data = np.array([1, 2, 91, 7])
        freq = np.array([100, 200, 300, 400])
        full_scale = -3
        nfft = 2**8
        fs = 1
        exp_return = {
            "sig": {
                "freq": 300,
                "bin": 2,
                "power": 91,
                "dB": 19.6,
                "dBFS": round(19.590 - full_scale, 1),
            },
            "spur": {
                "freq": 400,
                "bin": 3,
                "power": 7,
                "dB": 8.5,
                "dBFS": round(8.451 - full_scale, 1),
            },
            "noise": {
                "floor": 18 / nfft,
                "power": 9,
                "rms": 3,
                "dBHz": round(-11.529675 - full_scale, 1),
                "NSD": round(36.6351 - full_scale, 1),
            },
            "sndr": {
                "dBc": 10.0,
                "dBFS": round(full_scale - 9.542, 1),
            },
            "sfdr": {
                "dBc": 11.1,
                "dBFS": round(full_scale - 8.451, 1),
            },
            "enob": {
                "bits": round((full_scale - 11.3024) / 6.02, 1),
            },
        }

        result = spectrum.sndr_sfdr(data, freq, fs, nfft, 0, full_scale=full_scale)
        for key, val in exp_return.items():
            self.assertDictEqual(result[key], val, msg=key)
