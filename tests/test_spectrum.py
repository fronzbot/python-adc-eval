"""Test the spectrum module."""

import unittest
import numpy as np
from unittest import mock
from adc_eval import spectrum


class TestSpectrum(unittest.TestCase):
    """Test the spectrum module."""

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

    def test_calc_psd_two_sided(self):
        """Test calc_psd with dummy input."""
        sq2 = np.sqrt(2) / 4
        data = np.array([0, sq2, 0.5, sq2, 0, -sq2, -0.5, -sq2])
        exp_psd = np.array([0, 0.5, 0, 0, 0, 0, 0, 0.5])
        exp_freq = np.array([i / (len(data) - 1) for i in range(0, len(data))])
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=8, single_sided=False)

        for index in range(0, len(psd)):
            self.assertEqual(round(psd[index], 5), round(exp_psd[index], 5))
            self.assertEqual(round(freq[index], 5), round(exp_freq[index], 5))

    def test_calc_psd_one_sided(self):
        """Test calc_psd with dummy input."""
        sq2 = np.sqrt(2) / 4
        data = np.array([0, sq2, 0.5, sq2, 0, -sq2, -0.5, -sq2])
        exp_psd = 2 * np.array([0, 0.5, 0, 0])
        exp_freq = np.array([i / (len(data) - 1) for i in range(0, len(data))])
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=8, single_sided=True)

        for index in range(0, len(psd)):
            self.assertEqual(round(psd[index], 5), round(exp_psd[index], 5))
            self.assertEqual(round(freq[index], 5), round(exp_freq[index], 5))

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
