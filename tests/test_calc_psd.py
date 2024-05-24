"""Test the calc_psd method."""

import unittest
import numpy as np
from unittest import mock
from adc_eval import spectrum


class TestCalcPSD(unittest.TestCase):
    """Test the calc_psd method."""

    def setUp(self):
        """Initialize tests."""
        self.nfft = 2**8
        self.nlen = 2**18
        accuracy = 0.01
        self.bounds = [1 - accuracy, 1 + accuracy]
        np.random.seed(1)

    def test_calc_psd_randomized_dual(self):
        """Test calc_psd with random data."""
        for i in range(0, 10):
            data = np.random.randn(self.nlen)
            (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=False)
            mean_val = np.mean(psd)
            self.assertTrue(self.bounds[0] <= mean_val <= self.bounds[1], msg=mean_val)

    def test_calc_psd_randomized_single(self):
        """Test calc_psd with random data and single-sided."""
        for i in range(0, 10):
            data = np.random.randn(self.nlen)
            (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=True)
            mean_val = np.mean(psd)
            self.assertTrue(
                2 * self.bounds[0] <= mean_val <= 2 * self.bounds[1], msg=mean_val
            )

    def test_calc_psd_zeros_dual(self):
        """Test calc_psd with zeros."""
        data = np.zeros(self.nlen)
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=False)
        mean_val = np.mean(psd)
        self.assertTrue(
            self.bounds[0] - 1 <= mean_val <= self.bounds[1] - 1, msg=mean_val
        )

    def test_calc_psd_zeros_single(self):
        """Test calc_psd with zeros and single-sided.."""
        data = np.zeros(self.nlen)
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=True)
        mean_val = np.mean(psd)
        self.assertTrue(
            self.bounds[0] - 1 <= mean_val <= self.bounds[1] - 1, msg=mean_val
        )

    def test_calc_psd_ones_dual(self):
        """Test calc_psd with ones."""
        data = np.ones(self.nlen)
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=False)
        mean_val = np.mean(psd)
        self.assertTrue(self.bounds[0] <= mean_val <= self.bounds[1], msg=mean_val)

    def test_calc_psd_ones_single(self):
        """Test calc_psd with ones and single-sided."""
        data = np.ones(self.nlen)
        (freq, psd) = spectrum.calc_psd(data, 1, nfft=self.nfft, single_sided=True)
        mean_val = np.mean(psd)
        self.assertTrue(
            2 * self.bounds[0] <= mean_val <= 2 * self.bounds[1], msg=mean_val
        )

    def test_calc_psd_two_sine_dual(self):
        """Test calc_psd with two sine waves."""
        fs = 1
        fbin = fs / self.nfft
        f1 = 29 * fbin
        f2 = 97 * fbin
        a1 = 0.37
        a2 = 0.11
        t = 1 / fs * np.linspace(0, self.nlen - 1, self.nlen)
        data = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
        (freq, psd) = spectrum.calc_psd(data, fs, nfft=self.nfft, single_sided=False)
        exp_peaks = [
            round(a1**2 / 4 * self.nfft, 3),
            round(a2**2 / 4 * self.nfft, 3),
        ]
        exp_f1 = [round(f1, 2), round(fs - f1, 2)]
        exp_f2 = [round(f2, 2), round(fs - f2, 2)]

        peak1 = max(psd)
        ipeaks = np.where(psd >= peak1 * self.bounds[0])[0]
        fpeaks = [round(freq[ipeaks[0]], 2), round(freq[ipeaks[1]], 2)]

        self.assertEqual(round(peak1, 3), exp_peaks[0])
        self.assertListEqual(fpeaks, exp_f1)

        psd[ipeaks[0]] = 0
        psd[ipeaks[1]] = 0

        peak2 = max(psd)
        ipeaks = np.where(psd >= peak2 * self.bounds[0])[0]
        fpeaks = [round(freq[ipeaks[0]], 2), round(freq[ipeaks[1]], 2)]

        self.assertEqual(round(peak2, 3), exp_peaks[1])
        self.assertListEqual(fpeaks, exp_f2)

    def test_calc_psd_two_sine_single(self):
        """Test calc_psd with two sine waves, single-eided."""
        fs = 1
        fbin = fs / self.nfft
        f1 = 29 * fbin
        f2 = 97 * fbin
        a1 = 0.37
        a2 = 0.11
        t = 1 / fs * np.linspace(0, self.nlen - 1, self.nlen)
        data = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
        (freq, psd) = spectrum.calc_psd(data, fs, nfft=self.nfft, single_sided=True)
        exp_peaks = [
            round(a1**2 / 2 * self.nfft, 3),
            round(a2**2 / 2 * self.nfft, 3),
        ]
        exp_f1 = round(f1, 2)
        exp_f2 = round(f2, 2)

        peak1 = max(psd)
        ipeak = np.where(psd == peak1)[0][0]
        fpeak = round(freq[ipeak], 2)

        self.assertEqual(round(peak1, 3), exp_peaks[0])
        self.assertEqual(fpeak, exp_f1)

        psd[ipeak] = 0

        peak2 = max(psd)
        ipeak = np.where(psd == peak2)[0][0]
        fpeak = round(freq[ipeak], 2)

        self.assertEqual(round(peak2, 3), exp_peaks[1])
        self.assertEqual(fpeak, exp_f2)
