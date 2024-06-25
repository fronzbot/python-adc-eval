"""Test the spectrum plotting functions."""

import unittest
import numpy as np
from unittest import mock
from adc_eval import spectrum


class TestSpectrumPlotting(unittest.TestCase):
    """Test the spectrum module."""

    def setUp(self):
        """Initialize tests."""
        self.nlen = 2**16
        self.nfft = 2**12
        self.fs = 1
        self.bin = 13
        self.arms = 0.5 / np.sqrt(2)
        self.fin = self.fs / self.nfft * self.bin

    def gen_spectrum(self, harmonics):
        """Generate a wave with arbitrary harmonics."""
        t = 1 / self.fs * np.linspace(0, self.nlen - 1, self.nlen)
        vin = np.zeros(len(t))
        for i in range(1, harmonics + 1):
            vin += np.sqrt(2) * self.arms / i * np.sin(2 * np.pi * i * self.fin * t)

        return spectrum.get_spectrum(vin, fs=self.fs, nfft=self.nfft)

    def test_find_harmonics(self):
        """Test the find harmonics method."""
        for i in range(2, 10):
            (freq, pwr) = self.gen_spectrum(10)

            stats = spectrum.find_harmonics(
                pwr, freq, self.nfft, self.bin, self.arms, harms=i, leak=0
            )

            for x in range(2, i + 1):
                msg_txt = f"harm={i}, index={x}"
                self.assertEqual(stats["harm"][x]["bin"], x * self.bin, msg=msg_txt)
                self.assertEqual(
                    round(stats["harm"][x]["power"], 4),
                    round((self.arms / x) ** 2, 4),
                    msg=msg_txt,
                )
                self.assertEqual(
                    stats["harm"][x]["freq"],
                    round(freq[x * self.bin] / 1e6, 1),
                    msg=msg_txt,
                )

    def test_find_harmonics_with_leakage(self):
        """Test the find harmonics method with spectral leakage."""
        self.bin = 13.5
        leakage_bins = 5
        for i in range(2, 10):
            (freq, pwr) = self.gen_spectrum(10)

            stats = spectrum.find_harmonics(
                pwr, freq, self.nfft, self.bin, self.arms, harms=i, leak=leakage_bins
            )

            for x in range(2, i + 1):
                msg_txt = f"harm={i}, index={x}"
                self.assertTrue(
                    x * self.bin - leakage_bins
                    <= stats["harm"][x]["bin"]
                    <= x * self.bin + leakage_bins,
                    msg=msg_txt,
                )

    def test_find_harmonics_with_leakage_outside_bounds(self):
        """Test find harmonics with leakage bins exceeding array bounds."""
        self.bin = self.nfft / 4 - 0.5
        (freq, pwr) = self.gen_spectrum(5)
        leakage_bins = 2
        stats = spectrum.find_harmonics(
            pwr, freq, self.nfft, self.bin, self.arms, harms=2, leak=leakage_bins
        )
        self.assertTrue(self.nfft / 2 - 3 <= stats["harm"][2]["bin"], self.nfft / 2 - 1)

    def test_find_harmonics_on_fft_bound(self):
        """Test find harmonics with harmonics landing at nfft/2."""
        self.nfft = 2**12
        self.bin = self.nfft / 8
        (freq, pwr) = self.gen_spectrum(10)
        leakage_bins = 0
        stats = spectrum.find_harmonics(
            pwr, freq, self.nfft, self.bin, self.arms, harms=5, leak=leakage_bins
        )
        self.assertEqual(stats["harm"][2]["bin"], 2 * self.bin)
        self.assertEqual(stats["harm"][3]["bin"], 3 * self.bin)
        self.assertEqual(stats["harm"][4]["bin"], 0)
        self.assertEqual(stats["harm"][5]["bin"], self.nfft - 5 * self.bin)

    def test_plot_string(self):
        """Test proper return of plotting string."""
        self.bin = 13
        (freq, pwr) = self.gen_spectrum(3)
        stats = spectrum.sndr_sfdr(pwr, freq, 1, self.nfft, leak=0, full_scale=0)
        harms = spectrum.find_harmonics(
            pwr, freq, self.nfft, self.bin, self.arms, harms=3, leak=0
        )
        all_stats = {**stats, **harms}

        plt_str = spectrum.get_plot_string(
            all_stats, 0, self.fs, self.nfft, window="rectangular"
        )

        # Check for important information, not everything
        msg_txt = f"{all_stats}\n{plt_str}"
        self.assertTrue(f"NFFT = {self.nfft}" in plt_str, msg=msg_txt)
        self.assertTrue(
            f"ENOB = {all_stats['enob']['bits']} bits" in plt_str, msg=msg_txt
        )
        self.assertTrue(
            f"SNDR = {all_stats['sndr']['dBFS']} dBFS" in plt_str, msg=msg_txt
        )
        self.assertTrue(
            f"SFDR = {all_stats['sfdr']['dBFS']} dBFS" in plt_str, msg=msg_txt
        )
        self.assertTrue(
            f"Noise Floor = {all_stats['noise']['dBHz']} dBFS" in plt_str,
            msg=msg_txt,
        )
        self.assertTrue(
            f"HD2 = {round(all_stats['harm'][2]['dB'], 1)} dBFS @ {all_stats['harm'][2]['freq']} MHz"
            in plt_str,
            msg=msg_txt,
        )
        self.assertTrue(
            f"HD3 = {round(all_stats['harm'][3]['dB'], 1)} dBFS @ {all_stats['harm'][3]['freq']} MHz"
            in plt_str,
            msg=msg_txt,
        )
