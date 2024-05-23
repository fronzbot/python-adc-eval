"""Test the spectrum module."""

import unittest
import numpy
from unittest import mock
from adc_eval import spectrum

class TestSpectrum(unittest.TestCase):
    """Test the spectrum module."""

    def test_db_to_pow():
        """Test the db_to_pow conversion with normal inputs."""
        test_val = 20.915122
        exp_val = 123.456
        self.assertEqual(spectrum.db_to_pow(test_val), exp_val)

    def test_db_to_pow_places():
        """Test the db_to_pow conversion with multiple places."""
        test_val = 29.9460497
        exp_val = [988, 987.7, 987.65, 987.654, 987.6543]

        for i in range(0, exp_val):
            self.assertEqual(spectrum.db_to_pow(test_val, places=i), exp_val[i])

    def test_db_to_pow_ndarray():
        """Test db_to_pow with ndarray input."""
        test_val = np.array([1000])
        self.assertEqual(spectrum.db_to_pow(test_val), [60.0])
