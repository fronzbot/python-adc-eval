"""Generic simulator class for adc evaluation."""

import numpy as np
from tqdm import tqdm


class Simulator:
    """
    Class for handling simulation functions.

    Parameters
    ----------
    adc_obj : ADC.__class__
        An ADC object from the adc_eval.eval.adc class list.
    xarray : ndarray
        Input signal array to simulate the adc_obj with.


    Attributes
    ----------
    out : ndarray of ADC output values.
    adc : Reference to the input adc_obj.
    vin : xarray with global signal errors included as set by adj_obj.

    Methods
    -------
    run

    """

    def __init__(self, adc_obj, xarray):
        """Initialize the simulator class."""
        self.dval = []
        self.adc = adc_obj
        self.vin = self.calc_error(xarray)

    @property
    def out(self):
        """Return output value array."""
        return np.array(self.dval)

    def calc_error(self, vin):
        """Using the adc_obj, calculates global signal error before simulation."""
        vinx = vin

        # First calculate gain error
        vinx *= (1 + self.adc.err["gain"]) * self.adc.err["dist"][0]

        # Next calculate the harmonic distortion
        for index, gain in enumerate(self.adc.err["dist"]):
            if index > 0:
                vinx += gain * vin ** (index + 1)

        # Now add the offset
        vinx += self.adc.err["offset"]

        # Now add random noise
        vinx += self.adc.err["noise"] * np.random.randn(vin.size)

        return vinx

    def run(self):
        with tqdm(
            range(len(self.vin)), "RUNNING", unit=" samples", position=0, leave=True
        ) as pbar:
            for xval in self.vin:
                self.adc.vin = xval
                self.adc.run_step()
                self.dval.append(self.adc.dout)
                pbar.update()
