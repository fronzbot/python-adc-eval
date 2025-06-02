"""Implements filters and decimation."""

import numpy as np
from scipy.signal import remez, freqz
import matplotlib.pyplot as plt
from adc_eval import signals
from adc_eval.eval import spectrum
from adc_eval.eval import calc


class CICDecimate:
    """
    Generic CIC Decimator Object.

    Parameters
    ----------
    dec : int, default=2
        Output decimation factor.
    order : int, default=1
        Filter order.
    fs : int or float, default=1
        Sample rate for the filter in Hz.


    Attributes
    ----------
    gain : Gain normalization factor of CIC filter
    out : Filtered and decimated output data


    Methods
    -------
    run
    response

    """

    def __init__(self, dec=2, order=1, fs=1):
        """Initialize the CIC filter."""
        self._dec = dec
        self._order = order
        self.fs = fs
        self.gain = dec**order
        self._xout = None
        self._xfilt = None

    @property
    def dec(self):
        """Returns the decimation factor."""
        return self._dec

    @dec.setter
    def dec(self, value):
        """Sets the decimation factor."""
        self._dec = value
        self.gain = value**self._order

    @property
    def order(self):
        """Returns the order of the filter."""
        return self._order

    @order.setter
    def order(self, value):
        """Sets the filter order."""
        self._order = value
        self.gain = self.dec**value

    @property
    def out(self):
        """Filtered and decimated output data."""
        return np.array(self._xout)

    def filt(self, xarray):
        """CIC filtering routine."""
        yint = xarray

        # Integrate first
        for _ in range(self.order):
            yint = np.cumsum(yint)

        # Then comb, adding delays based on decimation factor
        xcomb = yint
        xcomb = np.insert(xcomb, 0, [0 for x in range(self.dec)])
        ycomb = xcomb
        for _ in range(self.order):
            ycomb = ycomb[self.dec :] - ycomb[0 : -self.dec]
            xcomb = ycomb
            xcomb = np.insert(xcomb, 0, [0 for x in range(self.dec)])
            ycomb = xcomb

        self._xfilt = ycomb / self.gain

    def decimate(self, xarray=None):
        """decimation routine."""
        if xarray is None:
            xarray = self._xfilt
        self._xout = xarray[:: self.dec]

    def run(self, xarray):
        """Runs filtering and decimation on input list."""
        self.filt(xarray)
        self.decimate()

    def response(self, fft, no_plot=False):
        """Plots the frequency response of the pre-decimated filter."""
        xin = signals.impulse(fft)
        self.filt(xin)
        (freq, psd, stats) = spectrum.analyze(
            self._xfilt * fft / np.sqrt(2),
            fft,
            fs=self.fs,
            dr=1,
            harmonics=0,
            leak=1,
            window="rectangular",
            no_plot=no_plot,
            yaxis="power",
            single_sided=True,
        )
        if not no_plot:
            ax = plt.gca()
            n1 = 0
            n2 = int(fft / (2 * self.dec))
            x = freq[n1:n2] / 1e6
            y1 = psd[n1:n2] - max(psd)
            y2 = -2000 * np.ones(np.size(x))
            ax.plot(x, y2, alpha=0)
            ax.plot(x, y1, alpha=0)
            ax.fill_between(x, y1, y2, color="green", alpha=0.1)
            ax.set_xticks(np.linspace(0, self.fs / 2e6, 9))
        return (freq, psd)


class FIRLowPass:
    """
    Generic FIR Low Pass Filter.

    Parameters
    ----------
    dec : int, optional
        Output decimation rate. The default is 1.
    fs : int or float, optional
        Sample rate for the filter in Hz. The default is 1Hz.
    bit_depth : int, optional
        Bit depth to store coefficients. The default is 16b.
    coeffs : list, optional
        List of coefficients if pre-determined.


    Attributes
    ----------
    out : Filtered and decimated output data.
    ntaps : Number of filter taps.


    Methods
    -------
    run
    response

    """

    def __init__(self, dec=1, fs=1, bit_depth=16, coeffs=None):
        """Initialize the FIR LowPass Class."""
        self.coeffs = coeffs
        self.dec = dec
        self.fs = fs
        self.bit_depth = bit_depth
        self.ntaps = np.size(coeffs) if coeffs is not None else 0
        self.yfilt = None
        self._out = None

    @property
    def out(self):
        """Filtered and decimated output datat."""
        return np.array(self._out)

    def generate_taps(self, fbw, pbripple=1, stopatt=-60, deltaf=None):
        """
        Generates FIR taps from key inputs.

        Parameters
        ----------
        fbw : float
            Bandwidth of the filter in Hz.
        pbripple : float, optional
            Acceptable passband ripple in percentage. The default is 1%.
        stopatt : float, optional
            Desired stop-band attenuation. The default is -60dB.
        deltaf : float, optional
            Desired transition band of the filter. The default is FS/100 if set to None.

        Returns
        -------
        (ntaps, coeffs) : tuple
            Minimum number of taps required, List of FIR tap coefficients.

        """
        if deltaf is None:
            deltaf = self.fs / 100
        x1 = pbripple / 100
        x2 = 10 ** (stopatt / 20)
        x3 = np.log10(1 / (10 * x1 * x2))
        _ntaps = 2 / 3 * x3 * self.fs / deltaf

        if self.ntaps > 0 and _ntaps > self.ntaps:
            print(
                f"WARNING: Required NTAPs calculated as {int(_ntaps)} but only {self.ntaps} were provided."
            )
        elif self.ntaps == 0:
            self.ntaps = int(_ntaps)

        _coeffs = remez(
            self.ntaps, [0, fbw, fbw + deltaf, self.fs / 2], [1, 0], fs=self.fs
        )
        self.coeffs = np.int32(_coeffs * 2**self.bit_depth).tolist()
        return (self.ntaps, self.coeffs)

    def filt(self, xarray):
        """Performs FIR filtering on input xarray."""
        _coeffs = np.array(self.coeffs) / 2**self.bit_depth
        self.yfilt = np.convolve(_coeffs, xarray, mode="same")

    def decimate(self, xarray=None):
        """Performs decimation on the input data."""
        if xarray is None:
            xarray = self.yfilt
        self._out = xarray[:: self.dec]

    def run(self, xarray):
        """Runs FIR filtering and decimation on input xarray data."""
        self.filt(xarray)
        self.decimate()

    def response(self, fft, no_plot=False):
        """Plots the frequency response of the pre-decimated filter."""
        freq, mag = freqz(self.coeffs, [1], worN=fft, fs=self.fs)
        yfft = calc.dBW(np.abs(mag))
        if not no_plot:
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(freq / 1e6, yfft)
            ax.grid(True)
            ax.set_ylabel("Filter Magnitude Response (dB)", fontsize=18)
            ax.set_xlabel("Frequency (MHz)", fontsize=16)
            ax.set_title("FIR Low Pass Response", fontsize=16)
            ax.set_xlim([0, self.fs / 2e6])
            ax.set_ylim([1.1 * min(yfft), 1])
            plt.show()
        return (freq, yfft)
