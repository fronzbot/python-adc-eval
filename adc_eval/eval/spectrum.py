"""Spectral analysis module."""

import numpy as np
import matplotlib.pyplot as plt
from adc_eval.eval import calc


def calc_psd(data, fs=1, nfft=2**12):
    """
    Calculate the PSD using the Bartlett method.

    Parameters
    ----------
    data : ndarray
        Time-series input data.
    fs : float, optional
        Sample frequency of the input time series data in Hz. Default is 1Hz.
    nfft : int, optional
        Number of FFT samples to use for PSD calculation. Default is 2^12.

    Returns
    -------
    list
        [freq_ss, psd_ss, freq_ds, psd_ds]
        List containing single and double-sided PSDs along with frequncy array.
    """
    nwindows = max(1, int(np.floor(len(data) / nfft)))
    nfft = int(nfft)
    xs = data[0 : int(nwindows * nfft)]
    xt = xs.reshape(nwindows, nfft).T
    XF = abs(np.fft.fft(xt, nfft, axis=0) / nfft) ** 2
    psd = np.mean(XF, axis=1) / (fs / nfft)  # average the ffts and divide by bin width
    psd += np.finfo(float).eps  # Prevents zeros in the PSD
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / fs))

    # For single sided we double all the bins, then we halve the DC bin
    psd_ss = 2 * psd[0 : int(nfft / 2)]
    psd_ss[0] /= 2
    freq_ss = freq[int(nfft / 2) :]

    # Need to rotate DS PSD so 0Hz is in middle of graph
    freq_ds = freq
    psd_ds = np.concatenate([psd[int(nfft / 2) :], psd[0 : int(nfft / 2)]])

    return [freq_ss, psd_ss, freq_ds, psd_ds]


def get_spectrum(data, fs=1, nfft=2**12, single_sided=True):
    """
    Get the power spectrum for an input signal.

    Parameters
    ----------
    data : ndarray
        Time-series input data.
    fs : float, optional
        Sample frequency of the input time series data in Hz. Default is 1Hz.
    nfft : int, optional
        Number of FFT samples to use for PSD calculation. Default is 2^12.
    single_sided : bool, optional
        Set to `True` for single-sided spectrum or `False` for double-sided.
        Default is `True`.

    Returns
    -------
    tuple
        (freq, psd)
        Tuple containing frequency array and PSD of input data.
    """
    (freq_ss, psd_ss, freq_ds, psd_ds) = calc_psd(np.array(data), fs=fs, nfft=nfft)
    if single_sided:
        return (freq_ss, psd_ss * fs / nfft)
    return (freq_ds, psd_ds * fs / nfft)


def window_data(data, window="rectangular"):
    """
    Applies a window to the time-domain data.

    Parameters
    ----------
    data : ndarray
        Time-series input data.
    window : str, optional
        Window to use for input data. Default is rectangular.

    Returns
    -------
    ndarray
        Windowed version of input data.
    """
    try:
        wsize = data.size
    except AttributeError:
        data = np.array(data)
        wsize = data.size

    windows = {
        "rectangular": (np.ones(wsize), 1.0),
        "hanning": (np.hanning(wsize), 1.633),
    }

    if window not in windows:
        print(f"WARNING: {window} not implemented. Defaulting to 'rectangular'.")
        window = "rectangular"

    wscale = windows[window][1]

    return data * windows[window][0] * wscale


def plot_spectrum(
    data,
    fs=1,
    nfft=2**12,
    dr=1,
    harmonics=7,
    leak=1,
    window="rectangular",
    no_plot=False,
    yaxis="power",
    single_sided=True,
    fscale=("MHz", 1e6),
):
    """
    Plot Power Spectrum for input signal.

    Parameters
    ----------
    data : ndarray
        Time-series input data.
    fs : float, optional
        Sample frequency of the input time series data in Hz. Default is 1Hz.
    nfft : int, optional
        Number of FFT samples to use for PSD calculation. Default is 2^12.
    dr : float, optional
        Dynamic range for input data to be referenced to. Default is 1.
    harmonics : int, optional
        Number of harmonics to calculate and annotate on plot. Default is 7.
    leak : int, optional
        Number of leakage bins to use in signal and harmonic calculation. Default is 1.
    window : str, optional
        Type of input window to use for input data. Default is rectangular.
    no_plot : bool, optional
        Selects whether to plot (`False`) or not (`True`). Default is `False`.
    yaxis : str, optional
        Selects y-axis reference units. Example: `power`, `fullscale`, etc. Default is `power`.
    single_sided : bool, optional
        Set to `True` for single-sided spectrum or `False` for double-sided.
        Default is `True`.
    fscale : tuple, optional
        Selects x-axis scaling and units. Default is ('MHz', 1e6).

    Returns
    -------
    tuple
        (freq, psd, stats)
        Tuple containing frequency array, PSD of input data, and calculated statstics dictionary.
    """
    (freq, pwr) = get_spectrum(data, fs=fs, nfft=nfft, single_sided=single_sided)

    # Calculate the fullscale range of the spectrum in Watts
    full_scale = calc.dBW(dr**2 / 8)

    # Determine what y-axis scaling to use
    yaxis_lut = {
        "power": [0, "dB"],
        "fullscale": [full_scale, "dBFS"],
        "normalize": [max(calc.dBW(pwr)), "dB Normalized"],
        "magnitude": [0, "W"],
    }

    lut_key = yaxis.lower()
    scalar = yaxis_lut[lut_key][0]
    yunits = yaxis_lut[lut_key][1]
    xscale = fscale[1]

    # Convert to dBW and perform scalar based on y-axis scaling input
    psd_out = calc.dBW(pwr, places=3) - scalar

    # Use Watts if magnitude y-axis scaling is desired
    if lut_key in ["magnitude"]:
        psd_out = pwr

    # Get single-sided spectrum for consistent SNDR and harmonic calculation behavior
    f_ss = freq
    psd_ss = pwr
    if not single_sided:
        # Get single-sided spectrum for SNDR and Harmonic stats
        (f_ss, psd_ss) = get_spectrum(data, fs=fs, nfft=nfft, single_sided=True)

    sndr_stats = calc.sndr_sfdr(
        psd_ss, f_ss, fs, nfft, leak=leak, full_scale=full_scale
    )
    harm_stats = calc.find_harmonics(
        psd_ss,
        f_ss,
        nfft,
        sndr_stats["sig"]["bin"],
        sndr_stats["sig"]["power"],
        harms=harmonics,
        leak=leak,
        fscale=xscale,
    )

    # Merge the two stat dictionaries into one for convenient access
    stats = {**sndr_stats, **harm_stats}

    # Change the x-axis minimum value based on single or dual-sided selection
    xmin = 0 if single_sided else -fs / 2e6

    # If plotting, prep plot and generate all required axis strings
    if not no_plot:
        plt_str = calc.get_plot_string(
            stats, full_scale, fs, nfft, window, xscale, fscale[0]
        )
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(freq / xscale, psd_out)
        ax.set_ylabel(f"Power Spectrum ({yunits})", fontsize=18)
        ax.set_xlabel(f"Frequency ({fscale[0]})", fontsize=16)
        ax.set_title("Output Power Spectrum", fontsize=16)
        ax.set_xlim([xmin, fs / (2 * xscale)])
        ax.set_ylim([1.1 * min(psd_out), 1])

        ax.annotate(
            plt_str,
            xy=(1, 1),
            xytext=(10, -80),
            xycoords=("axes fraction", "figure fraction"),
            textcoords="offset points",
            size=11,
            ha="left",
            va="top",
        )

        # Get noise floor in dB/Hz (not dBFS/Hz)
        noise_dB = stats["noise"]["dBHz"] + full_scale

        # Add points for harmonics and largest spur
        if not single_sided:
            scalar += 3
        for hindex in range(2, harmonics + 1):
            if stats["harm"][hindex]["dB"] > (noise_dB + 3):
                fharm = stats["harm"][hindex]["freq"]
                aharm = stats["harm"][hindex]["dB"] - scalar
                ax.plot(
                    fharm,
                    aharm,
                    marker="s",
                    mec="r",
                    ms=8,
                    fillstyle="none",
                    mew=3,
                )
                ax.text(
                    fharm,
                    aharm + 3,
                    f"HD{hindex}",
                    ha="center",
                    weight="bold",
                )
                if not single_sided:
                    ax.plot(
                        -fharm,
                        aharm,
                        marker="s",
                        mec="r",
                        ms=8,
                        fillstyle="none",
                        mew=3,
                    )
                    ax.text(
                        -fharm,
                        aharm + 3,
                        f"HD{hindex}",
                        ha="center",
                        weight="bold",
                    )
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.grid()

    return (freq, psd_out, stats)


def analyze(
    data,
    nfft,
    fs=1,
    dr=1,
    harmonics=11,
    leak=5,
    window="rectangular",
    no_plot=False,
    yaxis="fullscale",
    single_sided=True,
    fscale="MHz",
):
    """
    Perform spectral analysis on input waveform.

    Parameters
    ----------
    data : ndarray
        Time-series input data.
    nfft : int
        Number of FFT samples to use for PSD calculation.
    fs : float, optional
        Sample frequency of the input time series data in Hz. Default is 1Hz.
    dr : float, optional
        Dynamic range for input data to be referenced to. Default is 1.
    harmonics : int, optional
        Number of harmonics to calculate and annotate on plot. Default is 7.
    leak : int, optional
        Number of leakage bins to use in signal and harmonic calculation. Default is 1.
    window : str, optional
        Type of input window to use for input data. Default is rectangular.
    no_plot : bool, optional
        Selects whether to plot (`False`) or not (`True`). Default is `False`.
    yaxis : str, optional
        Selects y-axis reference units. Example: `power`, `fullscale`, etc. Default is `power`.
    single_sided : bool, optional
        Set to `True` for single-sided spectrum or `False` for double-sided.
        Default is `True`.
    fscale : str, optional
        Selects x-axis units. Default is 'MHz'.

    Returns
    -------
    tuple
        (freq, psd, stats)
        Tuple containing frequency array, PSD of input data, and calculated statstics dictionary.
    """
    fscalar = {
        "uHz": 1e-6,
        "mHz": 1e-3,
        "Hz": 1,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
        "THz": 1e12,
    }
    if fscale not in fscalar:
        print(f"WARNING: {fscale} not implemented. Defaulting to 'MHz'.")
        fscale = "MHz"

    # Window the data
    wdata = window_data(data, window=window)

    (freq, spectrum, stats) = plot_spectrum(
        wdata,
        fs=fs,
        nfft=nfft,
        dr=dr,
        harmonics=harmonics,
        leak=leak,
        window=window,
        no_plot=no_plot,
        yaxis=yaxis,
        single_sided=single_sided,
        fscale=(fscale, fscalar[fscale]),
    )

    return (freq, spectrum, stats)
