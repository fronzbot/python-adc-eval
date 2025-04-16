"""Spectral analysis module."""

import numpy as np
import matplotlib.pyplot as plt
from adc_eval.eval import calc


def calc_psd(data, fs, nfft=2**12):
    """Calculate the PSD using the Bartlett method."""
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
    """Get the power spectrum for an input signal."""
    (freq_ss, psd_ss, freq_ds, psd_ds) = calc_psd(np.array(data), fs=fs, nfft=nfft)
    if single_sided:
        return (freq_ss, psd_ss * fs / nfft)
    return (freq_ds, psd_ds * fs / nfft)


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
    fscale="MHz",
):
    """Plot Power Spectrum for input signal."""
    wsize = data.size
    windows = {
        "rectangular": np.ones(wsize),
        "hanning": np.hanning(wsize),
    }

    fscalar = {
        "uHz": 1e-6,
        "mHz": 1e-3,
        "Hz": 1,
        "kHz": 1e3,
        "MHz": 1e6,
        "GHz": 1e9,
        "THz": 1e12,
    }

    if window not in windows:
        print(f"WARNING: {window} not implemented. Defaulting to 'rectangular'.")
        window = "rectangular"

    if fscale not in fscalar:
        print(f"WARNING: {fscale} not implemented. Defaulting to 'MHz'.")
        fscale = "MHz"

    wscale = {
        "rectangular": 1.0,
        "hanning": 1.633,
    }[window]

    (freq, pwr) = get_spectrum(
        data * windows[window] * wscale, fs=fs, nfft=nfft, single_sided=single_sided
    )
    full_scale = calc.dBW(dr**2 / 8)

    yaxis_lut = {
        "power": [0, "dB"],
        "fullscale": [full_scale, "dBFS"],
        "normalize": [max(calc.dBW(pwr)), "dB Normalized"],
        "magnitude": [0, "W"],
    }

    lut_key = yaxis.lower()
    scalar = yaxis_lut[lut_key][0]
    yunits = yaxis_lut[lut_key][1]
    try:
        xscale = fscalar[fscale]
    except KeyError:
        print(
            f"WARNING: {fscale} not a valid option for fscale. Valid inputs are {fscalar.keys()}."
        )
        print("         Defaulting to Hz.")

    psd_out = calc.dBW(pwr, places=3) - scalar
    if lut_key in ["magnitude"]:
        psd_out = pwr

    f_ss = freq
    psd_ss = pwr
    if not single_sided:
        # Get single-sided spectrum for SNDR and Harmonic stats
        (f_ss, psd_ss) = get_spectrum(
            data * windows[window] * wscale, fs=fs, nfft=nfft, single_sided=True
        )

    sndr_stats = calc.sndr_sfdr(psd_ss, f_ss, fs, nfft, leak=leak, full_scale=full_scale)

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

    stats = {**sndr_stats, **harm_stats}

    xmin = 0 if single_sided else -fs / 2e6

    if not no_plot:
        plt_str = calc.get_plot_string(stats, full_scale, fs, nfft, window, xscale, fscale)
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(freq / xscale, psd_out)
        ax.set_ylabel(f"Power Spectrum ({yunits})", fontsize=18)
        ax.set_xlabel(f"Frequency ({fscale})", fontsize=16)
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
    """Perform spectral analysis on input waveform."""
    (freq, spectrum, stats) = plot_spectrum(
        data,
        fs=fs,
        nfft=nfft,
        dr=dr,
        harmonics=harmonics,
        leak=leak,
        window=window,
        no_plot=no_plot,
        yaxis=yaxis,
        single_sided=single_sided,
        fscale=fscale,
    )

    return (freq, spectrum, stats)