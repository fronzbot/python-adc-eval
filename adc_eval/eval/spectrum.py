"""Spectral analysis module."""

import numpy as np
import matplotlib.pyplot as plt


def db_to_pow(value, places=3):
    """Convert dBW to W."""
    if isinstance(value, np.ndarray):
        return round(10 ** (0.1 * value), places)
    return round(10 ** (0.1 * value), places)


def dBW(value, places=1):
    """Convert to dBW."""
    if isinstance(value, np.ndarray):
        return round(10 * np.log10(value), places)
    return round(10 * np.log10(value), places)


def enob(sndr, places=1):
    """Return ENOB for given SNDR."""
    return round((sndr - 1.76) / 6.02, places)


def sndr_sfdr(spectrum, freq, fs, nfft, leak, full_scale=0):
    """Get SNDR and SFDR."""

    # Zero the DC bin
    for i in range(0, leak + 1):
        spectrum[i] = 0
    bin_sig = np.argmax(spectrum)
    psig = sum(spectrum[i] for i in range(bin_sig - leak, bin_sig + leak + 1))
    spectrum_n = spectrum.copy()
    spectrum_n[bin_sig] = 0
    fbin = fs / nfft

    for i in range(bin_sig - leak, bin_sig + leak + 1):
        spectrum_n[i] = 0

    bin_spur = np.argmax(spectrum_n)
    pspur = spectrum[bin_spur]

    noise_power = sum(spectrum_n)
    noise_floor = 2 * noise_power / nfft

    stats = {}

    stats["sig"] = {
        "freq": freq[bin_sig],
        "bin": bin_sig,
        "power": psig,
        "dB": dBW(psig),
        "dBFS": round(dBW(psig) - full_scale, 1),
    }

    stats["spur"] = {
        "freq": freq[bin_spur],
        "bin": bin_spur,
        "power": pspur,
        "dB": dBW(pspur),
        "dBFS": round(dBW(pspur) - full_scale, 1),
    }
    stats["noise"] = {
        "floor": noise_floor,
        "power": noise_power,
        "rms": np.sqrt(noise_power),
        "dBHz": round(dBW(noise_floor, 3) - full_scale, 1),
        "NSD": round(dBW(noise_floor, 3) - full_scale - 2 * dBW(fbin, 3), 1),
    }
    stats["sndr"] = {
        "dBc": dBW(psig / noise_power),
        "dBFS": round(full_scale - dBW(noise_power), 1),
    }
    stats["sfdr"] = {
        "dBc": dBW(psig / pspur),
        "dBFS": round(full_scale - dBW(pspur), 1),
    }
    stats["enob"] = {"bits": enob(stats["sndr"]["dBFS"])}

    return stats


def find_harmonics(spectrum, freq, nfft, bin_sig, psig, harms=5, leak=20, fscale=1e6):
    """Get the harmonic contents of the data."""
    harm_stats = {"harm": {}}
    harm_index = 2
    for harm in bin_sig * np.arange(2, harms + 1):
        harm_stats["harm"][harm_index] = {}
        zone = np.floor(harm / (nfft / 2)) + 1
        if zone % 2 == 0:
            bin_harm = int(nfft / 2 - (harm - (zone - 1) * nfft / 2))
        else:
            bin_harm = int(harm - (zone - 1) * nfft / 2)

        # Make sure we pick the max bin where power is maximized; due to spectral leakage
        # if bin_harm == nfft/2, set to bin of 0
        if bin_harm == nfft / 2:
            bin_harm = 0
        pwr_max = spectrum[bin_harm]
        bin_harm_max = bin_harm
        for i in range(bin_harm - leak, bin_harm + leak + 1):
            try:
                pwr = spectrum[i]
                if pwr > pwr_max:
                    bin_harm_max = i
                    pwr_max = pwr
            except IndexError:
                # bin + leakage out of bounds, so stop looking
                break

        harm_stats["harm"][harm_index]["bin"] = bin_harm_max
        harm_stats["harm"][harm_index]["power"] = pwr_max
        harm_stats["harm"][harm_index]["freq"] = round(freq[bin_harm] / fscale, 1)
        harm_stats["harm"][harm_index]["dBc"] = dBW(pwr_max / psig)
        harm_stats["harm"][harm_index]["dB"] = dBW(pwr_max)

        harm_index = harm_index + 1

    return harm_stats


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
    full_scale = dBW(dr**2 / 8)

    yaxis_lut = {
        "power": [0, "dB"],
        "fullscale": [dBW(dr**2 / 8), "dBFS"],
        "normalize": [max(dBW(pwr)), "dB Normalized"],
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

    psd_out = 10 * np.log10(pwr) - scalar
    if lut_key in ["magnitude"]:
        psd_out = pwr

    f_ss = freq
    psd_ss = pwr
    if not single_sided:
        # Get single-sided spectrum for SNDR and Harmonic stats
        (f_ss, psd_ss) = get_spectrum(
            data * windows[window] * wscale, fs=fs, nfft=nfft, single_sided=True
        )

    sndr_stats = sndr_sfdr(psd_ss, f_ss, fs, nfft, leak=leak, full_scale=full_scale)

    harm_stats = find_harmonics(
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
        plt_str = get_plot_string(stats, full_scale, fs, nfft, window, xscale, fscale)
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


def get_plot_string(stats, full_scale, fs, nfft, window, xscale=1e6, fscale="MHz"):
    """Generate plot string from stats dict."""

    plt_str = "==== FFT ====\n"
    plt_str += f"NFFT = {nfft}\n"
    plt_str += f"fbin = {round(fs/nfft / 1e3, 2)} kHz\n"
    plt_str += f"window = {window}\n"
    plt_str += "\n"
    plt_str += "==== Signal ====\n"
    plt_str += f"FullScale = {full_scale} dB\n"
    plt_str += f"Psig = {stats['sig']['dBFS']} dBFS ({stats['sig']['dB']} dB)\n"
    plt_str += f"fsig = {round(stats['sig']['freq']/xscale, 2)} {fscale}\n"
    plt_str += f"fsamp = {round(fs/xscale, 2)} {fscale}\n"
    plt_str += "\n"
    plt_str += "====  SNDR/SFDR  ====\n"
    plt_str += f"ENOB = {stats['enob']['bits']} bits\n"
    plt_str += f"SNDR = {stats['sndr']['dBFS']} dBFS ({stats['sndr']['dBc']} dBc)\n"
    plt_str += f"SFDR = {stats['sfdr']['dBFS']} dBFS ({stats['sfdr']['dBc']} dBc)\n"
    plt_str += f"Pspur = {stats['spur']['dBFS']} dBFS\n"
    plt_str += f"fspur = {round(stats['spur']['freq']/xscale, 2)} {fscale}\n"
    plt_str += f"Noise Floor = {stats['noise']['dBHz']} dBFS\n"
    plt_str += f"NSD = {stats['noise']['NSD']} dBFS\n"
    plt_str += "\n"
    plt_str += "==== Harmonics ====\n"

    for hindex, hdata in stats["harm"].items():
        plt_str += f"HD{hindex} = {round(hdata['dB'] - full_scale, 1)} dBFS @ {hdata['freq']} {fscale}\n"

    plt_str += "\n"

    return plt_str


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
