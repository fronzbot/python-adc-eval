"""Spectral analysis helper module."""

import numpy as np


def db_to_pow(value, places=3):
    """
    Convert dBW to W.

    Parameters
    ----------
    value : float or ndarray
        Value to convert to power, in dBW.
    places : int, optional
        Number of places to round output value to. Default is 3.

    Returns
    -------
    float or ndarray
        Returns either the rounded and converted value, or the ndarray
    """
    if isinstance(value, np.ndarray):
        return np.round(10 ** (0.1 * value), places)
    return round(10 ** (0.1 * value), places)


def dBW(value, places=1):
    """
    Convert to dBW.

    Parameters
    ----------
    value : float or ndarray
        Value to convert to dBW, in W.
    places : int, optional
        Number of places to round output value to. Default is 1.

    Returns
    -------
    float or ndarray
        Returns either the rounded and converted value, or the ndarray
    """
    if isinstance(value, np.ndarray):
        return np.round(10 * np.log10(value), places)
    return round(10 * np.log10(value), places)


def enob(sndr, places=1):
    """
    Return ENOB for given SNDR.

    Parameters
    ----------
    sndr : float
        SNDR value in dBW to convert to ENOB.
    places : int, optional
        Number of places to round output value to. Default is 1.

    Returns
    -------
    float or ndarray
        Returns either the rounded and converted value, or the ndarray
    """
    return round((sndr - 1.76) / 6.02, places)


def sndr_sfdr(spectrum, freq, fs, nfft, leak=0, full_scale=0):
    """
    Get SNDR and SFDR.

    Parameters
    ----------
    spectrum : ndarray
        Power spectrum as ndarray in units of Watts.
    freq : ndarray
        Array of frequencies for the input power spectrum.
    fs : float
        Sample frequency of power spectrum in Hz.
    nfft : int
        Number of samples in the FFT.
    leak : int, optional
        Number of leakage bins to consider when looking for peaks. Default is 0.
    full_scale : float, optional
        Full scale reference value for spectrum in Watts.

    Returns
    -------
    dict
        Returns a dictionary of computed stats.
    """
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
    """
    Get the harmonic contents of the data.

    Parameters
    ----------
    spectrum : ndarray
        Power spectrum as ndarray in units of Watts.
    freq : ndarray
        Array of frequencies for the input power spectrum.
    nfft : int
        Number of samples in the FFT.
    bin_sig : int
        Frequency bin of the dominant signal.
    psig : float
        Power of dominant signal in spectrum.
    harms : int, optional
        Number of input harmonics to calculate. Default is 5.
    leak : int, optional
        Number of leakage bins to look at when finding harmonics. Default is 20.
    fscale : float, optional
        Value to scale frequencies by in Hz. Default is 1MHz.

    Returns
    -------
    dict
        Returns a dictionary of computed stats.
    """
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
