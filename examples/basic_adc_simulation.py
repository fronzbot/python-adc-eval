"""Runs a basic ADC simulation and plots the spectrum."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from adc_eval import signals
from adc_eval.adcs import basic
from adc_eval.eval import spectrum
from adc_eval.eval.simulate import Simulator


"""
Simulation Settings
"""
SEED = 42
NBITS = 10
FS = 200e6
NLEN = 2**16    # Larger than NFFT to enable Bartlett method for PSD
NFFT = 2**12
vref = 1
fbin = NFFT / 4 - 31
ftone = NFFT / 2 - 15
vin_amp = 0.707 * vref / 2


"""
VIN Generation
"""
(t, vin) = signals.tones(NLEN, [fbin, ftone], [vin_amp, vin_amp*0.2], offset=0, fs=FS, nfft=NFFT)

"""
ADC Architecture Creation
"""
adc_dut = basic.ADC(nbits=NBITS, vref=vref, fs=FS, seed=SEED)


"""
Global ADC Error Settings
"""
adc_dut.noise = 0  # No internal noise generation
adc_dut.offset = (0, 0)  # 10mV mean offset with no stdev
adc_dut.gain_err = (0, 0)  # No internal gain error
adc_dut.distortion = [1, 0, 0.3]  # HD3 only


"""
Run Simulation
"""
s = Simulator(adc_dut, vin)
s.run()


"""
Output Plotting
"""
(freq, ps, stats) = spectrum.analyze(
    s.out,
    NFFT,
    fs=FS,
    dr=2**NBITS,
    harmonics=7,
    leak=1,
    window="rectangular",
    no_plot=False,
    yaxis="fullscale",
    single_sided=True,
    fscale="MHz"
)
ax = plt.gca()
ax.set_title("ADC Spectrum")
ax.set_ylim([-100, 0])
ax.set_yticks(np.linspace(-100, 0, 11))
ax.set_xticks(np.linspace(0, FS/2e6, 9))
