python-adc-eval |Lint| |PyPi Version| |Codecov| |Codestyle|
=============================================================

A python-based ADC evaluation tool, suitable for standalone or library-based usage

Details
--------

Inspired by
`esynr3z/adc-eval <https://github.com/esynr3z/adc-eval>`__

Performs spectral analysis of a dataset utilizing the Bartlett method. Calculates SFDR, SNDR, as well as harmonics.

.. figure:: analyser.png
   :alt: analyser

   analyser


USAGE
=======

To load the library in a module:

.. code-block:: python

    import adc_eval


Given an array of values representing the output of an ADC, the spectrum can be analyzed with the following:

.. code-block:: python

    import adc_eval

    adc_eval.spectrum.analyze(
        <data>,
        <fft bins>,
        fs=<sample frequency>,
        dr=<dynamicrange/vref>,
        harmonics=<num of harmonics to find>,
        leak=<adjacent bins to filter>,
        window=<window type (rectangular/hanning)>,
        no_plot=<True/False>,
        yaxis=<"power"/"fullscale">
    )


.. |Lint| image:: https://github.com/fronzbot/python-adc-eval/workflows/Lint/badge.svg
   :target: https://github.com/fronzbot/python-adc-eval/actions?query=workflow%3ALint
.. |PyPi Version| image:: https://img.shields.io/pypi/v/python-adc-eval.svg
   :target: https://pypi.org/project/python-adc-eval
.. |Codestyle| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. |Codecov| image:: https://codecov.io/gh/fronzbot/python-adc-eval/graph/badge.svg?token=156GMQ4NNV 
 :target: https://codecov.io/gh/fronzbot/python-adc-eval
