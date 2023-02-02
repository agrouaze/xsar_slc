.. _atbd:

*******************
ATBD L1B Sentinel-1
*******************

algorithm overview
##################

processing steps to start from S1 SLC product to L1B

  - computing modulation
  - tiling the sub-swath
  - removing `bright targets`_
  - low pass filtering (gaussian filter) on digital numbers
  - land detection per tile
  - `Impulse Response`_ correction
  - periodograms definition
  - looks definitions
  - zero doppler estimation
  - FFT computation
  - normalization of looks energy




bright target
#############

For now no bright target detection is applied nor bright target removal.


Impulse Response
################

The Impulse Response (RI), is the spectral contribution of the sensor into the signal.
It is defined in range and azimuth.
Sentinel-1A has a default Impulse Response that leads to asymmetric and uncentered (skewed) spectra.
This RI has to be removed from each FFT performed on the SLC signal to respect hypothesis of looks centered on the zero Doppler.
Estimation of the RI is illustrated in these examples: `example_
The IW estimation of RI is performed using :py:func:`xsarslc.processing.impulseResponse.compute_IWS_subswath_Impulse_Response`
Same for WV in :py:func:`xsarslc.processing.impulseResponse.compute_WV_Impulse_Response .

.. _`bright targets`: ATBD.rst#bright target
.. _`Impulse Response`: ATBD.rst#Impulse Response