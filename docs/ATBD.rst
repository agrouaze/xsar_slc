.. _atbd:

***************************************
Algorithm L1B cross spectrum Sentinel-1
***************************************

This page stands as the ATBD (Algorithm Technical Baseline Document) for Sentinel-1 L1B IFREMER product.

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


computing modulation
####################

The modulation is computed from Digital Numbers (DN) (i.e. complexe Level-1 SLC annotated values).
The method :meth:`xsarslc.processing.xspectra.compute_modulation` performs this operation.

The equation is :
:math:`modulation = \frac{DN}{\sqrt{intensity_{lowpass}}}`

where :math:`intensity_{lowpass} = \frac{GaussianFilter(abs(DN^2))}{GaussianFilter(DN)}`


tiling of the sub-swathes
#########################

The tiling which is done within the bursts and inter-burst areas is given in meters by the user.
The philosophy behind the posting consists in setting the maximum of tiles within the valid pixels footprint.
Tiles can overlap or not. A maximum of 50% overllaping is present.
The periodograms size cannot be larger than tiles.

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