.. _cutoff:

*****************************
Azimuthal cut-off computation
*****************************

The azimuthal cut-off is a characteristic distance defining the maximum wavelengh that the SAR was able to recover in the azimuth distance.
It characterize its effective azimuthal resolution. The sea surface waves motion is responsible for a strong smearing in the azimuth direction.
This smearing effect largely increases with the wave motion and is a good proxy for wind velocity.

The azimuthal cut-off is computed as follow.

1. Computation of covariance function as the Inverse Fourier Transform of the cross-spectrum
2. Averaging the covariance function on the range axis (or taking a transect)
3. Normalize by it maximum
4. Fit a Gaussian function and returns its standard deviation


The covariance function writes:

.. math::
   \rho(rg,az) = IFT^{2D}\left[\Re e(XS^{n\tau})\right)

where :math:`\Re e` stands for the real part and where `n=2` in the baseline processing.
A Gaussian fit is applied on :math:`\underline{\rho}(az) = \dfrac{\rho(rg=0, az)}{\rho(rg=0, az=0)}` over the range span [-500,500] in the baseline processing.
In the literature, the Gaussian fit can also be done over the range averaged covariance function :math:`\left\langle\rho(rg,az)\right\rangle_{rg}`.
The Gaussian fit is realized with a least square difference cost function and a gradient descent methodology.
The azimuthal cut-off :math:`\lambda` is defined as the standard deviation of the fitted function:

.. math::

   \exp\left(\dfrac{-az^2}{2\lambda^2}\right)
