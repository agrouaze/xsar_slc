.. _normalizedvariance:

*******************************************************
Computation of the sigma0 normalized variance
*******************************************************

The normalized variance is the variance of the Digital number defined over a prescribed spatial extension.
In the baseline L1B SLC processor, the variance is computed at a tile level.

It writes:

.. math::
   nv\triangleq\dfrac{\left\langle\left(m-\left\langle m\right\rangle\right)^2\right\rangle}{\left\langle m\right\rangle^2}

where :math:`m=\left|\widetilde{DN}\right|^2` and :math:`\widetilde{DN}` is defined in equation :eq:`DNmod`
