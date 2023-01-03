######################################################################
xsarslc: functions to compute cross spectra from TOPS SLC SAR products
######################################################################

**xsarslc** is a library to compute cross spectrum from level 1 SAR SLC products. Objets manipulated are all `xarray`_.

The input `datatree`_ object can come from any reader library but the original design has been done using **`xsar`_**


.. jupyter-execute:: examples/intro.py


Documentation
-------------

Overview
........

    **xsarslc** can compute both intra burst and inter (i.e. overlapping bursts) burst cross spectrum.

    To have comparable cross spectrum among bursts and sub-swaths, we choose to have constant `dk` values,
    it means that the number of pixels used to compute the cross spectrum is not always the same.

    The algorithm is performing 4 different sub-setting in the original complex digital number images:

        1) bursts sub-setting
        2) tiles sub-setting
        3) periodograms sub-setting
        4) looks (in azimuth) sub-setting

    Default configuration is set to:
        * 20x20 km tiles in the bursts. (less in inter burst where we have about 3 km of overlapping).
        * 50% overlapping tiles
        * 2-tau saved cross spectrum

Examples
........

.. note::
    here are some examples of usage

* :doc:`examples/xspec_IW_intra_and_inter_burst`

Reference
.........

* :doc:`basic_api`

Get in touch
------------

- Report bugs, suggest features or view the source code `on github`_.

----------------------------------------------

Last documentation build: |today|

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   installing


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Examples

   examples/xspec_IW_intra_and_inter_burst

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   basic_api

.. _on github: https://github.com/umr-lops/xsar_slc
.. _xarray: http://xarray.pydata.org
.. _datatree: https://github.com/xarray-contrib/datatree
.. _xarray.Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html