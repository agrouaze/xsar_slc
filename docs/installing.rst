.. _installing:

************
Installation
************

`xsar_slc` use `xarray-datatree` object as input, any Sentinel-1 TOPS SLC reader could be used to retrieve digital
numbers and useful annotations (doppler estimates, bursts, FM-rates, orbits, ...). xsar_ is the reader used to develop
this cross spectra estimator library.
Installation in a conda_ environment is recommended.


conda install
#############


.. code-block::

    conda create -n xsar_slc
    conda activate xsar_slc
    conda install -c conda-forge xsar
    cd xsar_slc
    pip install .


Update xsar_slc to the latest version
#####################################


To be up to date with the development team, it's recommended to update the installation using pip:

.. code-block::

    pip install git+https://github.com/umr-lops/xsar-slc.git


.. _conda: https://docs.anaconda.com/anaconda/install/
.. _xsar: https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsar