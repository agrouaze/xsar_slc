import xsarslc
from yaml import load
import logging
from importlib import reload
import os
reload(logging)
logging.basicConfig(level=logging.INFO)
from yaml import CLoader as Loader

stream = open(os.path.join(os.path.dirname(xsarslc.__file__), 'config.yml'), 'r')
conf = load(stream, Loader=Loader)  # TODO : add argument to compute_subswath_xspectra(conf=conf)


def get_IR_file(unit, subswath, polarization, auxdir=conf['auxfiledir']):
    """
    parameters:
        unit str S1A or S1B or ...
        subswath str IW1, IW2, ... WV1, WV2
        polarization str VH HH VV
    """

    pathaux = os.path.abspath(os.path.join(os.path.dirname(xsarslc.__file__), 'scripts', auxdir,
                                           unit + '_IRs_' + subswath + '_' + polarization + '.nc'))
    logging.info('pathaux: %s', pathaux)
    if os.path.exists(pathaux):

        return pathaux
    else:
        logging.warning('RI file %s cannot be found', pathaux)
        return None

def get_production_version(auxdir=conf):
    """

    :param auxdir: str path of the config file
    :return:
         pv (str): product version (e.g. "1.4")
    """
    pv = str(auxdir['product_version'])
    return pv

def get_default_outputdir(auxdir=conf):
    """

    :param auxdir: str path of the config file
    :return:
         do (str): path of the outputdir where to store L1B files
    """
    do = auxdir['default_outputdir']
    return do

def get_default_landmask_dir(auxdir=conf):
    """

        :param auxdir: str path of the config file
        :return:
             lm (str): path of the landmask cartopy for instance (for offline access)
        """
    lm = auxdir['default_landmask_dir']
    return lm

def get_default_xspec_params(config_name='tiles20km',auxdir=conf):
    """

    :param config_name: str eg tiles2km
    :param auxdir: str path of the config file
    :return:
    """
    params = auxdir['xspec_configs'][config_name]
    return params


if __name__ == '__main__':
    get_IR_file(unit='S1A', subswath='WV2', polarization='VV', auxdir=conf['auxfiledir'])
    out_dir = '/home/datawork-cersat-public/project/sarwave/data/products/developments/aux_files/sar/impulse_response/'
    get_IR_file(unit='S1A', subswath='WV2', polarization='VV', auxdir=out_dir)