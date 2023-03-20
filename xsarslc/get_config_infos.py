import xsarslc
from yaml import load
import logging
import os
from yaml import CLoader as Loader

stream = open(os.path.join(os.path.dirname(xsarslc.__file__), 'config.yml'), 'r')
conf = load(stream, Loader=Loader)


def get_IR_file(unit, subswath, polarization):
    """
    parameters:
        unit str S1A or S1B or ...
        subswath str IW1, IW2, ... WV1, WV2
        polarization str VH HH VV
    """
    logging.debug('unit %s %s',unit,type(unit))
    logging.debug('subswath %s %s', subswath, type(subswath))
    logging.debug('polarization %s %s', polarization, type(polarization))
    pathaux = os.path.abspath(os.path.join(os.path.dirname(xsarslc.__file__),'..', 'auxdata',
                                           unit + '_IRs_' + subswath + '_' + polarization + '.nc'))
    logging.info('expected path for auxiliary Impulse Response file : %s', pathaux)
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

def get_default_outputdir(mode='iw',auxdir=conf):
    """

    :param auxdir: str path of the config file
    :return:
         do (str): path of the outputdir where to store L1B files
    """
    do = auxdir['default_outputdir_%s'%mode]
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
