#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A. Grouazel
13 April 2023
purpose: produce nc files from SAFE EW SLC containing cartesian x-spec computed with xsar and xsar_slc
 on intra and inter bursts
"""

import xsarslc.processing.xspectra as proc
import warnings
import xsar

# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
import numpy as np
import datetime
import logging
import os
import time
import xsarslc

print('xsarslc version:', xsarslc.__version__)
print('source ', xsarslc.__file__)
import argparse
from xsarslc.get_config_infos import get_IR_file, get_production_version, get_default_outputdir, \
    get_default_xspec_params, \
    get_default_landmask_dir

PRODUCT_VERSION = get_production_version()


def get_memory_usage():
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000. / 1000.
    except:  # on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go' % memory_used_go
    return str_mem


def generate_EW_L1Bxspec_product(slc_ew_path, output_filename, xspeconfigname, polarization=None, dev=False,
                                 landmask=None):
    """

    :param tiff: str full path
    :param output_filename : str full path
    :param xspeconfigname : str (eg 'tiles20km')
    :param polarization : str : VV VH HH HV [optional]
    :param dev: bool: allow to shorten the processing
    :param landmask : landmask obj (eg : cartopy.feature.NaturalEarthFeature() )
    :return:
    """
    safe = os.path.dirname(os.path.dirname(slc_ew_path))
    logging.info('start loading the datatree %s', get_memory_usage())
    tiff_number = os.path.basename(slc_ew_path).split('-')[1].replace('ew', '')
    str_gdal = 'SENTINEL1_DS:%s:EW%s' % (safe, tiff_number)
    bu = xsar.Sentinel1Meta(str_gdal)._bursts
    chunksize = {'line': int(bu['linesPerBurst'].values), 'sample': int(bu['samplesPerBurst'].values)}
    xsarobj = xsar.Sentinel1Dataset(str_gdal, chunks=chunksize)
    dt = xsarobj.datatree
    dt.load()  # took 4min to load and 35Go RAM
    logging.info('datatree loaded %s', get_memory_usage())
    xspec_params = get_default_xspec_params(config_name=xspeconfigname)
    tile_width_intra = xspec_params['tile_width_intra']
    tile_overlap_intra = xspec_params['tile_overlap_intra']
    periodo_width_intra = xspec_params['periodo_width_intra']
    periodo_overlap_intra = xspec_params['periodo_overlap_intra']
    tile_width_inter = xspec_params['tile_width_inter']
    tile_overlap_inter = xspec_params['tile_overlap_inter']
    periodo_width_inter = xspec_params['periodo_width_inter']
    periodo_overlap_inter = xspec_params['periodo_overlap_inter']
    logging.info('tile_width_intra : %s', tile_width_intra)
    unit = os.path.basename(safe)[0:3]
    subswath = str_gdal.split(':')[2]
    IR_path = get_IR_file(unit, subswath, polarization.upper())
    logging.info('impulse response file: %s', IR_path)
    if IR_path:
        one_subswath_xspectrum_dt = proc.compute_subswath_xspectra(dt, polarization=polarization.upper(),
                                                                   dev=dev, compute_intra_xspec=True,
                                                                   compute_inter_xspec=True,
                                                                   tile_width_intra=tile_width_intra,
                                                                   tile_overlap_intra=tile_overlap_intra,
                                                                   periodo_width_intra=periodo_width_intra,
                                                                   periodo_overlap_intra=periodo_overlap_intra,
                                                                   tile_width_inter=tile_width_inter,
                                                                   tile_overlap_inter=tile_overlap_inter,
                                                                   periodo_width_inter=periodo_width_inter,
                                                                   periodo_overlap_inter=periodo_overlap_inter
                                                                   , IR_path=IR_path, landmask=landmask)
    else:
        one_subswath_xspectrum_dt = proc.compute_subswath_xspectra(dt, polarization=polarization.upper(),
                                                                   dev=dev, compute_intra_xspec=True,
                                                                   compute_inter_xspec=True,
                                                                   tile_width_intra=tile_width_intra,
                                                                   tile_overlap_intra=tile_overlap_intra,
                                                                   periodo_width_intra=periodo_width_intra,
                                                                   periodo_overlap_intra=periodo_overlap_intra,
                                                                   tile_width_inter=tile_width_inter,
                                                                   tile_overlap_inter=tile_overlap_inter,
                                                                   periodo_width_inter=periodo_width_inter,
                                                                   periodo_overlap_inter=periodo_overlap_inter,
                                                                   landmask=landmask
                                                                   )
    if one_subswath_xspectrum_dt:
        logging.info('xspec intra and inter ready for %s', slc_ew_path)
        logging.debug('one_subswath_xspectrum = %s', one_subswath_xspectrum_dt)
        one_subswath_xspectrum_dt.attrs['version_xsar'] = xsar.__version__
        one_subswath_xspectrum_dt.attrs['version_xsarslc'] = xsarslc.__version__
        one_subswath_xspectrum_dt.attrs['processor'] = __file__
        one_subswath_xspectrum_dt.attrs['generation_date'] = datetime.datetime.today().strftime('%Y-%b-%d')
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), 0o0775)
            logging.info('makedir %s', os.path.dirname(output_filename))
        one_subswath_xspectrum_dt.to_netcdf(output_filename)
        logging.info('successfuly written %s', output_filename)
    else:
        logging.info('no inter nor intra xspectra available in this subswath')


def main():
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1BwaveIFR_EW_SLC')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--tiff', required=True, help='tiff file full path EW SLC')
    parser.add_argument('--outputdir', required=False, help='directory where to store output netCDF files',
                        default=get_default_outputdir(mode='ew'))
    parser.add_argument('--version',
                        help='set the output product version (e.g. 1.4) default version will be read from config.yml',
                        required=False, default=PRODUCT_VERSION)
    parser.add_argument('--dev', action='store_true', default=False, help='dev mode stops the computation early')
    parser.add_argument('--landmask', required=False, default=get_default_landmask_dir(),
                        help='landmask files (such as cartopy /.local/share/cartopy ) to have a landmask information '
                             'without web connexion , default value cromes from config.yml')
    parser.add_argument('--xspeconfigname', required=False, default='tiles20km',
                        help='name of the cross-spectra (tiles/periodogram) in config.yml e.g. "tiles20km" ')
    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    t0 = time.time()
    logging.info('product version to produce: %s', args.version)
    logging.info('outputdir will be: %s', args.outputdir)
    logging.info('xspeconfigname : %s', args.xspeconfigname)
    slc_ew_path = args.tiff
    if 'cartopy' in args.landmask:
        logging.info('landmask is a cartopy feature')
        import cartopy

        cartopy.config['pre_existing_data_dir'] = args.landmask
        landmask = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    else:
        landmask = None
    polarization_from_file = os.path.basename(slc_ew_path).split('-')[3]
    safe_basename = os.path.basename(os.path.dirname(os.path.dirname(slc_ew_path)))
    safe_basename = safe_basename.replace('SLC', 'XSP')
    output_filename = os.path.join(args.outputdir, args.version, safe_basename, os.path.basename(
        slc_ew_path).replace('.tiff', '') + '_L1B_xspec_IFR_' + args.version + '.nc')
    logging.info('mode dev is %s', args.dev)
    logging.info('output filename would be: %s', output_filename)
    if os.path.exists(output_filename) and args.overwrite is False:
        logging.info('%s already exists', output_filename)
    else:
        generate_EW_L1Bxspec_product(slc_ew_path=slc_ew_path, output_filename=output_filename,
                                     xspeconfigname=args.xspeconfigname, dev=args.dev,
                                     polarization=polarization_from_file, landmask=landmask)
    logging.info('peak memory usage: %s Mbytes', get_memory_usage())
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
