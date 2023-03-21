#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A. Grouazel
23 Jan 2023
purpose: produce nc files from SAFE WV SLC containing cartesian x-spec computed with xsar and xsarsea
"""
import pdb
import argparse
import xsarslc.processing.xspectra as proc
from xsarslc.tools import netcdf_compliant
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
from xsarslc.get_config_infos import get_IR_file, get_production_version, get_default_outputdir, \
    get_default_xspec_params, \
    get_default_landmask_dir

PRODUCT_VERSION = get_production_version()  # see  https://github.com/umr-lops/xsar_slc/wiki/IFR-WV-processings


def get_memory_usage():
    """
    Args:

    Returns:

    """
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000. / 1000.
    except:  # on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go' % memory_used_go
    return str_mem


def main():
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1BwaveIFR_WV_SLC')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--tiff', required=True, help='tiff file full path WV SLC')
    parser.add_argument('--outputdir', required=False, help='directory where to store output netCDF files',
                        default=get_default_outputdir(mode='wv'))
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
                            datefmt='%d/%m/%Y %H:%M:%S')
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S')
    t0 = time.time()

    slc_wv_path = args.tiff
    logging.info('product version to produce: %s', args.version)
    logging.info('product version to produce: %s', args.version)
    logging.info('outputdir will be: %s', args.outputdir)
    logging.info('xspeconfigname : %s', args.xspeconfigname)
    subswath_number = os.path.basename(slc_wv_path).split('-')[1]
    polarization_from_file = os.path.basename(slc_wv_path).split('-')[3].upper()
    subsath_nickname = '%s_%s' % (subswath_number, polarization_from_file)
    safe_basename = os.path.basename(os.path.dirname(os.path.dirname(slc_wv_path)))
    safe_basename = safe_basename.replace('SLC', 'XSP')
    output_filename = os.path.join(args.outputdir, args.version, safe_basename, os.path.basename(
        slc_wv_path).replace('.tiff', '') + '_L1B_xspec_IFR_' + args.version + '.nc')
    logging.info('mode dev is %s', args.dev)

    if 'cartopy' in args.landmask:
        logging.info('landmask is a cartopy feature')
        import cartopy

        cartopy.config['pre_existing_data_dir'] = args.landmask
        landmask = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    else:
        landmask = None
    logging.info('output filename would be: %s', output_filename)
    if os.path.exists(output_filename) and args.overwrite is False:
        logging.info('%s already exists', output_filename)
    else:
        generate_WV_L1Bxspec_product(slc_wv_path=slc_wv_path, output_filename=output_filename,
                                     xspeconfigname=args.xspeconfigname, dev=args.dev,
                                     polarization=polarization_from_file, landmask=landmask)
    logging.info('peak memory usage: %s Mbytes', get_memory_usage())
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)


def generate_WV_L1Bxspec_product(slc_wv_path, output_filename, xspeconfigname, polarization=None, dev=False,
                                 landmask=None):
    """
    Args:
        slc_wv_path: str full path of .tiff file
        output_filename : str full path
        xspeconfigname : str (eg 'tiles20km')
        polarization : str : VV VH HH HV [optional]
        dev: bool: allow to shorten the processing
        landmask : landmask obj (eg : cartopy.feature.NaturalEarthFeature() )
    Return:

    """
    safe = os.path.dirname(os.path.dirname(slc_wv_path))
    logging.info('start loading the datatree %s', get_memory_usage())

    xspec_params = get_default_xspec_params(config_name=xspeconfigname)
    tile_width_intra = xspec_params['tile_width_intra']
    tile_overlap_intra = xspec_params['tile_overlap_intra']
    periodo_width_intra = xspec_params['periodo_width_intra']
    periodo_overlap_intra = xspec_params['periodo_overlap_intra']
    logging.info('tile_width_intra : %s', tile_width_intra)

    imagette_number = os.path.basename(slc_wv_path).split('-')[-1].replace('.tiff', '')
    str_gdal = 'SENTINEL1_DS:%s:WV_%s' % (safe, imagette_number)
    chunksize = {'line': 6000, 'sample': 7000}
    xsarobj = xsar.Sentinel1Dataset(str_gdal, chunks=chunksize)
    dt = xsarobj.datatree
    dt.load()  # took ?min to load and ? Go RAM
    logging.info('datatree loaded %s', get_memory_usage())
    unit = os.path.basename(safe)[0:3]
    subswath = str(dt['image'].ds['swath_subswath'].values)
    IR_path = get_IR_file(unit, subswath, polarization.upper())
    xs0 = proc.compute_WV_intraburst_xspectra(dt=dt,
                                              polarization=polarization,
                                              tile_width_intra=tile_width_intra,
                                              tile_overlap_intra=tile_overlap_intra,
                                              periodo_width_intra=periodo_width_intra,
                                              periodo_overlap_intra=periodo_overlap_intra,
                                              IR_path=IR_path, dev=dev, landmask=landmask)
    # xs = xs0.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
    # xs = xspectra.symmetrize_xspectrum(xs, dim_range='k_rg', dim_azimuth='k_az')
    xs = netcdf_compliant(xs0)  # to split complex128 variables into real and imag part
    xs = xs.drop('pol')
    if 'spatial_ref' in xs:
        xs = xs.drop('spatial_ref')
    if xs:
        logging.info('xspec ready for %s', slc_wv_path)
        logging.debug('one_subswath_xspectrum = %s', xs)
        xs.attrs['version_xsar'] = xsar.__version__
        xs.attrs['version_xsarslc'] = xsarslc.__version__
        xs.attrs['processor'] = __file__
        xs.attrs['generation_date'] = datetime.datetime.today().strftime('%Y-%b-%d')
        if not os.path.exists(os.path.dirname(output_filename)):
            os.makedirs(os.path.dirname(output_filename), 0o0775)
            logging.info('makedir %s', os.path.dirname(output_filename))
        xs.attrs['footprint'] = str(xs.attrs['footprint'])
        xs.attrs['tile_width_sample'] = str(xs.attrs['tile_width_sample'].values)
        xs.attrs['multidataset'] = str(xs.attrs['multidataset'])
        xs.to_netcdf(output_filename)
        logging.info('successfuly written %s', output_filename)
    else:
        logging.info('no xspectra available in this subswath')


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    main()
