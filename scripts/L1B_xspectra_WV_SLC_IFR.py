# -*- coding: utf-8 -*-
"""
A. Grouazel
23 Jan 2023
purpose: produce nc files from SAFE WV SLC containing cartesian x-spec computed with xsar and xsarsea
"""

import xsarslc.processing.xspectra as proc
from xsarslc.processing import xspectra
from xsarslc.tools import netcdf_compliant
import warnings
import xsar
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
import numpy as np
import datetime
import logging
import os
import time
import xsarslc
import pdb
from get_RI_file import get_IR_file
PRODUCT_VERSION = '0.10' # see https://github.com/umr-lops/xsar_slc/wiki/processings
def get_memory_usage():
    try:
        import resource
        memory_used_go = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000./1000.
    except: #on windows resource is not usable
        import psutil
        memory_used_go = psutil.virtual_memory().used / 1000 / 1000 / 1000.
    str_mem = 'RAM usage: %1.1f Go'%memory_used_go
    return str_mem

def generate_WV_L1Bxspec_product(slc_wv_path,output_filename, polarization=None,dev=False):
    """

    :param tiff: str full path
    :param output_filename : str full path
    :param polarization : str : VV VH HH HV [optional]
    :apram dev: bool: allow to shorten the processing
    :return:
    """
    safe = os.path.dirname(os.path.dirname(slc_wv_path))
    logging.info('start loading the datatree %s', get_memory_usage())
    imagette_number = os.path.basename(slc_wv_path).split('-')[-1].replace('.tiff', '')
    str_gdal = 'SENTINEL1_DS:%s:WV_%s' % (safe, imagette_number)
    bu = xsar.Sentinel1Meta(str_gdal)._bursts
    chunksize = {'line': 6000, 'sample': 7000}
    xsarobj = xsar.Sentinel1Dataset(str_gdal, chunks=chunksize)
    dt = xsarobj.datatree
    dt.load() #took ?min to load and ? Go RAM
    logging.info('datatree loaded %s',get_memory_usage())
    unit = safe[0:3]
    subswath = str_gdal.split(':')[2]
    subswath = dt['image'].ds['swath_subswath'].values
    IR_dir = '/home/datawork-cersat-public/project/sarwave/data/products/developments/aux_files/sar/impulse_response/'
    IR_path = get_IR_file(unit, subswath, polarization.upper(), auxdir=IR_dir)
    xs0 = proc.compute_WV_intraburst_xspectra(dt=dt,
                                         polarization=polarization, periodo_width={"line": 2000, "sample": 2000},
                                         periodo_overlap={"line": 1000, "sample": 1000},IR_path=IR_path)
    xs = xs0.swap_dims({'freq_line': 'k_az', 'freq_sample': 'k_rg'})
    xs = xspectra.symmetrize_xspectrum(xs, dim_range='k_rg', dim_azimuth='k_az')
    xs = netcdf_compliant(xs) # to split complex128 variables into real and imag part
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
            os.makedirs(os.path.dirname(output_filename),0o0775)
            logging.info('makedir %s',os.path.dirname(output_filename))
        xs.to_netcdf(output_filename)
        logging.info('successfuly written %s', output_filename)
    else:
        logging.info('no xspectra available in this subswath')


if __name__ == '__main__':
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    import argparse
    time.sleep(np.random.rand(1, 1)[0][0])  # to avoid issue with mkdir
    parser = argparse.ArgumentParser(description='L1BwaveIFR_WV_SLC')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite the existing outputs [default=False]', required=False)
    parser.add_argument('--tiff', required=True, help='tiff file full path WV SLC')
    parser.add_argument('--outputdir', required=True, help='directory where to store output netCDF files')
    parser.add_argument('--dev', action='store_true', default=False,help='dev mode stops the computation early')
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
    subswath_number = os.path.basename(slc_wv_path).split('-')[1]
    polarization_from_file = os.path.basename(slc_wv_path).split('-')[3].upper()
    subsath_nickname = '%s_%s' % (subswath_number, polarization_from_file)
    safe_basename = os.path.basename(os.path.dirname(os.path.dirname(slc_wv_path)))
    output_filename = os.path.join(args.outputdir,safe_basename, os.path.basename(
        slc_wv_path).replace('.tiff','') + '_L1B_xspec_IFR_' + PRODUCT_VERSION + '.nc')
    logging.info('mode dev is %s',args.dev)
    logging.info('output filename would be: %s',output_filename)
    if os.path.exists(output_filename) and args.overwrite is False:
        logging.info('%s already exists', output_filename)
    else:
        generate_WV_L1Bxspec_product(slc_wv_path=slc_wv_path,output_filename=output_filename, dev=args.dev,
                                     polarization=polarization_from_file)
    logging.info('peak memory usage: %s Mbytes', get_memory_usage())
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)
