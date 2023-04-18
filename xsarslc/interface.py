#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
from xsarslc.tools import xndindex
import warnings
import xsar
import argparse

def get_low_res_tiles_from_L1BSLC(file_path, xspectra = 'intra', posting = {'sample':400,'line':400}, tile_width = {'sample':17600.,'line':17600.}, window='GAUSSIAN', **kwargs):
    """
    compute low resolution sigma0 tiles from L1B SLC product
    Args:
        file_path (str): path to the L1B SLC product
        xspectra (str): 'intra' or 'inter'
        posting (dict): Desired output posting. {name of dimension (str): spacing in [m] (float)}. 
        tile_width (dict): form {name of dimension (str): width in [m] (float)}. Desired width of the output tile (should be smaller or equal than provided data)
        window (str, optional): Name of window used to smooth out the data. 'GAUSSIAN' and 'RECT' are valid entries
    keyword Args:
        resolution (dict, optional): resolution for filter. default is twice the posting (Nyquist)
    Returns:
        (xarray.DataArray) low resolution tile with field "sigma0"
    """
    import datatree
    dt = datatree.open_datatree(file_path)
    L1B = dt[xspectra+'burst'].ds
    tiles = get_tiles_from_L1B_SLC(L1B)
    low_res_tiles = list()
    for mytile in tiles:
        mytile.load()
        if not mytile['land_flag']:
            incidence = mytile['incidence']
            spacing = {'sample':mytile['sampleSpacing']/np.sin(np.radians(incidence)), 'line':mytile['lineSpacing']}
            low_res_tiles.append(compute_low_res_tiles(mytile, spacing = spacing, posting = posting, tile_width=tile_width, window=window, **kwargs))
    res = xr.combine_by_coords([t.expand_dims(['burst', 'tile_sample', 'tile_line']) for t in low_res_tiles])
    res['land_flag'] = res['land_flag'].fillna(1).astype(bool)
    attrs = L1B.attrs.copy()
    attr_to_rm = ['comment','azimuth_time_interval','periodo_width_sample','periodo_width_line','periodo_overlap_sample','periodo_overlap_line']
    [attrs.pop(k,None) for k in attr_to_rm]
    res.attrs.update(attrs)
    return res

def compute_low_res_tiles(tile, spacing, posting, tile_width, resolution=None, window = 'GAUSSIAN'):
    """
    Compute low resolution tiles on defined ground spacing based on full resolution SLC tile.
    Code example:
    tiles = get_tiles_from_L1B_SLC(L1B)
    mytile = tiles[0]
    spacing = {'sample':mytile['sampleSpacing']/np.sin(np.radians(mytile['incidence'])), 'line':mytile['lineSpacing']}
    posting = {'sample':400,'line':400}
    tile_width = {'sample':17600.,'line':17600.}
    low_res_tile = compute_low_res_tiles(mytile, spacing = spacing, posting = posting, tile_width = tile_width)

    Args:
        tile (xarray.DataArray) : A tile dataArray (list element generated with get_tiles_from_L1B_SLC()).
        spacing (dict): GROUND spacing of provided tile. {name of dimension (str): spacing in [m] (float)}. 
        posting (dict): Desired output posting. {name of dimension (str): spacing in [m] (float)}. 
        tile_width (dict): form {name of dimension (str): width in [m] (float)}. Desired width of the output tile (should be smaller or equal than provided data)
        resolution (dict, optional): resolution for filter. default is twice the posting (Nyquist)
        window (str, optional): Name of window used to smooth out the data. 'GAUSSIAN' and 'RECT' are valid entries
    Returns:
        (xarray.Dataset) : dataset of filtered/resampled sigma0
    """
    from scipy.signal import fftconvolve
    from xsarslc.tools import gaussian_kernel, rect_kernel
    
    if resolution is None:
        resolution = {d:2*v for d,v in posting.items()}

    sigma0 = tile['sigma0']
    mask = np.isfinite(sigma0)
    if window.upper() == 'GAUSSIAN':
        kernel_filter = gaussian_kernel(width=resolution, spacing=spacing)
    elif window.upper() == 'RECT':
        kernel_filter = rect_kernel(width=resolution, spacing=spacing)
    else:
        raise ValueError('Unknown window: {}'.format(window))
    swap_dims = {d: d + '_' for d in resolution.keys()}
    kernel_filter = kernel_filter.rename(swap_dims)

    low_pass = xr.apply_ufunc(fftconvolve, sigma0.where(mask, 0.), kernel_filter,
                                        input_core_dims=[resolution.keys(), swap_dims.values()], vectorize=True,
                                        output_core_dims=[resolution.keys()], kwargs={'mode': 'same'}, dask='allowed')

    normal = xr.apply_ufunc(fftconvolve, mask, kernel_filter, input_core_dims=[resolution.keys(), swap_dims.values()],
                            vectorize=True, output_core_dims=[resolution.keys()], kwargs={'mode': 'same'}, dask='allowed')

    low_pass = low_pass / normal
    
    # ------- decimate -------
    Np = {d:np.rint(tile_width[d]/posting[d]).astype(int) for d in tile_width.keys()}
    new_line = xr.DataArray(int(low_pass['line'].isel(line=low_pass.sizes['line']//2))+np.arange(-Np['line']//2,Np['line']//2)*float(posting['line']/spacing['line']), dims='azimuth')
    new_sample = xr.DataArray(int(low_pass['sample'].isel(sample=low_pass.sizes['sample']//2))+np.arange(-Np['sample']//2,Np['sample']//2)*float(posting['sample']/spacing['sample']), dims='range')
    decimated = low_pass.interp(sample=new_sample, line=new_line, assume_sorted=True).rename('sigma0')
    corner_lat = tile['corner_latitude'].interp(c_sample = decimated['sample'][[0,-1]], c_line = decimated['line'][[0,-1]]).rename({'range':'c_range', 'azimuth':'c_azimuth'})
    corner_lon = tile['corner_longitude'].interp(c_sample = decimated['sample'][[0,-1]], c_line = decimated['line'][[0,-1]]).rename({'range':'c_range', 'azimuth':'c_azimuth'})
    decimated = decimated.drop_vars(['line','sample'])
    decimated.attrs.update(sigma0.attrs)
    corner_lat = corner_lat.drop_vars(['line','sample', 'c_line', 'c_sample'])
    corner_lon = corner_lon.drop_vars(['line','sample', 'c_line', 'c_sample'])
    range_spacing = xr.DataArray(posting['sample'], attrs={'units':'m', 'long_name':'ground range spacing'}, name='range_spacing')
    azimuth_spacing = xr.DataArray(posting['line'], attrs={'units':'m', 'long_name':'azimuth spacing'}, name='azimuth_spacing')
    added_variables = [tile[v].to_dataset() for v in ['incidence','ground_heading','land_flag']] # add variables from L1B to output
    decimated = xr.merge([decimated.to_dataset(),corner_lat.to_dataset(), corner_lon.to_dataset(), range_spacing.to_dataset(), azimuth_spacing.to_dataset(), *added_variables])
    decimated = decimated.transpose('azimuth', 'range', 'c_azimuth', 'c_range', ...)
    return decimated

def get_tiles_from_L1B_SLC(L1B, polarization=None):
    """
    Return list of tiles (sigma0) based on L1B informations. Open original SLC file, extract DN and calibrate sigma0
    Args:
        L1B (xarray.dataset): intraburst-like L1B SLC dataset
        polarization (str, optional) : polarization. Default is the one found in L1B
    Returns:
        (list): list of tiles with calibrated sigma0
    """
    slc_path = L1B.attrs.get('name')
    tiles_index = get_tiles_index_from_L1B_SLC(L1B)
    dt = xsar.open_datatree(slc_path)
    if polarization is None:
        if np.asarray(L1B['pol']).size>1:
            raise ValueError('More than one polarization found in provided L1B. Please, specify polarization explicitely')
        else:
            polarization = L1B['pol'].item()
    DN = dt['measurement']['digital_number'].sel(pol=polarization)
    sigma0_lut = dt['calibration']['sigma0_lut'].sel(pol=polarization)
    range_noise_lut = dt['noise_range'].ds['noise_lut'].sel(pol=polarization)
    azimuth_noise_lut = dt['noise_azimuth'].ds['noise_lut'].sel(pol=polarization)
    sample_spacing = dt['measurement']['sampleSpacing']
    line_spacing = dt['measurement']['lineSpacing']
    DN_tiles = new_get_tiles(DN, tiles_index)

    sigma0 = list()
    for DN in DN_tiles:
        range_noise_lut_mytile = range_noise_lut.interp_like(DN, assume_sorted=True)
        if np.any(np.isnan(range_noise_lut_mytile)):
            mid_burst_line = int(dt['bursts']['linesPerBurst'].item()*(DN['burst'].item()+0.5))
            range_noise_lut_mytile = range_noise_lut.sel(line=mid_burst_line, method='nearest').drop_vars('line') # taking closest line
            range_noise_lut_mytile = range_noise_lut_mytile.interp_like(DN, assume_sorted=True)
        noise = (azimuth_noise_lut.interp_like(DN, assume_sorted=True))*range_noise_lut_mytile
        calibrated_DN = ((np.abs(DN)**2-noise)/((sigma0_lut.interp_like(DN, assume_sorted=True))**2)).rename('sigma0')
        calibrated_DN.attrs.update({'long_name': 'calibrated sigma0', 'units': 'linear'})
        tile_coords = {'burst':calibrated_DN['burst'], 'tile_line':calibrated_DN['tile_line'], 'tile_sample':calibrated_DN['tile_sample']}
        calibrated_DN = calibrated_DN.assign_coords({'longitude':L1B['longitude'].sel(tile_coords).item(),'latitude':L1B['latitude'].sel(tile_coords).item()})
        added_variables = [L1B[v].sel(tile_coords).to_dataset() for v in ['incidence','corner_longitude', 'corner_latitude', 'ground_heading','land_flag']] # add variables from L1B to output
        calibrated_DN = xr.merge([calibrated_DN,*added_variables, sample_spacing.to_dataset(), line_spacing.to_dataset()])
        calibrated_DN = calibrated_DN.assign_coords({'c_sample':L1B.sel(tile_coords)['corner_sample'].data, 'c_line':L1B.sel(tile_coords)['corner_line'].data})
        sigma0.append(calibrated_DN)
    return sigma0

def get_tiles_index_from_L1B_SLC(L1B):
    """
    A tile indexer designed to slice original SLC measurement dataset based on corner information of L1B information.

    Args:
        L1B (xarray.dataset): intraburst-like L1B SLC dataset
    Returns:
        (dict):indexer for slicing tiles into original measurement dataset
    """
    corners2slice = lambda s:slice(s[0],s[1])
    line_slices = xr.apply_ufunc(corners2slice,L1B['corner_line'], input_core_dims=[['c_line']], vectorize=True)
    sample_slices = xr.apply_ufunc(corners2slice,L1B['corner_sample'], input_core_dims=[['c_sample']], vectorize=True)
    return {'sample':sample_slices, 'line':line_slices}

def new_get_tiles(ds, tiles_index):
    """
    Returns the list of all tiles taken over tiles_index. tiles_index can be generated using xtiling (uniform tiling) or get_tiles_index_from_L1B_SLC().
    Args:
        ds (xarray.dataset/xarray.DataArray) : dataset to slice
        tiles_index (dict): keys are dimensions of ds to be indexeded. Values are xarray.DataArray containing slices (or indexes)
    Returns
        (list) : a list of all tiles
    
    """
    alldims = set.union(*[set(v.dims) for v in tiles_index.values()])
    allcoords = {d:set() for d in alldims} # do not use dict.fromkeys()
    for dim_index in tiles_index.keys():
        for dim in alldims:
            if dim in tiles_index[dim_index].dims:
                allcoords[dim] = set.union(allcoords[dim],set(tiles_index[dim_index][dim].data))
    allcoords = {k:list(v) for k,v in allcoords.items()} # changing sets into lists
    allcoords_sizes = {k:len(v) for k,v in allcoords.items()}

    tiles_list = list()
    for c in xndindex(allcoords_sizes):
        r = {d:allcoords[d][i] for d,i in c.items()}
        indexer = dict()
        for dim_index in tiles_index.keys():
            rd = {k:v for k,v in r.items() if k in tiles_index[dim_index].dims}
            indexer.update({dim_index:tiles_index[dim_index].sel(rd).item()})
        tiles_list.append(ds[indexer].assign_coords(r))
    return tiles_list



if __name__ == '__main__':
    import datatree
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', action="store", help="L1B SLC file path ")
    args = parser.parse_args()
    file_path = args.file
    dt = datatree.open_datatree(file_path)
    L1B = dt['interburst_xspectra']
    tiles = get_tiles_from_L1B_SLC(L1B)
    posting = {'sample':400,'line':400}
    tile_width = {'sample':17600.,'line':17600.}
    low_res_tiles = list()
    for mytile in tiles:
        mytile.load()
        incidence = mytile['incidence']
        spacing = {'sample':mytile['sampleSpacing']/np.sin(np.radians(incidence)), 'line':mytile['lineSpacing']}
        low_res_tiles.append(compute_low_resolution(mytile, spacing = spacing, posting = posting, tile_width=tile_width))
    res = xr.combine_by_coords([t.expand_dims(['burst', 'tile_sample', 'tile_line']) for t in low_res_tiles])
