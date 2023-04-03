#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
import logging
from scipy.constants import c as celerity
from xsarslc.tools import xtiling, xndindex
import warnings
from tqdm import tqdm



def compute_HR_tile(dt, polarization, posting, resolution = None, **kwargs):
    """
    Main function to compute Hight Resolution (HR) map of cross-section
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to compute_HR_tile_per_burst(): landmask, truncate is valid entry
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.burst import crop_IW_burst, deramp_burst

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()
    nb_burst = dt['bursts'].sizes['burst']
    dev = kwargs.get('dev', False)
    
    if dev:
        logging.info('reduce number of burst -> 2')
        nb_burst = 1
        HRmap = list()
    for b in range(nb_burst):
        burst = crop_IW_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol=polarization)
        burst.load()
        sigma0 = get_calibrated_sigma0_from_burst(burst,  dt['calibration'], dt['noise_range'], dt['noise_azimuth'])
        burst = xr.merge([burst, sigma0], combine_attrs='drop_conflicts')
        burst.attrs.update(commons)
        burst_HRmap = compute_HR_tile_per_burst(burst, dt['geolocation_annotation'], dt['orbit'], dt['calibration'], dt['noise_range'], dt['noise_azimuth'], posting=posting, resolution=resolution, **kwargs)
        if burst_HRmap:
            HRmap.append(burst_HRmap.drop(['tile_line', 'tile_sample'])) # dropping coordinate is important to not artificially multiply the dimensions
    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    if HRmap:
        HRmap = xr.concat([x for x in HRmap], dim='burst')
        dims_to_transpose = [d for d in ['burst', 'tile_sample','tile_line', 'freq_sample','freq_line'] if d in HRmap.dims] # for homogeneous order of dimensions with interburst
        HRmap = HRmap.transpose(*dims_to_transpose, ...)
    return HRmap


def get_calibrated_sigma0_from_burst(burst, calibration, noise_range, noise_azimuth):
    sigma0_lut = calibration['sigma0_lut']
    range_noise_lut = noise_range['noise_lut']
    azimuth_noise_lut = noise_azimuth['noise_lut']
    polarization = burst.pol.item()
    sigma0_lut = sigma0_lut.sel(pol=polarization)
    range_noise_lut = range_noise_lut.sel(pol=polarization)
    azimuth_noise_lut = azimuth_noise_lut.sel(pol=polarization)
    noise = (azimuth_noise_lut.interp_like(burst, assume_sorted=True))*(range_noise_lut.interp_like(burst, assume_sorted=True))
    sigma0 = (np.abs(burst['digital_number'])**2-noise)/((sigma0_lut.interp_like(burst, assume_sorted=True))**2)
    sigma0 = sigma0.rename('sigma0')
    sigma0.attrs.update({'long_name': 'calibrated sigma0', 'units': 'linear'})
    return sigma0



def compute_HR_tile_per_burst(burst, geolocation_annotation, orbit, calibration, noise_range, noise_azimuth, posting = {'sample':1.e3, 'line':1.e3}, resolution = None, truncate = 3, **kwargs):

    """
    Divide burst in tiles and compute radar parameters at tile level

    Args:
        burst (xarray.Dataset): dataset with deramped digital number variable
        geolocation_annotation (xarray.Dataset): dataset of geolocation annotation
        orbit (xarray.Dataset): dataset of orbit annotation
        posting (dict): required posting. Dict {dim_name (str): width of tile [m](float)}
        resolution (dict): required resolution. Dict {dim_name (str): width of tile [m](float)}
        truncate (float, optional): how many standard deviation are used for filtering at required resolution
    
    Keyword Args:
        landmask (optional) : If provided, land mask passed to is_ocean().
    """
    from xsarslc.tools import get_tiles, get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation
    from xsarslc.processing.xspectra import compute_mean_sigma0

    if resolution is None:
        resolution = {d:2*p for d,p in posting.items()}

    # tile_width = {d:2*truncate*r for d,r in resolution.items()}
    tile_width = {d:r for d,r in resolution.items()}
    tile_overlap = {d:tile_width[d]-posting[d] for d in posting.keys()}

    azitime_interval = burst.attrs['azimuth_time_interval']
    azimuth_spacing = float(burst['lineSpacing'])

    if tile_width:
        nperseg_tile = {'line':int(np.rint(tile_width['line'] / azimuth_spacing))}
    else:
        nperseg_tile = {'line':burst.sizes['line']}
        tile_width = {'line':nperseg_tile['line']*azimuth_spacing}

    noverlap_tile = {'line': int(np.rint(tile_overlap['line'] / azimuth_spacing))}  # np.rint is important for homogeneity of point numbers between bursts


    # ------------- defining custom sample tiles_index because of non-constant ground range spacing -------
    incidenceAngle = FullResolutionInterpolation(burst['line'][{'line':slice(burst.sizes['line']//2, burst.sizes['line']//2+1)}], burst['sample'], 'incidenceAngle', geolocation_annotation, azitime_interval)
    cumulative_len = (float(burst['sampleSpacing'])*np.cumsum(1./np.sin(np.radians(incidenceAngle)))).rename('cumulative ground length').squeeze(dim='line')
    burst_width = cumulative_len[{'sample':-1}]
    tile_width.update({'sample':tile_width.get('sample',burst_width)})
    starts = np.arange(0.,burst_width,tile_width['sample']-tile_overlap['sample'])
    ends = starts+float(tile_width['sample'])
    starts = starts[ends<=float(burst_width)] # starting length restricted to available data
    ends = ends[ends<=float(burst_width)] # ending length restricted to available data
    istarts = np.searchsorted(cumulative_len,starts, side='right') # index of begining of tiles
    iends = np.searchsorted(cumulative_len,ends, side='left') # index of ending of tiles
    tile_sample = {'sample':xr.DataArray([slice(s,min(e+1,burst.sizes['sample'])) for s,e in zip(istarts,iends)], dims='tile_sample')}#, coords={'tile_sample':[(e+s)//2 for s,e in zip(istarts,iends)]})} # This is custom tile indexing along sample dimension to preserve constant tile width
    tile_sample_coords = get_middle_tile(tile_sample)
    tile_sample['sample'] = tile_sample['sample'].assign_coords({'tile_sample':burst['sample'][tile_sample_coords]})

    # ------------- defining regular line indexing --------
    tile_line = xtiling(burst['line'], nperseg=nperseg_tile, noverlap=noverlap_tile) # homogeneous tiling along line dimension can be done using xtiling()

    # ------------- customized indexes --------
    tiles_index = tile_sample.copy()
    tiles_index.update(tile_line)

    # ----- getting all tiles ------
    all_tiles = get_tiles(burst, tiles_index)


    # ---------Computing quantities at tile middle locations --------------------------
    tiles_middle = get_middle_tile(tiles_index) # this return the indexes, NOT the sample/line coord
    middle_sample = burst['sample'][{'sample': tiles_middle['sample']}]
    middle_line = burst['line'][{'line': tiles_middle['line']}]
    middle_lons = FullResolutionInterpolation(middle_line, middle_sample, 'longitude', geolocation_annotation,
                                              azitime_interval)
    middle_lats = FullResolutionInterpolation(middle_line, middle_sample, 'latitude', geolocation_annotation,
                                              azitime_interval)

    # ---------Computing quantities at tile corner locations  --------------------------
    tiles_corners = get_corner_tile(tiles_index) # returns index and not sample/line coordinates!
    # The two lines below can be called if longitude and latitude are already in burst dataset at full resolution
    # corner_lon = burst['longitude'][tiles_corners].rename('corner_longitude').drop(['line','sample'])
    # corner_lat = burst['latitude'][tiles_corners].rename('corner_latitude').drop(['line','sample'])

    # Having variables below at corner positions is sufficent for further calculations (and save memory space)
    corner_sample = burst['sample'][{'sample': tiles_corners['sample']}].rename('corner_sample')
    corner_sample = corner_sample.stack(flats=corner_sample.dims)
    corner_line = burst['line'][{'line': tiles_corners['line']}].rename('corner_line')
    corner_line = corner_line.stack(flatl=corner_line.dims)
    azitime_interval = burst.attrs['azimuth_time_interval']
    corner_lons = FullResolutionInterpolation(corner_line, corner_sample, 'longitude', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl']).rename(
        'corner_longitude').drop(['c_line', 'c_sample'])
    corner_lats = FullResolutionInterpolation(corner_line, corner_sample, 'latitude', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl']).rename(
        'corner_latitude').drop(['c_line', 'c_sample'])
    corner_incs = FullResolutionInterpolation(corner_line, corner_sample, 'incidenceAngle', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl'])
    corner_slantTimes = FullResolutionInterpolation(corner_line, corner_sample, 'slantRangeTime',
                                                    geolocation_annotation, azitime_interval).unstack(
        dim=['flats', 'flatl'])
    vel = np.sqrt(orbit['velocity_x'] ** 2 + orbit['velocity_y'] ** 2 + orbit['velocity_z'] ** 2)
    corner_time = burst['time'][{'line': tiles_corners['line']}]
    corner_velos = vel.interp(time=corner_time)

    # --------------------------------------------------------------------------------------






    res = list()  # np.empty(tuple(tiles_sizes.values()), dtype=object)
    # combinaison_selection_tiles = [yy for yy in xndindex(tiles_sizes)]
    combinaison_selection_tiles = all_tiles
    pbar = tqdm(range(len(all_tiles)), desc='start')
    for ii in pbar:
        pbar.set_description('loop on %s/%s tiles' % (ii+1,len(combinaison_selection_tiles)))
        sub = all_tiles[ii].swap_dims({'__line':'line', '__sample':'sample'})
        mytile = {'tile_sample':sub['tile_sample'], 'tile_line':sub['tile_line']}

        # ------ checking if we are over water only ------
        # if 'landmask' in kwargs:
        #     tile_lons = [float(corner_lons.sel(mytile)[{'c_line': j, 'c_sample': k}]) for j, k in
        #                  [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
        #     tile_lats = [float(corner_lats.sel(mytile)[{'c_line': j, 'c_sample': k}]) for j, k in
        #                  [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
        #     water_only = is_ocean((tile_lons, tile_lats), kwargs.get('landmask'))
        # else:
        #     water_only = True
        # logging.debug('water_only : %s', water_only)
        # ------------------------------------------------
        # if water_only:

        mean_incidence = float(corner_incs.sel(mytile).mean())

        # ------------- mean sigma0 ------------
        sigma0 = sub['sigma0'].mean(dim=['line','sample'])
        # ------------- mean incidence ------------
        mean_incidence = xr.DataArray(mean_incidence, name='incidence', attrs={'long_name':'incidence at tile middle', 'units':'degree'})
        # ------------- concatenate all variables ------------
        res.append(xr.merge([mean_incidence.to_dataset(), sigma0.to_dataset()]))

    if not res:  # All tiles are over land
        return
    
    # line below rearange res on (tile_sample, tile_line) grid and expand_dims ensures rearangment in combination by coords
    res = xr.combine_by_coords(
        [x.expand_dims(['tile_sample', 'tile_line']) for x in res],
        combine_attrs='drop_conflicts')

    # ------------------- Formatting returned dataset -----------------------------

    corner_sample = corner_sample.unstack(dim=['flats']).drop('c_sample')
    corner_line = corner_line.unstack(dim=['flatl']).drop('c_line')
    corner_sample.attrs.update({'long_name':'sample number in original digital number matrix'})
    corner_line.attrs.update({'long_name':'line number in original digital number matrix'})

    res = xr.merge([res, corner_lons.to_dataset(), corner_lats.to_dataset(), corner_line.to_dataset(), corner_sample.to_dataset()],
                  combine_attrs='drop_conflicts')
    res = res.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    res.attrs.update(burst.attrs)
    res.attrs.update({'tile_width_' + d: k for d, k in tile_width.items()})
    res.attrs.update({'tile_overlap_' + d: k for d, k in tile_overlap.items()})
    return res
