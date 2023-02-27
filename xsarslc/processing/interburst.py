#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
import logging
from scipy.constants import c as celerity
from xsarslc.tools import xtiling, xndindex


def tile_bursts_overlap_to_xspectra(burst0, burst1, geolocation_annotation, calibration, tile_width, tile_overlap,
                                    lowpass_width={'sample': 1000., 'line': 1000.},
                                    periodo_width={'sample': 2000., 'line': 1200.}, #2000 1200 en 20km# 1800 1200 en 2km
                                    periodo_overlap={'sample': 1000., 'line': 600.}, **kwargs):
    """
    Divide bursts overlaps in tiles and compute inter-burst cross-spectra using compute_interburst_xspectrum() function.

    Args:
        burst0 (xarray.Dataset): first burst (in time) dataset (No need of deramped digital number variable)
        burst1 (xarray.Dataset): second burst (in time) dataset (No need of deramped digital number variable)
        geolocation_annotation (xarray.Dataset): dataset of geolocation annotation
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        azimuth_steering_rate (float) : antenna azimuth steering rate [deg/s]
        azimuth_time_interval (float) : azimuth time spacing [s]
        lowpass_width (dict): width for low pass filtering [m]. Dict of form {dim_name (str): width (float)}
    
    Keyword Args:
        kwargs: keyword arguments passed to compute_interburst_xspectrum()
    """
    from xsarslc.tools import get_tiles, get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation
    from xsarslc.processing.xspectra import compute_modulation, compute_azimuth_cutoff, compute_normalized_variance, compute_mean_sigma0

    # find overlapping burst portion

    az0 = burst0['time'].load()
    az1 = burst1['time'][{'line': 0}].load()

    # az0 = burst0[{'sample':0}].azimuth_time.load()
    # az1 = burst1.isel(sample=0).azimuth_time[{'line':0}].load()

    frl = np.argwhere(az0.data >= az1.data)[0].item()  # first overlapping line of first burst
    # Lines below ensures we choose the closest index since azimuth_time are not exactly the same
    t0 = burst0[{'sample': 0, 'line': frl}].time
    t1 = burst1[{'sample': 0, 'line': 0}].time
    aziTimeDiff = np.abs(t0 - t1)

    if np.abs(burst0[{'sample': 0, 'line': frl - 1}].time - t1) < aziTimeDiff:
        frl -= 1
    elif np.abs(burst0[{'sample': 0, 'line': frl + 1}].time - t1) < aziTimeDiff:
        frl += 1
    else:
        pass

    burst0 = burst0[{'line': slice(frl, None)}]
    burst1 = burst1[{'line': slice(None, burst0.sizes['line'])}]

    # if overlap0.sizes!=overlap1.sizes:
    #     raise ValueError('Overlaps have different sizes: {} and {}'.format(overlap0.sizes, overlap1.sizes))

    burst0.load()  # loading ensures efficient tiling below
    burst1.load()  # loading ensures efficient tiling below

    burst = burst0  # reference burst for geolocation

# -----------------------------------------------------------------------------------


    azitime_interval = burst.attrs['azimuth_time_interval']
    azimuth_spacing = float(burst['lineSpacing'])

    if tile_width:
        nperseg_tile = {'line':int(np.rint(tile_width['line'] / azimuth_spacing))}
    else:
        nperseg_tile = {'line':burst.sizes['line']}
        tile_width = {'line':nperseg_tile['line']*azimuth_spacing}


    if tile_overlap in (0., None):
        tile_overlap = {'sample': 0., 'line': 0.}
        noverlap_tile = {'line': 0}
    else:
        noverlap_tile = {'line': int(np.rint(tile_overlap['line'] / azimuth_spacing))}  # np.rint is important for homogeneity of point numbers between bursts

    if np.any([tile_width[d]<periodo_width[d] for d in tile_width.keys()]):
        warnings.warn("One or all periodogram widths are larger than tile widths. Exceeding periodogram widths are reset to match tile width.")

    for d in tile_width.keys():
        periodo_width[d] = min(periodo_width[d], tile_width[d])

    if np.any([periodo_overlap[d]>0.5*periodo_width[d] for d in periodo_width.keys()]):
        warnings.warn("Periodogram overlap should not exceed half of the periodogram width.")

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
    all_tiles_0 = get_tiles(burst0, tiles_index)
    all_tiles_1 = get_tiles(burst1, tiles_index)

    # ---------Computing quantities at tile middle locations --------------------------
    tiles_middle = get_middle_tile(tiles_index)
    middle_sample = burst['sample'][{'sample': tiles_middle['sample']}]
    middle_line = burst['line'][{'line': tiles_middle['line']}]
    middle_lons = FullResolutionInterpolation(middle_line, middle_sample, 'longitude', geolocation_annotation,
                                              azitime_interval)
    middle_lats = FullResolutionInterpolation(middle_line, middle_sample, 'latitude', geolocation_annotation,
                                              azitime_interval)

    # ---------Computing quantities at tile corner locations  --------------------------
    tiles_corners = get_corner_tile(
        tiles_index)  # Having variables below at corner positions is sufficent for further calculations (and save memory space)
    corner_sample = burst['sample'][{'sample': tiles_corners['sample']}]
    corner_sample = corner_sample.stack(flats=corner_sample.dims)
    corner_line = burst['line'][{'line': tiles_corners['line']}]
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

    # --------------------------------------------------------------------------------------

    # tiles_corners = get_corner_tile(tiles_index)
    # corner_lon = burst['longitude'][tiles_corners].rename('corner_longitude').drop(['line','sample'])
    # corner_lat = burst['latitude'][tiles_corners].rename('corner_latitude').drop(['line','sample'])

    xs = list()  # np.empty(tuple(tiles_sizes.values()), dtype=object)
    for sub0, sub1 in zip(all_tiles_0, all_tiles_1):
    # for i in xndindex(tiles_sizes):
        sub0 = sub0.swap_dims({'__line':'line', '__sample':'sample'})
        sub1 = sub1.swap_dims({'__line':'line', '__sample':'sample'})
        mytile = {'tile_sample':sub0['tile_sample'], 'tile_line':sub0['tile_line']}

        # ------ checking if we are over water only ------
        if 'landmask' in kwargs:
            tile_lons = [float(corner_lons.sel(mytile)[{'c_line': j, 'c_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            tile_lats = [float(corner_lats.sel(mytile)[{'c_line': j, 'c_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            water_only = is_ocean((tile_lons, tile_lats), kwargs.get('landmask'))
        else:
            water_only = True
        logging.debug('water_only :  %s', water_only)
        # ------------------------------------------------
        if water_only:
            # sub0 = tiled_burst0[i].swap_dims({'__' + d: d for d in tile_width.keys()})
            # sub1 = tiled_burst1[i].swap_dims({'__' + d: d for d in tile_width.keys()})
            sub = sub0

            mean_incidence = float(corner_incs.sel(mytile).mean())
            mean_slant_range = float(corner_slantTimes.sel(mytile).mean()) * celerity / 2.
            slant_spacing = float(sub['sampleSpacing'])
            ground_spacing = slant_spacing / np.sin(np.radians(mean_incidence))
            azimuth_spacing = float(sub['lineSpacing'])

            periodo_spacing = {'sample': ground_spacing, 'line': azimuth_spacing}
            nperseg_periodo = {d: int(np.rint(periodo_width[d] / periodo_spacing[d])) for d in tile_width.keys()}
            noverlap_periodo = {d: int(np.rint(periodo_overlap[d] / periodo_spacing[d])) for d in tile_width.keys()}

            logging.debug("periodo_spacing %s",periodo_spacing)
            logging.debug("nperseg_periodo %s",nperseg_periodo)
            logging.debug("noverlap_periodo %s",noverlap_periodo)

            if np.any([sub0.sizes[d] < nperseg_periodo[d] for d in ['line', 'sample']]):
                raise ValueError(
                    'periodo_width ({}) is too large compared to available data (line : {} m, sample : {} m).'.format(
                        periodo_width, sub0.sizes['line'] * azimuth_spacing, sub0.sizes['sample'] * ground_spacing))

            mod0 = compute_modulation(np.abs(sub0['digital_number']), lowpass_width=lowpass_width,
                                      spacing={'sample': ground_spacing, 'line': azimuth_spacing})
            mod1 = compute_modulation(np.abs(sub1['digital_number']), lowpass_width=lowpass_width,
                                      spacing={'sample': ground_spacing, 'line': azimuth_spacing})

            xspecs = compute_interburst_xspectrum(mod0 ** 2, mod1 ** 2, mean_incidence, slant_spacing, azimuth_spacing,
                                                  nperseg=nperseg_periodo, noverlap=noverlap_periodo, **kwargs)
            xspecs_m = xspecs.mean(dim=['periodo_line', 'periodo_sample'],
                                   keep_attrs=True)  # averaging all the periodograms in each tile
            # ------------- tau ------------------
            antenna_velocity = np.radians(sub.attrs['azimuth_steering_rate']) * mean_slant_range
            ground_velocity = azimuth_spacing / sub.attrs['azimuth_time_interval']
            scan_velocity = (ground_velocity + antenna_velocity).item()
            dist0 = (sub0[{'line': sub0.sizes['line'] // 2}]['line'] - sub0['linesPerBurst'] * sub0[
                'burst']) * azimuth_spacing  # distance from begining of the burst
            dist1 = (sub1[{'line': sub1.sizes['line'] // 2}]['line'] - sub1['linesPerBurst'] * sub1[
                'burst']) * azimuth_spacing  # distance from begining of the burst
            tau = (sub1['sensingTime'] - sub0['sensingTime']) / np.timedelta64(1, 's') + (
                        dist1 - dist0) / scan_velocity  # The division by timedelta64(1,s) is to convert in seconds
            tau = xr.DataArray(float(tau), name='tau', attrs={'long_name': 'delay between two successive acquisitions', 'units': 's'})
            # ------------- cut-off --------------
            xs_cut = xspecs_m.swap_dims({'freq_sample': 'k_rg', 'freq_line': 'k_az'})
            cutoff = compute_azimuth_cutoff(xs_cut)
            # ------------- nv ------------
            nv = compute_normalized_variance(mod0)
            # ------------- mean sigma0 ------------
            sigma0 = compute_mean_sigma0(sub0['digital_number'], calibration['sigma0_lut'])
            # ------------- mean incidence ------------
            mean_incidence = xr.DataArray(mean_incidence, name='incidence', attrs={'long_name':'incidence at tile middle', 'units':'degree'})
            # ------------- concatenate all variables ------------
            xs.append(xr.merge([xspecs_m, tau.to_dataset(), cutoff.to_dataset(), mean_incidence.to_dataset(), nv.to_dataset(), sigma0.to_dataset()]))


    if not xs:  # All tiles are over land -> no xspectra available
        return
    
    # -------Returned xspecs have different shape in range (to keep same dk). Lines below only select common portions of xspectra-----
    Nfreq_min = min([x.sizes['freq_sample'] for x in xs])
    # line below rearange xs on (tile_sample, tile_line) grid and expand_dims ensures rearangment in combination by coords
    xs = xr.combine_by_coords(
        [x[{'freq_sample': slice(None, Nfreq_min)}].expand_dims(['tile_sample', 'tile_line']) for x in xs],
        combine_attrs='drop_conflicts')

    # -------Returned xspecs have different shape in range (to keep same dk). Lines below only select common portions of xspectra-----
    # Nfreq_min = min([xs[i].sizes['freq_sample'] for i in np.ndindex(xs.shape)])
    # for i in np.ndindex(xs.shape):
    # xs[i] = xs[i][{'freq_sample':slice(None,Nfreq_min)}]
    # ------------------------------------------------

    # xs = [list(a) for a in list(xs)] # must be generalized for larger number of dimensions
    # xs = xr.combine_nested(xs, concat_dim=tiles_sizes.keys(), combine_attrs='drop_conflicts')

    # taus.attrs.update({'long_name': 'delay between two successive acquisitions', 'units': 's'})
    # cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

    # tiles_middle = get_middle_tile(tiles_index)
    # middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    # middle_lat = burst['latitude'][tiles_middle].rename('latitude')
    # xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lons.to_dataset(), corner_lats.to_dataset()],
    #               combine_attrs='drop_conflicts')
    xs = xr.merge([xs, corner_lons.to_dataset(), corner_lats.to_dataset()], combine_attrs='drop_conflicts')
    
    xs = xs.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    xs.attrs.update(burst.attrs)
    xs.attrs.update({'tile_width_' + d: k for d, k in tile_width.items()})
    xs.attrs.update({'tile_overlap_' + d: k for d, k in tile_overlap.items()})
    xs.attrs.update({'periodo_width_' + d: k for d, k in periodo_width.items()})
    xs.attrs.update({'periodo_overlap_' + d: k for d, k in periodo_overlap.items()})
    return xs


def compute_interburst_xspectrum(mod0, mod1, mean_incidence, slant_spacing, azimuth_spacing, azimuth_dim='line',
                                 nperseg={'sample': 512, 'line': None}, noverlap={'sample': 256, 'line': 0},**kwargs):
    """
    Compute cross spectrum between mod0 and mod1 using a 2D Welch method (periodograms).
    
    Args:
        mod0 (xarray): modulation signal from burst0
        mod1 (xarray): modulation signal from burst1
        mean_incidence (float): mean incidence on slc
        slant_spacing (float): slant spacing
        azimuth_spacing (float): azimuth spacing
        azimuth_dim (str): name of azimuth dimension
        nperseg (dict of int): number of point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        noverlap (dict of int): number of overlapping point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        
    Returns:
        (xarray): concatenated cross_spectra
    """

    range_dim = list(set(mod0.dims) - set([azimuth_dim]))[0]  # name of range dimension
    freq_rg_dim = 'freq_' + range_dim
    freq_azi_dim = 'freq_' + azimuth_dim

    periodo_slices = xtiling(mod0, nperseg=nperseg, noverlap=noverlap, prefix='periodo_')

    periodo0 = mod0[periodo_slices]  # .swap_dims({'__'+d:d for d in [range_dim, azimuth_dim]})
    periodo1 = mod1[periodo_slices]  # .swap_dims({'__'+d:d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo0.sizes.items() if 'periodo_' in d}

    out = np.empty(tuple(periodo_sizes.values()), dtype=object)

    for i in xndindex(periodo_sizes):
        image0 = periodo0[i].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
        image1 = periodo1[i].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
        image0 = (image0 - image0.mean()) / image0.mean()
        image1 = (image1 - image1.mean()) / image1.mean()
        # xspecs = xr.DataArray(np.fft.fftshift(np.fft.fft2(image1)*np.conj(np.fft.fft2(image0))), dims=['freq_'+d for d in image0.dims])
        xspecs = xr.DataArray(np.fft.fft2(image1) * np.conj(np.fft.fft2(image0)),
                              dims=['freq_' + d for d in image0.dims])
        xspecs = xspecs[{freq_rg_dim: slice(None, xspecs.sizes[
            freq_rg_dim] // 2 + 1)}]  # keeping only half of the wavespectrum (positive wavenumbers)
        xspecs.data = np.fft.fftshift(xspecs.data,
                                      axes=xspecs.get_axis_num(freq_azi_dim))  # fftshifting azimuthal wavenumbers
        out[tuple(i.values())] = xspecs

    out = [list(a) for a in list(out)]  # must be generalized for larger number of dimensions
    out = xr.combine_nested(out, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts').rename(
        'xspectra')

    out = out.assign_coords(periodo0.drop(['line', 'sample']).coords)

    out.attrs.update({'long_name':'successive bursts overlap cross-spectra', 'mean_incidence': mean_incidence})

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(out.mean_incidence))
    k_rg = xr.DataArray(np.fft.rfftfreq(nperseg[range_dim], ground_range_spacing / (2 * np.pi)), dims=freq_rg_dim,
                        name='k_rg', attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[azimuth_dim], azimuth_spacing / (2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    # out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az}).swap_dims({'freq_'+range_dim:'k_rg', 'freq_'+azimuth_dim:'k_az'})
    out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az})
    # out = xr.merge([out, k_rg.to_dataset(), k_az.to_dataset()],
    #                combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})
    return out
