#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
import logging
from scipy.constants import c as celerity
from xsarslc.tools import xtiling, xndindex

import cartopy

cartopy.config['pre_existing_data_dir'] = '/home1/datahome/agrouaze/.local/share/cartopy'


def compute_subswath_xspectra(dt, **kwargs):
    """
    Main function to compute inter and intra burst spectra. It has to be modified to be able to change Xspectra options
    Keyword Args:
        kwargs (dict): keyword arguments passed to called functions. landmask, ...
    """
    import datatree
    import cartopy
    from xsarslc.tools import netcdf_compliant

    landmask = kwargs.pop('landmask', cartopy.feature.NaturalEarthFeature('physical', 'land', '10m'))

    intra_xs = compute_subswath_intraburst_xspectra(dt, landmask=landmask, **kwargs)

    intra_xs = intra_xs.drop('spatial_ref')
    intra_xs.attrs.update({'start_date': str(intra_xs.start_date)})
    intra_xs.attrs.update({'stop_date': str(intra_xs.stop_date)})
    intra_xs.attrs.update({'footprint': str(intra_xs.footprint)})
    intra_xs.attrs.pop('pixel_line_m')
    intra_xs.attrs.pop('pixel_sample_m')

    inter_xs = compute_subswath_interburst_xspectra(dt, landmask=landmask, **kwargs)

    inter_xs = inter_xs.drop('spatial_ref')
    inter_xs.attrs.update({'start_date': str(inter_xs.start_date)})
    inter_xs.attrs.update({'stop_date': str(inter_xs.stop_date)})
    inter_xs.attrs.update({'footprint': str(inter_xs.footprint)})
    inter_xs.attrs.pop('pixel_line_m')
    inter_xs.attrs.pop('pixel_sample_m')

    dt = datatree.DataTree.from_dict(
        {'interburst_xspectra': netcdf_compliant(inter_xs), 'intraburst_xspectra': netcdf_compliant(intra_xs)})
    return dt


def compute_subswath_intraburst_xspectra(dt, tile_width={'sample': 20.e3, 'line': 20.e3},
                                         tile_overlap={'sample': 10.e3, 'line': 10.e3}, **kwargs):
    """
    Compute IW subswath intra-burst xspectra per tile
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_burst_to_xspectra(), landmask can be added in kwargs
        
    Return:
        (xarray): xspectra.
    """

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()
    for b in range(dt['bursts'].sizes['burst']):
        burst = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol='VV')
        deramped_burst = deramp_burst(burst, dt)
        burst = xr.merge([burst, deramped_burst.drop('azimuthTime')], combine_attrs='drop_conflicts')
        burst.load()
        burst.attrs.update(commons)
        burst_xspectra = tile_burst_to_xspectra(burst, dt['geolocation_annotation'], dt['orbit'], tile_width,
                                                tile_overlap, **kwargs)
        if burst_xspectra:
            xspectra.append(burst_xspectra)#.drop(['tile_line', 'tile_sample']))

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    Nfreq_min = min([x.sizes['freq_sample'] for x in xspectra])
    xspectra = xr.combine_by_coords([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra],
                                    combine_attrs='drop_conflicts')  # rearange xs on burst
    # Nfreq_min = min([xs.sizes['freq_sample'] for xs in xspectra])
    # xspectra = [xs[{'freq_sample':slice(None, Nfreq_min)}] for xs in xspectra]
    # xspectra = xr.concat(xspectra, dim='burst')
    xspectra = xspectra.assign_coords({'k_rg': xspectra.k_rg, 'k_az': xspectra.k_az})  # move wavenumbers as coordinates
    return xspectra


def compute_subswath_interburst_xspectra(dt, tile_width={'sample': 20.e3, 'line': 20.e3},
                                         tile_overlap={'sample': 10.e3, 'line': 10.e3}, **kwargs):
    """
    Compute IW subswath inter-burst xspectra. No deramping is applied since only magnitude is used.
    
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Note: The overlap is short in azimuth (line) direction. Keeping nperseg = {'line':None} in Xspectra computation
    keeps maximum number of point in azimuth but is not ensuring the same number of overlapping point for all burst
    
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_bursts_overlap_to_xspectra()
        
    Return:
        (xarray): xspectra.
    """

    commons = {'azimuth_steering_rate': dt['image']['azimuthSteeringRate'].item(),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()
    for b in range(dt['bursts'].sizes['burst'] - 1):
        burst0 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True,
                            merge_burst_annotation=True).sel(pol='VV')
        burst1 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b + 1, valid=True,
                            merge_burst_annotation=True).sel(pol='VV')
        burst0.attrs.update(commons)
        burst1.attrs.update(commons)
        interburst_xspectra = tile_bursts_overlap_to_xspectra(burst0, burst1, dt['geolocation_annotation'], tile_width,
                                                              tile_overlap, **kwargs)
        if interburst_xspectra:
            xspectra.append(interburst_xspectra)#.drop(['tile_line', 'tile_sample']))

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    Nfreq_min = min([x.sizes['freq_sample'] for x in xspectra])
    xspectra = xr.combine_by_coords([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra],
                                    combine_attrs='drop_conflicts')  # rearange xs on burst
    # Nfreq_min = min([xs.sizes['freq_sample'] for xs in xspectra])
    # xspectra = [xs[{'freq_sample':slice(None, Nfreq_min)}] for xs in xspectra]
    # xspectra = xr.concat(xspectra, dim='burst')
    xspectra = xspectra.assign_coords({'k_rg': xspectra.k_rg, 'k_az': xspectra.k_az})  # move wavenumbers as coordinates
    return xspectra


def tile_burst_to_xspectra(burst, geolocation_annotation, orbit, tile_width, tile_overlap,
                           lowpass_width={'sample': 1000., 'line': 1000.},
                           periodo_width={'sample': 2000., 'line': 4000.},
                           periodo_overlap={'sample': 1000., 'line': 2000.}, **kwargs):
    """
    Divide burst in tiles and compute intra-burst cross-spectra using compute_intraburst_xspectrum() function.

    Args:
        burst (xarray.Dataset): dataset with deramped digital number variable
        geolocation_annotation (xarray.Dataset): dataset of geolocation annotation
        orbit (xarray.Dataset): dataset of orbit annotation
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        periodo_width (dict): approximative sizes of periodogram in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        periodo_overlap (dict): approximative sizes of periodogram overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        lowpass_width (dict): width for low pass filtering [m]. Dict of form {dim_name (str): width (float)}
    
    Keyword Args:
        landmask (optional) : If provided, land mask passed to is_ocean(). Otherwise xspectra are calculated by default
        kwargs: keyword arguments passed to compute_intraburst_xspectrum()
    """
    from xsarslc.tools import get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation

    burst.load()
    mean_ground_spacing = float(burst['sampleSpacing'] / np.sin(np.radians(burst.attrs['mean_incidence'])))
    azimuth_spacing = float(burst['lineSpacing'])
    spacing = {'sample': mean_ground_spacing, 'line': azimuth_spacing}

    nperseg_tile = {d: int(np.rint(tile_width[d] / spacing[d])) for d in tile_width.keys()}

    # print('mean_ground_spacing ',mean_ground_spacing)
    # print('azimuth_spacing ',azimuth_spacing)
    # print('spacing ',spacing)
    # print('nperseg_tile ',nperseg_tile)

    if tile_overlap in (0., None):
        noverlap = {d: 0 for k in nperseg_tile.keys()}
    else:
        noverlap = {d: int(np.rint(tile_overlap[d] / spacing[d])) for d in
                    tile_width.keys()}  # np.rint is important for homogeneity of point numbers between bursts

    # print('noverlap ',noverlap)
    # print('burst', burst)
    # print('burst sizes', burst.sizes)

    tiles_index = xtiling(burst, nperseg=nperseg_tile, noverlap=noverlap)
    tiled_burst = burst[tiles_index].drop(['sample', 'line']).swap_dims({'__' + d: d for d in tile_width.keys()})
    tiles_sizes = {d: k for d, k in tiled_burst.sizes.items() if 'tile_' in d}

    # ---------Computing quantities at tile middle locations --------------------------
    tiles_middle = get_middle_tile(tiles_index)
    # middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    # middle_lat = burst['latitude'][tiles_middle].rename('latitude')
    middle_sample = burst['sample'][{'sample': tiles_middle['sample']}]
    middle_line = burst['line'][{'line': tiles_middle['line']}]
    azitime_interval = burst.attrs['azimuth_time_interval']
    middle_lons = FullResolutionInterpolation(middle_line, middle_sample, 'longitude', geolocation_annotation,
                                              azitime_interval)
    middle_lats = FullResolutionInterpolation(middle_line, middle_sample, 'latitude', geolocation_annotation,
                                              azitime_interval)

    # ---------Computing quantities at tile corner locations  --------------------------
    tiles_corners = get_corner_tile(tiles_index)
    # The two lines below can be called if longitude and latitude are already in burst dataset at full resolution
    # corner_lon = burst['longitude'][tiles_corners].rename('corner_longitude').drop(['line','sample'])
    # corner_lat = burst['latitude'][tiles_corners].rename('corner_latitude').drop(['line','sample'])

    # Having variables below at corner positions is sufficent for further calculations (and save memory space)
    corner_sample = burst['sample'][{'sample': tiles_corners['sample']}]
    corner_sample = corner_sample.stack(flats=corner_sample.dims)
    corner_line = burst['line'][{'line': tiles_corners['line']}]
    corner_line = corner_line.stack(flatl=corner_line.dims)
    azitime_interval = burst.attrs['azimuth_time_interval']
    corner_lons = FullResolutionInterpolation(corner_line, corner_sample, 'longitude', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl']).rename(
        'corner_longitude')
    corner_lats = FullResolutionInterpolation(corner_line, corner_sample, 'latitude', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl']).rename(
        'corner_latitude')
    corner_incs = FullResolutionInterpolation(corner_line, corner_sample, 'incidenceAngle', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl'])
    corner_slantTimes = FullResolutionInterpolation(corner_line, corner_sample, 'slantRangeTime',
                                                    geolocation_annotation, azitime_interval).unstack(
        dim=['flats', 'flatl'])
    vel = np.sqrt(orbit['velocity_x'] ** 2 + orbit['velocity_y'] ** 2 + orbit['velocity_z'] ** 2)
    corner_time = burst['time'][{'line': tiles_corners['line']}]
    corner_velos = vel.interp(time=corner_time)
    # --------------------------------------------------------------------------------------

    xs = list()  # np.empty(tuple(tiles_sizes.values()), dtype=object)
    taus = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='tau')
    cutoff = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='cutoff')

    for i in xndindex(tiles_sizes):
        # ------ checking if we are over water only ------
        if 'landmask' in kwargs:
            tile_lons = [float(corner_lons[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            tile_lats = [float(corner_lats[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            water_only = is_ocean((tile_lons, tile_lats), kwargs.get('landmask'))
        else:
            water_only = True
        print('water_only : ', water_only)
        # ------------------------------------------------
        if water_only:
            # sub = tiled_burst[i].swap_dims({'n_line':'line','n_sample':'sample'})
            sub = tiled_burst[i]

            mean_incidence = float(corner_incs[i].mean())
            mean_slant_range = float(corner_slantTimes[i].mean()) * celerity / 2.
            mean_velocity = float(corner_velos[{'tile_line': i['tile_line']}].mean())

            # Below is old version when full resolution variables were systematically computed
            # mean_incidence = float(sub.incidence.mean())
            # mean_slant_range = float(sub.slant_range_time.mean())*celerity/2.
            # mean_velocity = float(sub.velocity.mean())

            slant_spacing = float(sub['sampleSpacing'])
            ground_spacing = slant_spacing / np.sin(np.radians(mean_incidence))
            periodo_spacing = {'sample': ground_spacing, 'line': azimuth_spacing}

            nperseg_periodo = {d: int(np.rint(periodo_width[d] / periodo_spacing[d])) for d in tile_width.keys()}
            noverlap_periodo = {d: int(np.rint(periodo_overlap[d] / periodo_spacing[d])) for d in tile_width.keys()}

            azimuth_spacing = float(sub['lineSpacing'])
            synthetic_duration = celerity * mean_slant_range / (
                        2 * burst.attrs['radar_frequency'] * mean_velocity * azimuth_spacing)
            mod = compute_modulation(sub['deramped_digital_number'], lowpass_width=lowpass_width,
                                     spacing={'sample': ground_spacing, 'line': azimuth_spacing})
            xspecs = compute_intraburst_xspectrum(mod, mean_incidence, slant_spacing, azimuth_spacing,
                                                  synthetic_duration, nperseg=nperseg_periodo,
                                                  noverlap=noverlap_periodo, **kwargs)
            xspecs_m = xspecs.mean(dim=['periodo_line', 'periodo_sample'],
                                   keep_attrs=True)  # averaging all the periodograms in each tile
            # xs[tuple(i.values())] = xspecs_m
            xs.append(xspecs_m)
            # ------------- tau ----------------
            taus[i] = float(xspecs.attrs['tau'])
            # ------------- cut-off ------------
            cutoff_tau = [str(i) + 'tau' for i in [1, 2, 3, 0] if str(i) + 'tau' in xspecs_m.dims][
                0]  # which tau is used to compute azimuthal cutoff
            k_rg = xspecs_m.k_rg
            k_az = xspecs_m.k_az
            xspecs_m = xspecs_m['xspectra_' + cutoff_tau].mean(dim=cutoff_tau)
            xspecs_m = xspecs_m.assign_coords({'k_rg': k_rg, 'k_az': k_az}).swap_dims(
                {'freq_sample': 'k_rg', 'freq_line': 'k_az'})
            cutoff[i] = compute_azimuth_cutoff(xspecs_m)

    if xs:  # at least one existing xspectra has been calculated
        # -------Returned xspecs have different shape in range (to keep same dk). Lines below only select common portions of xspectra-----
        # Nfreq_min = min([xs[i].sizes['freq_sample'] for i in np.ndindex(xs.shape)])
        Nfreq_min = min([x.sizes['freq_sample'] for x in xs])
        # line below rearange xs on (tile_sample, tile_line) grid and expand_dims ensures rearangment in combination by coords
        xs = xr.combine_by_coords(
            [x[{'freq_sample': slice(None, Nfreq_min)}].expand_dims(['tile_sample', 'tile_line']) for x in xs],
            combine_attrs='drop_conflicts')

        # for i in np.ndindex(xs.shape):
        # xs[i] = xs[i][{'freq_sample':slice(None,Nfreq_min)}]
        # ------------------------------------------------

        # xs = [list(a) for a in list(xs)] # must be generalized for larger number of dimensions
        # xs = xr.combine_nested(xs, concat_dim=tiles_sizes.keys(), combine_attrs='drop_conflicts')
        # xs = xs.assign_coords(tiles.coords)
        # tau
        taus.attrs.update({'long_name': 'delay between two successive looks', 'units': 's'})
        cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

        xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lons.to_dataset(), corner_lats.to_dataset()],
                      combine_attrs='drop_conflicts')
        xs = xs.assign_coords({'longitude': middle_lons,
                               'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
        xs.attrs.update(burst.attrs)
        xs.attrs.update({'tile_nperseg_' + d: k for d, k in nperseg_tile.items()})
        xs.attrs.update({'tile_noverlap_' + d: k for d, k in noverlap.items()})
    return xs


def burst_valid_indexes(ds):
    """
    Find indexes of valid portion of a burst. Returned line index are relative to burst only !
    
    Args:
        ds (xarray.Dataset): Dataset of one burst
    Return:
        (tuple of int): index of (first valid sample, last valid sample, first valid line, last valid line)
    """
    fvs = ds['firstValidSample']  # first valid samples
    valid_lines = np.argwhere(np.isfinite(fvs).data)  # valid lines
    fvl = int(valid_lines[0])  # first valid line
    lvl = int(valid_lines[-1])  # last valid line
    fvs = int(fvs.max(dim='line'))

    lvs = ds['lastValidSample']  # last valid samples
    valid_lines2 = np.argwhere(np.isfinite(lvs).data)  # valid lines
    if not np.all(valid_lines2 == valid_lines):
        raise ValueError("valid lines are not consistent between first and last valid samples")
    lvs = int(lvs.max(dim='line'))
    return fvs, lvs, fvl, lvl


def crop_burst(ds, burst_annotation, burst_number, valid=True, merge_burst_annotation=True):
    """
    Crop burst from the measurement dataset
    
    Args:
        ds (xarray.Dataset): measurement dataset
        burst_annotation (xarray.dataset): burst annotation dataset
        burst_number (int): burst number
        valid (bool, optional): If true: only return the valid part of the burst
        merge_burst_annotation (bool): If true: annotation of the burst are added to the returned dataset
        
    Return:
        xarray.Dataset : extraction of valid burst portion of provided datatree
    """

    lpb = int(burst_annotation['linesPerBurst'])

    if valid:
        fs, ls, fl, ll = burst_valid_indexes(
            burst_annotation.sel(burst=burst_number))  # first and last line are relative to burst
    else:
        fs, ls = None, None
        fl = 0  # relative to burst
        ll = lpb  # relative to burst

    myburst = ds[{'sample': slice(fs, ls, None), 'line': slice(burst_number * lpb + fl, burst_number * lpb + ll, None)}]

    if merge_burst_annotation:
        annotation = burst_annotation.sel(burst=burst_number)[{'line': slice(fl, ll, None)}]
        myburst = xr.merge([myburst, annotation])

    return myburst.assign_coords({'burst': burst_number})  # This ensures keeping burst number in coordinates


def deramp_burst(burst, dt):
    """
    Deramp burst. Return deramped digital numbers
    
    Args:
        burst (xarray.dataArray or xarray.Dataset): burst or portion of a burst
        dt (xarray.dataTree): datatree containing all informations of the SLC
    Return:
        (xarray.DataArray): deramped digital numbers
    """

    from xsarslc.deramping import compute_midburst_azimuthtime, compute_slant_range_time, compute_Doppler_centroid_rate, \
        compute_reference_time, compute_deramping_phase, compute_DopplerCentroid_frequency

    FMrate = dt['FMrate'].ds
    dcEstimates = dt['doppler_estimate'].ds
    orbit = dt['orbit'].ds
    radar_frequency = float(dt['image']['radarFrequency'])
    azimuth_steering_rate = float(dt['image']['azimuthSteeringRate'])
    azimuth_time_interval = float(dt['image']['azimuthTimeInterval'])

    midburst_azimuth_time = compute_midburst_azimuthtime(burst, azimuth_time_interval)  # mid burst azimuth time
    slant_range_time = compute_slant_range_time(burst, dt['image']['slantRangeTime'], dt['image']['rangeSamplingRate'])

    kt = compute_Doppler_centroid_rate(orbit, azimuth_steering_rate, radar_frequency, FMrate, midburst_azimuth_time,
                                       slant_range_time)
    fnc = compute_DopplerCentroid_frequency(dcEstimates, midburst_azimuth_time, slant_range_time)
    eta_ref = compute_reference_time(FMrate, dcEstimates, midburst_azimuth_time, slant_range_time,
                                     int(burst['samplesPerBurst']))
    phi = compute_deramping_phase(burst, kt, eta_ref, azimuth_time_interval)

    with xr.set_options(keep_attrs=True):
        deramped_signal = (burst['digital_number'] * np.exp(-1j * phi)).rename('deramped_digital_number')

    return deramped_signal


def compute_modulation(ds, lowpass_width, spacing):
    """
    Compute modulation map (sig0/low_pass_filtered_sig0)

    Args:
        ds (xarray) : array of (deramped) digital number
        lowpass_width (dict): form {name of dimension (str): width in [m] (float)}. width for low pass filtering [m]
        spacing (dict): form {name of dimension (str): spacing in [m] (float)}. spacing for each filtered dimension


    """
    from scipy.signal import fftconvolve
    from xsarslc.tools import gaussian_kernel

    # ground_spacing = float(ds['sampleSpacing'])/np.sin(np.radians(ds['incidence'].mean()))

    mask = np.isfinite(ds)
    gk = gaussian_kernel(width=lowpass_width, spacing=spacing)
    swap_dims = {d: d + '_' for d in lowpass_width.keys()}
    gk = gk.rename(swap_dims)

    low_pass_intensity = xr.apply_ufunc(fftconvolve, np.abs(ds.where(mask, 0.)) ** 2, gk,
                                        input_core_dims=[lowpass_width.keys(), swap_dims.values()], vectorize=True,
                                        output_core_dims=[lowpass_width.keys()], kwargs={'mode': 'same'})

    normal = xr.apply_ufunc(fftconvolve, mask, gk, input_core_dims=[lowpass_width.keys(), swap_dims.values()],
                            vectorize=True, output_core_dims=[lowpass_width.keys()], kwargs={'mode': 'same'})

    low_pass_intensity = low_pass_intensity / normal

    return ds / np.sqrt(low_pass_intensity)


def compute_intraburst_xspectrum(slc, mean_incidence, slant_spacing, azimuth_spacing, synthetic_duration,
                                 azimuth_dim='line', nperseg={'sample': 512, 'line': 512},
                                 noverlap={'sample': 256, 'line': 256}, **kwargs):
    """
    Compute SAR cross spectrum using a 2D Welch method. Looks are centered on the mean Doppler frequency
    If ds contains only one cycle, spectrum wavenumbers are added as coordinates in returned DataSet, otherwise, they are passed as variables (k_range, k_azimuth).
    
    Args:
        slc (xarray): digital numbers of Single Look Complex image.
        mean_incidence (float): mean incidence on slc
        slant_spacing (float): slant spacing
        azimuth_spacing (float): azimuth spacing
        synthetic_duration (float): synthetic aperture duration (to compute tau)
        azimuth_dim (str): name of azimuth dimension
        nperseg (dict of int): number of point per periodogram. Dict of form {dimension_name(str): number of point (int)}
        noverlap (dict of int): number of overlapping point per periodogram. Dict of form {dimension_name(str): number of point (int)}
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to compute_looks()
        
    Returns:
        (xarray): SLC cross_spectra
    """

    range_dim = list(set(slc.dims) - set([azimuth_dim]))[0]  # name of range dimension

    periodo_slices = xtiling(slc, nperseg=nperseg, noverlap=noverlap, prefix='periodo_')
    periodo = slc[periodo_slices].swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo.sizes.items() if 'periodo_' in d}

    out = np.empty(tuple(periodo_sizes.values()), dtype=object)

    for i in xndindex(periodo_sizes):
        image = periodo[i]
        xspecs = compute_looks(image, azimuth_dim=azimuth_dim, synthetic_duration=synthetic_duration,
                               **kwargs)  # .assign_coords(i)
        out[tuple(i.values())] = xspecs

    out = [list(a) for a in list(out)]  # must be generalized for larger number of dimensions
    out = xr.combine_nested(out, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts')
    # out = out.assign_coords(periodo_slices.coords)
    out = out.assign_coords(periodo.coords)

    out.attrs.update({'mean_incidence': mean_incidence})

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(out.mean_incidence))
    # k_rg = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(out.sizes['freq_'+range_dim], ground_range_spacing/(2*np.pi))), dims='freq_'+range_dim, name = 'k_rg', attrs={'long_name':'wavenumber in range direction', 'units':'rad/m'})
    k_rg = xr.DataArray(np.fft.rfftfreq(nperseg[range_dim], ground_range_spacing / (2 * np.pi)),
                        dims='freq_' + range_dim, name='k_rg',
                        attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(
        np.fft.fftfreq(out.sizes['freq_' + azimuth_dim], azimuth_spacing / (out.attrs.pop('look_width') * 2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    # out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az}).swap_dims({'freq_'+range_dim:'k_rg', 'freq_'+azimuth_dim:'k_az'})
    out = xr.merge([out, k_rg.to_dataset(), k_az.to_dataset()],
                   combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})

    return out.drop(['freq_' + range_dim, 'freq_' + azimuth_dim])


def compute_looks(slc, azimuth_dim, synthetic_duration, nlooks=3, look_width=0.2, look_overlap=0., look_window=None,
                  **kwargs):
    """
    Compute the N looks of an slc DataArray.
    Spatial coverage of the provided slc must be small enough to enure an almost constant ground spacing.
    Meaning: ground_spacing ~= slant_spacing / sin(mean_incidence)
    
    Args:
        slc (xarray.DataArray): (bi-dimensional) array to process
        azimuth_dim (str) : name of the azimuth dimension (dimension used to extract the look)
        nlooks (int): number of look
        look_width (float): in [0,1.] width of a look [percentage of full bandwidth] (nlooks*look_width must be < 1)
        look_overlap (float): in [0,1.] look overlapping [percentage of a look]
        look_window (str or tuple): instance that can be passed to scipy.signal.get_window()
    
    Return:
        (dict) : keys are '0tau', '1tau', ... and values are list of corresponding computed spectra
    """
    # import matplotlib.pyplot as plt
    import xrft

    if nlooks < 1:
        raise ValueError('Number of look must be greater than 0')
    if (nlooks * look_width) > 1.:
        raise ValueError('Number of look times look_width must be lower than 1.')

    range_dim = list(set(slc.dims) - set([azimuth_dim]))[0]  # name of range dimension
    freq_azi_dim = 'freq_' + azimuth_dim
    freq_rg_dim = 'freq_' + range_dim

    Np = slc.sizes[azimuth_dim]  # total number of point in azimuth direction
    nperlook = int(np.rint(look_width * Np))  # number of point perlook in azimuth direction
    noverlap = int(np.rint(look_overlap * look_width * Np))  # number of overlap point

    mydop = xrft.fft(slc, dim=[azimuth_dim], detrend=None, window=None, shift=True, true_phase=True,
                     true_amplitude=True)

    # Finding an removing Doppler centroid
    weight = xr.DataArray(np.hanning(100), dims=['window'])  # window for smoothing
    weight /= weight.sum()
    smooth_dop = np.abs(mydop).mean(dim=range_dim).rolling(**{freq_azi_dim: len(weight), 'center': True}).construct(
        'window').dot(weight)
    i0 = int(np.abs(mydop[freq_azi_dim]).argmin())  # zero frequency indice
    ishift = int(smooth_dop.argmax()) - i0  # shift of Doppler centroid
    mydop = mydop.roll(**{freq_azi_dim: -ishift, 'roll_coords': False})

    # Extracting the useful part of azimuthal Doppler spectrum
    # It removes points on each side to be sure that tiling will operate correctly
    Nused = nlooks * nperlook - (nlooks - 1) * noverlap
    left = (Np - Nused) // 2  # useless points on left side
    mydop = mydop[{freq_azi_dim: slice(left, left + Nused)}]
    look_tiles = xtiling(mydop, nperseg={freq_azi_dim: nperlook}, noverlap={freq_azi_dim: noverlap}, prefix='look_')

    if look_window is not None:
        raise ValueError('Look windowing is not available.')

    looks_spec = list()
    looks = mydop[look_tiles].drop(['freq_' + azimuth_dim]).swap_dims({'__' + d: d for d in ['freq_' + azimuth_dim]})
    looks_sizes = {d: k for d, k in looks.sizes.items() if 'look_' in d}

    # for l in range(look_tiles.sizes[freq_azi_dim]):
    for l in xndindex(looks_sizes):
        look = looks[l]
        look = xrft.ifft(look.assign_coords({freq_azi_dim: np.arange(-(nperlook // 2),
                                                                     -(nperlook // 2) + nperlook) * float(
            mydop[freq_azi_dim].spacing)}), dim=freq_azi_dim, detrend=None, window=False, shift=True, true_phase=False,
                         true_amplitude=True)
        look = np.abs(look) ** 2
        look = look / look.mean(dim=slc.dims)
        look = xrft.fft(look, dim=slc.dims, detrend='constant', true_phase=True, true_amplitude=True,
                        shift=False)  # shift=False is to keep zero at begnining for easier selection of half the spectrum
        look.data = np.fft.fftshift(look.data,
                                    axes=look.get_axis_num(freq_azi_dim))  # fftshifting azimuthal wavenumbers
        looks_spec.append(
            look[{freq_rg_dim: slice(None, look.sizes[freq_rg_dim] // 2 + 1)}])  # Only half of the spectrum is kept
        # looks_spec.append(xrft.fft(np.abs(look)**2, dim=slc.dims, detrend='linear'))

    looks_spec = xr.concat(looks_spec, dim='look')

    xspecs = {str(i) + 'tau': [] for i in range(nlooks)}  # using .fromkeys() do not work because of common empylist
    for l1 in range(nlooks):
        for l2 in range(l1, nlooks):
            df = float(looks_spec[{'look': l2}][freq_azi_dim].spacing * looks_spec[{'look': l2}][freq_rg_dim].spacing)
            xspecs[str(l2 - l1) + 'tau'].append(looks_spec[{'look': l2}] * np.conj(looks_spec[{'look': l1}]) * df)

    # compute tau = time difference between looks
    look_sep = look_width * (1. - look_overlap)
    tau = synthetic_duration * look_sep

    merged_xspecs = list()
    for i in range(nlooks):
        concat_spec = xr.concat(xspecs[str(i) + 'tau'], dim=str(i) + 'tau').rename('xspectra_{}tau'.format(i))
        concat_spec.attrs.update(
            {'nlooks': nlooks, 'look_width': look_width, 'look_overlap': look_overlap, 'look_window': str(look_window),
             'tau': tau})
        merged_xspecs.append(concat_spec.to_dataset())  # adding to_dataset() ensures promote_attrs=False per default

    merged_xspecs = xr.merge(merged_xspecs, combine_attrs='drop_conflicts')
    merged_xspecs.attrs.update({'look_width': look_width, 'tau': tau})
    return merged_xspecs


def compute_azimuth_cutoff(spectrum, definition='drfab'):
    """
    compute azimuth cutoff
    Args:
        spectrum (xarray): Xspectrum with coordinates k_rg and k_az
        definition (str, optional): ipf (covariance is averaged over range) or drfab (covariance taken at range = 0)
    Return:
        (float): azimuth cutoff [m]
    """
    import xrft
    from scipy.optimize import curve_fit

    if not np.any(spectrum['k_rg'] < 0.).item():  # only half spectrum with positive wavenumber has been passed
        spectrum = symmetrize_xspectrum(spectrum)

    coV = xrft.ifft(spectrum, dim=('k_rg', 'k_az'), shift=True, prefix='k_')
    coV = coV.assign_coords({'rg': 2 * np.pi * coV.rg, 'az': 2 * np.pi * coV.az})
    if definition == 'ipf':
        coVRm = coV.real.mean(dim='rg')
    elif definition == 'drfab':
        coVRm = np.real(coV).sel(rg=0.0)
    else:
        raise ValueError("Unknow definition '{}' for azimuth cutoff. It must be 'drfab' or 'ipf'".format(definition))
    coVRm /= coVRm.max()
    coVfit = coVRm.where(np.abs(coVRm.az) < 500, drop=True)

    def fit_gauss(x, a, l):
        return a * np.exp(-(np.pi * x / l) ** 2)

    p, r = curve_fit(fit_gauss, coVfit.az, coVfit.data, p0=[1., 227.])
    return p[1]


def tile_bursts_overlap_to_xspectra(burst0, burst1, geolocation_annotation, tile_width, tile_overlap,
                                    lowpass_width={'sample': 1000., 'line': 1000.},
                                    periodo_width={'sample': 2000., 'line': 1400.},
                                    periodo_overlap={'sample': 1000., 'line': 700.}, **kwargs):
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
    from xsarslc.tools import get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation

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
    mean_ground_spacing = float(burst['sampleSpacing'] / np.sin(np.radians(burst.attrs['mean_incidence'])))

    azimuth_spacing = float(burst['lineSpacing'])
    spacing = {'sample': mean_ground_spacing, 'line': azimuth_spacing}
    nperseg = {d: int(np.rint(tile_width[d] / spacing[d])) for d in tile_width.keys()}

    if tile_overlap in (0., None):
        noverlap = {d: 0 for k in nperseg.keys()}
    else:
        noverlap = {d: int(np.rint(tile_overlap[d] / spacing[d])) for d in tile_width.keys()}
    tiles_index = xtiling(burst, nperseg=nperseg, noverlap=noverlap)
    tiled_burst0 = burst0[tiles_index]  # .drop(['sample','line']).swap_dims({'__'+d:d for d in tile_width.keys()})
    tiled_burst1 = burst1[tiles_index]  # .drop(['sample','line']).swap_dims({'__'+d:d for d in tile_width.keys()})
    tiles_sizes = {d: k for d, k in tiled_burst0.sizes.items() if 'tile_' in d}

    # ---------Computing quantities at tile middle locations --------------------------
    tiles_middle = get_middle_tile(tiles_index)
    middle_sample = burst['sample'][{'sample': tiles_middle['sample']}]
    middle_line = burst['line'][{'line': tiles_middle['line']}]
    azitime_interval = burst.attrs['azimuth_time_interval']
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
        'corner_longitude')
    corner_lats = FullResolutionInterpolation(corner_line, corner_sample, 'latitude', geolocation_annotation,
                                              azitime_interval).unstack(dim=['flats', 'flatl']).rename(
        'corner_latitude')
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
    taus = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='tau')
    cutoff = xr.DataArray(np.empty(tuple(tiles_sizes.values()), dtype='float'), dims=tiles_sizes.keys(), name='cutoff')

    for i in xndindex(tiles_sizes):
        # ------ checking if we are over water only ------
        if 'landmask' in kwargs:
            tile_lons = [float(corner_lons[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            tile_lats = [float(corner_lats[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            water_only = is_ocean((tile_lons, tile_lats), kwargs.get('landmask'))
        else:
            water_only = True
        logging.debug('water_only :  %s', water_only)
        # ------------------------------------------------
        if water_only:
            sub0 = tiled_burst0[i].swap_dims({'__' + d: d for d in tile_width.keys()})
            sub1 = tiled_burst1[i].swap_dims({'__' + d: d for d in tile_width.keys()})
            sub = sub0

            mean_incidence = float(corner_incs[i].mean())
            mean_slant_range = float(corner_slantTimes[i].mean()) * celerity / 2.
            # mean_incidence = float(sub.incidence.mean())
            # mean_slant_range = float(sub.slant_range_time.mean())*celerity/2.
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
            # xs[tuple(i.values())] = xspecs_m
            xs.append(xspecs_m)
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
            taus[i] = tau.item()
            # ------------- cut-off --------------
            k_rg = xspecs_m.k_rg
            k_az = xspecs_m.k_az
            xspecs_m = xspecs_m['interburst_xspectra']
            xspecs_m = xspecs_m.assign_coords({'k_rg': k_rg, 'k_az': k_az}).swap_dims(
                {'freq_sample': 'k_rg', 'freq_line': 'k_az'})
            # return xspecs
            cutoff[i] = compute_azimuth_cutoff(xspecs_m)

    if xs:  # at least one existing xspectra has been calculated
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

    taus.attrs.update({'long_name': 'delay between two successive acquisitions', 'units': 's'})
    cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

    # tiles_middle = get_middle_tile(tiles_index)
    # middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    # middle_lat = burst['latitude'][tiles_middle].rename('latitude')

    xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lons.to_dataset(), corner_lats.to_dataset()],
                  combine_attrs='drop_conflicts')
    xs = xs.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    xs.attrs.update(burst.attrs)
    xs.attrs.update({'tile_nperseg_' + d: k for d, k in nperseg.items()})
    xs.attrs.update({'tile_noverlap_' + d: k for d, k in noverlap.items()})
    return xs


def compute_interburst_xspectrum(mod0, mod1, mean_incidence, slant_spacing, azimuth_spacing, azimuth_dim='line',
                                 nperseg={'sample': 512, 'line': None}, noverlap={'sample': 256, 'line': 0}):
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
        'interburst_xspectra')

    out = out.assign_coords(periodo0.drop(['line', 'sample']).coords)

    out.attrs.update({'mean_incidence': mean_incidence})

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(out.mean_incidence))
    k_rg = xr.DataArray(np.fft.rfftfreq(nperseg[range_dim], ground_range_spacing / (2 * np.pi)), dims=freq_rg_dim,
                        name='k_rg', attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[azimuth_dim], azimuth_spacing / (2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    # out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az}).swap_dims({'freq_'+range_dim:'k_rg', 'freq_'+azimuth_dim:'k_az'})
    out = xr.merge([out, k_rg.to_dataset(), k_az.to_dataset()],
                   combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    out.attrs.update({'periodogram_nperseg_' + range_dim: nperseg[range_dim],
                      'periodogram_nperseg_' + azimuth_dim: nperseg[azimuth_dim],
                      'periodogram_noverlap_' + range_dim: noverlap[range_dim],
                      'periodogram_noverlap_' + azimuth_dim: noverlap[azimuth_dim]})
    return out


def symmetrize_xspectrum(xs, dim_range='k_rg', dim_azimuth='k_az'):
    """
    Symmetrize half-xspectrum around origin point. xspectrum is assumed to contain only positive wavenumbers in range.
    
    Args:
        xs (xarray.DataArray or xarray.Dataset): complex xspectra to be symmetrized
    Return:
        (xarray.DataArray or xarray.Dataset): symmetrized spectra (as they were generated in the first place using fft)
    """
    if (dim_range not in xs.coords) or (dim_azimuth not in xs.coords):
        raise ValueError(
            'Symmetry can not be applied because {} or {} do not have coordinates. Swap dimensions prior to symmetrization.'.format(
                dim_range, dim_azimuth))

    if not xs.sizes['k_az'] % 2:
        xs = xr.concat([xs, xs[{'k_az': 0}].assign_coords({'k_az': -xs.k_az[0].data})], dim='k_az')

    mirror = np.conj(xs[{dim_range: slice(None, 0, -1)}])
    mirror = mirror.assign_coords({dim_range: -mirror[dim_range], dim_azimuth: -mirror[dim_azimuth]})
    res = xr.concat([mirror, xs], dim=dim_range)[{dim_range: slice(None, -1), dim_azimuth: slice(None, -1)}]
    return res
