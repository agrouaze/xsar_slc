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

    #landmask = kwargs.pop('landmask', cartopy.feature.NaturalEarthFeature('physical', 'land', '10m'))
    kwargs['landmask'] = cartopy.feature.NaturalEarthFeature('physical', 'land', '10m')
    intra_xs = compute_subswath_intraburst_xspectra(dt, **kwargs)
    if 'spatial_ref' in intra_xs:
        intra_xs = intra_xs.drop('spatial_ref')
        #intra_xs.attrs.update({'start_date': str(intra_xs.start_date)})
        #intra_xs.attrs.update({'stop_date': str(intra_xs.stop_date)})
        intra_xs.attrs.update({'footprint': str(intra_xs.footprint)})
        #intra_xs.attrs.pop('pixel_line_m')
        #intra_xs.attrs.pop('pixel_sample_m')

    inter_xs = compute_subswath_interburst_xspectra(dt, **kwargs)
    if 'spatial_ref' in inter_xs:
        inter_xs = inter_xs.drop('spatial_ref')
        #inter_xs.attrs.update({'start_date': str(inter_xs.start_date)})
        #inter_xs.attrs.update({'stop_date': str(inter_xs.stop_date)})
        inter_xs.attrs.update({'footprint': str(inter_xs.footprint)})
        #inter_xs.attrs.pop('pixel_line_m')
        #inter_xs.attrs.pop('pixel_sample_m')
    if not inter_xs and not intra_xs:
        dt = None
    else:
        dt_dict={}
        if inter_xs:
            dt_dict.update({'interburst_xspectra': netcdf_compliant(inter_xs)})
        if intra_xs:
            dt_dict.update({'intraburst_xspectra': netcdf_compliant(intra_xs)})
        dt = datatree.DataTree.from_dict(dt_dict)
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
    from xsarslc.processing.intraburst import tile_burst_to_xspectra
    from xsarslc.burst import crop_burst, deramp_burst

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()
    nb_burst = dt['bursts'].sizes['burst']
    dev = kwargs.get('dev', False)
    pol = kwargs.get('pol', 'VV')
    if dev:
        logging.info('reduce number of burst -> 2')
        nb_burst = 2
    for b in range(nb_burst):
        burst = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol=pol)
        deramped_burst = deramp_burst(burst, dt)
        burst = xr.merge([burst, deramped_burst.drop('azimuthTime')], combine_attrs='drop_conflicts')
        burst.load()
        burst.attrs.update(commons)
        burst_xspectra = tile_burst_to_xspectra(burst, dt['geolocation_annotation'], dt['orbit'], tile_width,
                                                tile_overlap, **kwargs)
        if burst_xspectra:
            xspectra.append(burst_xspectra.drop(['tile_line', 'tile_sample']))

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    if xspectra:
        Nfreq_min = min([x.sizes['freq_sample'] for x in xspectra])
        # xspectra = xr.combine_by_coords([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra],
                                        # combine_attrs='drop_conflicts')  # rearange xs on burst
        # Nfreq_min = min([xs.sizes['freq_sample'] for xs in xspectra])
        # xspectra = [xs[{'freq_sample':slice(None, Nfreq_min)}] for xs in xspectra]
        xspectra = xr.concat([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra], dim='burst')
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
    from xsarslc.processing.interburst import tile_bursts_overlap_to_xspectra
    from xsarslc.burst import crop_burst

    commons = {'azimuth_steering_rate': dt['image']['azimuthSteeringRate'].item(),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()
    pol = kwargs.get('pol', 'VV')
    nb_burst = dt['bursts'].sizes['burst'] - 1
    dev = kwargs.get('dev', False)
    if dev:
        logging.info('reduce number of burst -> 2')
        nb_burst = 2
    for b in range(nb_burst):
        burst0 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True,
                            merge_burst_annotation=True).sel(pol=pol)
        burst1 = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b + 1, valid=True,
                            merge_burst_annotation=True).sel(pol=pol)
        burst0.attrs.update(commons)
        burst1.attrs.update(commons)
        interburst_xspectra = tile_bursts_overlap_to_xspectra(burst0, burst1, dt['geolocation_annotation'], tile_width,
                                                              tile_overlap, **kwargs)
        if interburst_xspectra:
            xspectra.append(interburst_xspectra.drop(['tile_line', 'tile_sample']))

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    if xspectra:
        Nfreq_min = min([x.sizes['freq_sample'] for x in xspectra])
        # xspectra = xr.combine_by_coords([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra],
                                        # combine_attrs='drop_conflicts')  # rearange xs on burst
        # Nfreq_min = min([xs.sizes['freq_sample'] for xs in xspectra])
        # xspectra = [xs[{'freq_sample':slice(None, Nfreq_min)}] for xs in xspectra]
        xspectra = xr.concat([x[{'freq_sample': slice(None, Nfreq_min)}] for x in xspectra], dim='burst')
        xspectra = xspectra.assign_coords({'k_rg': xspectra.k_rg, 'k_az': xspectra.k_az})  # move wavenumbers as coordinates
    return xspectra


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