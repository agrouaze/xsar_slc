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

def compute_subswath_xspectra(dt, polarization, tile_width_intra, tile_width_inter, tile_overlap_intra,
                              tile_overlap_inter,
                              periodo_width_intra, periodo_width_inter, periodo_overlap_intra, periodo_overlap_inter,
                              **kwargs):
    """
    Main function to compute IW inter and intra burst spectra. It has to be modified to be able to change Xspectra options

    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        tile_width_intra (dict, optional): approximate sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)} for intra burst.
        tile_width_inter (dict, optional): approximate sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)} for inter burst.
        tile_overlap_intra (dict, optional): approximate sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)} for intra burst.
        tile_overlap_inter (dict, optional): approximate sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)} for inter burst.
        periodo_width_intra (dict): approximate sizes of periodogram in meters. Dict of shape {dim_name (str): width of tile [m](float)} for intra burst.
        periodo_width_inter (dict): approximate sizes of periodogram in meters. Dict of shape {dim_name (str): width of tile [m](float)} for inter burst.
        periodo_overlap_intra (dict): approximate sizes of periodogram overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)} for intra burst.
        periodo_overlap_inter (dict): approximate sizes of periodogram overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)} for inter burst.
        polarization (str, optional): polarization to be selected for xspectra computation

    Keyword Args:
        kwargs (dict): keyword arguments passed to called functions. landmask, IR_path ...
    """
    import datatree
    from xsarslc.tools import netcdf_compliant

    intra_xs = compute_IW_subswath_intraburst_xspectra(dt, polarization=polarization, tile_width=tile_width_intra,
                                                       tile_overlap=tile_overlap_intra,
                                                       periodo_overlap=periodo_overlap_intra,
                                                       periodo_width=periodo_width_intra,
                                                       **kwargs)
    if 'spatial_ref' in intra_xs:
        intra_xs = intra_xs.drop('spatial_ref')
        # intra_xs.attrs.update({'start_date': str(intra_xs.start_date)})
        # intra_xs.attrs.update({'stop_date': str(intra_xs.stop_date)})
    if isinstance(intra_xs, xr.Dataset):
        intra_xs.attrs.update({'footprint': str(intra_xs.footprint)})
        if 'multidataset' in intra_xs.attrs:
            intra_xs.attrs.update({'multidataset': str(intra_xs.multidataset)})
            # intra_xs.attrs.pop('pixel_line_m')
            # intra_xs.attrs.pop('pixel_sample_m')

    inter_xs = compute_IW_subswath_interburst_xspectra(dt, polarization=polarization, tile_width=tile_width_inter,
                                                       tile_overlap=tile_overlap_inter,
                                                       periodo_overlap=periodo_overlap_inter,
                                                       periodo_width=periodo_width_inter,
                                                       **kwargs)
    if 'spatial_ref' in inter_xs:
        inter_xs = inter_xs.drop('spatial_ref')
    if isinstance(inter_xs, xr.Dataset):
        if 'multidataset' in inter_xs.attrs:
            inter_xs.attrs.update({'multidataset': str(inter_xs.multidataset)})
            # inter_xs.attrs.update({'start_date': str(inter_xs.start_date)})
            # inter_xs.attrs.update({'stop_date': str(inter_xs.stop_date)})
        if 'footprint' in inter_xs.attrs:
            inter_xs.attrs.update({'footprint': str(inter_xs.footprint)})
        # inter_xs.attrs.pop('pixel_line_m')
        # inter_xs.attrs.pop('pixel_sample_m')
    if not inter_xs and not intra_xs:
        dt = None
    else:
        dt_dict = {}
        if inter_xs:
            dt_dict.update({'interburst': netcdf_compliant(inter_xs)})
        if intra_xs:
            dt_dict.update({'intraburst': netcdf_compliant(intra_xs)})
        dt = datatree.DataTree.from_dict(dt_dict)
    return dt


def compute_WV_intraburst_xspectra(dt, polarization, tile_width=None, tile_overlap=None, landmask=None, IR_path=None, **kwargs):
    """
    Main function to compute WV x-spectra (tiling is available)
    Note: If requested tile is larger than the size of available data (or set to None). tile will be set to maximum available size
    Args:
        dt (xarray.Datatree): datatree containing sub-swath information
        polarization (str, optional): polarization to be selected for x-spectra computation
        tile_width (dict, optional): approximate sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}. Default is all imagette
        tile_overlap (dict, optional): approximate sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}. Default is no overlap
        landmask (cartopy, optional) : a landmask to be used for land discrimination
        IR_path (str, optional) : a path to the Impulse Response file
    
    Keyword Args:
        kwargs (dict): optional keyword arguments : No valid entries for now
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.processing.intraburst import tile_burst_to_xspectra
    from xsarslc.burst import crop_WV

    if not IR_path:
        warnings.warn('Impulse Reponse not found in keyword argument. No IR correction will be applied.')

    if not landmask:
        warnings.warn('Landmask not found in keyword argument. X-spectra will be evaluated everywhere.')

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval']),
               'swath': dt.attrs['swath']}

    burst = dt['measurement'].ds.sel(pol=polarization)
    burst.load()
    burst = crop_WV(burst)
    burst.attrs.update(commons)
    xspectra = tile_burst_to_xspectra(burst, dt['geolocation_annotation'], dt['orbit'], dt['calibration'],
                                      dt['noise_range'], dt['noise_azimuth'], tile_width, tile_overlap, landmask=landmask, IR_path=IR_path, **kwargs)
    xspectra = xspectra.drop(
        ['tile_line', 'tile_sample'])  # dropping coordinate is important to not artificially multiply the dimensions
    if xspectra.sizes['tile_line'] == xspectra.sizes[
        'tile_sample'] == 1:  # In WV mode, it will probably be only one tile
        xspectra = xspectra.squeeze(['tile_line', 'tile_sample'])
    xspectra = xs_formatting(xspectra)
    return xspectra


def compute_IW_subswath_intraburst_xspectra(dt, polarization, periodo_width={'sample': 2000, 'line': 2000},
                                            periodo_overlap={'sample': 1000 ,'line': 1000},
                                            tile_width={'sample': 20.e3, 'line': 20.e3},
                                            tile_overlap={'sample': 10.e3, 'line': 10.e3}, landmask=None, IR_path=None, **kwargs):
    """
    Compute IW subswath intra-burst xspectra per tile
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        polarization (str, optional): polarization to be selected for xspectra computation
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        polarization (str, optional): polarization to be selected for xspectra computation
        periodo_width (dict): approximate sizes of periodogram in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        periodo_overlap (dict): approximate sizes of periodogram overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        landmask (cartopy, optional) : a landmask to be used for land discrimination
        IR_path (str, optional) : a path to the Impulse Response file
    
    Keyword Args:
        kwargs (dict): optional keyword arguments : burst_list (list), dev (bool) are valid entries
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.processing.intraburst import tile_burst_to_xspectra
    from xsarslc.burst import crop_IW_burst, deramp_burst

    if not IR_path:
        warnings.warn('Impulse Reponse not found in keyword argument. No IR correction will be applied.')

    if not landmask:
        warnings.warn('Landmask not found in keyword argument. X-spectra will be evaluated everywhere.')

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval']),
               'swath': dt.attrs['swath']}
    xspectra = list()

    burst_list = kwargs.pop('burst_list', dt['bursts'].ds['burst'].data) # this is a list of burst number (not burst index)

    if kwargs.pop('dev', False):
        logging.info('reduce number of burst -> 1')
        burst_list = [burst_list[0]] if len(burst_list)>0 else []
    pbar = tqdm(range(len(burst_list)))
    for ii in pbar:
        b = burst_list[ii]
        pbar.set_description('intrabursts')
        burst = crop_IW_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol=polarization)
        deramped_burst = deramp_burst(burst, dt)
        burst = xr.merge([burst, deramped_burst.drop('azimuthTime')], combine_attrs='drop_conflicts')
        burst.load()
        burst.attrs.update(commons)
        burst_xspectra = tile_burst_to_xspectra(burst, dt['geolocation_annotation'], dt['orbit'], dt['calibration'],
                                                dt['noise_range'], dt['noise_azimuth'], tile_width, tile_overlap,
                                                periodo_width=periodo_width, periodo_overlap=periodo_overlap,
                                                landmask=landmask, IR_path=IR_path, **kwargs)
        if burst_xspectra:
            xspectra.append(burst_xspectra.drop(['tile_line',
                                                 'tile_sample']))  # dropping coordinate is important to not artificially multiply the dimensions

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    Nfreqs = [x.sizes['freq_sample'] if 'freq_sample' in x.dims else np.nan for x in xspectra if 'freq_sample' in x.dims]
    if np.any(np.isfinite(Nfreqs)):
        # -------Returned xspecs have different shape in range (to keep same dk). Lines below only select common portions of xspectra-----
        Nfreq_min = min(Nfreqs)
        xspectra = [x[{'freq_sample': slice(None, Nfreq_min)}] if 'freq_sample' in x.dims else x for x in xspectra]

    xspectra = [x.assign_coords({'tile_sample':range(x.sizes['tile_sample']), 'tile_line':range(x.sizes['tile_line'])}) for x in xspectra] # coords assignement is for alignment below
    dims_not_align = set()
    for x in xspectra:
        dims_not_align=dims_not_align.union(set(x.dims))
    dims_not_align = dims_not_align-set(['tile_sample', 'tile_line'])
    xspectra = xr.align(*xspectra, exclude=dims_not_align, join='outer') # tile sample/line are aligned (thanks to their coordinate value) to avoid bug in combine_by_coords below
    xspectra = xr.combine_by_coords([x.drop(['tile_sample', 'tile_line']).reset_coords(['line','sample','longitude','latitude']).expand_dims('burst') for x in xspectra], combine_attrs='drop_conflicts')
    xspectra = xspectra.assign_coords({d:xspectra[d] for d in ['line','sample','longitude','latitude'] if d in xspectra}) # reseting and reassigning theses variables avoid some bug in combine_by_coords with missing variables between datasets
    xspectra = xs_formatting(xspectra)
    return xspectra


def compute_IW_subswath_interburst_xspectra(dt, polarization, periodo_width={'sample': 2000, 'line': 1200},
                                            periodo_overlap={'sample': 1000 ,'line': 600},
                                            tile_width={'sample': 20.e3, 'line': 1.5e3},
                                            tile_overlap={'sample': 10.e3, 'line': 0.75e3}, landmask=None, IR_path=None, **kwargs):
    """
    Compute IW subswath inter-burst xspectra. No deramping is applied since only magnitude is used.
    
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Note: The overlap is short in azimuth (line) direction. Keeping nperseg = {'line':None} in Xspectra computation
    keeps maximum number of point in azimuth but is not ensuring the same number of overlapping point for all burst
    
    Args:
        dt (xarray.Datatree): datatree containing sub-swath information
        polarization (str, optional): polarization to be selected for xspectra computation
        tile_width (dict): approximate sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximate sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        polarization (str, optional): polarization to be selected for xspectra computation
        periodo_width (dict): approximate sizes of periodogram in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        periodo_overlap (dict): approximate sizes of periodogram overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        landmask (cartopy, optional) : a landmask to be used for land discrimination
        IR_path (str, optional) : a path to the Impulse Response file
    
    Keyword Args:
        kwargs (dict): optional keyword arguments : burst_list (list), dev (bool) are valid entries
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.processing.interburst import tile_bursts_overlap_to_xspectra
    from xsarslc.burst import crop_IW_burst

    if not IR_path:
        warnings.warn('Impulse Reponse not found in keyword argument. No IR correction will be applied.')

    if not landmask:
        warnings.warn('Landmask not found in keyword argument. X-spectra will be evaluated everywhere.')


    commons = {'azimuth_steering_rate': dt['image']['azimuthSteeringRate'].item(),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval'])}
    xspectra = list()

    burst_list = kwargs.pop('burst_list', dt['bursts'].ds['burst'].data) # this is a list of burst number (not burst index)

    if kwargs.pop('dev', False):
        logging.info('reduce number of burst -> 1')
        burst_list = [burst_list[0]] if len(burst_list)>0 else []


    #for b in burst_list[:-1]:
    pbar = tqdm(range(len(burst_list[:-1])))
    for ii in pbar:
        b = burst_list[ii]
        pbar.set_description('interbursts')
        burst0 = crop_IW_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True,
                            merge_burst_annotation=True).sel(pol=polarization)
        burst1 = crop_IW_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b + 1, valid=True,
                            merge_burst_annotation=True).sel(pol=polarization)
        burst0.attrs.update(commons)
        burst1.attrs.update(commons)
        interburst_xspectra = tile_bursts_overlap_to_xspectra(burst0, burst1, dt['geolocation_annotation'],
                                                              dt['calibration'], dt['noise_range'], dt['noise_azimuth'],
                                                              tile_width=tile_width,
                                                              tile_overlap=tile_overlap,
                                                              periodo_width=periodo_width,
                                                              periodo_overlap=periodo_overlap,
                                                              landmask=landmask, IR_path=IR_path, **kwargs)
        if interburst_xspectra:
            xspectra.append(interburst_xspectra.drop(['tile_line', 'tile_sample']))

    # -------Returned xspecs have different shape in range (between burst). Lines below only select common portions of xspectra-----
    Nfreqs = [x.sizes['freq_sample'] if 'freq_sample' in x.dims else np.nan for x in xspectra if 'freq_sample' in x.dims]
    if np.any(np.isfinite(Nfreqs)):
        # -------Returned xspecs have different shape in range (to keep same dk). Lines below only select common portions of xspectra-----
        Nfreq_min = min(Nfreqs)
        xspectra = [x[{'freq_sample': slice(None, Nfreq_min)}] if 'freq_sample' in x.dims else x for x in xspectra]

    xspectra = [x.assign_coords({'tile_sample':range(x.sizes['tile_sample']), 'tile_line':range(x.sizes['tile_line'])}) for x in xspectra] # coords assignement is for alignment below
    dims_not_align = set()
    for x in xspectra:
        dims_not_align=dims_not_align.union(set(x.dims))
    dims_not_align = dims_not_align-set(['tile_sample', 'tile_line'])
    xspectra = xr.align(*xspectra, exclude=dims_not_align, join='outer') # tile sample/line are aligned (thanks to their coordinate value) to avoid bug in combine_by_coords below
    xspectra = xr.combine_by_coords([x.drop(['tile_sample', 'tile_line']).reset_coords(['line','sample','longitude','latitude']).expand_dims('burst') for x in xspectra], combine_attrs='drop_conflicts')
    xspectra = xspectra.assign_coords({d:xspectra[d] for d in ['line','sample','longitude','latitude'] if d in xspectra}) # reseting and reassigning theses variables avoid some bug in combine_by_coords with missing variables between datasets
    xspectra = xs_formatting(xspectra)
    return xspectra


def xs_formatting(xspectra):
    """
    Format final returned xspectra. Transposing dimensions, checking data type, ...
    Args:
        xspectra (xarray.Dataset): xspectra to format
    Return:
        (xarray.Dataset): formatted xspectra

    """
    dims_to_transpose = [d for d in ['burst', 'tile_line', 'tile_sample', 'freq_line', 'freq_sample'] if
                         d in xspectra.dims]  # for homogeneous order of dimensions with intraburst
    xspectra = xspectra.transpose(*dims_to_transpose, ...)
    if 'land_flag' in xspectra:
        xspectra['land_flag'] = xspectra['land_flag'].astype(bool)
    if 'corner_line' in xspectra:
        xspectra['corner_line'] = xspectra['corner_line'].astype(int)
        xspectra['corner_sample'] = xspectra['corner_sample'].astype(int)
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

    coV = xrft.ifft(spectrum.real, dim=('k_rg', 'k_az'), shift=True, prefix='k_')
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

    try:
        p, pcov = curve_fit(fit_gauss, coVfit.az, coVfit.data, p0=[1., 227.])
        cutoff = p[1]
        relat_err = np.abs((np.sqrt(np.diag(pcov))/p)[1])
    except:
        cutoff, relat_err = np.nan, np.nan

    cutoff = xr.DataArray(float(cutoff), name='azimuth_cutoff', attrs={'long_name': 'Azimuthal cut-off', 'units': 'm'})
    cutoff_error = xr.DataArray(float(relat_err), name='azimuth_cutoff_error', attrs={'long_name': 'normalized azimuthal cut-off error std'})
    return cutoff, cutoff_error


def compute_normalized_variance(modulation):
    """
    compute normalized variance from modulation of digital numbers
    Args:
        modulation (xarray): output from compute_modulation()
    Return:
        (float): normalized variance
    """
    detected_mod = np.abs(modulation) ** 2.
    nv = detected_mod.var(dim=['line', 'sample']) / ((detected_mod.mean(dim=['line', 'sample'])) ** 2)
    nv = nv.rename('normalized_variance')
    nv.attrs.update({'long_name': 'normalized variance', 'units': ''})
    return nv


def compute_mean_sigma0_interp(DN, sigma0_lut, range_noise_lut, azimuth_noise_lut):
    """
    Compute mean calibrated sigma0 and mean noise equivalent sigma0. This is a version doing interpolation over the LUT instead of taking closest LUT
    Args:
        DN (xarray): digital number
        sigma0_lut (xarray) : calibration LUT of dataset
    Return:
        (xarray): calibrated mean sigma0 (single value)
        (xarray): noise equivalent sigma0 (single value)
    """
    polarization = DN.pol.item()
    sigma0_lut = sigma0_lut.sel(pol=polarization)
    range_noise_lut = range_noise_lut.sel(pol=polarization)
    azimuth_noise_lut = azimuth_noise_lut.sel(pol=polarization)
    noise = (azimuth_noise_lut.interp_like(DN, assume_sorted=True)) * (
        range_noise_lut.interp_like(DN, assume_sorted=True))
    
    calib = (sigma0_lut.interp_like(DN, assume_sorted=True)) ** 2
    sigma0 = (np.abs(DN) ** 2 - noise) / calib
    sigma0 = sigma0.mean(dim=['line', 'sample']).rename('sigma0')
    sigma0.attrs.update({'long_name': 'calibrated sigma0', 'units': 'linear'})

    nesz = (noise / calib).mean(dim=['line', 'sample']).rename('nesz')
    nesz.attrs.update({'long_name': 'noise-equivalent sigma zero', 'units': 'linear'})

    return sigma0, nesz

def compute_mean_sigma0_closest(DN, linesPerBurst, sigma0_lut, range_noise_lut, azimuth_noise_lut):
    """
    Compute mean calibrated sigma0 and mean noise equivalent sigma0. This version uses closest line of the burst
    Args:
        DN (xarray): digital number
        sigma0_lut (xarray) : calibration LUT of dataset
    Return:
        (xarray): calibrated mean sigma0 (single value)
        (xarray): noise equivalent sigma0 (single value)
    """
    polarization = DN.pol.item()
    sigma0_lut = sigma0_lut.sel(pol=polarization)
    mid_burst_line = int(linesPerBurst*(DN['burst'].item()+0.5))
    range_noise_lut = range_noise_lut.sel(pol=polarization).sel(line=mid_burst_line, method='nearest').drop_vars('line') # taking closest line
    azimuth_noise_lut = azimuth_noise_lut.sel(pol=polarization)
    noise = (azimuth_noise_lut.interp_like(DN, assume_sorted=True)) * (
        range_noise_lut.interp_like(DN, assume_sorted=True))
    
    calib = (sigma0_lut.interp_like(DN, assume_sorted=True)) ** 2
    sigma0 = (np.abs(DN) ** 2 - noise) / calib
    sigma0 = sigma0.mean(dim=['line', 'sample']).rename('sigma0')
    sigma0.attrs.update({'long_name': 'calibrated sigma0', 'units': 'linear'})

    nesz = (noise / calib).mean(dim=['line', 'sample']).rename('nesz')
    nesz.attrs.update({'long_name': 'noise-equivalent sigma zero', 'units': 'linear'})
    return sigma0, nesz


def symmetrize_xspectrum(xs, dim_range='k_rg', dim_azimuth='k_az'):
    """
    Symmetrize half-xspectrum around origin point. For correct behaviour, xs has to be complex and assumed to contain only positive wavenumbers in range.
    
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


def get_centroid(spectrum, dim, width=0.5, method='firstmoment'):
    """
    Find Doppler centroid of provided spectrum value. If the provided spectrum has more than one dimension, they will be averaged.
    Args:
        spectrum (xarray.DataAArray) : Doppler azimuthal spectrum
        dim (str): dimension name of azimuth
        width (float, optional): percentage of Doppler bandwidth to be used for the fit in [0.,1.[
        method (str, optional): Method to compute the centroid
                                If 'maxfit': a gaussian fit around maximum spectrum value is used.
                                If 'firstmoment': the first moment of the spectrum (centered around its maximum) is used
    Return:
        (float) : Doppler centroid value in same unit than dim
    """
    if spectrum.dtype == complex:
        raise ValueError('Doppler centroid can not be estimated on complex data')
    if len(spectrum.sizes) > 1:
        spectrum = spectrum.mean(list(set(spectrum.sizes.keys()) - set([dim])))
    if dim not in spectrum.coords:
        raise ValueError('{} are not in {}'.format(dim, spectrum.coords))

    imax = int(spectrum.argmax())
    N = int(width * spectrum.sizes[dim])
    fit_spectrum = spectrum[{dim: slice(max(0, imax - N // 2), min(imax + N // 2, spectrum.sizes[
        dim]))}]  # selecting portion of interest (width)

    if method == 'firstmoment':
        centroid = float((fit_spectrum * fit_spectrum[dim]).sum() / (fit_spectrum).sum())

    elif method == 'maxfit':
        from scipy.optimize import curve_fit
        fit_gauss = lambda x, a, c, l: a * np.exp(-(x - c) ** 2 / (l ** 2))
        a_estimate = float(spectrum[{dim: imax}])
        c_estimate = float(spectrum[dim][{dim: imax}])
        l_estimate = 0.25 * float(spectrum[dim].max() - spectrum[dim].min())
        try:
            p, r = curve_fit(fit_gauss, fit_spectrum[dim], fit_spectrum.data, p0=[a_estimate, c_estimate, l_estimate])
            centroid = p[1]
        except:
            centroid = np.nan
        
    else:
        raise ValueError('Unknown method : {}'.format(method))

    return centroid
