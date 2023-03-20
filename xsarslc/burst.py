#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr

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


def crop_IW_burst(ds, burst_annotation, burst_number, valid=True, merge_burst_annotation=True):
    """
    Crop IW burst from the measurement dataset
    
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

    from xsarslc.processing.deramping import compute_midburst_azimuthtime, compute_slant_range_time, compute_Doppler_centroid_rate, \
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

def crop_WV(burst):
    """
    Crop WV data. Removes portion of WV with zero values only and removes 25 points on each side to remove windowing effect
    """
    DN = burst['digital_number'].where(np.abs(burst['digital_number'])!=0., drop=True)
    DN = DN[{'line':slice(25,-25), 'sample':slice(25,-25)}]
    b,_= xr.align(burst,DN, join='inner')
    return b