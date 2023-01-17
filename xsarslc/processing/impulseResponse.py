#!/usr/bin/env python
# coding=utf-8
"""
"""
import numpy as np
import xarray as xr
import logging
from scipy.constants import c as celerity
from xsarslc.tools import xtiling, xndindex


def generate_IW_AUX_file_ImpulseReponse(subswathes, subswath_number):
    """
    Compute IR for each file listed in subswathes. Average over all files, bursts, tiles and returm mean range and azimuth Impulse Response.
    All listed subswath/burst should be on homogeneous zone

    Args:
        subswathes (dict): keys are SAFE file path (str), and values are list of burst number. Ex {'/home/my_directory/my_file.SAFE', [0,2,6]}
    """
    product_IRs = list()
    for SAFE_path, burst_list in subswathes.items():
        slc_iw_path = 'SENTINEL1_DS:'+SAFE_path+':IW'+str(subswath_number)
        dt = xsar.open_datatree(slc_iw_path)
        IRs = compute_subswath_Impulse_Response(dt, burst_list = burst_list)
        product_IRs.append(IRs)
        # product_IRs.append(IRs.mean(dim=['burst','tile_line', 'tile_sample']))
    # product_IRs = xr.concat(product_IRs, dim='product')
    return product_IRs#.mean(dim='product')

def generate_WV_AUX_file_ImpulseReponse(swathes):
    """
    Compute IR for each file listed in swathes. Average over all files and returm mean range and azimuth Impulse Response.
    All listed swath - imagette should be on homogeneous zone

    Args:
        swathes (dict): keys are SAFE file path (str), and values are list of imagette number. Ex {'/home/my_directory/my_file.SAFE', ['045','046']}
    """
    import xsar
    product_IRs = list()
    for SAFE_path, WV_list in swathes.items():
        for iWV in WV_list:
            slc_wv_path = 'SENTINEL1_DS:'+SAFE_path+':WV_'+str(iWV)
            dt = xsar.open_datatree(slc_wv_path)
            IRs = compute_WV_Impulse_Response(dt)
            product_IRs.append(IRs)
    return product_IRs

def compute_subswath_Impulse_Response(dt, burst_list=None, tile_width={'sample': 20.e3, 'line': 20.e3},
                                         tile_overlap={'sample': 10.e3, 'line': 10.e3}, polarization='VV', **kwargs):
    """
    Compute IW subswath range and azimuth Impulse Response. This function must be applied on homogeneous zone (amazonia, ...)
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        burst_list (list of int, optional): list of burst number to process. Default is all
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        polarization (str, optional): polarization to be selected for IR computation
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_burst_to_IR(), landmask can be added in kwargs. Can contain polarisation
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.processing.impulseResponse import tile_burst_to_IR
    from xsarslc.burst import crop_burst, deramp_burst

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval']),
               'swath': dt.attrs['swath']}
    IRs = list()
    dev = kwargs.get('dev', False)
    
    if not burst_list:
        burst_list = np.arange(dt['bursts'].sizes['burst'])
    if dev:
        logging.info('reduce number of burst -> 2')
        burst_list = [0,1]
    
    for b in burst_list:      
        burst = crop_burst(dt['measurement'].ds, dt['bursts'].ds, burst_number=b, valid=True).sel(pol=polarization)
        deramped_burst = deramp_burst(burst, dt)
        burst = xr.merge([burst, deramped_burst.drop('azimuthTime')], combine_attrs='drop_conflicts')
        burst.load()
        burst.attrs.update(commons)
        IR_range, IR_azimuth = tile_burst_to_IR(burst, dt['geolocation_annotation'], dt['orbit'], tile_width, tile_overlap,
                           lowpass_width={'sample': 1000., 'line': 1000.},
                           periodo_width={'sample': 2000., 'line': 4000.},
                           periodo_overlap={'sample': 1000., 'line': 2000.})
        IRs.append(xr.merge([IR_range, IR_azimuth]).drop(['tile_line', 'tile_sample']))
    IRs = xr.concat(IRs, dim='burst')
    return IRs

def compute_WV_Impulse_Response(dt, tile_width=None, tile_overlap=None, polarization='VV', **kwargs):
    """
    Compute WV range and azimuth Impulse Response. This function must be applied on homogeneous zone (amazonia, ...)
    Note: If requested tile is larger than the size of availabe data. tile will be set to maximum available size
    Args:
        dt (xarray.Datatree): datatree contraining subswath information
        burst_list (list of int, optional): list of burst number to process. Default is all
        tile_width (dict): approximative sizes of tiles in meters. Dict of shape {dim_name (str): width of tile [m](float)}
        tile_overlap (dict): approximative sizes of tiles overlapping in meters. Dict of shape {dim_name (str): overlap [m](float)}
        polarization (str, optional): polarization to be selected for IR computation
    
    Keyword Args:
        kwargs (dict): keyword arguments passed to tile_burst_to_IR(), landmask can be added in kwargs. Can contain polarisation
        
    Return:
        (xarray): xspectra.
    """
    from xsarslc.processing.impulseResponse import tile_burst_to_IR

    commons = {'radar_frequency': float(dt['image']['radarFrequency']),
               'mean_incidence': float(dt['image']['incidenceAngleMidSwath']),
               'azimuth_time_interval': float(dt['image']['azimuthTimeInterval']),
               'swath': dt.attrs['swath']}
    
    burst = dt['measurement'].ds.sel(pol=polarization)
    burst.load()
    burst.attrs.update(commons)
    IR_range, IR_azimuth = tile_burst_to_IR(burst, dt['geolocation_annotation'], dt['orbit'], tile_width, tile_overlap,
                           lowpass_width={'sample': 1000., 'line': 1000.},
                           periodo_width={'sample': 2000., 'line': 4000.},
                           periodo_overlap={'sample': 1000., 'line': 2000.})
    IRs = xr.merge([IR_range, IR_azimuth]).drop(['tile_line', 'tile_sample'])
    
    return IRs

def tile_burst_to_IR(burst, geolocation_annotation, orbit, tile_width, tile_overlap,
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
        kwargs: keyword arguments passed to compute_intraburst_xspectrum()
    """
    from xsarslc.tools import get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation
    from xsarslc.processing.xspectra import compute_modulation, compute_azimuth_cutoff

    burst.load()
    mean_ground_spacing = float(burst['sampleSpacing'] / np.sin(np.radians(burst.attrs['mean_incidence'])))
    azimuth_spacing = float(burst['lineSpacing'])
    spacing = {'sample': mean_ground_spacing, 'line': azimuth_spacing}

    if tile_width:
        nperseg_tile = {d: int(np.rint(tile_width[d] / spacing[d])) for d in tile_width.keys()}
    else:
        nperseg_tile = burst.sizes
        tile_width = {d:nperseg_tile[d]*spacing[d] for d in nperseg_tile.keys()}

    if tile_overlap in (0., None):
        noverlap = {d: 0 for d in nperseg_tile.keys()}
    else:
        noverlap = {d: int(np.rint(tile_overlap[d] / spacing[d])) for d in
                    tile_width.keys()}  # np.rint is important for homogeneity of point numbers between bursts

    tiles_index = xtiling(burst, nperseg=nperseg_tile, noverlap=noverlap)
    dev = kwargs.get('dev', False)
    if dev:
        logging.info('reduce number of burst for dev: 2')
        tiles_index['sample'] = tiles_index['sample'].isel({'tile_sample': slice(0, 2)})
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

    IR_range = list()
    IR_azimuth = list()

    for i in xndindex(tiles_sizes):
        
        sub = tiled_burst[i]

        mean_incidence = float(corner_incs[i].mean())

        slant_spacing = float(sub['sampleSpacing'])
        ground_spacing = slant_spacing / np.sin(np.radians(mean_incidence))
        periodo_spacing = {'sample': ground_spacing, 'line': azimuth_spacing}

        nperseg_periodo = {'sample':2048, 'line':256}
        noverlap_periodo = {'sample':1024, 'line':128}

        azimuth_spacing = float(sub['lineSpacing'])

        mod = sub['digital_number'] if sub.swath=='WV' else sub['deramped_digital_number']
        mod = compute_modulation(mod, lowpass_width=lowpass_width,
                                     spacing={'sample': ground_spacing, 'line': azimuth_spacing})
        rg_ir, azi_ir = compute_rg_az_response(mod, mean_incidence, slant_spacing, azimuth_spacing,
                                                  nperseg=nperseg_periodo,
                                                  noverlap=noverlap_periodo, **kwargs)
        mean_incidence = xr.DataArray(mean_incidence, name='mean_incidence', attrs={'long_name':'incidence at middle tile', 'units':'degree'})
        rg_ir = xr.merge([rg_ir, mean_incidence.to_dataset()])
        azi_ir = xr.merge([azi_ir, mean_incidence.to_dataset()])
            
        IR_range.append(rg_ir.mean(dim=['periodo_sample','periodo_line'], keep_attrs=True))
        IR_azimuth.append(azi_ir.mean(dim=['periodo_sample','periodo_line'], keep_attrs=True))

    if not IR_range: 
        return


    # line below rearange xs on (tile_sample, tile_line) grid and expand_dims ensures rearangment in combination by coords
    IR_range = xr.combine_by_coords(
        [x.expand_dims(['tile_sample', 'tile_line']) for x in IR_range],
        combine_attrs='drop_conflicts')
    IR_azimuth = xr.combine_by_coords(
        [x.expand_dims(['tile_sample', 'tile_line']) for x in IR_azimuth],
        combine_attrs='drop_conflicts')
    

    print(IR_range)
    print(IR_azimuth)


    IR_range = IR_range.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE


    IR_azimuth = IR_azimuth.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE

    print('-----------------')
    print(IR_range)
    print(IR_azimuth)

    
    IR_range.attrs.update(burst.attrs)
    IR_azimuth.attrs.update(burst.attrs)

    return IR_range, IR_azimuth


def compute_rg_az_response(slc, mean_incidence, slant_spacing, azimuth_spacing,
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

    range_IR = np.empty(tuple(periodo_sizes.values()), dtype=object)
    azimuth_IR = np.empty(tuple(periodo_sizes.values()), dtype=object)

    for i in xndindex(periodo_sizes):
        image = periodo[i]
        azi_spec, rg_spec = compute_IR(image, azimuth_dim=azimuth_dim,**kwargs)
        range_IR[tuple(i.values())] = rg_spec
        azimuth_IR[tuple(i.values())] = azi_spec

    range_IR = [list(a) for a in list(range_IR)]  # must be generalized for larger number of dimensions
    range_IR = xr.combine_nested(range_IR, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts')
    azimuth_IR = [list(a) for a in list(azimuth_IR)]  # must be generalized for larger number of dimensions
    azimuth_IR = xr.combine_nested(azimuth_IR, concat_dim=periodo_sizes.keys(), combine_attrs='drop_conflicts')
    
    range_IR = range_IR.assign_coords(periodo.coords)
    azimuth_IR = azimuth_IR.assign_coords(periodo.coords)

    # dealing with wavenumbers
    ground_range_spacing = slant_spacing / np.sin(np.radians(mean_incidence))

    k_rg = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[range_dim], ground_range_spacing / (2 * np.pi))),
                        dims='freq_' + range_dim, name='k_rg',
                        attrs={'long_name': 'wavenumber in range direction', 'units': 'rad/m'})
    k_srg = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[range_dim], slant_spacing / (2 * np.pi))),
                        dims='freq_' + range_dim, name='k_rg',
                        attrs={'long_name': 'wavenumber in slant range direction', 'units': 'rad/m'})
    k_az = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[azimuth_dim], azimuth_spacing / (2 * np.pi))),
                        dims='freq_' + azimuth_dim, name='k_az',
                        attrs={'long_name': 'wavenumber in azimuth direction', 'units': 'rad/m'})
    
#     range_IR = xr.merge([range_IR, k_rg.to_dataset()], combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
#     azimuth_IR = xr.merge([azimuth_IR, k_az.to_dataset()], combine_attrs='drop_conflicts')  # Adding .to_dataset() ensures promote_attrs=False
    
    range_IR = range_IR.drop(['freq_' + range_dim]).assign_coords({'k_rg':k_rg, 'k_srg':k_srg})
    azimuth_IR = azimuth_IR.drop(['freq_' + azimuth_dim]).assign_coords({'k_az':k_az})
    range_IR.attrs.update({'long_name':'Impulse response in range direction'})
    azimuth_IR.attrs.update({'long_name':'Impulse response in azimuth direction'})
    return range_IR, azimuth_IR
#     return range_IR.drop(['freq_' + range_dim]), azimuth_IR.drop(['freq_' + azimuth_dim])


def compute_IR(slc, azimuth_dim,**kwargs):
    """
    Compute Impulse response of slc by applying FFT in range and azimuth direction
    
    Args:
        slc (xarray.DataArray): (bi-dimensional) array to process
        azimuth_dim (str) : name of the azimuth dimension (dimension used to extract the look)
    
    Return:
        (xr.DataArray, xr.DataArray) : azimuth IR, range IR
    """
    
    import xrft

    range_dim = list(set(slc.dims) - set([azimuth_dim]))[0]  # name of range dimension
    freq_azi_dim = 'freq_' + azimuth_dim
    freq_rg_dim = 'freq_' + range_dim
    
    azi_spec = xrft.power_spectrum(slc, dim=azimuth_dim).mean(dim=range_dim).rename('azimuth_IR')
    rg_spec = xrft.power_spectrum(slc, dim=range_dim).mean(dim=azimuth_dim).rename('range_IR')
    

    return azi_spec, rg_spec