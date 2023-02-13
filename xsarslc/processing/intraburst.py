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

def tile_burst_to_xspectra(burst, geolocation_annotation, orbit, tile_width, tile_overlap,
                           lowpass_width={'sample': 1000., 'line': 1000.},
                           periodo_width={'sample': 4000., 'line': 4000.},
                           periodo_overlap={'sample': 2000., 'line': 2000.}, **kwargs):
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
    from xsarslc.tools import get_tiles, get_corner_tile, get_middle_tile, is_ocean, FullResolutionInterpolation
    from xsarslc.processing.xspectra import compute_modulation, compute_azimuth_cutoff



    # burst.load()
    azitime_interval = burst.attrs['azimuth_time_interval']
    mean_ground_spacing = float(burst['sampleSpacing'] / np.sin(np.radians(burst.attrs['mean_incidence'])))
    azimuth_spacing = float(burst['lineSpacing'])
    spacing = {'sample': mean_ground_spacing, 'line': azimuth_spacing}

    if tile_width: # tile width is defined for both sample and line but only line will be used (sample will have heterogeneous number of point)
        nperseg_tile = {d: int(np.rint(tile_width[d] / spacing[d])) for d in tile_width.keys()}
    else:
        nperseg_tile = burst.sizes
        tile_width = {d:nperseg_tile[d]*spacing[d] for d in nperseg_tile.keys()}

    if tile_overlap in (0., None):
        noverlap_tile = {d: 0 for d in nperseg_tile.keys()}
    else:
        noverlap_tile = {d: int(np.rint(tile_overlap[d] / spacing[d])) for d in
                    tile_width.keys()}  # np.rint is important for homogeneity of point numbers between bursts

    if np.any([tile_width[d]<periodo_width[d] for d in tile_width.keys()]):
        warnings.warn("One or all periodogram widths are larger than tile widths. Exceeding periodogram widths are reset to match tile width.")

    for d in tile_width.keys():
        periodo_width[d] = min(periodo_width[d], tile_width[d])

    if np.any([periodo_overlap[d]>0.5*periodo_width[d] for d in periodo_width.keys()]):
        warnings.warn("Periodogram overlap should not exceed half of the periodogram width.")

    # tiles_index = xtiling(burst, nperseg=nperseg_tile, noverlap=noverlap_tile)

    # ------------- defining custom sample tiles_index because of variable ground range spacing -------
    nperseg_tile.pop('sample')
    noverlap_tile.pop('sample')
    incidenceAngle = FullResolutionInterpolation(burst['line'][{'line':slice(burst.sizes['line']//2, burst.sizes['line']//2+1)}], burst['sample'], 'incidenceAngle', geolocation_annotation, azitime_interval)
    cumulative_len = (float(burst['sampleSpacing'])*np.cumsum(1./np.sin(np.radians(incidenceAngle)))).rename('cumulative ground length').squeeze(dim='line')
    burst_width = cumulative_len[{'sample':-1}]
    starts = np.arange(0.,burst_width,tile_width['sample']-tile_overlap['sample'])
    ends = starts+tile_width['sample']
    starts = starts[ends<float(burst_width)] # starting length restricted to available data
    ends = ends[ends<float(burst_width)] # ending length restricted to available data
    istarts = np.searchsorted(cumulative_len,starts) # index of begining of tiles
    iends = np.searchsorted(cumulative_len,ends) # index of ending of tiles
    tile_sample = {'sample':xr.DataArray([slice(s,e+1) for s,e in zip(istarts,iends)], dims='tile_sample')}#, coords={'tile_sample':[(e+s)//2 for s,e in zip(istarts,iends)]})} # This is custom tile indexing along sample dimension to preserve constant tile width
    tile_sample_coords = get_middle_tile(tile_sample)
    tile_sample['sample'] = tile_sample['sample'].assign_coords({'tile_sample':burst['sample'][tile_sample_coords]})

    # ------------- defining regular line indexing --------
    tile_line = xtiling(burst['line'], nperseg=nperseg_tile, noverlap=noverlap_tile) # homogeneous tiling along line dimension can be done using xtiling()

    # ------------- customized indexes --------
    tiles_index = tile_sample.copy()
    tiles_index.update(tile_line)

    # ----- getting all tiles ------
    all_tiles = get_tiles(burst, tiles_index)


    # dev = kwargs.get('dev', False)
    # if dev:
    #     logging.info('reduce number of burst for dev: 2')
    #     tiles_index['sample'] = tiles_index['sample'].isel({'tile_sample': slice(0, 2)})

    # tiled_burst = burst[tiles_index].drop(['sample', 'line']).swap_dims({'__' + d: d for d in tile_width.keys()})
    # tiles_sizes = {d: k for d, k in tiled_burst.sizes.items() if 'tile_' in d}

    # ---------Computing quantities at tile middle locations --------------------------
    tiles_middle = get_middle_tile(tiles_index) # this return the indexes, NOT the sample/line coord
    # middle_lon = burst['longitude'][tiles_middle].rename('longitude')
    # middle_lat = burst['latitude'][tiles_middle].rename('latitude')
    middle_sample = burst['sample'][{'sample': tiles_middle['sample']}]
    middle_line = burst['line'][{'line': tiles_middle['line']}]
    # azitime_interval = burst.attrs['azimuth_time_interval']
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
    # combinaison_selection_tiles = [yy for yy in xndindex(tiles_sizes)]
    combinaison_selection_tiles = all_tiles
    pbar = tqdm(range(len(all_tiles)), desc='start')
    for ii in pbar:
        pbar.set_description('loop on %s/%s tiles' % (ii,len(combinaison_selection_tiles)))
        sub = all_tiles[ii].swap_dims({'__line':'line', '__sample':'sample'})
        mytile = {'tile_sample':sub['tile_sample'], 'tile_line':sub['tile_line']}

        # ------ checking if we are over water only ------
        if 'landmask' in kwargs:
            tile_lons = [float(corner_lons[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            tile_lats = [float(corner_lats[i][{'corner_line': j, 'corner_sample': k}]) for j, k in
                         [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]]
            water_only = is_ocean((tile_lons, tile_lats), kwargs.get('landmask'))
        else:
            water_only = True
        logging.debug('water_only : %s', water_only)
        # ------------------------------------------------
        if water_only:
            # sub = tiled_burst[i]
            mean_incidence = float(corner_incs.sel(mytile).mean())
            mean_slant_range = float(corner_slantTimes.sel(mytile).mean()) * celerity / 2.
            mean_velocity = float(corner_velos.sel({'tile_line':sub['tile_line']}).mean())

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

            mod = sub['digital_number'] if sub.swath=='WV' else sub['deramped_digital_number']
            mod = compute_modulation(mod, lowpass_width=lowpass_width,
                                     spacing={'sample': ground_spacing, 'line': azimuth_spacing})
            xspecs = compute_intraburst_xspectrum(mod, mean_incidence, slant_spacing, azimuth_spacing,
                                                  synthetic_duration, nperseg=nperseg_periodo,
                                                  noverlap=noverlap_periodo, **kwargs)
            xspecs_m = xspecs.mean(dim=['periodo_line', 'periodo_sample'],
                                   keep_attrs=True)  # averaging all the periodograms in each tile
            
            # ------------- tau ----------------
            tau = float(xspecs_m.attrs.pop('tau'))
            tau = xr.DataArray(float(tau), name='tau', attrs={'long_name': 'delay between two successive looks', 'units': 's'})
            # ------------- cut-off ------------
            cutoff_tau = [str(i) + 'tau' for i in [1, 2, 3, 0] if str(i) + 'tau' in xspecs_m.dims][0]  # tau used to compute azimuthal cutoff
            xs_cut = xspecs_m['xspectra_' + cutoff_tau].mean(dim=cutoff_tau).swap_dims(
                {'freq_sample': 'k_rg', 'freq_line': 'k_az'})
            cutoff = compute_azimuth_cutoff(xs_cut)
            cutoff = xr.DataArray(float(cutoff), name='cutoff', attrs={'long_name': 'Azimuthal cut-off', 'units': 'm'})
            mean_incidence = xr.DataArray(mean_incidence, name='incidence', attrs={'long_name':'incidence at tile middle', 'units':'degree'})
            xs.append(xr.merge([xspecs_m, tau.to_dataset(), cutoff.to_dataset(), mean_incidence.to_dataset()]))

    if not xs:  # All tiles are over land -> no xspectra available
        return
    
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
    # taus.attrs.update({'long_name': 'delay between two successive looks', 'units': 's'})
    # cutoff.attrs.update({'long_name': 'Azimuthal cut-off', 'units': 'm'})

    xs = xr.merge([xs, corner_lons.to_dataset(), corner_lats.to_dataset()],
                  combine_attrs='drop_conflicts')
    # xs = xr.merge([xs, taus.to_dataset(), cutoff.to_dataset(), corner_lons.to_dataset(), corner_lats.to_dataset()],
    #               combine_attrs='drop_conflicts')
    xs = xs.assign_coords({'longitude': middle_lons,
                           'latitude': middle_lats})  # This line also ensures adding line/sample coordinates too !! DO NOT REMOVE
    xs.attrs.update(burst.attrs)
    # xs.attrs.update({'tile_nperseg_' + d: k for d, k in nperseg_tile.items()})
    # xs.attrs.update({'tile_noverlap_' + d: k for d, k in noverlap.items()})
    xs.attrs.update({'tile_width_' + d: k for d, k in tile_width.items()})
    xs.attrs.update({'tile_overlap_' + d: k for d, k in tile_overlap.items()})
    xs.attrs.update({'periodo_width_' + d: k for d, k in periodo_width.items()})
    xs.attrs.update({'periodo_overlap_' + d: k for d, k in periodo_overlap.items()})
    return xs



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
    periodo = slc[periodo_slices]
    periodo=periodo.drop([range_dim, azimuth_dim]).swap_dims({'__' + d: d for d in [range_dim, azimuth_dim]})
    periodo_sizes = {d: k for d, k in periodo.sizes.items() if 'periodo_' in d}

    if 'IR_path' in kwargs: # Impulse Response has been provided
        IR = xr.load_dataset(kwargs.pop('IR_path'))
        IR['range_IR'] = IR['range_IR'].where(IR['range_IR']>IR['range_IR'].max()/100, np.nan) # discarding portion where range IR is very low
        freq_line = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(IR.sizes['k_az'])), dims='k_az')
        freq_sample = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(IR.sizes['k_srg'])), dims='k_srg')
        IR  = IR.assign_coords({'freq_line':freq_line, 'freq_sample':freq_sample}).swap_dims({'k_srg':'freq_sample', 'k_az':'freq_line'})
        freq_line = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[azimuth_dim])), dims='freq_line')
        freq_sample = xr.DataArray(np.fft.fftshift(np.fft.fftfreq(nperseg[range_dim])), dims='freq_sample')
        IR = IR.interp(freq_line=freq_line).interp(freq_sample=freq_sample)
        IR = np.sqrt(IR['azimuth_IR'])*np.sqrt(IR['range_IR'])
        kwargs.update({'IR':IR})

    out = list()

    for i in xndindex(periodo_sizes):
        image = periodo[i]
        xspecs = compute_looks(image, azimuth_dim=azimuth_dim, synthetic_duration=synthetic_duration,**kwargs) 
        out.append(xspecs)

    out = xr.combine_by_coords([x.expand_dims(['periodo_sample', 'periodo_line']) for x in out], combine_attrs='drop_conflicts')
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
    
    with xr.set_options(keep_attrs=True): # centroid has been evaluated on freq_line. It has to be converted on azimuth dimension
        out['centroid'] = out['centroid']*(2.*np.pi/azimuth_spacing)
        out['centroid'].attrs.update({'units':'rad/m'})
    out = out.assign_coords({'k_rg':k_rg, 'k_az':k_az})
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
    from xsarslc.processing.xspectra import get_centroid

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

    # mydop = xrft.fft(slc, dim=[azimuth_dim], detrend=None, window=None, shift=True, true_phase=True,
                     # true_amplitude=True)
    # Finding an removing Doppler centroid
    # weight = xr.DataArray(np.hanning(min(20,Np//10+1)), dims=['window'])  # window for smoothing
    # weight /= weight.sum()
    # smooth_dop = np.abs(mydop).mean(dim=range_dim).rolling(**{freq_azi_dim: len(weight), 'center': True}).construct(
    #     'window').dot(weight)
    # i0 = int(np.abs(mydop[freq_azi_dim]).argmin())  # zero frequency indice
    # ishift = int(smooth_dop.argmax()) - i0  # shift of Doppler centroid
    # mydop = mydop.roll(**{freq_azi_dim: -ishift, 'roll_coords': False})
    # centroid = float((smooth_dop*smooth_dop['freq_line']).sum()/(smooth_dop).sum())

    mydop = xrft.power_spectrum(slc, dim=azimuth_dim)
    centroid = get_centroid(mydop, dim=freq_azi_dim, method='maxfit')
    
    if 'IR' not in kwargs: # No Impulse Response has been provided
        mydop = xrft.fft(slc*np.exp(-1j*2*np.pi*centroid*slc[azimuth_dim]), dim=[azimuth_dim], detrend=None, window=None, shift=True, true_phase=True, true_amplitude=True)
    else: # Provided IR is used to normalize slc spectra
        mydop = xrft.fft(slc*np.exp(-1j*2*np.pi*centroid*slc[azimuth_dim]), dim=[azimuth_dim, range_dim], detrend=None, window=None, shift=True, true_phase=True, true_amplitude=True)   
        mydop = (mydop/kwargs.get('IR')).fillna(0.)
        mydop = xrft.ifft(mydop, dim='freq_'+range_dim, true_phase=True, true_amplitude=True)
        mydop = mydop.drop(['k_az', 'k_srg'])

    # Extracting the useful part of azimuthal Doppler spectrum
    # It removes points on each side to be sure that tiling will operate correctly
    Nused = nlooks * nperlook - (nlooks - 1) * noverlap
    left = (Np - Nused) // 2  # useless points on left side
    mydop = mydop[{freq_azi_dim: slice(left, left + Nused)}]
    look_tiles = xtiling(mydop, nperseg={freq_azi_dim: nperlook}, noverlap={freq_azi_dim: noverlap}, prefix='look_')

    if look_window is not None:
        raise ValueError('Look windowing is not available.')

    looks_spec = list()
    looks = mydop[look_tiles].drop(['freq_' + azimuth_dim]).swap_dims({'__' + d: d for d in ['freq_' + azimuth_dim]}).drop('look_freq_line')
    looks_sizes = {d: k for d, k in looks.sizes.items() if 'look_' in d}

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

    xspecs = {str(i) + 'tau': [] for i in range(nlooks)}  # using .fromkeys() do not work because of common emptylist
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
    merged_xspecs = merged_xspecs.merge(xr.DataArray(centroid, name='centroid', attrs={'long_name':'Doppler centroid', 'units':''}).to_dataset())
    merged_xspecs.attrs.update({'look_width': look_width, 'tau': tau})
    return merged_xspecs