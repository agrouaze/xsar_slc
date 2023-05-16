#!/usr/bin/env python
# coding=utf-8
"""
"""
import logging
import numpy as np
import xarray as xr
from shapely.geometry import Polygon, GeometryCollection


def FullResolutionInterpolation(lines, samples, field, geolocation_annotation, azimuthTimeInterval):
    """
    Interpolate a field of geolocation annotation at given (line, sample) locations. Azimuth time non-monotonous variations have to be carrefully taken into account.
    Args:
        lines (xarray.DataArray) : line coordinates where to interpole
        samples (xarray.DataArray) : sample coordinates where to interpole
        field (str) : name of field in geolocation_annotation to interpolate
        geolocation_annotation (xarray.Dataset) : dataset of geolocation annotation
        azimuthTimeInterval (float): azimuth time interval [s]
    Returns:
        (xarray.DataArray): interpolated values of field at provided (lines, samples) coordinates
    """
    from scipy.interpolate import RectBivariateSpline
    geolocated_lines = line2geolocline(lines, geolocation_annotation, azimuthTimeInterval)  # Find geolocated lines
    LR_field = geolocation_annotation[field]  # Low resolution field
    field_interpolator = RectBivariateSpline(LR_field['line'].data, LR_field['sample'].data,
                                             LR_field.transpose('line', 'sample').data, kx=3,
                                             ky=3)  # interpolator of low resolution field
    out = from_HDresampler(geolocated_lines, samples, field_interpolator).rename(field)
    out.attrs.update(LR_field.attrs)
    return out


def line2geolocline(lines, geolocation_annotation, azimuth_time_interval):
    """
    line values in geoloc_annotation do not allow correct handling of overlapping area.
    In geoloc_annotaion, low resolution quantities are at beginning of each burst and last line of last burst.
    Interpolation on Full resolution measurement has to be done using the azimuth Time.
    Args:
        geolocation_annotation (xarray.Dataset): dataset of geolocation annotation
        azimuth_time_interval (float) : azimuth time interval [s]
        lines (np.1darray or xarray.DataArray) : array of line number
    Returns:
        geoloc_lines (np.1darray) : Equivalent geolocated line number to interpolate over low resolution geolocation quantities
    """
    aziTime = geolocation_annotation['azimuthTime']
    geolines = geolocation_annotation['line']

    i_ref = np.clip(np.searchsorted(geolines, lines.data, side='left') - 1, 0,
                    geolines.sizes['line'] - 2)  # indice of burst containing lines (reference burst number)
    l_ref = geolines.isel(line=i_ref).data  # line number of the first line of reference burst
    az_ref = aziTime.isel(line=i_ref, sample=0).data  # azimuth time of the first line of reference burst
    az = az_ref + ((lines.data - l_ref) * azimuth_time_interval * 1e9).astype('<m8[ns]')  # azimuth time of the lines

    i_ref = np.clip(np.searchsorted(aziTime.isel(sample=0), az, side='left') - 1, 0,
                    geolines.sizes['line'] - 2)  # indice of reference burst is updated for the overlapping parts only
    az_ref = aziTime.isel(line=i_ref,
                          sample=0).data  # azimuth time of the first line of reference burst is updated for the overlapping parts
    l_ref = geolines.isel(
        line=i_ref).data  # line number of the first line of reference burst is updated for the overlapping parts

    az_ref2 = aziTime.isel(line=i_ref + 1, sample=0).data
    l_ref2 = geolines.isel(line=i_ref + 1).data
    delta = (az_ref2 - az_ref) / (
            l_ref2 - l_ref)  # rate of azimuth time variation VS line number in the low resolution geolocation annotation
    geoloc_lines = l_ref + (az - az_ref) / delta
    geoloc_lines = np.where(i_ref!=geolines.sizes['line'] - 2, geoloc_lines,lines.data) # Ensuring lines and geolines match on last burst only !
    if isinstance(lines, xr.DataArray):
        geoloc_lines = xr.DataArray(geoloc_lines, dims=lines.dims, coords=lines.coords).rename('geolocated_line')

    return geoloc_lines


def from_HDresampler(geoloc_lines, samples, HD_resampler):
    """
    Call the interplator taking care of the strucutre of geoloc_lines and sample
    """
    if geoloc_lines.sizes == samples.sizes:
        res = HD_resampler(geoloc_lines.values, samples.values, grid=False)
        res = xr.DataArray(res, dims=geoloc_lines.dims).assign_coords(geoloc_lines.coords)
    else:
        l, s = np.meshgrid(geoloc_lines, samples, indexing='ij')
        l, s = l.reshape((1, np.prod(l.shape))), s.reshape((1, np.prod(s.shape)))
        res = HD_resampler(l, s, grid=False)
        res = res.reshape((len(geoloc_lines), len(samples)))
        res = xr.DataArray(res, dims=geoloc_lines.dims + samples.dims)
        res = res.assign_coords(geoloc_lines.coords)
        res = res.assign_coords(samples.coords)
    return res


def is_ocean(polygon, landmask):
    """
    Check if polygon only contains ocean (no land)
    Args: 
        polygon (2-tuple of lists or shapely.geometry.Polygon) : tuple of form (list of lons, list of lats) or cartopy.polygon to be checked
        landmask (shapely.geometry.MultiPolygon) : the land mask to be used (eg cartopy.feature.NaturalEarthFeature)
    Return:
        (bool): True if polygon is only over water
    """
    if not isinstance(polygon, Polygon):
        polygon = Polygon(zip(*polygon))

    land_geom = list(landmask.geometries())
    gseries = GeometryCollection(land_geom)
    land_intersection = gseries.intersection(polygon)
    return land_intersection.area == 0.0


def netcdf_compliant(dataset):
    """
    Create a dataset that can be written on disk with xr.Dataset.to_netcdf() function. It split complex variable in real and imaginary variable

    Args:
        dataset (xarray.Dataset): dataset to be transform
    """
    var_to_rm = list()
    var_to_add = list()
    for i in dataset.variables.keys():
        if dataset[i].dtype == complex:
            re = dataset[i].real
            # re.encoding['_FillValue'] = 9.9692099683868690e+36
            im = dataset[i].imag
            # im.encoding['_FillValue'] = 9.9692099683868690e+36
            var_to_add.append({str(i) + '_Re': re, str(i) + '_Im': im})
            var_to_rm.append(str(i))
    ds_to_save = xr.merge([dataset.drop_vars(var_to_rm), *var_to_add], compat='override')
    for vv in ds_to_save.variables.keys():
        if ds_to_save[vv].dtype == 'int64':  # to avoid ncview: netcdf_dim_value: unknown data type (10) for corner_line ...
            ds_to_save[vv] = ds_to_save[vv].astype(np.int16)
        elif ds_to_save[vv].dtype == 'float64':
            ds_to_save[vv] = ds_to_save[vv].astype(np.float32) # to reduce volume of output files
        else:
            logging.info('%s is dtype %s',vv,ds_to_save[vv].dtype)
    # for vv in dataset.variables.keys():
    #     if dataset[vv].dtype == 'float64':
    #         dataset[vv] = dataset[vv].astype(np.float32)
    #         dataset[vv].encoding['_FillValue'] = 9.9692099683868690e+36
    # if 'pol' in dataset:
    #     dataset['pol'] = dataset['pol'].astype('S1')
    #     dataset['pol'].encoding['_FillValue'] = ''
    return ds_to_save


def gaussian_kernel(width, spacing, truncate=3.):
    """
    Compute a Gaussian kernel for filtering. The width correspond to the wavelength that is needed to be kept. The standard deviation of the gaussian has to be width/(2 pi)
    
    Args:
        width (dict): form {name of dimension (str): width in [m] (float)}
        spacing (dict): form {name of dimension (str): spacing in [m] (float)}
        truncate (float): gaussian shape is truncate at +/- (truncate x width) value
    """
    gk = 1.
    width= {d:w/(2*np.pi) for d,w in width.items()} # frequency cut off has a 2 pi factor
    for d in width.keys():
        coord = np.arange(-truncate * width[d], truncate * width[d], spacing[d])
        coord = xr.DataArray(coord, dims=d, coords={d: coord})
        gk = gk * np.exp(-coord ** 2 / (2 * width[d] ** 2))
    gk /= gk.sum()
    return gk

def rect_kernel(width, spacing):
    """
    Compute a rectangular window kernel for filtering
    
    Args:
        width (dict): form {name of dimension (str): width in [m] (float)}
        spacing (dict): form {name of dimension (str): spacing in [m] (float)}
    """
    wk = 1.
    for d in width.keys():
        coord = np.arange(-width[d]/2, width[d]/2, spacing[d])
        win = xr.DataArray(np.ones_like(coord), dims=d, coords={d: coord})
        wk = wk * win
    wk /= wk.sum()
    return wk

def haversine(lon1, lat1, lon2, lat2):
    """
    Compute great circle distance and bearing starting from (lon1, lat1)
    
    Args:
        lon1 (float): initial longitude
        lat1 (float): initial latitude
        lon2 (float or array of float): final longitude
        lat2 (float or array of float): final latitude
    
    Returns:
        (tuple of array): (great cicle distance, bearing [def] from North clockwise)
    """
    lat2 = np.array(lat2, ndmin=1)
    lon2 = np.array(lon2, ndmin=1)
    Re = 6371e3 # Radius of earth
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    v1 = np.stack([np.cos(lat1)*np.cos(lon1), np.cos(lat1)*np.sin(lon1), np.sin(lat1)])[:, np.newaxis]
    v2 = np.stack([np.cos(lat2)*np.cos(lon2), np.cos(lat2)*np.sin(lon2), np.sin(lat2)])
    v1v2 = (v1*v2).sum(axis=0, keepdims=True)
    v1v1 = (v1*v1).sum(axis=0, keepdims=True)
    w = times(v1,v2)
    w/=np.linalg.norm(w, axis=0, keepdims=True)
    e = times(w,v1)
    n = np.stack([-np.sin(lat1)*np.cos(lon1), -np.sin(lat1)*np.sin(lon1), np.cos(lat1)])[:, np.newaxis]# Axis pointing to North and tangent to the sphere
    nte = times(n,e)
    sinb = -np.linalg.norm(nte, axis=0, keepdims=True)*np.sign((nte*v1).sum(axis=0, keepdims=True))
    cosb = (n*e).sum(axis=0, keepdims=True)
    a = np.arccos(v1v2)
    b = np.degrees(np.arctan2(sinb, cosb))
    return Re*a.squeeze(),b.squeeze()

def times(a,b, axis=0):
    """
    cross product:  a x b assuming space dimension is on indices [0,1,2] on axis =  axis
    a and b must have the same shape
    
    Args:
        a (ndarray): first argument
        b (ndarray): second argument
    
    Returns:
        (ndarray): same shape as a and b
    """
    return np.stack([a.take(1, axis=axis)*b.take(2, axis=axis)-a.take(2, axis=axis)*b.take(1, axis=axis),
    a.take(2, axis=axis)*b.take(0, axis=axis)-a.take(0, axis=axis)*b.take(2, axis=axis),
    a.take(0, axis=axis)*b.take(1, axis=axis)-a.take(1, axis=axis)*b.take(0, axis=axis)], axis=axis)



def xndindex(sizes):
    """
    xarray equivalent of np.ndindex iterator with defined dimension names
    
    Args:
        sizes (dict): dict of form {dimension_name (str): size(int)}
    Return:
        iterator over dict
    """
    from itertools import repeat

    for d, k in zip(repeat(tuple(sizes.keys())), zip(np.ndindex(tuple(sizes.values())))):
        yield {k: l for k, l in zip(d, k[0])}


def xtiling(ds, nperseg, noverlap=0, centering=False, side='left', prefix='tile_'):
    """
    Define tiles indexes of an xarray of abritrary shape. Name of returned coordinates are prefix+keys of nperseg
    
    Note1: Coordinates of returned arrays depends on type of ds.
    If ds is an xarray instance, returned coordinates are in ds coordinate referential.
    If ds is a dict of shape, returned coordinates are assumed starting from zero.
    
    Note2 : If nperseg is a dict and contains one value set as None (or zero), a tile dimension is created along the corresponding dimension with nperseg set as
    the corresponding shape of ds. This is a different behaviour than not providing the dimension in nperseg
    
   
    Args:
        ds (xarray.DataArray or xarray.Dataset or dict). Array to tile or dict with form {dim1 (str):size1 (int), dim2 (str):size2 (int), ...}
        nperseg (int or dict): number of point per tile for each dimension. If defined as int, nperseg is applied on all dimensions of ds. If ds is a dict, form must be {tile_dim1 (str):number_of_point_per_segment(int), tile_dim2 (str):number_of_point_per_segment(int), ...}
        noverlap (int or dict, optional): Number of overlapping points between tiles for each dimension (default is zero). dict of form {tile_dim1 (str):number_of_overlapping_point(int), tile_dim2 (str):number_of_overlapping_point(int), ...}
        centering (dict of bool, optional): If False, first tile starts at first index. If True, number of unused points are equal on each side (see side parameter)
        side (str): 'left' or 'right' If centering is True and unused number of points is odd, side parameter defines on which side there is one more ununsed point
        prefix (str, optional): prefix to add to tiles dimension name in order to define tile coordinates
    Return:
        (dict of xarray.DataArray): keys are the same as nperseg keys and values are xarray indexes of tile in the corresponding dimension.
    """
    import warnings

    sizes = ds.sizes if isinstance(ds, xr.DataArray) or isinstance(ds, xr.Dataset) else ds

    if isinstance(nperseg, int):
        nperseg = dict.fromkeys(sizes.keys(), nperseg)

    if isinstance(noverlap, int):
        noverlap = dict.fromkeys(nperseg, noverlap)
    elif nperseg.keys() != noverlap.keys():
        noverlap = {d: 0 if d not in noverlap.keys() else noverlap[d] for d in nperseg.keys()}
    else:
        pass

    if centering is True:
        centering = dict.fromkeys(nperseg, True)
    elif centering is False:
        centering = dict.fromkeys(nperseg, False)
    elif nperseg.keys() != centering.keys():
        centering = {d: False if d not in centering.keys() else centering[d] for d in nperseg.keys()}
    else:
        pass

    if side == 'left':
        side = dict.fromkeys(nperseg, 'left')
    elif side == 'right':
        side = dict.fromkeys(nperseg, 'right')
    elif nperseg.keys() != side.keys():
        side = {d: 'left' if d not in side.keys() else side[d] for d in nperseg.keys()}
    else:
        pass

    dims = nperseg.keys()  # list of dimensions to work on

    for d in dims:
        if nperseg[d] in (0, None):
            nperseg[d] = sizes[d]
            noverlap[d] = 0
        if sizes[d] < nperseg[d]:
            warnings.warn(
                "Dimension '{}' ({}) is smaller than required nperseg :{}. nperseg is adjusted accordingly and noverlap forced to zero".format(
                    d, sizes[d], nperseg[d]))
            nperseg[d] = sizes[d]
            noverlap[d] = 0

    steps = {d: nperseg[d] - noverlap[d] for d in dims}  # step between each tile
    
    if np.any([steps[d]<1 for d in dims]):
        raise ValueError("noverlap can not be equal or larger than nperseg")

    indices = {d: np.arange(0, sizes[d] - nperseg[d] + 1, steps[d]) for d in dims}  # index of first point of each tile

    # For centering option:
    for d in dims:
        if centering[d]:
            ishift = {'left': 1, 'right': 0}[side[d]]
            indices[d] = indices[d] + (sizes[d] - indices[d][-1] - nperseg[d] + ishift) // 2

    tiles_index = {d: xr.DataArray(np.empty([len(indices[d]), nperseg[d]], dtype=int), dims=[d, '__' + d]) for d, ind in
                   nperseg.items()}

    for d in dims:
        for i in range(tiles_index[d].sizes[d]):
            tiles_index[d][i] = np.arange(indices[d][i], indices[d][i] + nperseg[d])

    if isinstance(ds, xr.DataArray) or isinstance(ds, xr.Dataset):
        coords = {d: ds[d][indices[d]].data + nperseg[d] // 2 for d in dims}
    else:
        coords = {d: indices[d] + nperseg[d] // 2 for d in dims}

    tiles_index = {d: k.assign_coords({d: coords[d]}).rename({d: prefix + d}).rename('tile_{}_index'.format(d)) for d, k
                   in tiles_index.items()}

    for d, v in tiles_index.items():
        v[prefix + d].attrs.update(
            {'long_name': 'index of tile middle point', 'nperseg': nperseg[d], 'noverlap': noverlap[d]})

    return tiles_index


def get_corner_tile(tiles):
    """
    Extract corner indexes of tiles.. Returns an index, not the coordinate !
    Args:
        tiles (dict of xarray) : xtiling() output or {key:value} with values being DataArray of slices
    Return:
        (dict of xarray): same keys as tiles, values ares corners indexes only
    """
    # function below if used if tiles contains list of slices instead of index values
    slice_bound_indexes = lambda slices: np.array([[s.start, s.stop-1] for s in slices])
    
    corners = dict()
    for d, v in tiles.items():
        if v.dtype!=int:
            corners[d] = xr.apply_ufunc(slice_bound_indexes, v, input_core_dims=[['tile_'+d]],output_core_dims=[['tile_'+d,'c_'+d]])
        else:
            corners[d] = v[{'__' + d: [0, -1]}].rename({'__' + d: 'c_' + d})
    return corners

def get_middle_tile(tiles):
    """
    Extract middle indexes of tiles. Returns an index, not the coordinate !
    Args:
        tiles (dict of xarray) : xtiling() output or {key:value} with values being DataArray of slices
    Return:
        (dict of xarray): same keys as tiles, values ares middle indexes
    """
    # function below if used if tiles contains list of slices instead of index values
    slice_middle_indexes = lambda slices:np.array([(s.stop+s.start)//2 for s in slices])
    
    middle = dict()
    for d, v in tiles.items():
        if v.dtype!=int:
            middle[d] = xr.apply_ufunc(slice_middle_indexes, v, input_core_dims=[['tile_'+d]],output_core_dims=[['tile_'+d]])
        else:
            middle[d] = v[{'__' + d: v.sizes['__' + d] // 2}]
    return middle

def get_tiles(ds, tiles_index):
    """
    Returns the list of all tiles taken over tiles_index. tiles_index could be providing using xtiling (uniform tiling) or be custom using slices instead of integers.
    Args:
        ds (xarray.dataset/xarray.DataArray)
        tiles_index (dict): keys are dimensions of ds to be indexeded. Values are xarray.DataArray containing indexes or slices
    Returns
        (list) : a list of all tiles
    
    """
    uniform_tiles_index = {}
    non_uniform_tiles_index = {}
    for d,v in tiles_index.items():
        if v.dtype!=int:
            non_uniform_tiles_index.update({d:v})
        else:
            uniform_tiles_index.update({d:v})    
    
    uds = ds[uniform_tiles_index] # taking tiles over all uniform dimensions
    
    uniform_tile_sizes = dict() # sizes of uniform tile dimensions
    for d,k in uniform_tiles_index.items():
        uniform_tile_sizes.update({b:j for b,j in k.sizes.items() if 'tile_' in b})
    
    if non_uniform_tiles_index: # taking all the tiles over all non-uniform dimensions
        uds = {'tile_'+d:[uds[{d:v[s].item()}].assign_coords({td:v[td][{td:tv}].item() for td,tv in s.items()}).swap_dims({d:'__'+d}) for s in xndindex(v.sizes)] for d,v in non_uniform_tiles_index.items()}
    
    # concatenation of all possible tiles
    all_tiles_list = list()
    if non_uniform_tiles_index:
        for heterogen_dim, tile_list in uds.items():
            for ts in tile_list:
                all_tiles_list = all_tiles_list+[ts[i] for i in xndindex(uniform_tile_sizes)]
    else:
        all_tiles_list = [uds[i] for i in xndindex(uniform_tile_sizes)]
        
    return all_tiles_list


# def get_corner_tile(tiles):
#     """
#     Extract corner indexes of tiles
#     Args:
#         tiles (dict of xarray) : xtiling()
#     Return:
#         (dict of xarray): same keys as tiles, values ares corners indexes only
#     """
#     corners = dict()
#     for d, v in tiles.items():
#         corners[d] = v[{'__' + d: [0, -1]}].rename({'__' + d: 'corner_' + d})
#     return corners


# def get_middle_tile(tiles):
#     """
#     Extract middle indexes of tiles
#     Args:
#         tiles (dict of xarray) : xtiling()
#     Return:
#         (dict of xarray): same keys as tiles, values ares middle indexes
#     """
#     middle = dict()
#     for d, v in tiles.items():
#         middle[d] = v[{'__' + d: v.sizes['__' + d] // 2}]
#     return middle
