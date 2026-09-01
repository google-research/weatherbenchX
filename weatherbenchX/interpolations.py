# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Definition of interpolation classes."""

import abc
from collections.abc import Iterable
import dataclasses
from typing import Hashable, Mapping, Optional, Sequence, Union
import numpy as np
import scipy.spatial
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import spatial
from weatherbenchX.metrics import wrappers
import xarray as xr


class Interpolation(abc.ABC):
  """Interpolation base class."""

  @abc.abstractmethod
  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    """Implementation of the interpolation function for a single variable."""

  def interpolate(
      self,
      ds: Mapping[Hashable, xr.DataArray],
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    """Interpolates dataset, potentially according to a reference dataset.

    Args:
      ds: Xarray dataset to be interpolated.
      reference: Optional reference dataset, e.g. target.

    Returns:
      interpolated_ds: Interpolated dataset.
    """
    if reference is None:
      return xarray_tree.map_structure(self.interpolate_data_array, ds)
    else:
      return xarray_tree.map_structure(
          self.interpolate_data_array, ds, reference
      )


@dataclasses.dataclass
class MultipleInterpolation(Interpolation):
  """Applies multiple interpolations to a dataset in sequence.

  Attributes:
    interpolations: List of interpolations to be applied in sequence.
  """

  interpolations: Sequence[Interpolation]

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    for interpolation in self.interpolations:
      da = interpolation.interpolate_data_array(da, reference)
    return da


def pad_longitude(da: xr.DataArray) -> xr.DataArray:
  """Pad longitude values to allow for wrapped interpolation."""
  left = da.isel(longitude=[-1])
  left = left.assign_coords(longitude=left.longitude.values - 360)
  right = da.isel(longitude=[0])
  right = right.assign_coords(longitude=right.longitude.values + 360)
  return xr.concat([left, da, right], 'longitude')


def interpolate_to_coords(
    da: xr.DataArray,
    dim_args: Mapping[str, Union[xr.DataArray, np.ndarray]],
    method: str,
    extrapolate_out_of_bounds: bool = True,
) -> xr.DataArray:
  """Interpolate to a fixed set of coordinates."""
  if extrapolate_out_of_bounds:
    # See xarray documentation for interpolation behaviour.
    # https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html
    if len(dim_args) > 1:
      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interpn.html
      interp_kwargs = {'fill_value': None}
    else:
      # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
      interp_kwargs = {'fill_value': 'extrapolate'}
  else:
    interp_kwargs = None

  out = da.interp(
      **dim_args,  # pyrefly: ignore[bad-argument-type]
      method=method,  # pyrefly: ignore[bad-argument-type]
      kwargs=interp_kwargs,
  )  # pytype: disable=wrong-arg-types
  return out


class CropToBox(Interpolation):
  """Crops the dataset to the given bounding box.

  Since interpolation is called before compute(), this can be useful to reduce
  the amount of data that is read into memory when you are only interested in
  a particular area.

  This is essentially a wrapper around an xarray.Dataset.sel() call.
  """

  def __init__(
      self,
      lat_min: float,
      lat_max: float,
      lon_min: float,
      lon_max: float,
  ):
    """Init.

    Args:
      lat_min: Minimum latitude to crop to (inclusive).
      lat_max: Maximum latitude to crop to (inclusive).
      lon_min: Minimum longitude to crop to (exclusive).
      lon_max: Maximum longitude to crop to (exclusive).
    """
    if lat_min > lat_max:
      raise ValueError(f'Invalid latitudes: {lat_min} and {lat_max}')
    if lon_min > lon_max:
      raise ValueError(f'Invalid longitudes: {lon_min} and {lon_max}')
    self._lat_min = lat_min
    self._lat_max = lat_max
    self._lon_min = lon_min
    self._lon_max = lon_max

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    # Some datasets have latitude in the descending order, or longitude that
    # wraps around, so just in case, we will sort by those coordinates first.
    da = da.sortby('longitude', ascending=True)
    da = da.sortby('latitude', ascending=True)
    da = da.sel(
        latitude=slice(self._lat_min, self._lat_max),
        longitude=slice(self._lon_min, self._lon_max),
    )
    return da


class InterpolateToFixedCoords(Interpolation):
  """Interpolate to a fixed set of coordinates.

  Interplation is done using xarray's built-in interp method:
  https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html
  """

  def __init__(
      self,
      method: str,
      coords: Mapping[str, Union[xr.DataArray, np.ndarray]],
      wrap_longitude: bool = False,
      extrapolate_out_of_bounds: bool = True,
  ):
    """Init.

    Args:
      method: Interpolation method to be passed to xarray's interpolation API.
      coords: Dictionary of coordinate names and values to interpolate to.
      wrap_longitude: If True, perform a wrapped interpolation in the longitude
        dimension. Default: False
      extrapolate_out_of_bounds: If True, extrapolate to out of bounds values
        using the chosen interpolation method. Default: True
    """
    self._method = method
    self._coords = coords
    self._wrap_longitude = wrap_longitude
    self._extrapolate_out_of_bounds = extrapolate_out_of_bounds

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:

    if self._wrap_longitude:
      # TODO(srasp): Raise error if this isn't True but seems like it should be.
      da = pad_longitude(da)

    interpolated_da = interpolate_to_coords(
        da,
        self._coords,
        self._method,
        self._extrapolate_out_of_bounds,
    )
    return interpolated_da


class InterpolateToReferenceCoords(Interpolation):
  """Interpolate to a reference dataset.

  Interplation is done using xarray's built-in interp method:
  https://docs.xarray.dev/en/latest/generated/xarray.DataArray.interp.html
  """

  def __init__(
      self,
      method: str,
      dims: Optional[Sequence[str]] = None,
      wrap_longitude: bool = False,
      clip_reference_coords: Optional[Iterable[str]] = None,
      extrapolate_out_of_bounds: bool = True,
  ):
    """Init.

    Args:
      method: Interpolation method to be passed to xarray's interpolation API.
      dims: (Optional) Dimensions over which to interpolate. If None (default),
        infer dimensions from intersect of DataArray dimensions and reference
        coordinates.
      wrap_longitude: If True, perform a wrapped interpolation in the longitude
        dimension. Default: False
      clip_reference_coords: Clip the reference dataset to the maximum extent of
        the data to be interpolated in the given dimensions, e.g. ['latitude',
        'longitude']. Note that this can potentially lead to errors in the
        reference go unnoticed. It is preferred to use a fixed interpolation
        instead or ensure that the reference extent matches beforehand. Default:
        None.
      extrapolate_out_of_bounds: If True, extrapolate to out of bounds values
        using the chosen interpolation method. Default: True
    """
    self._method = method
    self._dims = dims
    self._wrap_longitude = wrap_longitude
    self._clip_reference_coords = clip_reference_coords
    self._extrapolate_out_of_bounds = extrapolate_out_of_bounds

  def interpolate_data_array(  # pyrefly: ignore[bad-override]
      self,
      da: xr.DataArray,
      reference: xr.DataArray,  # pytype: disable=signature-mismatch
  ) -> xr.DataArray:

    if self._wrap_longitude:
      da = pad_longitude(da)

    if self._clip_reference_coords is not None:
      for coord in self._clip_reference_coords:
        reference = reference.sel(
            {coord: slice(da[coord].min(), da[coord].max())}
        )

    # If dims not explicit, interpolate all dims that have a corresponding
    # coordinate in the reference.
    if self._dims is None:
      dims = [d for d in da.dims if d in reference.coords]
    else:
      dims = self._dims

    # Catch case where reference doesn't contain any data.
    if reference.size == 0:
      # Need to make sure to retain any dimensions that are not being
      # interpolated.
      da_dims_to_retain = set(da.dims) - set(dims)
      return reference.copy().expand_dims({d: da[d] for d in da_dims_to_retain})

    dim_args = {dim: reference[dim] for dim in dims}

    da_like_reference = interpolate_to_coords(
        da,
        dim_args,  # pyrefly: ignore[bad-argument-type]
        self._method,
        self._extrapolate_out_of_bounds,
    )
    return da_like_reference


LAPSE_RATE_K_PER_M = -0.0065  # Standard atmosphere lapse rate.


class GridToSparseWithAltitudeAdjustment(InterpolateToReferenceCoords):
  """Applies altitude adjustment to 2m_temperature and 10m_wind_speed.

  Alititude adjustments are based on the difference of the grid elevation to the
  station elevation. Reference:
  https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2372, Section 3.3.

  Assumes that elevations are in meters and an 'elevation' coordinate exists on
  the reference dataset. Requires passing a DataArray with the grid elevation
  corresponding to the dataset to be interpolated. Variables must be named
  '2m_temperature' and '10m_wind_speed'. Other variables will be left unchanged.

  Note:
    The same interpolation is applied to the grid_elevation as to the data,
    so in the case of linear interpolation, the elevation difference will also
    be based on the grid elevation linearly interpolated to the reference
    coordinates.
  """

  def __init__(
      self,
      method: str,
      grid_elevation: xr.DataArray,
      dims: Optional[Sequence[str]] = None,
      wrap_longitude: bool = False,
      extrapolate_out_of_bounds: bool = True,
      max_alititude_diff_in_m: float = 1500,
  ):
    """Init.

    Args:
      method: Interpolation method to be passed to xarray's interpolation API.
      grid_elevation: DataArray matching the dataset coordinates specifying the
        grid box elevation in m.
      dims: (Optional) Dimensions over which to interpolate. If None (default),
        infer dimensions from intersect of DataArray dimensions and reference
        coordinates.
      wrap_longitude: If True, perform a wrapped interpolation in the longitude
        dimension. Default: False
      extrapolate_out_of_bounds: If True, extrapolate to out of bounds values
        using the chosen interpolation method. Default: True
      max_alititude_diff_in_m: No adjustment is applied for elevation
        differences greater than this value. Large values can appear because of
        errors in the station dataset, e.g. elevation reported in ft instead of
        m. Default: 1500.
    """
    self._grid_elevation = grid_elevation
    self._max_alititude_diff_in_m = max_alititude_diff_in_m
    super().__init__(
        method=method,
        dims=dims,
        wrap_longitude=wrap_longitude,
        extrapolate_out_of_bounds=extrapolate_out_of_bounds,
    )

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: xr.DataArray,  # pytype: disable=signature-mismatch
  ) -> xr.DataArray:
    if da.name in ['2m_temperature', '10m_wind_speed']:
      # Sometimes coordinates don't match exactly, so we reassign them here.
      grid_elevation = self._grid_elevation.compute()
      xr.testing.assert_allclose(grid_elevation.latitude, da.latitude)
      xr.testing.assert_allclose(grid_elevation.longitude, da.longitude)
      grid_elevation = grid_elevation.assign_coords(
          latitude=da.latitude,
          longitude=da.longitude,
      )
      da.coords['grid_elevation'] = grid_elevation

    da_like_reference = super().interpolate_data_array(da, reference)
    if (
        da.name in ['2m_temperature', '10m_wind_speed']
        and da_like_reference.size > 0
    ):
      # Positive if station is higher than grid.
      sparse_higher_than_grid_m = (
          da_like_reference['elevation'] - da_like_reference['grid_elevation']
      )
      # Set "unrealistic" differences to 0.
      sparse_higher_than_grid_m = sparse_higher_than_grid_m.where(
          np.abs(sparse_higher_than_grid_m) < self._max_alititude_diff_in_m, 0
      )
      if da.name == '2m_temperature':
        adjustment = sparse_higher_than_grid_m * LAPSE_RATE_K_PER_M
        da_like_reference += adjustment
      elif da.name == '10m_wind_speed':
        # Only adjust stations > 100m above model orography.
        adjustment_factor = xr.ones_like(sparse_higher_than_grid_m)
        # Subtract 100m from the difference. I couldn't find this in the paper
        # but it does make sense so that the different regimes overlap.
        dz = sparse_higher_than_grid_m - 100
        adjustment_factor = adjustment_factor.where(
            sparse_higher_than_grid_m < 100,
            1 + 0.002 * dz,
        )
        adjustment_factor = adjustment_factor.where(
            sparse_higher_than_grid_m < 1100, 3
        )
        da_like_reference *= adjustment_factor
    return da_like_reference


def _latlon_to_cartesian(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
  """Converts latitude and longitude in degrees to 3D unit sphere Cartesian coordinates."""
  phi = np.deg2rad(lats)
  theta = np.deg2rad(lons)
  x = np.cos(phi) * np.cos(theta)
  y = np.cos(phi) * np.sin(theta)
  z = np.sin(phi)
  return np.stack([x, y, z], axis=-1)


def _compute_nearest_weights(
    grid_xyz: np.ndarray,
    target_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Computes nearest neighbor indices and unit weights."""
  _, nx, _ = grid_xyz.shape
  ns = target_xyz.shape[0]
  kdtree = scipy.spatial.cKDTree(grid_xyz.reshape(-1, 3))
  _, flat_idx = kdtree.query(target_xyz, k=1)
  yc = flat_idx // nx
  xc = flat_idx % nx
  corner_y = yc[np.newaxis, :]
  corner_x = xc[np.newaxis, :]
  weights = np.ones((1, ns), dtype=np.float32)
  valid_mask = np.ones(ns, dtype=bool)
  return corner_y, corner_x, weights, valid_mask


def _compute_linear_weights(
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    target_lats: np.ndarray,
    target_lons: np.ndarray,
    extrapolate_out_of_bounds: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Computes Delaunay simplex corner indices and barycentric weights."""
  ny, nx = grid_lats.shape
  lon_center = float(np.nanmedian(grid_lons))
  grid_lons_c = (grid_lons - lon_center + 180.0) % 360.0 - 180.0
  target_lons_c = (target_lons - lon_center + 180.0) % 360.0 - 180.0

  grid_pts = np.column_stack([grid_lats.ravel(), grid_lons_c.ravel()])
  target_pts = np.column_stack([target_lats.ravel(), target_lons_c.ravel()])

  tri = scipy.spatial.Delaunay(grid_pts)
  simplex_indices = tri.find_simplex(target_pts)
  valid_mask = simplex_indices >= 0

  safe_simplex = np.where(valid_mask, simplex_indices, 0)
  vertex_indices = tri.simplices[safe_simplex].copy()

  va = grid_pts[vertex_indices[:, 0]]
  vb = grid_pts[vertex_indices[:, 1]]
  vc = grid_pts[vertex_indices[:, 2]]

  v0 = vb - va
  v1 = vc - va
  v2 = target_pts - va

  det = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
  det_safe = np.where(np.abs(det) < 1e-12, 1e-12, det)
  wb = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / det_safe
  wc = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / det_safe
  wa = 1.0 - wb - wc

  if extrapolate_out_of_bounds and not np.all(valid_mask):
    kdtree = scipy.spatial.cKDTree(grid_pts)
    _, nn_idx = kdtree.query(target_pts[~valid_mask], k=1)
    vertex_indices[~valid_mask, 0] = nn_idx
    vertex_indices[~valid_mask, 1] = nn_idx
    vertex_indices[~valid_mask, 2] = nn_idx
    wa[~valid_mask] = 1.0
    wb[~valid_mask] = 0.0
    wc[~valid_mask] = 0.0
    valid_mask = np.ones_like(valid_mask)
  else:
    wa = np.where(valid_mask, wa, 0.0)
    wb = np.where(valid_mask, wb, 0.0)
    wc = np.where(valid_mask, wc, 0.0)

  corner_y = (vertex_indices // nx).T
  corner_x = (vertex_indices % nx).T
  weights = np.stack([wa, wb, wc], axis=0).astype(np.float32)

  return corner_y, corner_x, weights, valid_mask


def _lookup_coord(
    da: xr.DataArray, name: str, fallbacks: Sequence[str]
) -> xr.DataArray:
  """Finds a coordinate in da.coords, checking fallbacks if name is missing."""
  if name in da.coords:
    return da.coords[name]
  for fb in fallbacks:
    if fb in da.coords:
      return da.coords[fb]
  raise KeyError(
      f'Coordinate {name!r} not found in da.coords (available coords: '
      f'{list(da.coords.keys())})'
  )


def _get_grid_lat_lon(
    da: xr.DataArray,
    lat_coord_name: str,
    lon_coord_name: str,
    spatial_dims: Optional[Sequence[str]],
) -> tuple[np.ndarray, np.ndarray, Sequence[str]]:
  """Extracts 2D grid latitude, longitude arrays and spatial dimension names."""
  lat_da = _lookup_coord(da, lat_coord_name, ('latitude', 'lat'))
  lon_da = _lookup_coord(da, lon_coord_name, ('longitude', 'lon'))
  if lat_da.ndim == 1:
    grid_lats, grid_lons = np.meshgrid(
        lat_da.values, lon_da.values, indexing='ij'
    )
    src_spatial_dims = spatial_dims or (
        str(lat_da.dims[0]),
        str(lon_da.dims[0]),
    )
  elif lat_da.ndim == 2:
    grid_lats = np.asarray(lat_da.values)
    grid_lons = np.asarray(lon_da.values)
    src_spatial_dims = spatial_dims or tuple(str(d) for d in lat_da.dims)
  else:
    raise ValueError(
        f'Expected 1D or 2D coordinate for {lat_coord_name}, got {lat_da.ndim}D'
    )
  return grid_lats, grid_lons, src_spatial_dims


def _build_output_data_array(
    da: xr.DataArray,
    interpolated_vals: np.ndarray,
    src_spatial_dims: Sequence[str],
    ref_dim: str,
    reference: xr.DataArray,
) -> xr.DataArray:
  """Constructs the output DataArray with preserved non-spatial coordinates."""
  non_spatial_dims = [
      str(d) for d in da.dims if d not in src_spatial_dims and d != ref_dim
  ]
  out_dims = (*non_spatial_dims, ref_dim)
  out_coords = {}
  for c in da.coords:
    if (
        not any(d in da.coords[c].dims for d in src_spatial_dims)
        and c != ref_dim
    ):
      if c not in reference.coords:
        out_coords[c] = da.coords[c]
  for c in reference.coords:
    if ref_dim in reference.coords[c].dims:
      out_coords[c] = reference.coords[c]

  return xr.DataArray(
      data=interpolated_vals,
      dims=out_dims,
      coords=out_coords,
      name=da.name,
      attrs=da.attrs,
  )


class GridToSparseInterpolation(Interpolation):
  """Interpolates gridded data to sparse point reference coordinates.

  Supports both 1D regular grids (latitude, longitude) and 2D curvilinear
  or projected grids (e.g. Lambert Conformal (y, x)) using Delaunay
  triangulation and barycentric interpolation or nearest neighbor interpolation.

  Grid geometry and interpolation weights are computed once and cached on the
  interpolator instance, enabling fast O(1) evaluation for subsequent time slices
  and variables.
  """

  def __init__(
      self,
      method: str = 'linear',
      spatial_dims: Optional[Sequence[str]] = None,
      lat_coord_name: str = 'latitude',
      lon_coord_name: str = 'longitude',
      extrapolate_out_of_bounds: bool = False,
  ):
    """Init.

    Args:
      method: Interpolation method: 'linear' (Delaunay barycentric) or 'nearest'.
      spatial_dims: (Optional) The spatial dimensions of the source grid, e.g.
        ('y', 'x') or ('latitude', 'longitude'). If None, inferred from the
        latitude coordinate dimensions.
      lat_coord_name: Name of the latitude coordinate. Default: 'latitude'.
      lon_coord_name: Name of the longitude coordinate. Default: 'longitude'.
      extrapolate_out_of_bounds: If True, extrapolate out of bounds points to
        the nearest boundary point. If False, set out of bounds points to NaN.
        Default: False.
    """
    if method not in ('linear', 'nearest'):
      raise ValueError(f"method must be 'linear' or 'nearest', got {method}")
    self._method = method
    self._spatial_dims = (
        tuple(spatial_dims) if spatial_dims is not None else None
    )
    self._lat_coord_name = lat_coord_name
    self._lon_coord_name = lon_coord_name
    self._extrapolate_out_of_bounds = extrapolate_out_of_bounds
    self._cached_weights = None
    self._cached_key = None

  def _compute_weights(
      self,
      grid_lats: np.ndarray,
      grid_lons: np.ndarray,
      target_lats: np.ndarray,
      target_lons: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if self._method == 'nearest':
      grid_xyz = _latlon_to_cartesian(grid_lats, grid_lons)
      target_xyz = _latlon_to_cartesian(target_lats, target_lons)
      return _compute_nearest_weights(grid_xyz, target_xyz)
    return _compute_linear_weights(
        grid_lats,
        grid_lons,
        target_lats,
        target_lons,
        extrapolate_out_of_bounds=self._extrapolate_out_of_bounds,
    )

  def interpolate_data_array(  # pyrefly: ignore[bad-override]
      self,
      da: xr.DataArray,
      reference: xr.DataArray,  # pytype: disable=signature-mismatch
  ) -> xr.DataArray:
    if reference is None:
      raise ValueError(
          'Reference DataArray with target coordinates is required.'
      )

    target_lat_da = _lookup_coord(
        reference, self._lat_coord_name, ('latitude', 'lat')
    )
    target_lon_da = _lookup_coord(
        reference, self._lon_coord_name, ('longitude', 'lon')
    )
    ref_dim = str(target_lat_da.dims[0])
    grid_lats, grid_lons, src_spatial_dims = _get_grid_lat_lon(
        da, self._lat_coord_name, self._lon_coord_name, self._spatial_dims
    )

    if reference.size == 0 or reference.sizes.get(ref_dim, 0) == 0:
      non_spatial_dims = [
          str(d) for d in da.dims if d not in src_spatial_dims and d != ref_dim
      ]
      shape = tuple(da.sizes[d] for d in non_spatial_dims) + (0,)
      interpolated_vals = np.empty(shape, dtype=da.dtype)
      return _build_output_data_array(
          da, interpolated_vals, src_spatial_dims, ref_dim, reference
      )

    target_lats = np.asarray(target_lat_da.values)
    target_lons = np.asarray(target_lon_da.values)

    cache_key = (
        grid_lats.shape,
        target_lats.shape,
        float(grid_lats[0, 0]) if grid_lats.size > 0 else 0.0,
        float(target_lats[0]) if target_lats.size > 0 else 0.0,
        self._method,
        self._extrapolate_out_of_bounds,
    )
    if self._cached_key != cache_key or self._cached_weights is None:
      self._cached_weights = self._compute_weights(
          grid_lats, grid_lons, target_lats, target_lons
      )
      self._cached_key = cache_key

    corner_y, corner_x, weights, valid_mask = self._cached_weights

    non_spatial_dims = [str(d) for d in da.dims if d not in src_spatial_dims]
    matching_coords = {
        d: xr.DataArray(reference.coords[d].values, dims=[ref_dim])
        for d in non_spatial_dims
        if d in reference.coords
        and ref_dim in reference.coords[d].dims
        and da.sizes.get(d, 1) > 1
    }
    if matching_coords:
      da = da.sel(matching_coords)

    if ref_dim in da.dims:
      remaining_dims = [
          str(d) for d in da.dims if d not in src_spatial_dims and d != ref_dim
      ]
      da_ordered = da.transpose(*remaining_dims, ref_dim, *src_spatial_dims)
      vals = da_ordered.values
      idx = np.arange(target_lats.shape[0])[None, :]
      corners_val = vals[..., idx, corner_y, corner_x]
    else:
      remaining_dims = [str(d) for d in da.dims if d not in src_spatial_dims]
      da_ordered = da.transpose(*remaining_dims, *src_spatial_dims)
      vals = da_ordered.values
      corners_val = vals[..., corner_y, corner_x]

    interpolated_vals = np.sum(corners_val * weights, axis=-2)

    if not self._extrapolate_out_of_bounds and not np.all(valid_mask):
      interpolated_vals = interpolated_vals.astype(
          np.result_type(interpolated_vals.dtype, np.float32)
      )
      interpolated_vals[..., ~valid_mask] = np.nan

    return _build_output_data_array(
        da, interpolated_vals, src_spatial_dims, ref_dim, reference
    )


class NeighborhoodThresholdProbabilities(Interpolation):
  """Converts a deterministic forecast to a probabilistic one by neighborhood averaging.

  For a given threshold, the probability is devined as the fraction of the
  fraction of pixels in a square neighborhood that exceeds the threshold. This
  is the same computation as in the Fraction Skill Score.
  """

  def __init__(
      self,
      neighborhood_sizes,
      thresholds,
      threshold_dim='threshold_value',
      wrap_longitude: bool = False,
  ):
    """Init.

    Args:
      neighborhood_sizes: List of neighborhood sizes to be used in pixels. Must
        be odd.
      thresholds: List of thresholds to be used to binarize data.
      threshold_dim: Dimension name of the thresholds. Default:
        'threshold_value'
      wrap_longitude: If True, perform a wrapped convolution in the longitude
        dimension. Default: False
    """
    self._neighborhood_sizes = neighborhood_sizes
    self._thresholds = thresholds
    self._threshold_dim = threshold_dim
    self._wrap_longitude = wrap_longitude

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    da = wrappers.binarize_thresholds(
        da, thresholds=self._thresholds, threshold_dim=self._threshold_dim
    )
    out = []
    for n in self._neighborhood_sizes:
      out.append(
          spatial.neighborhood_averaging_for_single_size(
              da, n, wrap_longitude=self._wrap_longitude
          )
      )
    out = xr.concat(
        out,
        dim=xr.DataArray(
            self._neighborhood_sizes, dims=['smoothing_neighborhood']
        ),
    )
    return out  # pyrefly: ignore[bad-return]


class Subsample(Interpolation):
  """Subsample a DataArray along specified dimensions.

  This is useful for reducing the resolution of a dataset without interpolation,
  e.g. for faster evaluation at lower resolution.
  """

  def __init__(
      self,
      dims: Sequence[str],
      stride: int,
  ):
    """Init.

    Args:
      dims: Dimensions along which to subsample.
      stride: Stride for subsampling. Must be a positive integer.
    """
    if stride < 1:
      raise ValueError(f'stride must be >= 1, got {stride}')
    self._dims = dims
    self._stride = stride

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    isel_kwargs = {
        dim: slice(None, None, self._stride)
        for dim in self._dims
        if dim in da.dims
    }
    return da.isel(**isel_kwargs)  # pyrefly: ignore[bad-argument-type]
