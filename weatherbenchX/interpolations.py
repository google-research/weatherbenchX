# Copyright 2026 Google LLC
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
  )
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
      raise ValueError('Invalid latitudes: {lat_min} and {lat_max}')
    if lon_min > lon_max:
      raise ValueError('Invalid longitudes: {lon_min} and {lon_max}')
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
      reference: xr.DataArray,
  ) -> xr.DataArray:
    # If dims not explicit, interpolate all dims that have a corresponding
    # coordinate in the reference.
    if self._dims is None:
      dims = [d for d in da.dims if d in reference.coords]
    else:
      dims = self._dims

    if reference.size > 0 and all(
        dim in da.coords
        and da[dim].shape == reference[dim].shape
        and (
            np.allclose(da[dim].values, reference[dim].values, atol=1e-4)
            if np.issubdtype(da[dim].dtype, np.floating)
            else np.array_equal(da[dim].values, reference[dim].values)
        )
        for dim in dims
    ):
      return da.assign_coords({dim: reference[dim] for dim in dims})

    if self._wrap_longitude:
      da = pad_longitude(da)

    if self._clip_reference_coords is not None:
      for coord in self._clip_reference_coords:
        reference = reference.sel(
            {coord: slice(da[coord].min(), da[coord].max())}
        )

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
      reference: xr.DataArray,
  ) -> xr.DataArray:
    if da.name in ['2m_temperature', '10m_wind_speed']:
      # Sometimes coordinates (e.g. latitude ordering North-South vs South-North)
      # don't match exactly, so we assert they cover the same coordinates and
      # reindex to match da.
      grid_elevation = self._grid_elevation.compute()
      xr.testing.assert_allclose(
          grid_elevation.latitude.sortby('latitude'),
          da.latitude.sortby('latitude'),
      )
      xr.testing.assert_allclose(
          grid_elevation.longitude.sortby('longitude'),
          da.longitude.sortby('longitude'),
      )
      if (
          not grid_elevation.latitude.equals(da.latitude)
          or not grid_elevation.longitude.equals(da.longitude)
      ):
        grid_elevation = grid_elevation.reindex(
            latitude=da.latitude, longitude=da.longitude
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


def _cell_edges_1d(
    coords: np.ndarray,
    wrap_longitude: bool = False,
    shared_bounds: Optional[tuple[Optional[float], Optional[float]]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes 1D cell lower/upper edges in ascending coordinate order."""
  raw = np.asarray(coords)
  coords_f64 = raw.astype(np.float64)
  if raw.dtype == np.float32:
    # Eliminate float32 representation jitter (for example,
    # float32(359.9) == 359.899994) before computing cell boundaries in float64.
    coords_f64 = np.round(coords_f64, 5)
  sort_order = np.argsort(coords_f64)
  u = coords_f64[sort_order]
  midpoints = 0.5 * (u[:-1] + u[1:])
  if wrap_longitude:
    first_edge = 0.5 * ((u[-1] - 360.0) + u[0])
    last_edge = 0.5 * (u[-1] + (u[0] + 360.0))
  else:
    first_edge = u[0] - 0.5 * (u[1] - u[0])
    last_edge = u[-1] + 0.5 * (u[-1] - u[-2])
    if shared_bounds is not None:
      min_bound, max_bound = shared_bounds
      if min_bound is not None:
        first_edge = max(first_edge, min_bound)
      if max_bound is not None:
        last_edge = min(last_edge, max_bound)
  edges_lo = np.concatenate([[first_edge], midpoints])
  edges_hi = np.concatenate([midpoints, [last_edge]])
  return sort_order, edges_lo, edges_hi


def _coarsen_1d(
    da: xr.DataArray,
    dim: str,
    target_coord: xr.DataArray,
    wrap_longitude: bool = False,
) -> xr.DataArray:
  """Coarsens `da` along `dim` onto `target_coord` via box-overlap averaging."""
  src_raw = np.asarray(da[dim].values)
  tgt_raw = np.asarray(target_coord.values)
  src_vals = (
      np.round(src_raw.astype(np.float64), 4)
      if src_raw.dtype == np.float32
      else src_raw.astype(np.float64)
  )
  tgt_vals = (
      np.round(tgt_raw.astype(np.float64), 4)
      if tgt_raw.dtype == np.float32
      else tgt_raw.astype(np.float64)
  )

  if len(src_vals) == len(tgt_vals) and np.allclose(
      src_vals, tgt_vals, atol=1e-5
  ):
    return da.assign_coords({dim: target_coord})

  if len(src_vals) < 2 or len(tgt_vals) < 2:
    raise ValueError(
        f'Coarsen requires at least 2 points along {dim!r}, got'
        f' {len(src_vals)} source and {len(tgt_vals)} target points.'
    )

  is_wrapped = wrap_longitude and dim == 'longitude'
  if not is_wrapped:
    src_min, src_max = float(np.min(src_vals)), float(np.max(src_vals))
    tgt_min, tgt_max = float(np.min(tgt_vals)), float(np.max(tgt_vals))
    min_bound = (
        0.5 * (src_min + tgt_min) if abs(src_min - tgt_min) < 1e-4 else None
    )
    max_bound = (
        0.5 * (src_max + tgt_max) if abs(src_max - tgt_max) < 1e-4 else None
    )
    shared_bounds = (min_bound, max_bound)
  else:
    shared_bounds = None

  src_order, src_lo, src_hi = _cell_edges_1d(
      src_vals, wrap_longitude=is_wrapped, shared_bounds=shared_bounds
  )
  tgt_order, tgt_lo, tgt_hi = _cell_edges_1d(
      tgt_vals, wrap_longitude=is_wrapped, shared_bounds=shared_bounds
  )

  src_step = float(np.median(src_hi - src_lo))
  tgt_step = float(np.median(tgt_hi - tgt_lo))
  if src_step > tgt_step * (1.0 + 1e-4):
    raise ValueError(
        f'Cannot coarsen along {dim!r} from coarser source resolution'
        f' ({src_step:g}) to finer target resolution ({tgt_step:g}).'
    )

  inv_tgt_order = np.argsort(tgt_order)
  tgt_lo = tgt_lo[inv_tgt_order]
  tgt_hi = tgt_hi[inv_tgt_order]

  n_tgt = len(tgt_vals)
  n_src = len(src_vals)
  if is_wrapped:
    overlap_sorted = np.zeros((n_tgt, n_src), dtype=np.float64)
    for shift in (-360.0, 0.0, 360.0):
      s_lo = src_lo[None, :] + shift
      s_hi = src_hi[None, :] + shift
      overlap_sorted += np.maximum(
          0.0,
          np.minimum(tgt_hi[:, None], s_hi) - np.maximum(tgt_lo[:, None], s_lo),
      )
  else:
    overlap_sorted = np.maximum(
        0.0,
        np.minimum(tgt_hi[:, None], src_hi[None, :])
        - np.maximum(tgt_lo[:, None], src_lo[None, :]),
    )

  tol = 1e-5 * min(src_step, tgt_step)
  overlap_sorted[overlap_sorted < tol] = 0.0
  row_sum = overlap_sorted.sum(axis=1, keepdims=True)
  if np.any(row_sum <= 0.0):
    raise ValueError(
        f'Target cells along {dim!r} have zero overlap with source grid.'
    )
  weights_sorted = overlap_sorted / row_sum

  counts = np.count_nonzero(weights_sorted > 0.0, axis=1)
  max_k = int(np.max(counts))
  idx_table = np.zeros((max_k, n_tgt), dtype=np.intp)
  weight_table = np.zeros((max_k, n_tgt), dtype=np.float64)
  for j in range(n_tgt):
    nz = np.flatnonzero(weights_sorted[j] > 0.0)
    k = len(nz)
    idx_table[:k, j] = src_order[nz]
    weight_table[:k, j] = weights_sorted[j, nz]
    if k < max_k:
      idx_table[k:, j] = idx_table[k - 1, j]

  aux_coords = [c for c in da.coords if c != dim and dim in da.coords[c].dims]
  da_clean = da.drop_vars(aux_coords) if aux_coords else da

  out_dtype = (
      da.dtype if np.issubdtype(da.dtype, np.floating) else np.dtype(np.float32)
  )
  target_coord_da = (
      target_coord
      if isinstance(target_coord, xr.DataArray)
      else xr.DataArray(target_coord, dims=[dim])
  )
  acc: Optional[xr.DataArray] = None
  for k in range(max_k):
    w_da = xr.DataArray(
        weight_table[k].astype(out_dtype),
        dims=[dim],
        coords={dim: target_coord_da},
    )
    sliced = da_clean.isel({dim: idx_table[k]}).assign_coords(
        {dim: target_coord_da}
    )
    term = sliced * w_da
    acc = term if acc is None else (acc + term)

  assert acc is not None
  out = acc.transpose(*da.dims).astype(out_dtype)
  out.attrs = dict(da.attrs)
  out.name = da.name
  return out


class Coarsen(Interpolation):
  """Coarsens a DataArray to a coarser reference or fixed coordinate grid.

  Uses 1D first-order conservative box-overlap averaging along each specified
  dimension, supporting both integer and non-integer grid resolution ratios
  (such as coarsening a 0.1 deg forecast onto a 0.25 deg target grid).
  """

  def __init__(
      self,
      dims: Optional[Sequence[str]] = None,
      wrap_longitude: bool = False,
      coords: Optional[Mapping[str, Union[xr.DataArray, np.ndarray]]] = None,
  ):
    """Init.

    Args:
      dims: (Optional) Dimensions over which to coarsen. If None (default),
        infer dimensions from the intersection of DataArray dimensions and
        reference/target coordinates.
      wrap_longitude: If True, treat the 'longitude' dimension as periodic with
        period 360 degrees. Default: False.
      coords: (Optional) Dictionary of fixed target coordinates to coarsen to
        when `reference` is not passed to `interpolate_data_array`.
    """
    self._dims = dims
    self._wrap_longitude = wrap_longitude
    self._coords = coords

  def interpolate(
      self,
      ds: Mapping[Hashable, xr.DataArray],
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    if reference is None:
      return xarray_tree.map_structure(self.interpolate_data_array, ds)
    if isinstance(reference, xr.DataArray):
      ref_da = reference
    elif isinstance(reference, xr.Dataset):
      ref_da = (
          next(iter(reference.data_vars.values()))
          if reference.data_vars
          else xr.DataArray(coords=reference.coords)
      )
    else:
      ref_da = next(iter(reference.values()), None)
    def _interpolate_var(da: xr.DataArray) -> xr.DataArray:
      if (
          isinstance(reference, Mapping)
          and da.name is not None
          and da.name in reference
      ):
        var_ref = reference[da.name]
      else:
        var_ref = ref_da
      return self.interpolate_data_array(da, var_ref)

    return xarray_tree.map_structure(_interpolate_var, ds)

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: Optional[xr.DataArray] = None,
  ) -> xr.DataArray:
    if reference is not None:
      target_coords = reference.coords
    elif self._coords is not None:
      target_coords = self._coords
    else:
      raise ValueError(
          'Coarsen requires either a reference DataArray or fixed coords.'
      )

    if self._dims is None:
      dims = [str(d) for d in da.dims if d in target_coords]
    else:
      dims = [str(d) for d in self._dims if d in da.dims and d in target_coords]

    if reference is not None and reference.size == 0:
      da_dims_to_retain = set(da.dims) - set(dims)
      return reference.copy().expand_dims({d: da[d] for d in da_dims_to_retain})

    out = da
    for dim in dims:
      tgt_coord = target_coords[dim]
      if not isinstance(tgt_coord, xr.DataArray):
        tgt_coord = xr.DataArray(tgt_coord, dims=[dim])
      out = _coarsen_1d(
          out,
          dim=dim,
          target_coord=tgt_coord,
          wrap_longitude=self._wrap_longitude,
      )
    return out
