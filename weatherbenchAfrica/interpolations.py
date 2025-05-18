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
from weatherbenchAfrica import xarray_tree
from weatherbenchAfrica.metrics import spatial
from weatherbenchAfrica.metrics import wrappers
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
      **dim_args,
      method=method,
      kwargs=interp_kwargs,
  )  # pytype: disable=wrong-arg-types
  return out


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

  def interpolate_data_array(
      self,
      da: xr.DataArray,
      reference: xr.DataArray,  # pytype: disable=signature-mismatch
  ) -> xr.DataArray:

    # Catch case where reference doesn't contain any data.
    if len(reference) == 0:
      return reference.copy()

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
    dim_args = {dim: reference[dim] for dim in dims}

    da_like_reference = interpolate_to_coords(
        da,
        dim_args,
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
      da.coords['grid_elevation'] = self._grid_elevation.compute()

    da_like_reference = super().interpolate_data_array(da, reference)
    if da.name in ['2m_temperature', '10m_wind_speed']:
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
    return out
