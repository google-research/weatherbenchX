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
"""Binning class definitions."""

import abc
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import xarray as xr


class Binning(abc.ABC):
  """Binning base class."""

  def __init__(self, bin_dim_name: str):
    """Init.

    Args:
      bin_dim_name: Name of binning dimension.
    """
    self.bin_dim_name = bin_dim_name

  @abc.abstractmethod
  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Creates a bin mask for a statistic.

    It is assumed that all information required to compute bins is included in
    the statistics element.

    Args:
      statistic: Individual DataArray with statistic values.

    Returns:
      bin_mask: Boolean mask with shape that boradcasts against the statistic
        DataArray.
    """


def _region_to_mask(
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat_lims: Tuple[int, int],
    lon_lims: Tuple[int, int],
) -> xr.DataArray:
  """Computes a boolean mask for a lat/lon limits region."""
  if lat_lims[0] >= lat_lims[1]:
    raise ValueError(
        f'`lat_lims[0]` must be smaller than `lat_lims[1]`, got {lat_lims}`'
    )
  lat_mask = np.logical_and(lat >= lat_lims[0], lat <= lat_lims[1])

  # Make sure we are in the [0, 360] interval.
  lon = np.mod(lon, 360)
  lon_lims = np.mod(lon_lims[0], 360), np.mod(lon_lims[1], 360)

  if lon_lims[1] > lon_lims[0]:
    # Same as the latitude.
    lon_mask = np.logical_and(lon >= lon_lims[0], lon <= lon_lims[1])
  else:
    # In this case it means we need to wrap longitude around the other side of
    # the globe.
    lon_mask = np.logical_or(lon <= lon_lims[1], lon >= lon_lims[0])
  return np.logical_and(lat_mask, lon_mask)


class Regions(Binning):
  """Class for rectangular region binning.

  Note that coordinate must be named `latitude` and `longitude`.
  """

  def __init__(
      self,
      regions: Mapping[Hashable, Tuple[Tuple[int, int], Tuple[int, int]]],
      bin_dim_name: str = 'region',
      land_sea_mask: Optional[xr.DataArray] = None,
  ):
    """Init.

    Args:
      regions: Dictionary specifying {name: ((lat_lims), (lon_lims))}.
      bin_dim_name: Name of binning dimension. Default: 'region'
      land_sea_mask: (Optional) Boolean mask (land = True) with same
        latitude/longitude coordinates as the statistic. If provided, for each
        region will add a new land-onlybin with the name {region}_land.
    """
    super().__init__(bin_dim_name)
    self._regions = regions
    self._land_sea_mask = land_sea_mask

  def _regions_to_masks(
      self,
      lat: xr.DataArray,
      lon: xr.DataArray,
  ) -> xr.DataArray:
    """Computes and stacks masks for all regions."""
    masks = []
    for region_name, (lat_lims, lon_lims) in self._regions.items():
      mask = _region_to_mask(lat, lon, lat_lims, lon_lims)
      mask = mask.expand_dims(dim=self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = np.array([region_name])
      masks.append(mask)
    return xr.concat(masks, dim=self.bin_dim_name)

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    masks = self._regions_to_masks(statistic.latitude, statistic.longitude)
    if self._land_sea_mask is not None:
      assert np.array_equal(
          np.sort(masks.latitude), np.sort(self._land_sea_mask.latitude)
      ) and np.array_equal(
          masks.longitude, self._land_sea_mask.longitude
      ), 'Land/sea mask coordinates do not match.'
      land_masks = masks * self._land_sea_mask.astype(bool)
      region_names = [f'{r}_land' for r in masks.coords[self.bin_dim_name].data]
      land_masks.coords[self.bin_dim_name] = np.array(region_names)
      masks = xr.concat([masks, land_masks], dim=self.bin_dim_name)
    return masks


def vectorized_coord_mask(
    coord: xr.DataArray,
    coord_name: str,
    bin_dim_name: str,
    add_global_bin: bool = False,
) -> xr.DataArray:
  """Helper to create bin masks for unique coordinate values."""
  unique_coord = np.unique(coord)
  ndims = len(coord.dims)
  # Use vectorized equal. This also works in the case of empty statistic.
  masks = xr.DataArray(
      np.equal(coord.values, unique_coord.reshape((-1,) + (1,) * ndims)),
      coords={bin_dim_name: unique_coord}
      | {dim: coord[dim] for dim in coord.dims},
      dims=[bin_dim_name] + list(coord.dims),
  )
  if add_global_bin:
    mask = (
        xr.ones_like(coord.astype(bool))
        .drop(coord_name)  # Drop the coordinate
        .expand_dims(bin_dim_name)  # Add as a dimension
    )
    mask.coords[bin_dim_name] = ['global']
    # Dtypes of bin coordinates need to match. If they don't cast both to
    # str.
    if mask[bin_dim_name].dtype != masks[bin_dim_name].dtype:
      masks.coords[bin_dim_name] = masks[bin_dim_name].astype('str')
      mask.coords[bin_dim_name] = mask[bin_dim_name].astype('str')
    masks = xr.concat([mask, masks], dim=bin_dim_name)
  return masks


class ByExactCoord(Binning):
  """Binning by unique coordinate values.

  This will create a bin for each unique coordinate value, for example for each
  unique lead time in the case of sparse forecasts where lead_time is a
  coordinate but not a dimension.
  """

  def __init__(self, coord: str, add_global_bin: bool = False):
    """Init.

    Args:
      coord: Name of coordinate to bin by.
      add_global_bin: If True, add a global bin containing all data. Default:
        False.
    """
    super().__init__(coord)
    self.coord = coord
    self.add_global_bin = add_global_bin

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    assert (
        self.coord not in statistic.dims
    ), 'For dimensions, specify reduce_dims in aggregation.'
    coord = statistic[self.coord]
    # Coord name and bin_dim_name are the same in this case.
    masks = vectorized_coord_mask(
        coord, self.coord, self.coord, self.add_global_bin
    )
    return masks


class ByTimeUnit(Binning):
  """Bin by time unit for given axis.

  This uses the .dt datetime accessor in xarray, and will work with both
  datetime64 and timedelta64 coordinates. However, the units should be in the
  datetime64 convention, i.e. 'second', 'minute', 'hour', etc.

  See:
  https://docs.xarray.dev/en/latest/generated/xarray.core.accessor_dt.DatetimeAccessor.html

  Example:
    ```
    unit = 'hour'
    time_dim = 'init_time'
    ```
    This will aggregate together all data initialized at the same time of day,
    e.g. [0, 1, 2, .., 23].
  """

  def __init__(self, unit: str, time_dim: str, add_global_bin: bool = False):
    # TODO(srasp): Add support for sequence of units.
    """Init.

    Args:
      unit: Time unit to bin by.
      time_dim: Time dimension to bin by.
      add_global_bin: If True, add a global bin containing all data. Default:
        False.
    """

    super().__init__(f'{time_dim}_{unit}')
    self.unit = unit
    self.time_dim = time_dim
    self.add_global_bin = add_global_bin

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    dt = statistic[self.time_dim].dt
    if isinstance(dt, xr.core.accessor_dt.TimedeltaAccessor):
      coord = statistic[self.time_dim].dt.total_seconds()
      if self.unit == 'minute':
        coord = coord // (60)
      elif self.unit == 'hour':
        coord = coord // (60 * 60)
      elif self.unit == 'day':
        coord = coord // (60 * 60 * 24)
      elif self.unit == 'week':
        coord = coord // (60 * 60 * 24 * 7)
      elif self.unit == 'year':
        coord = coord // (60 * 60 * 24 * 365)
      elif self.unit != 'second':
        raise ValueError(f'Unsupported unit: {self.unit}')
    else:
      assert isinstance(dt, xr.core.accessor_dt.DatetimeAccessor)
      coord = getattr(statistic[self.time_dim].dt, self.unit)
    masks = vectorized_coord_mask(
        coord,
        self.time_dim,
        f'{self.time_dim}_{self.unit}',
        self.add_global_bin,
    )
    return masks


class ByTimeUnitFromSeconds(Binning):
  """Similar to ByTimeUnit, but with the coordinate in seconds as a scalar.

  The seconds values will be converted to the desired time unit.

  This is useful if you want to wrap the computation in jax.jit, which does not
  support datetime64/timedelta64 coordinates.
  """

  def __init__(
      self, unit: str, time_dim: str, bins: Sequence[int] | None = None
  ):
    """Init.

    Args:
      unit: Time unit to bin by, one of 'second', 'minute', 'hour'.
      time_dim: Time dimension to bin by.
      bins: Sequence of bins to bin by. If None, will use default bins depending
      on the unit (e.g. 0 through 23 for hour). Note that these defaults won't
      always make sense (e.g. if binning by lead time, hours can be > 23).
    """

    super().__init__(f'{time_dim}_{unit}')
    self.unit = unit
    self.time_dim = time_dim
    self.bins = bins

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    coord = statistic[self.time_dim]
    bins = self.bins

    if self.unit == 'second':
      bins = bins if bins is not None else np.arange(0, 60)
    elif self.unit == 'minute':
      coord = coord // (60)
      bins = bins if bins is not None else np.arange(0, 60)
    elif self.unit == 'hour':
      coord = coord // (60 * 60)
      bins = bins if bins is not None else np.arange(0, 24)
    else:
      raise ValueError(f'Unsupported unit: {self.unit}')

    bin_dim_name = f'{self.time_dim}_{self.unit}'
    masks = (
        coord ==
        xr.DataArray(bins, dims=[bin_dim_name]).broadcast_like(coord)
    )
    masks = masks.assign_coords({bin_dim_name: bins})
    return masks


class ByCoordBins(Binning):
  """Binning by specified bins over a coordinate."""

  def __init__(self, dim_name: str, bin_edges: np.ndarray):
    """Init.

    Args:
      dim_name: Name of dimension to bin by.
      bin_edges: Bin edges to bin by.
    """
    super().__init__(dim_name)
    self.dim_name = dim_name
    self.bin_edges = bin_edges

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    masks = []

    # TODO(srasp): Potentially optimize using np.digitize.
    for start, stop in zip(self.bin_edges[:-1], self.bin_edges[1:]):
      mask = np.logical_and(
          statistic.coords[self.dim_name] >= start,
          statistic.coords[self.dim_name] < stop,
      )
      mask = mask.drop([self.dim_name]).expand_dims(self.dim_name, axis=0)
      mask.coords[self.dim_name] = np.array([start])
      mask.assign_coords({
          self.dim_name
          + '_left_edge': xr.DataArray([start], dims=[self.dim_name]),
          self.dim_name
          + '_right_edge': xr.DataArray([stop], dims=[self.dim_name]),
      })
      masks.append(mask)
    if not masks:  # Catch possibility of empty input arrays.
      dtype = statistic[self.dim_name].dtype
      masks = (
          xr.ones_like(statistic)
          .drop(self.dim_name)
          .expand_dims(
              {
                  self.dim_name: xr.DataArray([], dims=[self.dim_name]).astype(
                      dtype
                  )
              },
              axis=0,
          )
      )
      return masks
    else:
      return xr.concat(masks, self.dim_name)


class BySets(Binning):
  """Bin by sets of values along a coordinate.

  This is, for example, useful for binning by different sets of station names.
  """

  def __init__(
      self,
      sets: Mapping[str, Sequence[Any]],
      coord_name: str,
      bin_dim_name: Optional[str] = None,
      add_global_bin: bool = False,
  ):
    """Init.

    Args:
      sets: Dictionary specifying sets of values to bin by.
      coord_name: Name of coordinate to bin over.
      bin_dim_name: Name of binning dimension. Default: `dim_name`
      add_global_bin: If True, add a global bin containing all data. Default:
        False.
    """
    if bin_dim_name is None or bin_dim_name == coord_name:
      raise ValueError(
          'bin_dim_name must be defined and be different from coord_name.'
      )
    super().__init__(bin_dim_name)
    self.sets = sets
    self.coord_name = coord_name
    self.add_global_bin = add_global_bin

  def create_bin_mask(
      self,
      statistic: Union[xr.DataArray, xr.Dataset],
  ) -> xr.DataArray:
    masks = []

    for name, s in self.sets.items():
      mask = statistic[self.coord_name].isin(s)
      mask = mask.expand_dims(self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = [name]
      masks.append(mask)
    if self.add_global_bin:
      mask = xr.full_like(
          statistic[self.coord_name], True, dtype=bool
      ).expand_dims(
          self.bin_dim_name
      )  # Add as a dimension
      mask.coords[self.bin_dim_name] = ['global']
      masks.append(mask)
    return xr.concat(masks, self.bin_dim_name)
