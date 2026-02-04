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


def _create_lat_mask(
    lat: xr.DataArray, lat_lims: Tuple[int, int]
) -> xr.DataArray:
  """Computes a boolean mask for a latitude limits region."""
  if lat_lims[0] >= lat_lims[1]:
    raise ValueError(
        f'`lat_lims[0]` must be smaller than `lat_lims[1]`, got {lat_lims}`'
    )
  return np.logical_and(lat >= lat_lims[0], lat <= lat_lims[1])


def _create_lon_mask(
    lon: xr.DataArray, lon_lims: Tuple[int, int]
) -> xr.DataArray:
  """Computes a boolean mask for a longitude limits region."""
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
  return lon_mask


def _region_to_mask(
    lat: xr.DataArray,
    lon: xr.DataArray,
    lat_lims: Tuple[int, int],
    lon_lims: Tuple[int, int],
) -> xr.DataArray:
  """Computes a boolean mask for a lat/lon limits region."""
  lat_mask = _create_lat_mask(lat, lat_lims)
  lon_mask = _create_lon_mask(lon, lon_lims)
  return np.logical_and(lat_mask, lon_mask)


class LandSea(Binning):
  """Class for land/sea mask binning."""

  def __init__(
      self,
      land_sea_fraction: xr.DataArray,
      land_sea_threshold: float = 0.5,
      bin_dim_name: str = 'land_sea',
      include_global_mask: bool = False,
  ):
    """Init.

    Args:
      land_sea_fraction: Floating point land-sea fraction with same latitude/
        longitude coordinates as the statistic. 100% land is represented as 1
        and 100% sea as 0.
      land_sea_threshold: Threshold to classify as land. Computed as
        land_sea_fraction >= land_sea_threshold. (Default of 0.5 follows ECMWF
        convention).
      bin_dim_name: Name of binning dimension. Default: 'land_sea'
      include_global_mask: If True, the output mask will consist of ['land',
        'sea', 'global'], otherwise ['land', 'sea']. 'global' is the union of
        land and sea. Default: False.
    """
    super().__init__(bin_dim_name)
    # Force to bool to make sure it is a boolean mask.
    self._land_mask = land_sea_fraction >= land_sea_threshold
    self._include_global_mask = include_global_mask

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Creates a bin mask for a statistic.

    Args:
      statistic: Individual DataArray with statistic values.

    Returns:
      bin_mask: Boolean mask with output bins: ['land', 'sea', 'global'].
    """
    masks = [self._land_mask, 1 - self._land_mask]
    labels = ['land', 'sea']
    if self._include_global_mask:
      masks.append(xr.ones_like(self._land_mask))
      labels.append('global')

    masks = xr.concat(
        masks,
        dim=self.bin_dim_name,
    )
    masks.coords[self.bin_dim_name] = np.array(labels)
    return masks


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


class LatitudeBins(Binning):
  """Class for binning by latitude bands."""

  def __init__(
      self,
      degrees: float,
      lat_range: Tuple[int, int] = (-90, 90),
      bin_dim_name: str = 'latitude_bins',
  ):
    """Init.

    Args:
      degrees: Grid spacing in degrees.
      lat_range: Tuple of (min_lat, max_lat).
      bin_dim_name: Name of binning dimension.
    """
    super().__init__(bin_dim_name)
    self._degrees = degrees
    self._lat_bins = np.arange(
        lat_range[0], lat_range[1] + self._degrees, self._degrees
    )

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Creates a bin mask for a statistic."""
    masks = []
    for lat_start in self._lat_bins[:-1]:
      lat_end = lat_start + self._degrees
      mask = _create_lat_mask(
          statistic.latitude,
          (lat_start, lat_end),
      )
      # Broadcast the mask to the shape of statistic
      mask = mask.broadcast_like(statistic)
      mask = mask.expand_dims(dim=self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = np.array([lat_start])
      masks.append(mask)
    return xr.concat(masks, dim=self.bin_dim_name)


class LongitudeBins(Binning):
  """Class for binning by longitude bands."""

  def __init__(
      self,
      degrees: float,
      lon_range: Tuple[int, int] = (0, 360),
      bin_dim_name: str = 'longitude_bins',
  ):
    """Init.

    Args:
      degrees: Grid spacing in degrees.
      lon_range: Tuple of (min_lon, max_lon).
      bin_dim_name: Name of binning dimension.
    """
    super().__init__(bin_dim_name)
    self._degrees = degrees
    lon_end = lon_range[1]
    if lon_range[0] >= lon_range[1]:
      lon_end += 360
    self._lon_bins = np.arange(
        lon_range[0], lon_end + self._degrees, self._degrees
    )

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Creates a bin mask for a statistic."""
    masks = []
    for lon_start in self._lon_bins[:-1]:
      lon_end = lon_start + self._degrees
      mask = _create_lon_mask(
          statistic.longitude,
          (lon_start, lon_end),
      )
      # Broadcast the mask to the shape of statistic
      mask = mask.broadcast_like(statistic)
      mask = mask.expand_dims(dim=self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = np.array([np.mod(lon_start, 360)])
      masks.append(mask)
    return xr.concat(masks, dim=self.bin_dim_name)


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


def _extract_time_unit(
    time_coord: xr.DataArray,
    unit: str,
) -> xr.DataArray:
  """Extract time unit values from a datetime/timedelta coordinate.

  Args:
    time_coord: A datetime64 or timedelta64 xarray DataArray.
    unit: Time unit to extract, e.g. 'second', 'minute', 'hour', 'day', 'week',
      'year', 'month', 'dayofyear', etc.

  Returns:
    DataArray containing the extracted time unit values.

  Raises:
    ValueError: If the unit is not supported for timedelta coordinates.
  """
  dt = time_coord.dt
  if isinstance(dt, xr.core.accessor_dt.TimedeltaAccessor):
    coord = time_coord.dt.total_seconds()
    if unit == 'minute':
      coord = coord // (60)
    elif unit == 'hour':
      coord = coord // (60 * 60)
    elif unit == 'day':
      coord = coord // (60 * 60 * 24)
    elif unit == 'week':
      coord = coord // (60 * 60 * 24 * 7)
    elif unit == 'year':
      coord = coord // (60 * 60 * 24 * 365)
    elif unit != 'second':
      raise ValueError(f'Unsupported unit for timedelta: {unit}')
  else:
    assert isinstance(dt, xr.core.accessor_dt.DatetimeAccessor)
    coord = getattr(time_coord.dt, unit)
  return coord


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
    coord = _extract_time_unit(statistic[self.time_dim], self.unit)
    masks = vectorized_coord_mask(
        coord,
        self.time_dim,
        f'{self.time_dim}_{self.unit}',
        self.add_global_bin,
    )
    return masks


class ByTimeUnitSets(Binning):
  """Bin by sets of time unit values for a given axis.

  This combines the time unit extraction logic of ByTimeUnit with the set-based
  binning logic of BySets. It allows grouping by arbitrary sets of time unit
  values, for example grouping hours 0 and 12 together, and hours 6 and 18
  together.

  Example:
    ```
    sets = {'00/12': [0, 12], '06/18': [6, 18]}
    unit = 'hour'
    dim = 'init_time'
    ```
    This will create two bins: one for data initialized at hours 0 or 12, and
    another for data initialized at hours 6 or 18.
  """

  def __init__(
      self,
      sets: Mapping[str, Sequence[Any] | Any],
      unit: str,
      dim: str,
      bin_dim_name: Optional[str] = None,
      add_global_bin: bool = False,
  ):
    """Init.

    Args:
      sets: Dictionary specifying sets of time unit values to bin by. Keys are
        bin names, values are sequences of time unit values (e.g. hours).
      unit: Time unit to extract, e.g. 'hour', 'day', 'month', 'dayofyear'.
      dim: Time dimension/coordinate to bin by.
      bin_dim_name: Name of binning dimension. Default: `{dim}_{unit}_sets`.
      add_global_bin: If True, add a global bin containing all data. Default:
        False.
    """
    if bin_dim_name is None:
      bin_dim_name = f'{dim}_{unit}_sets'
    super().__init__(bin_dim_name)
    self.sets = sets
    self.unit = unit
    self.dim = dim
    self.add_global_bin = add_global_bin

  def create_bin_mask(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    time_unit_values = _extract_time_unit(statistic[self.dim], self.unit)

    masks = []
    for name, s in self.sets.items():
      if isinstance(s, (Sequence,)) and not isinstance(s, str):
        s = list(s)
      else:
        s = [s]
      s = np.array(s)
      mask = time_unit_values.isin(s)
      mask = mask.expand_dims(self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = [name]
      masks.append(mask)

    if self.add_global_bin:
      mask = xr.full_like(time_unit_values, True, dtype=bool).expand_dims(
          self.bin_dim_name
      )
      mask.coords[self.bin_dim_name] = ['global']
      masks.append(mask)

    return xr.concat(masks, self.bin_dim_name)


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
    masks = coord == xr.DataArray(bins, dims=[bin_dim_name]).broadcast_like(
        coord
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
                  self.dim_name: (
                      xr.DataArray([], dims=[self.dim_name]).astype(dtype)
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
      sets: Mapping[str, Sequence[Any] | Any],
      coord_name: str,
      bin_dim_name: Optional[str] = None,
      add_set_complements: bool = False,
      add_global_bin: bool = False,
  ):
    """Init.

    Args:
      sets: Dictionary specifying sets of values to bin by.
      coord_name: Name of coordinate to bin over.
      bin_dim_name: Name of binning dimension. Default: `dim_name`
      add_set_complements: If True, for each set, also add a bin for all values
        not in the set.
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
    self.add_set_complements = add_set_complements
    self.add_global_bin = add_global_bin

  def create_bin_mask(
      self,
      statistic: Union[xr.DataArray, xr.Dataset],
  ) -> xr.DataArray:
    masks = []

    for name, s in self.sets.items():
      # Convert s to a numpy array to handle different input types and
      # ensure compatibility with isin and JAX.
      if isinstance(s, (Sequence,)) and not isinstance(s, str):
        s = list(s)
      else:
        s = [s]
      s = np.array(s)
      mask = statistic[self.coord_name].isin(s)
      mask = mask.expand_dims(self.bin_dim_name, axis=0)
      mask.coords[self.bin_dim_name] = [name]
      masks.append(mask)
      if self.add_set_complements:
        not_in_mask = ~mask.copy()
        not_in_mask.coords[self.bin_dim_name] = [f'not_in_{name}']
        masks.append(not_in_mask)
    if self.add_global_bin:
      mask = xr.full_like(
          statistic[self.coord_name], True, dtype=bool
      ).expand_dims(
          self.bin_dim_name
      )  # Add as a dimension
      mask.coords[self.bin_dim_name] = ['global']
      masks.append(mask)
    return xr.concat(masks, self.bin_dim_name)
