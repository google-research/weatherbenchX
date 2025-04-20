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
"""Data loaders for reading gridded Zarr files."""

from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Union
import numpy as np
from weatherbenchX import interpolations
from weatherbenchX.data_loaders import base
import xarray as xr


def _rename_dataset(
    ds: xr.Dataset,
    rename_dimensions: Optional[Union[Mapping[str, str], str]] = 'ecmwf',
    rename_variables: Optional[Mapping[str, str]] = None,
    convert_lat_lon_to_latitude_longitude: bool = True,
) -> xr.Dataset:
  """Rename dimensions and variables of Zarr dataset."""
  # Rename dimensions
  if convert_lat_lon_to_latitude_longitude:
    if 'lat' in ds.coords and 'lon' in ds.coords:
      ds = ds.rename({'lat': 'latitude', 'lon': 'longitude'})
  if rename_dimensions == 'ecmwf':  # ECMWF standard
    if 'prediction_timedelta' in ds.coords:  # Is forecast dataset
      ds = ds.rename({'time': 'init_time', 'prediction_timedelta': 'lead_time'})
    else:  # Is (re-)analysis dataset
      ds = ds.rename({'time': 'valid_time'})
  elif isinstance(rename_dimensions, Mapping):
    ds = ds.rename(rename_dimensions)
  elif rename_dimensions is None:
    pass
  else:
    raise ValueError(
        'rename_dimensions must be either "ecmwf", a dict or None.'
    )
  # Rename variables
  if rename_variables is not None:
    ds = ds.rename(rename_variables)
  return ds


class XarrayDataLoader(base.DataLoader):
  """Base class for Xarray data loaders."""

  def __init__(
      self,
      path: Optional[str] = None,
      ds: Optional[xr.Dataset] = None,
      variables: Optional[Iterable[str]] = None,
      sel_kwargs: Optional[Mapping[str, Any]] = None,
      rename_dimensions: Optional[Union[Mapping[str, str], str]] = 'ecmwf',
      automatically_convert_lat_lon_to_latitude_longitude: bool = True,
      rename_variables: Optional[Mapping[str, str]] = None,
      interpolation: Optional[interpolations.Interpolation] = None,
      compute: bool = True,
      add_nan_mask: bool = False,
      preprocessing_fn: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
  ):
    """Init.

    Args:
      path: (Optional) Path to xarray dataset to open. If it ends with '.zarr',
        it is opened using xr.open_zarr. Otherwise, it is opened using
        xr.open_dataset.
      ds: (Optional) Already opened xarray dataset. Either path or ds must be
        specified.
      variables: (Optional) List of variables to load (after renaming). Default:
        Load all variables.
      sel_kwargs: (Optional) Keyword arguments to pass to .sel() after renaming.
      rename_dimensions: (Optional) Dictionary of dimensions to rename. The data
        loaders expect the following time dimensions: `init_time` and
        `lead_time` for a forecast dataset; `valid_time` for target datasets
        (e.g. reanalyses). rename_dimensions='ecmwf' (default) assumes ECMWF
        standard names, {'time': 'init_time', 'prediction_timedelta':
        'lead_time'} for prediction datasets and {'time': 'valid_time'} for
        analysis datasets.
      automatically_convert_lat_lon_to_latitude_longitude: (Optional) Whether to
        automatically convert 'lat' and 'lon' dimensions to 'latitude' and
        'longitude'. Default: True.
      rename_variables: (Optional) Dictionary of variables to rename.
      interpolation: (Optional) Interpolation instance.
      compute: Whether to load data into memory. Default: True.
      add_nan_mask: Adds a boolean coordinate named 'mask' to each variable
        (variables will be split into DataArrays if they aren't already), with
        False indicating NaN values. To be used for masked aggregation. Default:
        False.
      preprocessing_fn: (Optional) A function that is applied to the dataset
        right after it is opened.
    """
    if path is not None and ds is not None:
      raise ValueError('Only one of path or ds can be specified, not both.')
    if path is not None:
      if path.rstrip('/').endswith('.zarr'):
        self._ds = xr.open_zarr(path)
      else:
        self._ds = xr.open_dataset(path)
    elif ds is not None:
      self._ds = ds
    else:
      raise ValueError('Either path or ds must be specified.')
    if preprocessing_fn is not None:
      self._ds = preprocessing_fn(self._ds)
    self._ds = _rename_dataset(
        self._ds,
        rename_dimensions,
        rename_variables,
        automatically_convert_lat_lon_to_latitude_longitude,
    )
    if variables is not None:
      self._ds = self._ds[list(variables)]
    if sel_kwargs is not None:
      self._ds = self._ds.sel(**sel_kwargs)
    self._variables = variables
    super().__init__(
        interpolation=interpolation,
        compute=compute,
        add_nan_mask=add_nan_mask,
    )

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    raise NotImplementedError()


class PredictionsFromXarray(XarrayDataLoader):
  """Data loader for reading prediction datasets from Xarray.

  Example:
      >>> init_times, lead_times
      (array(['2020-01-01T00:00:00.000000000', '2020-01-01T12:00:00.000000000'],
       dtype='datetime64[ns]'), array([0, 6], dtype='timedelta64[h]'))
      >>> variables = ['2m_temperature', '10m_wind_speed']
      >>> prediction_data_loader = PredictionsFromXarray(
      >>>     path=<PATH>,
      >>>     variables=variables,
      >>> )
      >>> prediction_data_loader.load_chunk(init_times, lead_times)
      <xarray.Dataset>
      Dimensions:         (latitude: 32, longitude: 64, lead_time: 2, init_time:
      2)
      Coordinates:
        * latitude        (latitude) float64 -87.19 -81.56 -75.94 ... 81.56
        87.19
        * longitude       (longitude) float64 0.0 5.625 11.25 ... 343.1 348.8
        354.4
        * lead_time       (lead_time) timedelta64[ns] 00:00:00 06:00:00
        * init_time       (init_time) datetime64[ns] 2020-01-01
        2020-01-01T12:00:00
      Data variables:
          10m_wind_speed  (init_time, lead_time, longitude, latitude) float32
          2.29 ...
          2m_temperature  (init_time, lead_time, longitude, latitude) float32
          247.4...
  """

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    # Exact lead times or lead time slice.
    if lead_times is not None:
      chunk = self._ds.sel(init_time=init_times, lead_time=lead_times)

    # No lead times specified, return all.
    else:
      chunk = self._ds.sel(init_time=init_times)
    return chunk


class TargetsFromXarray(XarrayDataLoader):
  """Data loader for reading target datasets from Xarray.

  Example:
      >>> init_times, lead_times
      (array(['2020-01-01T00:00:00.000000000', '2020-01-01T12:00:00.000000000'],
      dtype='datetime64[ns]'), array([0, 6], dtype='timedelta64[h]'))
      >>> variables = ['2m_temperature', '10m_wind_speed']
      >>> target_data_loader = gridded_zarr.TargetsFromXarray(
      >>>     path=<PATH>,
      >>>     variables=variables,
      >>> )
      >>> target_data_loader.load_chunk(init_times, lead_times)
      <xarray.Dataset>
      Dimensions:         (latitude: 32, longitude: 64, init_time: 2, lead_time:
      2)
      Coordinates:
        * latitude        (latitude) float64 -87.19 -81.56 -75.94 ... 81.56
        87.19
        * longitude       (longitude) float64 0.0 5.625 11.25 ... 343.1 348.8
        354.4
          valid_time      (init_time, lead_time) datetime64[ns] 2020-01-01 ...
          2020...
        * init_time       (init_time) datetime64[ns] 2020-01-01
        2020-01-01T12:00:00
        * lead_time       (lead_time) timedelta64[ns] 00:00:00 06:00:00
      Data variables:
          10m_wind_speed  (init_time, lead_time, longitude, latitude) float32
          2.221...
          2m_temperature  (init_time, lead_time, longitude, latitude) float32
          248.5...
  """

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    # Exact lead times.
    if isinstance(lead_times, Iterable):
      # Construct valid times from init and lead time combination.
      valid_time = xr.DataArray(
          init_times, coords={'init_time': init_times}
      ) + xr.DataArray(lead_times, coords={'lead_time': lead_times})
      chunk = self._ds.sel(valid_time=valid_time)
    # Lead time slice: not allowed.
    elif isinstance(lead_times, slice):
      raise ValueError('Lead time slice not supported for target data loaders.')
    # No lead time slice, in this case treat the init times as valid times.
    else:
      chunk = self._ds.sel(valid_time=init_times)
    return chunk


class ClimatologyFromXarray(XarrayDataLoader):
  """Reads a climatology dataset as a predictions dataset."""

  def __init__(
      self,
      climatology_time_coords: Iterable[str] = ('dayofyear', 'hour'),
      rename_dimensions: Optional[Union[Mapping[str, str], str]] = None,
      **kwargs
  ):
    """Init.

    Args:
      climatology_time_coords: The time coordinates of the climatology dataset
        to select. Default: ('dayofyear', 'hour').
      rename_dimensions: (Optional) Dictionary of dimensions to rename. Default:
        None.
      **kwargs: Other arguments to pass to XarrayDataLoader.
    """
    super().__init__(rename_dimensions=rename_dimensions, **kwargs)
    self._climatology_time_coords = climatology_time_coords

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    # Exact lead times.
    if isinstance(lead_times, Iterable):
      # Construct valid times from init and lead time combination.
      valid_time = xr.DataArray(
          init_times, coords={'init_time': init_times}
      ) + xr.DataArray(lead_times, coords={'lead_time': lead_times})
      sel_kwargs = {}
      for coord in self._climatology_time_coords:
        sel_kwargs[coord] = getattr(valid_time.dt, coord)
    # Lead time slice: not allowed.
    elif isinstance(lead_times, slice):
      raise ValueError(
          'Lead time slice not yet supported for climatology data loaders.'
      )
    # No lead time slice, in this case treat the init times as valid times.
    else:
      init_times = xr.DataArray(init_times, coords={'init_time': init_times})
      sel_kwargs = {}
      for coord in self._climatology_time_coords:
        sel_kwargs[coord] = getattr(init_times.dt, coord)
    chunk = self._ds.sel(sel_kwargs)
    return chunk


class PersistenceFromXarray(XarrayDataLoader):
  """Reads a target dataset as a prediction dataset by replicating data along lead times."""

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    if lead_times is None or isinstance(lead_times, slice):
      raise ValueError(
          'Exact lead times must be specified for persistence data loader.'
      )
    chunk = self._ds.sel(valid_time=init_times).expand_dims(
        {'lead_time': lead_times}
    )
    return chunk.rename({'valid_time': 'init_time'})


class ProbabilisticClimatologyFromXarray(XarrayDataLoader):
  """Reads a target dataset and treats every year as an ensemble member.

  For each valid_time, take the corresponding value for the same day of the year
  and hour of the day from the target dataset between start and end year and
  treat it as an ensemble member.

  When querying the last day of a leap year, the loader will return the first
  day of the following year for non-leap years.

  This is used as a probablistic baseline for the WeatherBench website.
  """

  def __init__(self, start_year: int, end_year: int, **kwargs):
    """Init.

    Args:
      start_year: The first year to include in the climatology.
      end_year: The last year (incl.) to include in the climatology.
      **kwargs: Other arguments to pass to XarrayDataLoader.
    """
    super().__init__(**kwargs)
    self._start_year = start_year
    self._end_year = end_year

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    if lead_times is None or isinstance(lead_times, slice):
      raise ValueError(
          'Exact lead times must be specified for persistence data loader.'
      )
    init_times = xr.DataArray(
        init_times, dims=['init_time'], coords={'init_time': init_times}
    )
    lead_times = xr.DataArray(
        lead_times, dims=['lead_time'], coords={'lead_time': lead_times}
    )
    valid_times = init_times + lead_times
    doy = valid_times.dt.dayofyear
    hod = valid_times.dt.hour
    cat_times = []
    for year in range(self._start_year, self._end_year + 1):
      cat_times.append(
          np.datetime64(str(year))
          + ((doy - 1) * 24 + hod)
          * np.timedelta64(1, 'h').astype('timedelta64[ns]')
      )
    cat_times = xr.concat(cat_times, dim='number')
    chunk = self._ds.sel(valid_time=cat_times)
    return chunk
