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
"""Data loaders for tabular data stored in Parquet format."""

import functools
import os
from typing import Callable, Hashable, Mapping, Optional, Sequence, Union
import numpy as np
import pandas as pd
import pyarrow
from weatherbenchX import interpolations
from weatherbenchX.data_loaders import base
import xarray as xr


def get_parquet_files_subset(
    path: str,
    time_start: np.timedelta64,
    time_end: np.timedelta64,
    partition_by: str,
):
  """Get subset of parquet files for a given time interval."""
  if partition_by == 'month':
    unit = 'M'
  elif partition_by == 'day':
    unit = 'D'
  elif partition_by == 'hour':
    unit = 'h'
  else:
    raise NotImplementedError(f'{partition_by} not implemented.')
  time_start = np.datetime64(time_start, unit)
  time_end = np.datetime64(time_end, unit)
  td = np.timedelta64(1, unit)
  times = np.arange(time_start, time_end + td, td)
  files = []
  for time in times:
    fn = parquet_filename_for_time(path, time, unit)
    files.append(fn)
  return files


def parquet_filename_for_time(path: str, time: np.datetime64, unit: str) -> str:
  """Return parquet partition filename for a given time."""
  year = time.item().year
  month = time.item().month
  if unit == 'M':
    fn = f'year={year}/month={month}/{year}-{str(month).zfill(2)}.parquet'
  elif unit == 'D':
    day = time.item().day
    fn = f'year={year}/month={month}/day={day}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.parquet'
  elif unit == 'h':
    day = time.item().day
    hour = time.item().hour
    fn = f'year={year}/month={month}/day={day}/hour={hour}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}T{str(hour).zfill(2)}.parquet'
  else:
    raise NotImplementedError
  fn = os.path.join(path, fn)
  return fn


class SparseObservationsFromParquet(base.DataLoader):
  """Reads general sparse observation data stored in Parquet format.

  It is assumed that the data is partitioned by month, day or hour. A daily
  partition would follow the following directory structure:
  <PATH>/year=2020/month=1/day=1/2020-01-01.parquet

  Since auto-discovery of files can take a long time, this data loader assumes
  this format to quickly query the desired sub-files for a given time interval.

  Currently, this assumes there are no missing files.
  """

  def __init__(
      self,
      path: str,
      partitioned_by: str,
      time_dim: str,
      variables: Sequence[str],
      coordinate_variables: Sequence[str] = (),
      split_variables: bool = False,
      dropna: bool = False,
      add_nan_mask: bool = False,
      tolerance: Optional[np.timedelta64] = None,
      rename_variables: Optional[Mapping[str, str]] = None,
      include_slice_end_time: bool = False,
      remove_duplicates: bool = False,
      pick_closest_duplicate_by: Optional[str] = None,
      observation_dim: Optional[str] = None,
      file_tolerance: np.timedelta64 = np.timedelta64(1, 'h'),
      preprocessing_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
      interpolation: Optional[interpolations.Interpolation] = None,
  ):
    """Init.

    Args:
      path: Path to Parquet dataset.
      partitioned_by: How the Parquet file is partitioned. 'hour', 'day' or
        'month'.
      time_dim: Time dimension on Parquet files (before renaming) to use for
        time filtering.
      variables: Variables to load (after renaming).
      coordinate_variables: Coordinate variables to load. These will be
        converted to an xarray coordinates. 'valid_time' is always a coordinate
        and represents the original value of the time_dim coordinate. Default:
        ()
      split_variables: Whether to return the loaded data as a dictionary of
        DataArrays. Default: False.
      dropna: Whether to drop missing values. If split_variables is True, values
        will be dropped for each variable separately. Otherwise, only indices
        where all variables are non-NaN will be returned.
      add_nan_mask: Adds a boolean coordinate named 'mask' to each variable
        (variables will be split into DataArrays if they aren't already), with
        False indicating NaN values. To be used for masked aggregation. Default:
        False.
      tolerance: (Optional) Tolerance around the given valid time. Data within
        valid_time +/- tolerance will be returned. This is only supported for
        exact lead_times. The resulting init and lead time coordinates will be
        those requested. The valid_time dimension will reflect the original time
        for each observation.
      rename_variables: (Optional) Renaming dictionary.
      include_slice_end_time: Whether slice end time is included. Default: False
      remove_duplicates: For exact lead times, whether duplicate stations
        (specified by `observation_dim`) for the same valid time are removed. If
        True, this will pick the closest time specified by
        `pick_closest_duplicate_by` to the valid_time and keep it. Default:
        False
      pick_closest_duplicate_by: (Optional) Time dimension to use to pick the
        closest duplicate.
      observation_dim: (Optional) Dimension identifying e.g. station names. This
        is used to remove duplicate observations.
      file_tolerance: 'timeObs' does not always align with the time on the
        partition. To make sure all required times are read, open the files with
        +/- file_tolerance. The 'timeObs' of most observations are within a one
        hour window of the nominal time. 'timeNominal' will be equal to the
        partition time and would therefore not require a file_tolerane. Default:
        1h
      preprocessing_fn: (Optional) Function to apply to the dataframe after
        reading.
      interpolation: (Optional) Interpolation to be applied to the data.
    """

    super().__init__(
        interpolation=interpolation,
        compute=False,  # Data is already loaded.
        add_nan_mask=add_nan_mask,
    )
    self._path = path
    if partitioned_by not in ['hour', 'day', 'month']:
      raise ValueError(f'Unsupported partitioned_by: {partitioned_by}')
    self._partitioned_by = partitioned_by
    self._time_dim = time_dim
    self._variables = variables
    self._coordinate_variables = list(coordinate_variables) + ['valid_time']
    self._split_variables = split_variables
    self._dropna = dropna
    if tolerance == np.timedelta64(0, 'h'):
      raise ValueError(
          'Tolerance should not be zero. This will always return an emptyarray.'
      )
    self._tolerance = tolerance
    self._rename_variables = rename_variables
    self._include_slice_end_time = include_slice_end_time
    self._remove_duplicates = remove_duplicates
    self._pick_closest_duplicate_by = pick_closest_duplicate_by
    if remove_duplicates:
      if observation_dim is None:
        raise ValueError(
            'station_dim must be specified if remove_duplicates is True.'
        )
    self._observation_dim = observation_dim
    self._file_tolerance = file_tolerance
    self._preprocessing_fn = preprocessing_fn

  def _pick_closest_from_duplicates(
      self, df: pd.DataFrame, valid_time: np.datetime64
  ):
    """Pick row where `_pick_closest_duplicate_by` is closest to valid_time."""
    if self._pick_closest_duplicate_by is not None:
      df['time_diff'] = np.abs(df[self._pick_closest_duplicate_by] - valid_time)
      df = df.sort_values('time_diff', ascending=True)
    non_duplicated = df[~df[self._observation_dim].duplicated(keep='first')]
    return non_duplicated

  def _load_data_for_single_time(
      self,
      valid_time: Optional[np.datetime64],
      lead_time_slice: Optional[slice] = None,
  ) -> xr.Dataset:
    """Load data for some valid time.

    If lead_time_slice is given, load data for valid_time +/- lead_time_slice.
    Otherwise, tolerance and file tolerance are applied around valid_time.

    Args:
      valid_time: Base time to load data for.
      lead_time_slice: (Optional) If given, load data for valid_time +/-
        lead_time_slice.

    Returns:
      xarray.Dataset with data for the given valid_time.
    """

    if self._tolerance is None:
      if lead_time_slice is None:
        start_time = valid_time
        stop_time = None
      else:
        start_time = valid_time - lead_time_slice.start
        stop_time = valid_time + lead_time_slice.stop

    else:
      start_time = valid_time - self._tolerance
      stop_time = valid_time + self._tolerance

    # Get subset of files since filtering can take a very long time.
    # Also create additional filters to exactly get required times.
    if stop_time is None:
      file_start_time = start_time - self._file_tolerance
      file_stop_time = start_time + self._file_tolerance

      ts = pd.Timestamp(start_time)
      filters = [(self._time_dim, '=', ts)]

    else:
      file_start_time = start_time - self._file_tolerance
      file_stop_time = stop_time + self._file_tolerance

      ts_start = pd.Timestamp(start_time)
      ts_stop = pd.Timestamp(stop_time)
      if self._include_slice_end_time:
        filters = [
            (self._time_dim, '>=', ts_start),
            (self._time_dim, '<=', ts_stop),
        ]
      else:
        filters = [
            (self._time_dim, '>=', ts_start),
            (self._time_dim, '<', ts_stop),
        ]
    files = get_parquet_files_subset(
        self._path, file_start_time, file_stop_time, self._partitioned_by
    )

    def _read_single_file(fn):
      # Filters don't work for empty files. Catch this error but make sure the
      # file is empty.
      try:
        df = pd.read_parquet(fn, filters=filters)
      except pyarrow.lib.ArrowTypeError:
        df = pd.read_parquet(fn)
        assert len(df) == 0, 'This should only happen if the file is empty.'  # pylint: disable=g-explicit-length-test
      return df

    df = pd.concat([_read_single_file(fn) for fn in files], ignore_index=True)

    if self._preprocessing_fn is not None:
      df = self._preprocessing_fn(df)

    if self._remove_duplicates:
      assert (
          lead_time_slice is None
      ), 'Removing duplicates not compatible with slice lead_time.'
      df = self._pick_closest_from_duplicates(df, valid_time)

    if self._rename_variables is not None:
      df = df.rename(columns=self._rename_variables)

    df = df.rename(columns={self._time_dim: 'valid_time'})

    return df.loc[
        :,
        self._variables + self._coordinate_variables,  # pytype: disable=unsupported-operands
    ]

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:

    dfs = []
    # Case #1: Exact lead times or no lead_times
    if not isinstance(lead_times, slice):
      # Get data for each valid time
      for init_time in init_times:
        # Case #1.1: No lead times, i.e. init_time = valid_time
        if lead_times is None:
          if self._tolerance is None:
            df = self._load_data_for_single_time(init_time)
          else:
            df = self._load_data_for_single_time(init_time)

          dfs.append(df)
        # Case #1.2: Exact init_times given
        else:
          for lead_time in lead_times:
            valid_time = init_time + lead_time
            if self._tolerance is None:
              df = self._load_data_for_single_time(valid_time)
            else:

              df = self._load_data_for_single_time(valid_time)

            df['init_time'] = init_time
            df['lead_time'] = lead_time
            dfs.append(df)

    # Case #2: Lead time slice
    else:
      assert (
          self._tolerance is None
      ), 'Tolerance not compatible with lead_time slice.'

      for init_time in init_times:

        df = self._load_data_for_single_time(
            init_time, lead_time_slice=lead_times
        )
        df['init_time'] = init_time
        df['lead_time'] = df.valid_time - df.init_time
        dfs.append(df)

    # Combine dataframes
    combined_df = pd.concat(dfs)
    combined_df.index = range(len(combined_df))
    time_coords = [] if lead_times is None else ['init_time', 'lead_time']
    ds = combined_df.to_xarray().set_coords(
        self._coordinate_variables + time_coords
    )

    if self._split_variables:
      dic = dict(ds)
      if self._dropna:
        for v, da in dic.items():
          dic[v] = da.dropna('index')
      return dic
    else:
      if self._dropna:
        ds = ds.dropna('index')
      return ds


# METAR constants
METAR_TO_ERA5_NAMES = {
    'seaLevelPress': 'mean_sea_level_pressure',
    'temperature': '2m_temperature',
    'dewpoint': '2m_dewpoint_temperature',
    'windSpeed': '10m_wind_speed',
    'windGust': '10m_wind_gust',
    'windDir': '10m_wind_direction',
    'minTemp24Hour': 'min_2m_temperature_24hr',
    'maxTemp24Hour': 'max_2m_temperature_24hr',
    'precip1Hour': 'total_precipitation_1hr',
    'precip3Hour': 'total_precipitation_3hr',
    'precip6Hour': 'total_precipitation_6hr',
    'precip24Hour': 'total_precipitation_24hr',
    'precipRate': 'precipitation_rate',
}
ERA5_TO_METAR_NAMES = {v: k for k, v in METAR_TO_ERA5_NAMES.items()}

METAR_QC_SUFFIX = 'DD'

METAR_BAD_QUALITY_FLAGS = ('Z', 'B', 'X', 'Q', 'k')

METAR_COORDINATE_VARIABLES = (
    'latitude',
    'longitude',
    'elevation',
    'stationName',
)


# METAR preprocessing functions
def set_bad_quality_to_nan(
    df: pd.DataFrame,
    variables: Sequence[str],
    qc_suffix: str,
    bad_quality_flags: Sequence[str],
):
  for variable in variables:
    df[variable] = df[variable].where(
        ~np.isin(df[variable + qc_suffix], bad_quality_flags), np.nan
    )
  return df


def convert_longitude_to_0_to_360(
    df: pd.DataFrame, longitude_dim: str = 'longitude'
):
  df[longitude_dim] = np.mod(df[longitude_dim], 360)
  return df


class METARFromParquet(SparseObservationsFromParquet):
  """Reads METAR data stored in Parquet format.

  This implementation of SparseObservationsFromParquet sets all the default
  values for METAR and adds METAR-specific preprocessing functions.

  - Bad quality flags are set to NaN: ('Z', 'B', 'X', 'Q', 'k')
  - Longitude is converted to 0 to 360.
  - Elevation with fill values 9.999e+03 is set to NaN.

  Example:
      >>> init_times, lead_times
      (array(['2020-01-01T00:00:00.000000000', '2020-01-01T12:00:00.000000000'],
       dtype='datetime64[ns]'), array([ 6, 12], dtype='timedelta64[h]'))
      >>> target_data_loader = sparse_parquet.METARFromParquet(
      >>> path=<PATH>,
      >>> variables=['2m_temperature', '10m_wind_speed'],
      >>> split_variables=False,
      >>> partitioned_by='month',
      >>> dropna=True,
      >>> time_dim='timeNominal',
      >>> )
      >>> target_data_loader.load_chunk(init_times, lead_times)
      <xarray.Dataset>
      Dimensions:           (index: 31478)
      Coordinates:
        * index             (index) int64 0 1 2 3 4 ... 33021 33022 33023 33024
        33025
          2m_temperatureDD  (index) object 'S' 'S' 'S' 'S' 'S' ... 'V' 'S' 'S'
          'S' 'S'
          10m_wind_speedDD  (index) object 'S' 'S' 'S' 'S' 'S' ... 'V' 'S' 'S'
          'S' 'S'
          latitude          (index) float32 -77.87 -53.8 -33.38 ... 46.55 49.82
          49.83
          longitude         (index) float32 167.0 292.2 289.2 ... 299.0 285.0
          295.7
          elevation         (index) float32 8.0 22.0 476.0 141.0 ... 13.0 381.0
          53.0
          valid_time        (index) datetime64[ns] 2020-01-01T06:00:00 ...
          2020-01-02
          stationName       (index) object 'NZCM' 'SAWE' 'SCEL' ... 'CWUK'
          'CWBY'
          init_time         (index) datetime64[ns] 2020-01-01 ...
          2020-01-01T12:00:00
          lead_time         (index) timedelta64[ns] 06:00:00 06:00:00 ...
          12:00:00
      Data variables:
          2m_temperature    (index) float32 273.1 282.1 291.1 ... 274.1 268.1
          272.9
          10m_wind_speed    (index) float32 4.1 5.1 2.1 2.1 1.0 ... 12.4 9.3 2.1
          2.1
  """

  def __init__(
      self,
      path: str,
      variables: Sequence[str],
      time_dim: str,
      split_variables: bool = False,
      dropna: bool = False,
      add_nan_mask: bool = False,
      tolerance: Optional[np.timedelta64] = None,
      partitioned_by: str = 'month',
      rename_variables: Optional[Mapping[str, str]] = None,
      include_slice_end_time: bool = False,
      remove_duplicates: bool = False,
      pick_closest_duplicate_by: Optional[str] = None,
      file_tolerance: np.timedelta64 = np.timedelta64(1, 'h'),
      preprocessing_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
      interpolation: Optional[interpolations.Interpolation] = None,
  ):
    def metar_preprocessing_fn(
        df: pd.DataFrame,
        preprocessing_fn: Optional[
            Callable[[pd.DataFrame], pd.DataFrame]
        ] = None,
    ):
      if preprocessing_fn is not None:
        df = preprocessing_fn(df)
      df = set_bad_quality_to_nan(
          df,
          # Rename to raw variables since this happens before renaming.
          [ERA5_TO_METAR_NAMES[v] for v in variables],
          METAR_QC_SUFFIX,
          METAR_BAD_QUALITY_FLAGS,
      )
      df = convert_longitude_to_0_to_360(df)
      # Set elevation with fill values 9.999e+03 to NaN.
      df['elevation'] = df['elevation'].where(
          df['elevation'] < 9.999e03, np.nan
      )
      return df

    super().__init__(
        path=path,
        variables=variables,
        time_dim=time_dim,
        coordinate_variables=METAR_COORDINATE_VARIABLES,
        observation_dim='stationName',
        split_variables=split_variables,
        dropna=dropna,
        add_nan_mask=add_nan_mask,
        tolerance=tolerance,
        partitioned_by=partitioned_by,
        rename_variables=METAR_TO_ERA5_NAMES,
        include_slice_end_time=include_slice_end_time,
        remove_duplicates=remove_duplicates,
        pick_closest_duplicate_by=pick_closest_duplicate_by,
        file_tolerance=file_tolerance,
        preprocessing_fn=functools.partial(
            metar_preprocessing_fn, preprocessing_fn=preprocessing_fn
        ),
        interpolation=interpolation,
    )
