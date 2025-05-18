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

from absl.testing import absltest
import numpy as np
from weatherbenchAfrica import test_utils
from weatherbenchAfrica.data_loaders import xarray_loaders
import xarray as xr


class XarrayLoadersTest(absltest.TestCase):

  def test_prediction_target_dimension_match(self):
    # Create and save mock prediction and target datasets.
    target = test_utils.mock_target_data(
        time_start='2020-01-01T00', time_stop='2020-01-20T00'
    )
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00'
    )

    target_path = self.create_tempdir('target.zarr').full_path
    prediction_path = self.create_tempdir('prediction.zarr').full_path

    target.to_zarr(target_path)
    prediction.to_zarr(prediction_path)

    # Initialize data loaders.
    variables = ['geopotential', '2m_temperature']
    target_data_loader = xarray_loaders.TargetsFromXarray(
        path=target_path, variables=variables
    )
    prediction_data_loader = xarray_loaders.PredictionsFromXarray(
        path=prediction_path, variables=variables
    )

    init_times = np.arange(
        '2020-01-01T00',
        '2020-01-02T00',
        np.timedelta64(24, 'h'),
        dtype='datetime64[ns]',
    )
    lead_times = np.arange(3, dtype='timedelta64[D]').astype('timedelta64[ns]')
    target_chunk = target_data_loader.load_chunk(init_times, lead_times)
    prediction_chunk = prediction_data_loader.load_chunk(init_times, lead_times)

    for d in target_chunk.dims:
      xr.testing.assert_equal(target_chunk[d], prediction_chunk[d])

  def test_climatology_loader(self):
    target = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
    ).rename(time='init_time', prediction_timedelta='lead_time')
    climatology = target.isel(init_time=0, lead_time=0, drop=True).expand_dims(
        dayofyear=366, hour=4
    )
    init_times = np.arange(
        '2020-01-01T00',
        '2020-01-02T00',
        np.timedelta64(24, 'h'),
        dtype='datetime64[ns]',
    )
    lead_times = np.arange(0, 3, 1, dtype='timedelta64[D]')
    variables = ['geopotential', '2m_temperature']
    climatology_data_loader = xarray_loaders.ClimatologyFromXarray(
        ds=climatology,
        variables=variables,
        climatology_time_coords=['dayofyear', 'hour'],
    )
    climatology_chunk = climatology_data_loader.load_chunk(
        init_times, lead_times
    )
    self.assertEqual(
        set(climatology_chunk.dims),
        {'init_time', 'lead_time', 'level', 'latitude', 'longitude'},
    )

  def test_persistence_loader(self):
    target = test_utils.mock_target_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
    )
    init_times = np.arange(
        '2020-01-01T00',
        '2020-01-02T00',
        np.timedelta64(24, 'h'),
        dtype='datetime64[ns]',
    )
    lead_times = np.arange(0, 3, 1, dtype='timedelta64[D]')
    variables = ['geopotential', '2m_temperature']
    persistence_data_loader = xarray_loaders.PersistenceFromXarray(
        ds=target,
        variables=variables,
    )
    persistence_chunk = persistence_data_loader.load_chunk(
        init_times, lead_times
    )
    self.assertEqual(
        set(persistence_chunk.dims),
        {'init_time', 'lead_time', 'level', 'latitude', 'longitude'},
    )

  def test_probabilistic_climatology_loader(self):
    target = test_utils.mock_target_data(
        time_start='2015-01-01T00',
        time_stop='2021-01-01T00',
        variables_3d=[],
        variables_2d=['2m_temperature'],
    )
    init_times = np.arange(
        '2020-12-30T00',
        '2021-01-01T00',
        np.timedelta64(24, 'h'),
        dtype='datetime64[ns]',
    )
    lead_times = np.arange(0, 3, 1, dtype='timedelta64[D]')
    loader = xarray_loaders.ProbabilisticClimatologyFromXarray(
        ds=target, start_year=2015, end_year=2019
    )
    chunk = loader.load_chunk(init_times, lead_times)
    self.assertEqual(
        set(chunk.dims),
        {'number', 'init_time', 'lead_time', 'latitude', 'longitude'},
    )
    self.assertLen(chunk.number, 5)


if __name__ == '__main__':
  absltest.main()
