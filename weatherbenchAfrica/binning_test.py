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

from importlib import resources
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchAfrica import binning
from weatherbenchAfrica import test_utils
from weatherbenchAfrica.data_loaders import sparse_parquet
import xarray as xr


class BinningTest(parameterized.TestCase):

  def test_region_binning(self):

    statistic_values = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00'
    )['2m_temperature']
    statistic_base_shape = (
        statistic_values.latitude.shape + statistic_values.longitude.shape
    )

    regions = {
        'region1': ((20, 90), (-180, 180)),
    }

    bins = binning.Regions(regions=regions)
    # Since predictions and targets aren't used, just use the same array.
    mask = bins.create_bin_mask(statistic_values)
    self.assertEqual(mask.shape, (1,) + statistic_base_shape)

    regions = {
        'region1': ((20, 90), (-180, 180)),
        'region2': ((-90, -20), (-180, 180)),
    }

    bins = binning.Regions(regions=regions)
    mask = bins.create_bin_mask(statistic_values)
    self.assertEqual(mask.shape, (2,) + statistic_base_shape)

    # With a land_sea_mask
    land_sea_mask = xr.ones_like(mask.isel(region=0, drop=True)).where(
        mask.latitude > 0, False
    )
    bins = binning.Regions(regions=regions, land_sea_mask=land_sea_mask)
    mask = bins.create_bin_mask(statistic_values)
    self.assertEqual(mask.shape, (4,) + statistic_base_shape)

  def test_by_exact_coord_binning(self):
    target_path = resources.files('weatherbenchX').joinpath(
        'test_data/metar-timeNominal-by-month'
    )

    target_loader = sparse_parquet.METARFromParquet(
        path=target_path,
        variables=['2m_temperature'],
        partitioned_by='month',
        split_variables=True,
        dropna=True,
        time_dim='timeNominal',
        file_tolerance=np.timedelta64(1, 'h'),
        remove_duplicates=True,
    )
    init_times = np.array(
        ['2020-01-02T00', '2020-01-02T12'], dtype='datetime64[ns]'
    )
    lead_times = np.array([6, 12], dtype='timedelta64[h]')

    statistic = target_loader.load_chunk(init_times, lead_times)[
        '2m_temperature'
    ]

    bins = binning.ByExactCoord(coord='lead_time')
    mask = bins.create_bin_mask(statistic)
    np.testing.assert_allclose(mask.lead_time, lead_times)

    bins = binning.ByExactCoord(coord='stationName', add_global_bin=True)
    mask = bins.create_bin_mask(statistic)
    self.assertLen(mask.stationName, len(np.unique(statistic.stationName)) + 1)

    # Test empty input
    mask = bins.create_bin_mask(statistic.isel(index=[]))
    self.assertEqual(mask.size, 0)

  def test_by_time_unit_binning_with_with_datetime64(self):
    statistic_values = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-01T12',
        time_resolution='1 hr',
    )['2m_temperature']
    bins = binning.ByTimeUnit('hour', 'time')
    mask = bins.create_bin_mask(statistic_values)
    np.testing.assert_equal(mask.time_hour, np.arange(0, 12))

  @parameterized.parameters(
      ('second', '1 second', '6 second'),
      ('minute', '1 minute', '6 minute'),
      ('hour', '1 hour', '6 hour'),
      ('day', '1 day', '6 day'),
      ('week', '7 day', f'{7*6} day'),
      ('year', '365 day', f'{365*6} day'),
      ('hour', '15 minute', '6 hour'),
  )
  def test_by_time_unit_binning_with_with_timedelta64(
      self, unit, lead_resolution, lead_stop
  ):
    statistic_values = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-01T01',
        time_resolution='1 hr',
        lead_resolution=lead_resolution,
        lead_stop=lead_stop,
    )['2m_temperature']
    bins = binning.ByTimeUnit(unit, 'prediction_timedelta')
    mask = bins.create_bin_mask(statistic_values)
    np.testing.assert_equal(
        mask[f'prediction_timedelta_{unit}'], np.arange(0, 6 + 1)
    )

  def test_by_time_unit_from_seconds_binning(self):
    statistic_values = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-01T01',
        time_resolution='1 hr',
        lead_resolution='1 minute',
        lead_stop='6 hour',
    )['2m_temperature']
    statistic_values = statistic_values.assign_coords({
        'prediction_timedelta_sec': (
            statistic_values.prediction_timedelta.dt.total_seconds()
        )
    })
    bins = binning.ByTimeUnitFromSeconds(
        'hour', 'prediction_timedelta_sec', bins=np.arange(6)
    )
    mask = bins.create_bin_mask(statistic_values)
    np.testing.assert_equal(
        mask['prediction_timedelta_sec_hour'], np.arange(0, 6)
    )

  def test_by_coord_bins(self):
    target_path = resources.files('weatherbenchX').joinpath(
        'test_data/metar-timeNominal-by-month'
    )
    target_loader = sparse_parquet.METARFromParquet(
        path=target_path,
        variables=['2m_temperature'],
        partitioned_by='month',
        split_variables=True,
        dropna=True,
        time_dim='timeObs',
        file_tolerance=np.timedelta64(1, 'h'),
    )

    init_times = np.array(
        ['2020-01-02T00', '2020-01-02T12'], dtype='datetime64[ns]'
    )
    lead_times = slice(np.timedelta64(1, 'h'), np.timedelta64(6, 'h'))

    statistic = target_loader.load_chunk(init_times, lead_times)[
        '2m_temperature'
    ]
    bins = binning.ByCoordBins(
        'lead_time', np.arange(1, 7, dtype='timedelta64[h]')
    )
    mask = bins.create_bin_mask(statistic)
    self.assertTrue(np.all(mask.mean('index') > 0))

  def test_by_sets(self):
    target_path = resources.files('weatherbenchX').joinpath(
        'test_data/metar-timeNominal-by-month'
    )
    target_loader = sparse_parquet.METARFromParquet(
        path=target_path,
        variables=['2m_temperature'],
        partitioned_by='month',
        split_variables=True,
        dropna=True,
        time_dim='timeObs',
        file_tolerance=np.timedelta64(1, 'h'),
    )

    init_times = np.array(
        ['2020-01-02T00', '2020-01-02T12'], dtype='datetime64[ns]'
    )
    lead_times = slice(np.timedelta64(1, 'h'), np.timedelta64(6, 'h'))

    statistic = target_loader.load_chunk(init_times, lead_times)[
        '2m_temperature'
    ]

    bins = binning.BySets(
        {
            'set1': statistic.stationName[:10],
            'set2': statistic.stationName[10:20],
            'empty_set': [],
            'wrong_set': [1, 2, 3, 4],
        },
        coord_name='stationName',
        bin_dim_name='station_subset',
        add_global_bin=True,
    )

    mask = bins.create_bin_mask(statistic)
    self.assertLen(mask.station_subset, 5)
    self.assertGreaterEqual(mask.sum('index').sel(station_subset='set1'), 10)
    self.assertGreaterEqual(mask.sum('index').sel(station_subset='set2'), 10)
    self.assertEqual(mask.sum('index').sel(station_subset='empty_set'), 0)
    self.assertEqual(mask.sum('index').sel(station_subset='wrong_set'), 0)
    self.assertLen(statistic, mask.sum('index').sel(station_subset='global'))


if __name__ == '__main__':
  absltest.main()
