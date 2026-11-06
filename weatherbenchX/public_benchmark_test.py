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

"""Regression tests for the public benchmark's fractional land mask."""

import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from public_benchmark import run_benchmark_evaluation as benchmark
from weatherbenchX import binning
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import deterministic
import xarray as xr


class BenchmarkLandMaskTest(parameterized.TestCase):

  def _run_benchmark(self, land_fraction):
    config = types.ModuleType('_land_mask_test_config')
    config.deterministic_prediction_configs = {
        'fixture_fixture_2020': {
            'path': 'predictions.zarr',
            'variables': ['2m_temperature'],
        }
    }
    config.target_configs = {
        'era5_fixture': {'path': 'era5.zarr', 'variables': ['2m_temperature']}
    }
    config.climatology_configs = {
        'era5_fixture_2020': {'path': 'climatology.zarr'}
    }
    climatology = xr.Dataset({'2m_temperature': xr.zeros_like(land_fraction)})
    era5 = xr.Dataset({'land_sea_mask': land_fraction})

    def open_dataset(path):
      return {'climatology.zarr': climatology, 'era5.zarr': era5}[path]

    with (
        flagsaver.as_parsed(
            config=config.__name__,
            prediction='fixture',
            target='era5',
            resolution='fixture',
            year='2020',
            time_start='2020-01-01T00',
            time_stop='2020-01-01T12',
            lead_time_start='0',
            lead_time_stop='6',
            output_dir='unused',
            runner='DirectRunner',
            time_frequency='12',
            lead_time_frequency='6',
            init_time_chunk_size='1',
            lead_time_chunk_size='12',
            temporal='false',
        ),
        mock.patch.dict(sys.modules, {config.__name__: config}),
        mock.patch.object(benchmark.xr, 'open_zarr', side_effect=open_dataset),
        mock.patch.object(benchmark.xarray_loaders, 'PredictionsFromXarray'),
        mock.patch.object(benchmark.xarray_loaders, 'TargetsFromXarray'),
        mock.patch.object(benchmark.beam, 'Pipeline'),
        mock.patch.object(
            benchmark.beam_pipeline, 'define_pipeline'
        ) as pipeline,
    ):
      benchmark.main([])
    # Inspect the real Aggregator passed by the actual benchmark entrypoint.
    return pipeline.call_args.args[5]

  @parameterized.parameters(False, True)
  def test_land_fraction_threshold_and_coordinates(self, reversed_latitude):
    fractions = xr.DataArray(
        [
            [0.0, 1e-12, 0.49, 0.5, 0.51, 1.0, np.nan],
            [1.0, 0.51, 0.5, 0.49, 1e-12, 0.0, np.nan],
        ],
        dims=('latitude', 'longitude'),
        coords={'latitude': [-30.0, 30.0], 'longitude': np.arange(7) * 50.0},
    )
    if reversed_latitude:
      fractions = fractions.isel(latitude=slice(None, None, -1))
    aggregator = self._run_benchmark(fractions)
    masks = aggregator.bin_by[0].create_bin_mask(fractions)
    expected_land = (fractions >= 0.5).transpose('latitude', 'longitude')
    xr.testing.assert_equal(
        masks.sel(region='global_land', drop=True), expected_land
    )
    landsea_mask = binning.LandSea(fractions).create_bin_mask(fractions)
    xr.testing.assert_equal(
        masks.sel(region='global_land', drop=True),
        landsea_mask.sel(land_sea='land', drop=True).astype(bool),
    )
    unmasked = binning.Regions(benchmark.REGIONS).create_bin_mask(fractions)
    for region in benchmark.REGIONS:
      xr.testing.assert_equal(
          masks.sel(region=region, drop=True),
          unmasked.sel(region=region, drop=True),
      )
      xr.testing.assert_equal(
          masks.sel(region=f'{region}_land', drop=True),
          unmasked.sel(region=region, drop=True) & expected_land,
      )

  def test_binary_mask_preserves_existing_regions(self):
    fractions = xr.DataArray(
        [[0.0, 1.0], [1.0, 0.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [-30.0, 30.0], 'longitude': [0.0, 180.0]},
    )
    aggregator = self._run_benchmark(fractions)
    actual = aggregator.bin_by[0].create_bin_mask(fractions)
    expected = binning.Regions(
        benchmark.REGIONS, land_sea_mask=fractions.astype(bool)
    ).create_bin_mask(fractions)
    xr.testing.assert_equal(actual, expected)

  def test_land_rmse_excludes_mostly_ocean_cells(self):
    fractions = xr.DataArray(
        [[0.1, 0.5, 1.0], [0.1, 0.5, 1.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [-30.0, 30.0], 'longitude': [0.0, 120.0, 240.0]},
    )
    aggregator = self._run_benchmark(fractions)
    predictions = xr.Dataset(
        {
            '2m_temperature': xr.DataArray(
                [[10.0, 2.0, 2.0], [10.0, 2.0, 2.0]],
                dims=fractions.dims,
                coords=fractions.coords,
            ).expand_dims(init_time=[np.datetime64('2020-01-01')])
        }
    )
    targets = xr.zeros_like(predictions)
    metrics = {'rmse': deterministic.RMSE()}
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, predictions, targets
    )
    values = aggregator.aggregate_statistics(statistics).metric_values(metrics)
    self.assertAlmostEqual(
        float(values['rmse.2m_temperature'].sel(region='global_land')), 2.0
    )
    self.assertAlmostEqual(
        float(values['rmse.2m_temperature'].sel(region='global')), 6.0
    )


if __name__ == '__main__':
  absltest.main()
