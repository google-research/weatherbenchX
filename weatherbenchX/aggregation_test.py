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

from absl.testing import absltest
from weatherbenchX import aggregation
from weatherbenchX import binning
from weatherbenchX import test_utils
from weatherbenchX import weighting
from weatherbenchX import xarray_tree
from weatherbenchX.data_loaders import base as data_loaders_base
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import deterministic
import xarray as xr


class AggregationTest(absltest.TestCase):

  def _get_test_data(self):
    template = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        lead_start='0 days',
        lead_stop='1 day',
    ).rename({'time': 'init_time', 'prediction_timedelta': 'lead_time'})
    predictions = xr.zeros_like(template)
    targets = xr.ones_like(template)
    return predictions, targets

  def _get_example_aggregation_state(self):
    return aggregation.AggregationState(
        sum_weighted_statistics={
            'stat_name': {
                'var1': xr.DataArray([1, 2], dims=['x']),
                'var2': xr.DataArray([3, 4], dims=['x']),
            }
        },
        sum_weights={
            'stat_name': {
                'var1': xr.DataArray([5, 6], dims=['x']),
                'var2': xr.DataArray([7, 8], dims=['x']),
            }
        },
    )

  def _aggregate(self, all_metrics, predictions, targets, aggregation_kwargs):

    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        all_metrics, predictions, targets
    )

    aggregation_method = aggregation.Aggregator(
        **aggregation_kwargs,
    )
    aggregation_state = aggregation_method.aggregate_statistics(statistics)

    return aggregation_state

  def test_expected_output(self):
    """Run through a simple example and test expected output."""
    predictions, targets = self._get_test_data()

    all_metrics = {'rmse': deterministic.RMSE()}

    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude']},
    )
    actual = aggregation_state.metric_values(all_metrics)

    # Summing should, in this case, not change the result.
    actual_summed = (aggregation_state + aggregation_state).metric_values(
        all_metrics
    )

    expected = xr.Dataset({
        'rmse.2m_temperature': xr.DataArray(
            [1.0, 1.0], coords={'lead_time': predictions.lead_time}
        ),
        'rmse.geopotential': xr.DataArray(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            coords={
                'lead_time': predictions.lead_time,
                'level': predictions.level,
            },
        ),
    })

    for v in expected:
      xr.testing.assert_allclose(actual[v], expected[v])
      xr.testing.assert_allclose(actual_summed[v], expected[v])

  def test_missing_reduce_dims(self):
    predictions, targets = self._get_test_data()

    all_metrics = {'rmse': deterministic.RMSE()}

    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['level', 'latitude', 'longitude']},
    )
    values = aggregation_state.metric_values(all_metrics)

    # Note that 2m_temperature is missing level, so it's excluded.
    self.assertCountEqual(values.data_vars.keys(), ['rmse.geopotential'])

  def test_nan_handling(self):
    predictions, targets = self._get_test_data()

    targets = targets.where(targets.latitude > 0)
    targets = data_loaders_base.add_nan_mask_to_data(targets)

    all_metrics = {'rmse': deterministic.RMSE()}

    # Result should be all NaNs.
    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude']},
    )
    actual = aggregation_state.metric_values(all_metrics)
    self.assertTrue(actual['rmse.geopotential'].isnull().all())

    # With mask, results should be finite.
    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude'], 'masked': True},
    )
    actual = aggregation_state.metric_values(all_metrics)
    self.assertFalse(actual['rmse.geopotential'].isnull().any())

    # With skipa=True, should be finite.
    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude'], 'skipna': True},
    )
    actual = aggregation_state.metric_values(all_metrics)
    self.assertFalse(actual['rmse.geopotential'].isnull().any())

    # With mask only on one variable, only that one should be finite.
    targets['2m_temperature'] = targets['2m_temperature'].drop('mask')
    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude'], 'masked': True},
    )
    actual = aggregation_state.metric_values(all_metrics)
    self.assertFalse(actual['rmse.geopotential'].isnull().any())
    self.assertTrue(actual['rmse.2m_temperature'].isnull().any())

  def test_weighting(self):

    predictions, targets = self._get_test_data()

    all_metrics = {'rmse': deterministic.RMSE()}

    # Create a fake 2x weighting.
    class TestWeighting(weighting.Weighting):

      def weights(
          self,
          statistic: xr.DataArray,
      ) -> xr.DataArray:
        return xr.ones_like(statistic) * 2

    # Use it twice, should result in 4x weights.
    weigh_by = [TestWeighting(), TestWeighting()]

    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {'reduce_dims': ['init_time', 'latitude', 'longitude']},
    )
    actual = aggregation_state.metric_values(all_metrics)

    aggregation_state_4x = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {
            'reduce_dims': ['init_time', 'latitude', 'longitude'],
            'weigh_by': weigh_by,
        },
    )
    actual_4x = aggregation_state_4x.metric_values(all_metrics)

    # Aggregation state should be 4x.
    for stat in aggregation_state.sum_weighted_statistics:
      for variable in aggregation_state.sum_weighted_statistics[stat]:
        xr.testing.assert_allclose(
            aggregation_state.sum_weighted_statistics[stat][variable] * 4,
            aggregation_state_4x.sum_weighted_statistics[stat][variable],
        )
        xr.testing.assert_allclose(
            aggregation_state.sum_weights[stat][variable] * 4,
            aggregation_state_4x.sum_weights[stat][variable],
        )
    # Results should be the same.
    for variable in actual:
      xr.testing.assert_allclose(actual[variable], actual_4x[variable])

  def test_binning(self):
    predictions, targets = self._get_test_data()
    all_metrics = {'rmse': deterministic.RMSE()}

    regions1 = {'north': ((0, 90), (0, 360)), 'south': ((-90, 0), (0, 360))}
    regions2 = {'east': ((-90, 90), (0, 180)), 'west': ((-90, 90), (180, 360))}
    bin_by = [
        binning.Regions(regions1, bin_dim_name='bins1'),
        binning.Regions(regions2, bin_dim_name='bins2'),
    ]
    aggregation_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {
            'reduce_dims': ['init_time', 'latitude', 'longitude'],
            'bin_by': bin_by,
        },
    )
    actual = aggregation_state.metric_values(all_metrics)
    # Test for correct dimensions.
    self.assertEqual(
        set(actual.dims), set(['bins1', 'bins2', 'lead_time', 'level'])
    )

  def test_spatial_coarsening(self):
    predictions, targets = self._get_test_data()
    all_metrics = {'rmse': deterministic.RMSE()}

    orig_lat_size = predictions.sizes['latitude']
    orig_lon_size = predictions.sizes['longitude']
    target_spatial_resolution = 20.0

    agg_state = self._aggregate(
        all_metrics,
        predictions,
        targets,
        {
            'reduce_dims': ['init_time'],
            'target_spatial_resolution': target_spatial_resolution,
        },
    )
    actual = agg_state.metric_values(all_metrics)
    # Native resolution is 10.0 degrees; target 20.0 degrees gives window
    # size 2.
    self.assertEqual(
        actual['rmse.2m_temperature'].sizes['latitude'],
        orig_lat_size // 2,
    )
    self.assertEqual(
        actual['rmse.2m_temperature'].sizes['longitude'],
        orig_lon_size // 2,
    )

  def test_spatial_coarsening_multiple_resolution_inputs(self):
    pred_fine = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        lead_start='0 days',
        lead_stop='1 day',
        spatial_resolution_in_degrees=5.0,
    ).rename({'time': 'init_time', 'prediction_timedelta': 'lead_time'})
    pred_coarse = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        lead_start='0 days',
        lead_stop='1 day',
        spatial_resolution_in_degrees=10.0,
    ).rename({'time': 'init_time', 'prediction_timedelta': 'lead_time'})

    predictions = {
        'fine_var': xr.zeros_like(pred_fine['2m_temperature']),
        'coarse_var': xr.zeros_like(pred_coarse['2m_temperature']),
    }
    targets = {
        'fine_var': xr.ones_like(pred_fine['2m_temperature']),
        'coarse_var': xr.ones_like(pred_coarse['2m_temperature']),
    }
    all_metrics = {'rmse': deterministic.RMSE()}
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        all_metrics, predictions, targets
    )
    aggregator = aggregation.Aggregator(
        reduce_dims=['init_time'],
        target_spatial_resolution=20.0,
    )
    agg_state = aggregator.aggregate_statistics(statistics)
    mean_stats = agg_state.mean_statistics()

    fine_stat = mean_stats['SquaredError']['fine_var']
    coarse_stat = mean_stats['SquaredError']['coarse_var']

    fine_lat_res = float(
        abs(fine_stat['latitude'][1] - fine_stat['latitude'][0])
    )
    coarse_lat_res = float(
        abs(coarse_stat['latitude'][1] - coarse_stat['latitude'][0])
    )
    self.assertEqual(fine_lat_res, 20.0)
    self.assertEqual(coarse_lat_res, 20.0)

    fine_lon_res = float(
        abs(fine_stat['longitude'][1] - fine_stat['longitude'][0])
    )
    coarse_lon_res = float(
        abs(coarse_stat['longitude'][1] - coarse_stat['longitude'][0])
    )
    self.assertEqual(fine_lon_res, 20.0)
    self.assertEqual(coarse_lon_res, 20.0)

    self.assertEqual(fine_stat.sizes['latitude'], coarse_stat.sizes['latitude'])
    self.assertEqual(
        fine_stat.sizes['longitude'], coarse_stat.sizes['longitude']
    )

  def test_aggregation_state_round_trip_data_tree(self):
    aggregation_state = self._get_example_aggregation_state()
    data_tree = aggregation_state.to_data_tree()
    roundtripped = aggregation.AggregationState.from_data_tree(data_tree)
    xarray_tree.map_structure(
        xr.testing.assert_allclose,
        (aggregation_state.sum_weighted_statistics,
         aggregation_state.sum_weights),
        (roundtripped.sum_weighted_statistics,
         roundtripped.sum_weights),
    )

  def test_aggregation_state_round_trip_dataset(self):
    aggregation_state = self._get_example_aggregation_state()
    data_tree = aggregation_state.to_dataset()
    roundtripped = aggregation.AggregationState.from_dataset(data_tree)
    xarray_tree.map_structure(
        xr.testing.assert_allclose,
        (aggregation_state.sum_weighted_statistics,
         aggregation_state.sum_weights),
        (roundtripped.sum_weighted_statistics,
         roundtripped.sum_weights),
    )


if __name__ == '__main__':
  absltest.main()
