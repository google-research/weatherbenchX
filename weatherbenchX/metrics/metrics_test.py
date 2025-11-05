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
"""Unit tests for metrics."""

import dataclasses
import itertools
from typing import Hashable, Mapping
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import test_utils
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import categorical
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import probabilistic
from weatherbenchX.metrics import spatial
from weatherbenchX.metrics import wrappers
import xarray as xr


# Multivariate metric for testing.
@dataclasses.dataclass
class SampleMultivariateStatistic(metrics_base.Statistic):
  """Simple multivariate statistic that adds two variables of the predictions."""

  var1: str
  var2: str
  out_name: str

  @property
  def unique_name(self) -> str:
    return f'SampleMultivariateStatistic_{self.out_name}_from_{self.var1}_and_{self.var2}'

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    return {self.out_name: predictions[self.var1] + predictions[self.var2]}


@dataclasses.dataclass
class SampleMultivariateMetric(metrics_base.Metric):
  """Simple multivariate metric that adds two variables of the predictions."""

  var1: str
  var2: str
  out_name: str

  @property
  def statistics(self) -> Mapping[Hashable, metrics_base.Statistic]:
    return {
        'SampleMultivariateStatistic': SampleMultivariateStatistic(
            var1=self.var1, var2=self.var2, out_name=self.out_name
        ),
    }

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return statistic_values['SampleMultivariateStatistic']


def compute_precipitation_metric(metrics, metric_name, prediction, target):
  """Helper to compute metric values."""
  stats = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, prediction, target
  )
  stats = xarray_tree.map_structure(
      lambda x: x.mean(
          ('time', 'prediction_timedelta', 'latitude', 'longitude'),
          skipna=False,
      ),
      stats,
  )
  return metrics_base.compute_metric_from_statistics(
      metrics[metric_name], stats
  )['total_precipitation_1hr']


def compute_all_metrics(metrics, predictions, targets, reduce_dims):
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions, targets
  )
  aggregator = aggregation.Aggregator(
      reduce_dims=reduce_dims,
  )
  aggregation_state = aggregator.aggregate_statistics(statistics)
  results = aggregation_state.metric_values(metrics)
  return results


class MetricsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('split_variables=False', False),
      ('split_variables=True', True),
  )
  def test_statistics_computation(self, split_variables):
    target = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-20T00',
        variables_2d=['2m_temperature', '10m_wind_speed'],
    )
    prediction = (
        test_utils.mock_prediction_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-03T00',
            variables_2d=['2m_temperature', '10m_wind_speed'],
        )
        + 1
    )
    if split_variables:
      target = dict(target)
      prediction = dict(prediction)
    metrics = {
        'rmse': deterministic.RMSE(),
        'multivariate_metric': SampleMultivariateMetric(
            var1='2m_temperature', var2='10m_wind_speed', out_name='test'
        ),
    }
    stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, prediction, target
    )

    # Some basic sanity checks
    # 1. Variables remain the same
    self.assertSetEqual(set(stats['SquaredError']), set(target))
    # 2. The statistics are computed correctly
    self.assertEqual(stats['SquaredError']['2m_temperature'].mean(), 1.0)
    # 3. Dimension remain the same
    self.assertEqual(
        stats['SquaredError']['geopotential'].shape,
        prediction['geopotential'].shape,
    )
    # 4. Test value from mean statistics
    # Dict of DataArrays.
    for v in stats['SquaredError']:
      xr.testing.assert_equal(
          metrics_base.compute_metric_from_statistics(metrics['rmse'], stats)[
              v
          ],
          stats['SquaredError'][v],
      )
    # 5. Test multivariate metric
    self.assertEqual(
        list(
            metrics_base.compute_metric_from_statistics(
                metrics['multivariate_metric'], stats
            )
        ),
        ['test'],
    )

  def test_far(self):
    ds = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        variables_2d=['total_precipitation_1hr'],
        variables_3d=[],
    )
    metrics = {'far': categorical.FalseAlarmRate()}

    # 1. Only True Negatives, should be NaN
    self.assertTrue(
        np.isnan(compute_precipitation_metric(metrics, 'far', ds, ds))
    )

    # 2. Only True Positives, should be 0
    tmp = ds.copy(deep=True) + 1
    self.assertEqual(compute_precipitation_metric(metrics, 'far', tmp, tmp), 0)

    # 3. Only False Positives, should be 1
    self.assertEqual(compute_precipitation_metric(metrics, 'far', tmp, ds), 1)

    # 4. Half False Positives, should be 0.5
    tmp2 = ds.copy(deep=True)
    # Time dimension is size 2 in position 1.
    tmp2['total_precipitation_1hr'][{'time': 0}] = 1
    self.assertEqual(
        compute_precipitation_metric(metrics, 'far', tmp, tmp2), 0.5
    )

    # 5. Input NaNs should result in NaN
    tmp = ds.copy(deep=True) + 1
    tmp['total_precipitation_1hr'][{'time': 0}] = np.nan
    self.assertTrue(
        np.isnan(compute_precipitation_metric(metrics, 'far', ds, tmp))
    )

  def test_csi(self):
    ds = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        variables_2d=['total_precipitation_1hr'],
        variables_3d=[],
    )
    metrics = {'csi': categorical.CSI()}

    # 1. Only True Negatives, should be NaN
    self.assertTrue(
        np.isnan(compute_precipitation_metric(metrics, 'csi', ds, ds))
    )

    # 2. Only True Positives, should be 1
    tmp = ds.copy(deep=True) + 1
    self.assertEqual(compute_precipitation_metric(metrics, 'csi', tmp, tmp), 1)

    # 3. No True Positives, should be 0
    self.assertEqual(compute_precipitation_metric(metrics, 'csi', tmp, ds), 0)

    # 4. Half True Positives, should be 0.5
    tmp2 = ds.copy(deep=True)
    # Time dimension is size 2 in position 1.
    tmp2['total_precipitation_1hr'][{'time': 0}] = 1
    self.assertEqual(
        compute_precipitation_metric(metrics, 'csi', tmp, tmp2), 0.5
    )

    # 5. Input NaNs should result in NaN
    tmp = ds.copy(deep=True) + 1
    tmp['total_precipitation_1hr'][{'time': 0}] = np.nan
    self.assertTrue(
        np.isnan(compute_precipitation_metric(metrics, 'csi', ds, tmp))
    )

  def test_fss(self):
    prediction = xr.DataArray(
        [1, 0, 1, 0, 0, 1], dims=['longitude'], name='precipitation'
    )
    target = xr.DataArray(
        [1, 0, 0, 1, 0, 1], dims=['longitude'], name='precipitation'
    )
    prediction = prediction.expand_dims(latitude=3).to_dataset()
    target = target.expand_dims(latitude=3).to_dataset()
    metrics = {
        'fss_no_wrap': spatial.FSS(
            neighborhood_size_in_pixels=[1, 3], wrap_longitude=False
        ),
        'fss_wrap': spatial.FSS(
            neighborhood_size_in_pixels=[1, 3], wrap_longitude=True
        ),
    }
    stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, prediction, target
    )
    stats = xarray_tree.map_structure(
        lambda x: x.mean(['latitude', 'longitude']), stats
    )
    fss_no_wrap = metrics_base.compute_metric_from_statistics(
        metrics['fss_no_wrap'], stats
    )['precipitation']
    fss_wrap = metrics_base.compute_metric_from_statistics(
        metrics['fss_wrap'], stats
    )['precipitation']

    # For n=1, both should be similar = 4/6 correct pixels
    np.testing.assert_allclose(
        fss_no_wrap.sel(neighborhood_size=1).values, 4 / 6
    )
    np.testing.assert_allclose(fss_wrap.sel(neighborhood_size=1).values, 4 / 6)

    # For n=3, wrap version should be higher.
    self.assertGreater(
        fss_wrap.sel(neighborhood_size=3).values,
        fss_no_wrap.sel(neighborhood_size=3).values,
    )

    # Also test NaN handling of neighborhood averaging.
    # Reason for this is that we originally used ndimage.uniform_filter, which
    # does not handle NaNs correctly.
    x = np.ones((5, 5))
    x[0, 0] = np.nan
    neighborhood_size = 3
    out = spatial.convolve2d_wrap_longitude(x, neighborhood_size)
    correct_result = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, np.nan, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    np.testing.assert_allclose(out, correct_result)

  def test_wrapped_metric(self):
    target = (
        test_utils.mock_prediction_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-20T00',
            variables_2d=['total_precipitation_1hr'],
            variables_3d=[],
        )
        + 100
    )
    prediction = (
        test_utils.mock_prediction_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-03T00',
            ensemble_size=10,
            variables_2d=['total_precipitation_1hr'],
            variables_3d=[],
        )
        + 10
    )
    prediction = prediction.copy(deep=True)
    # Value = 100 for half the ensemble members and half the domain,
    # otherwise value = 1
    prediction['total_precipitation_1hr'][
        {'realization': slice(0, 5), 'longitude': slice(0, 18)}
    ] = 100

    metrics = {
        'csi': wrappers.WrappedMetric(
            categorical.CSI(),
            [
                wrappers.ContinuousToBinary(
                    which='both',
                    threshold_value=[0, 50],
                    threshold_dim='threshold_value',
                ),
                wrappers.EnsembleMean(
                    which='predictions', ensemble_dim='realization'
                ),
                wrappers.ContinuousToBinary(
                    which='predictions',
                    threshold_value=[0.25, 0.75],
                    threshold_dim='threshold_probability',
                ),
            ],
        )
    }
    metric_values = compute_precipitation_metric(
        metrics, 'csi', prediction, target
    )
    # For threshold value of 0, there should be only True Positives.
    self.assertTrue((metric_values.sel(threshold_value=0) == 1).all())
    # For a threshold value of 50 and a probability of 25%, half the domain
    # should be True Positives.
    self.assertEqual(
        metric_values.sel(threshold_value=50, threshold_probability=0.25), 0.5
    )
    # For a threshold value of 50 and a probability of 75%, there should be no
    # True Positives.
    self.assertEqual(
        metric_values.sel(threshold_value=50, threshold_probability=0.75), 0
    )

  def test_variable_subselection_wrapper(self):
    target = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-20T00',
        variables_2d=['2m_temperature', '10m_wind_speed'],
        variables_3d=['geopotential'],
    )
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        ensemble_size=10,
        variables_2d=['2m_temperature', '10m_wind_speed'],
        variables_3d=['geopotential'],
    )
    metrics = {
        'rmse': wrappers.SubselectVariables(
            deterministic.RMSE(), ['2m_temperature', 'geopotential']
        ),
        'mae': wrappers.SubselectVariables(
            deterministic.MAE(), ['10m_wind_speed']
        ),
    }
    results = compute_all_metrics(
        metrics, prediction, target, reduce_dims=['latitude', 'longitude']
    )
    self.assertSetEqual(
        set(results),
        {'mae.10m_wind_speed', 'rmse.2m_temperature', 'rmse.geopotential'},
    )

  def test_wind_vector_rmse(self):
    target = (
        test_utils.mock_prediction_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-20T00',
            variables_2d=['10m_u_component_of_wind', '10m_v_component_of_wind'],
            variables_3d=['u_component_of_wind', 'v_component_of_wind'],
        )
        + 1
    )
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        variables_2d=['10m_u_component_of_wind', '10m_v_component_of_wind'],
        variables_3d=['u_component_of_wind', 'v_component_of_wind'],
    )
    metrics = {
        'vector_rmse': deterministic.WindVectorRMSE(
            ['u_component_of_wind', '10m_u_component_of_wind'],
            ['v_component_of_wind', '10m_v_component_of_wind'],
            ['wind', '10m_wind'],
        ),
    }
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, prediction, target
    )
    aggregator = aggregation.Aggregator(
        reduce_dims=['time', 'latitude', 'longitude'],
    )
    aggregation_state = aggregator.aggregate_statistics(statistics)
    results = aggregation_state.metric_values(metrics)
    expected_results = xr.Dataset({
        'vector_rmse.10m_wind': xr.ones_like(
            target['10m_u_component_of_wind'].isel(
                latitude=0, longitude=0, time=0, drop=True
            )
        ),
        'vector_rmse.wind': xr.ones_like(
            target['u_component_of_wind'].isel(
                latitude=0, longitude=0, time=0, drop=True
            )
        ),
    }) * np.sqrt(2)
    xr.testing.assert_allclose(results, expected_results)

  def test_seeps(self):
    target = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        variables_2d=['total_precipitation_6hr', 'total_precipitation_24hr'],
        variables_3d=[],
    ).rename(time='init_time', prediction_timedelta='lead_time')
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        variables_2d=['total_precipitation_6hr', 'total_precipitation_24hr'],
        variables_3d=[],
    ).rename(time='init_time', prediction_timedelta='lead_time')

    climatology = target.isel(init_time=0, lead_time=0, drop=True).expand_dims(
        dayofyear=366, hour=4
    )
    for variable in ['total_precipitation_6hr', 'total_precipitation_24hr']:
      climatology[f'{variable}_seeps_dry_fraction'] = (
          climatology[variable] + 0.4
      )
      climatology[f'{variable}_seeps_threshold'] = climatology[variable] + 1

    seeps = categorical.SEEPS(
        climatology=climatology,
        variables=['total_precipitation_6hr', 'total_precipitation_24hr'],
    )
    # Test that perfect forecast results in SEEPS = 0
    statistic = seeps.compute(prediction, target)
    for variable in ['total_precipitation_6hr', 'total_precipitation_24hr']:
      np.testing.assert_allclose(statistic[variable].values, 0, atol=1e-4)

    # Test that obs_cat = dry and fc_cat = light = 1/p1 = 0.5 * 1 / 0.4 = 1.25
    # This means the scoring matrix is correctly oriented
    prediction += 0.5
    statistic = seeps.compute(prediction, target)
    for variable in ['total_precipitation_6hr', 'total_precipitation_24hr']:
      np.testing.assert_allclose(statistic[variable].values, 1.25, atol=1e-4)

    # Also test case where different parameters are used.
    seeps = categorical.SEEPS(
        climatology=climatology,
        variables=['total_precipitation_6hr', 'total_precipitation_24hr'],
        dry_threshold_mm=[0.25, 0.25],
        min_p1=[0.1, 0.1],
        max_p1=[0.85, 0.85],
    )
    statistic2 = seeps.compute(prediction, target)
    xr.testing.assert_allclose(
        statistic['total_precipitation_6hr'],
        statistic2['total_precipitation_6hr'],
    )
    xr.testing.assert_allclose(
        statistic['total_precipitation_24hr'],
        statistic2['total_precipitation_24hr'],
    )

  def _crps_spread_brute_force(self, ds: xr.Dataset, fair: bool) -> xr.Dataset:
    # This version is simple enough that we can use it as a reference.
    n_ensemble = ds.sizes['realization']
    return abs(ds - ds.rename({'realization': 'dummy'})).mean(
        dim=('latitude', 'longitude', 'realization', 'dummy'), skipna=False
    ) * (n_ensemble / (n_ensemble - int(fair)))

  @parameterized.named_parameters(
      dict(
          testcase_name=f'EnsembleSize{size}_{sort=}_{fair=}',
          ensemble_size=size,
          use_sort=sort,
          fair=fair,
      )
      for size, sort, fair in itertools.product(
          [4, 5], [False, True], [True, False]
      )
  )
  def test_crps(self, ensemble_size: int, use_sort: bool, fair: bool):
    def _crps_brute_force(
        forecast: xr.Dataset, truth: xr.Dataset
    ) -> xr.Dataset:
      """The eFAIR version of CRPS from Zamo & Naveau over a chunk of data."""
      spread = self._crps_spread_brute_force(forecast, fair=fair)
      skill = abs(truth - forecast).mean(
          ('latitude', 'longitude', 'realization'), skipna=False
      )

      return {
          'score': skill - 0.5 * spread,  # CRPS
          'spread': spread,
          'skill': skill,
      }

    targets = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00', random=True
    )
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        random=True,
        ensemble_size=ensemble_size,
    )

    # Test equivalience to brute force results.
    metrics = {
        'crps': probabilistic.CRPSEnsemble(
            ensemble_dim='realization', use_sort=use_sort, fair=fair
        )
    }
    results = compute_all_metrics(
        metrics, predictions, targets, reduce_dims=['latitude', 'longitude']
    )
    expected_results = _crps_brute_force(predictions, targets)
    for v in ['2m_temperature', 'geopotential']:
      xr.testing.assert_allclose(
          expected_results['score'][v], results[f'crps.{v}']
      )

  @parameterized.named_parameters(
      dict(
          testcase_name=f'EnsembleSize{size}_{sort=}_{fair=}',
          ensemble_size=size,
          use_sort=sort,
          fair=fair,
      )
      for size, sort, fair in itertools.product(
          [4, 5], [False, True], [True, False]
      )
  )
  def test_crps_ensemble_distance(
      self, ensemble_size: int, use_sort: bool, fair: bool
  ):

    targets = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        random=True,
        # Ensure targets have different ensemble than forecast... to make
        # dimension errors noisy.
        ensemble_size=ensemble_size + 1,
        seed=0,
    )
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        random=True,
        ensemble_size=ensemble_size,
        seed=1,
    )

    targets_no_ensemble = targets.isel(realization=0, drop=True)
    targets_no_spread = targets_no_ensemble.expand_dims(
        realization=np.arange(targets.sizes['realization'])
    )

    # Test equivalience to brute force results.
    ensemble_dist_metrics = {
        'crps': probabilistic.CRPSEnsembleDistance(
            ensemble_dim='realization', use_sort=use_sort, fair=fair
        )
    }
    metrics = {
        'crps': probabilistic.CRPSEnsemble(
            ensemble_dim='realization', use_sort=use_sort, fair=fair
        )
    }
    reduce_dims = ['latitude', 'longitude', 'time']
    predictions_vs_targets = compute_all_metrics(
        ensemble_dist_metrics,
        predictions,
        targets,
        reduce_dims=reduce_dims,
    )
    predictions_vs_targets_no_spread = compute_all_metrics(
        ensemble_dist_metrics,
        predictions,
        targets_no_spread,
        reduce_dims=reduce_dims,
    )
    predictions_vs_targets_no_ensemble = compute_all_metrics(
        metrics,
        predictions,
        targets_no_spread,
        reduce_dims=reduce_dims,
    )

    stderr = 1 / np.sqrt(
        np.prod([ensemble_size * targets.sizes[d] for d in reduce_dims])
    )

    # Since predictions and targets both come from Normal(0, I), the values of
    # the distance should be close to zero.
    if fair:
      for v in ['2m_temperature', 'geopotential']:
        np.testing.assert_allclose(
            predictions_vs_targets[f'crps.{v}'],
            0,
            atol=5 * stderr,
        )

    # The targets with no spread (due to all realizations the same) should give
    # the same CRPS value as standard CRPS.
    for v in ['2m_temperature', 'geopotential']:
      xr.testing.assert_allclose(
          predictions_vs_targets_no_spread[f'crps.{v}'],
          predictions_vs_targets_no_ensemble[f'crps.{v}'],
          atol=5 * stderr,
          check_dim_order=False,
      )

  def test_direct_rps(self):
    # CDFs
    predictions = xr.DataArray(
        [0.0, 0.0, 1.0],
        coords={'sample': np.arange(3)},
    )
    targets = xr.DataArray(
        [0.0, 1.0, 1.0],
        coords={'sample': np.arange(3)},
    )
    rps = categorical.RankedProbabilityScore(bin_dim='sample')
    result = rps.compute({'x': predictions}, {'x': targets})['x']
    self.assertEqual(result.values, 1.0)

  @parameterized.named_parameters(
      dict(
          testcase_name=f'fair_{fair}_targ_temp_{targ_temp}',
          fair=fair,
          targ_temp=targ_temp,
          expected_rps=expected_rps,
      )
      # Expected values for the RPS metric were computed by hand on the
      # dataset provided by test_utils.
      for fair, targ_temp, expected_rps in [
          (False, 0.1, 0.76),
          (False, 0.2, 0.76),
          (False, 0.7, 1.36),
          (False, 0.9, 1.96),
          (True, 0.1, 0.60),
          (True, 0.2, 0.60),
          (True, 0.7, 1.20),
          (True, 0.9, 1.80),
      ]
  )
  def test_ensemble_rps_on_handwritten_small_data(
      self,
      expected_rps: float,
      targ_temp: float,
      fair: bool,
  ):
    """Tests that RPS calculation is correct on a small dataset."""

    # Create dummy data for variables
    pred_temp = [0.1, 0.3, 0.3, 0.4, 0.9]
    num_samples = len(pred_temp)

    # Create predictions and targets datasets.
    pred = xr.Dataset(
        {'temperature': (('sample',), pred_temp)},
        coords={'sample': np.arange(num_samples)},
    )
    targ = xr.Dataset({'temperature': ((), targ_temp)})

    bin_thresholds_np = np.linspace(0.2, 0.8, 4)
    bin_thresholds_ds = xr.Dataset(
        data_vars={var: (['bin'], bin_thresholds_np) for var in targ.data_vars},
        coords={'bin': np.arange(len(bin_thresholds_np))},
    )
    statistic = probabilistic.EnsembleRankedProbabilityScore(
        prediction_bin_thresholds=bin_thresholds_ds,
        target_bin_thresholds=bin_thresholds_ds,
        unique_name_suffix='test',
        bin_dim='bin',
        ensemble_dim='sample',
        fair=fair,
    ).compute(pred, targ)

    # Check calculation matches the expected RPS, which has been determined by
    # writing out the cacluation manually.
    np.testing.assert_allclose(statistic['temperature'].values, expected_rps)

  def test_wasserstein_distance_simple(self):
    predictions_ds = xr.Dataset({'var1': ('realization', np.array([0.0, 1.0]))})
    targets_ds = xr.Dataset({'var1': ('realization', np.array([1.0, 2.0]))})
    statistic = probabilistic.WassersteinDistance(ensemble_dim='realization')
    results = xr.Dataset(statistic.compute(predictions_ds, targets_ds))
    expected = xr.Dataset({'var1': 1.0})
    xr.testing.assert_allclose(results, expected)

  def test_wasserstein_distance_different_ensemble_sizes(self):
    predictions_ds = xr.Dataset({'var1': (('realization',), np.array([2, 2]))})
    targets_ds = xr.Dataset({'var1': (('realization',), np.array([1, 1, 1]))})
    statistic = probabilistic.WassersteinDistance(ensemble_dim='realization')
    results = xr.Dataset(statistic.compute(predictions_ds, targets_ds))
    expected = xr.Dataset({'var1': 1.0})
    xr.testing.assert_allclose(results, expected)

  def test_wasserstein_distance_missing_ensemble_dim(self):
    predictions_ds = xr.Dataset({'var1': ('realization', np.array([0.0, 1.0]))})
    targets_ds = xr.Dataset({'var1': ('realization', np.array([1.0, 2.0]))})
    statistic = probabilistic.WassersteinDistance(ensemble_dim='realization')

    predictions_no_ens = predictions_ds.isel(realization=0)
    with self.assertRaisesRegex(
        ValueError, "Ensemble dimension 'realization' not found in predictions"
    ):
      statistic.compute(predictions_no_ens, targets_ds)

    targets_no_ens = targets_ds.isel(realization=0)
    with self.assertRaisesRegex(
        ValueError, "Ensemble dimension 'realization' not found in targets"
    ):
      statistic.compute(predictions_ds, targets_no_ens)

  def test_spread_skill_ratio(self):
    targets = test_utils.mock_target_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        variables_3d=[],
        random=True,
    )
    # Predictions centered at 0, which should result in an error of zero.
    predictions = test_utils.mock_target_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        variables_3d=[],
        ensemble_size=5,
        random=True,
    )

    metrics = {
        'unbiased_spread_skill': probabilistic.UnbiasedSpreadSkillRatio(
            ensemble_dim='realization'
        ),
        'spread_skill': probabilistic.SpreadSkillRatio(
            ensemble_dim='realization'
        ),
    }
    results = compute_all_metrics(
        metrics,
        predictions,
        targets,
        reduce_dims=['time', 'latitude', 'longitude'],
    )
    # Expected error: 1 / sqrt(sample size) + 1 / ensemble size
    atol = 4 * (
        1 / np.sqrt(np.prod(list(targets.sizes.values())))
        + 1 / predictions.realization.size
    )
    xr.testing.assert_allclose(results, xr.ones_like(results), atol=atol)

  def test_acc(self):
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
    ).rename(time='init_time', prediction_timedelta='lead_time')
    target = prediction.copy()
    climatology = (
        target.isel(init_time=0, lead_time=0, drop=True).expand_dims(
            dayofyear=366, hour=4
        )
        - 1
    )  # -1 because otherwise the anomalies will all be 0
    metrics = {
        'acc': deterministic.ACC(climatology=climatology),
    }
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, prediction, target
    )
    aggregator = aggregation.Aggregator(
        reduce_dims=['latitude', 'longitude'],
    )
    aggregation_state = aggregator.aggregate_statistics(statistics)
    results = aggregation_state.metric_values(metrics)
    xr.testing.assert_allclose(results, xr.ones_like(results))

  def test_prediction_passthrough(self):
    predictions = xr.DataArray(
        np.array([[1.0, 2.0], [np.nan, 4.0]]), dims=['x', 'y']
    )
    targets = xr.DataArray(
        np.array([[5.0, np.nan], [7.0, 8.0]]), dims=['x', 'y']
    )
    result = deterministic.PredictionPassthrough(
        copy_nans_from_targets=False
    )._compute_per_variable(predictions, targets)
    expected_result = xr.DataArray(
        np.array([[1.0, 2.0], [np.nan, 4.0]]), dims=['x', 'y']
    )
    xr.testing.assert_allclose(result, expected_result)

    result = deterministic.PredictionPassthrough(
        copy_nans_from_targets=True
    )._compute_per_variable(predictions, targets)
    expected_result = xr.DataArray(
        np.array([[1.0, np.nan], [np.nan, 4.0]]), dims=['x', 'y']
    )
    xr.testing.assert_allclose(result, expected_result)

  def test_error_exceedance(self):
    predictions = xr.DataArray(np.array([0, -1, 1, np.nan]), dims=['x'])
    targets = xr.DataArray(np.array([0, 0, 0, 0]), dims=['x'])
    result = deterministic.ErrorExceedance(
        thresholds=xr.DataArray([0, 0.5, 1, np.nan], dims=['y'])
    )._compute_per_variable(predictions, targets)
    expected_result = xr.DataArray(
        np.array([
            [0, 0, 0, np.nan],
            [1, 1, 0, np.nan],
            [1, 1, 0, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
        ]),
        dims=['x', 'y'],
    )
    xr.testing.assert_allclose(result, expected_result)

  def _get_opportunism_test_data(self):
    preds = test_utils.mock_prediction_data(
        ensemble_size=10,
        time_start='2020-01-01T00',
        time_stop='2020-01-01T00',
        variables_2d=['2m_temperature'],
        variables_3d=[],
    ).rename(time='init_time', prediction_timedelta='lead_time')

    # Climatology: 2m_temperature quantiles 0.1, 0.5, 0.9 are 0, 1, 2.
    # Spread q0.9-q0.1 = 2.
    def make_clim_var(data):
      return xr.full_like(
          preds['2m_temperature'].isel(
              init_time=0, lead_time=0, realization=0, drop=True
          ),
          fill_value=data,
      )

    clim_da = xr.concat(
        [make_clim_var(0), make_clim_var(1), make_clim_var(2)], dim='quantile'
    ).assign_coords(quantile=[0.1, 0.5, 0.9])
    clim_da = clim_da.expand_dims(dayofyear=list(range(1, 12)), hour=[0])
    clim = xr.Dataset({'2m_temperature': clim_da})

    # Predictions: 5 members at 0.9, 5 members at 1.1.
    # q0.1=0.9, mean=1, q0.9=1.1. Spread=0.2.
    preds['2m_temperature'] = xr.concat(
        [
            xr.full_like(
                preds['2m_temperature'].isel(realization=slice(0, 5)), 0.9
            ),
            xr.full_like(
                preds['2m_temperature'].isel(realization=slice(5, 10)), 1.1
            ),
        ],
        dim='realization',
    )
    targs = preds.mean('realization')
    return preds, targs, clim

  def test_confident(self):
    preds, targs, clim = self._get_opportunism_test_data()
    confident_stat = categorical.Confident(
        ensemble_dim='realization',
        climatology=clim,
        confidence_threshold=0.7,
    )
    res = confident_stat.compute(preds, targs)['2m_temperature']
    self.assertTrue(res.all())

    not_confident_stat = categorical.Confident(
        ensemble_dim='realization',
        climatology=clim,
        confidence_threshold=0.01,
    )
    res = not_confident_stat.compute(preds, targs)['2m_temperature']
    self.assertFalse(res.all())

  def test_covered(self):
    preds, targs, _ = self._get_opportunism_test_data()
    covered_stat = categorical.Covered(ensemble_dim='realization')
    res = covered_stat.compute(preds, targs)['2m_temperature']
    self.assertTrue(res.all())

    targs['2m_temperature'] = 0.0
    res = covered_stat.compute(preds, targs)['2m_temperature']
    self.assertFalse(res.all())

  def test_jaccard_distant(self):
    preds, targs, clim = self._get_opportunism_test_data()
    jd_stat = categorical.JaccardDistant(
        ensemble_dim='realization',
        climatology=clim,
        threshold=0.75,
    )
    res = jd_stat.compute(preds, targs)['2m_temperature']
    self.assertTrue(res.all())
    not_jd_stat = categorical.JaccardDistant(
        ensemble_dim='realization',
        climatology=clim,
        threshold=0.95,
    )
    res = not_jd_stat.compute(preds, targs)['2m_temperature']
    self.assertFalse(res.all())

  @parameterized.named_parameters(
      dict(
          testcase_name='all_true',
          is_confident=True,
          is_covered=True,
          is_jaccard_distant=True,
          expected_result=1.0,
      ),
      dict(
          testcase_name='jaccard_false',
          is_confident=True,
          is_covered=True,
          is_jaccard_distant=False,
          expected_result=0.0,
      ),
      dict(
          testcase_name='covered_false',
          is_confident=True,
          is_covered=False,
          is_jaccard_distant=True,
          expected_result=0.0,
      ),
      dict(
          testcase_name='confident_false',
          is_confident=False,
          is_covered=True,
          is_jaccard_distant=True,
          expected_result=0.0,
      ),
  )
  def test_opportunism(
      self, is_confident, is_covered, is_jaccard_distant, expected_result
  ):
    preds, targs, clim = self._get_opportunism_test_data()

    # For default parameters spread_quantile_boundaries=(0.1, 0.9),
    # confidence_threshold=0.7:
    # pred_spread = 0.2. clim_spread = 2.
    # 0.2 < 0.7 * 2 = 1.4 -> Confident=True.

    # For Covered statistic with default quantiles=(0.1, 0.9):
    # pred_q0.1=0.9, pred_q0.9=1.1. target=1.
    # 0.9 <= 1 <= 1.1 -> Covered=True.

    # For JaccardDistant statistic with threshold=0.75:
    # jaccard distance = 0.9.
    # 0.9 > 0.75 -> JaccardDistant=True.

    metrics = {
        'opp2': categorical.Opportunism(
            ensemble_dim='realization',
            climatology=clim,
            is_confident=is_confident,
            is_covered=is_covered,
            is_jaccard_distant=is_jaccard_distant,
        )
    }
    results = compute_all_metrics(
        metrics,
        preds,
        targs,
        reduce_dims=['init_time', 'lead_time', 'latitude', 'longitude'],
    )
    self.assertEqual(results['opp2.2m_temperature'], expected_result)

  @parameterized.named_parameters(
      dict(
          testcase_name=f'EnsembleSize{size}_{sort=}_{fair=}',
          ensemble_size=size,
          use_sort=sort,
          fair=fair,
      )
      for size, sort, fair in itertools.product(
          [4, 5], [False], [True, False]
      )
  )
  def test_crps_with_nans(
      self, ensemble_size: int, use_sort: bool, fair: bool
  ):
    targets = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00', random=True
    )
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        random=True,
        ensemble_size=ensemble_size,
    )

    # Introduce a NaN value into the predictions
    predictions_with_nan = predictions.copy(deep=True)
    predictions_with_nan['2m_temperature'][{'realization': 0}] = np.nan

    # Compute metrics with skipna_ensemble=True
    metrics = {
        'crps': probabilistic.CRPSEnsemble(
            ensemble_dim='realization',
            use_sort=use_sort,
            fair=fair,
            skipna_ensemble=True,
        )
    }
    results = compute_all_metrics(
        metrics,
        predictions_with_nan,
        targets,
        reduce_dims=['latitude', 'longitude'],
    )

    # Compute metrics on the non-NaN data with skipna_ensemble=False
    metrics_no_nan = {
        'crps': probabilistic.CRPSEnsemble(
            ensemble_dim='realization',
            use_sort=use_sort,
            fair=fair,
            skipna_ensemble=False,
        )
    }
    expected_results = compute_all_metrics(
        metrics_no_nan,
        predictions.isel(realization=slice(1, None)),
        targets,
        reduce_dims=['latitude', 'longitude'],
    )

    for v in ['2m_temperature', 'geopotential']:
      if v == '2m_temperature':
        xr.testing.assert_allclose(
            results[f'crps.{v}'], expected_results[f'crps.{v}']
        )
      else:  # Other variables should not be affected by the NaN
        # Recompute expected results without a nan
        expected_results_no_nan = compute_all_metrics(
            metrics_no_nan,
            predictions,
            targets,
            reduce_dims=['latitude', 'longitude'],
        )
        xr.testing.assert_allclose(
            results[f'crps.{v}'], expected_results_no_nan[f'crps.{v}']
        )

  def test_ensemble_averaged_metric(self):
    targets = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00', random=True
    )
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        random=True,
        ensemble_size=5,
    )

    # With explicit averaging over the ensemble
    metrics = {
        'rmse': deterministic.RMSE()
    }
    results_expected = compute_all_metrics(
        metrics, predictions, targets,
        reduce_dims=['latitude', 'longitude', 'realization']
    )

    # With EnsembleAveragedMetric.
    metrics_with_ensemble_averaged_metric = {
        'rmse': probabilistic.EnsembleAveragedMetric(
            deterministic.RMSE(),
            ensemble_dim='realization',
        )
    }
    results_actual = compute_all_metrics(
        metrics_with_ensemble_averaged_metric, predictions, targets,
        reduce_dims=['latitude', 'longitude']
    )

    xr.testing.assert_allclose(results_actual, results_expected)


if __name__ == '__main__':
  absltest.main()
