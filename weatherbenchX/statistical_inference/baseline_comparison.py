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
"""Facilitates baseline comparisons and paired significance testing.

('is this model significantly better/worse than the baseline?')
"""

from typing import Callable, Hashable, Mapping

from weatherbenchX import aggregation
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import wrappers
import xarray as xr


MetricResult = Mapping[Hashable, xr.DataArray]
Comparison = Callable[[MetricResult, MetricResult], MetricResult]


def difference(
    main_result: MetricResult,
    baseline_result: MetricResult,
) -> MetricResult:
  return {
      k: main_result[k] - baseline_result[k]
      for k in main_result.keys() & baseline_result.keys()
  }


class BaselineComparison(metrics_base.Metric):
  """Metric which compares values of an underlying metric to a baseline.

  Typically by taking the difference in metric values for the main model vs the
  baseline model on the same set of forecast initializations.

  This is a slightly unusual Metric in that it is only constructed post-hoc
  after the statistics of the underlying metrics have already been computed
  and aggregated.

  It should not be used directly to evaluate a single set of forecasts -- or at
  least in that case it will just compute the difference of the two metrics on
  the same forecasts, which may not be very interesting.

  The main application is to pass it to a StatisticalInferenceMethod
  together with an AggregationState computed using `combine_aggregation_states`
  below, from the separate AggregationStates computed for the main and baseline
  models. This then allows the StatisticalInferenceMethod to compute confidence
  intervals, p-values etc for the difference (or other comparison) in metric
  values between the two models, without having to include specific
  baseline-comparison logic in every inference method.
  """

  def __init__(
      self,
      metric: metrics_base.Metric,
      baseline_metric: metrics_base.Metric | None = None,
      comparison: Comparison = difference,
      ):
    """Initializer.

    Args:
      metric: The Metric whose values we are comparing to a baseline.
      baseline_metric: The Metric used to compute equivalent values for the
        baseline. If None, we assume the exact same Metric is used in both
        cases. In general it can be different, e.g. if the same metric needs to
        be computed differently for the main vs baseline models, although of
        course if you want a fair apples-to-apples comparison then you will need
        to be careful about this.
      comparison: A function that takes the results of the two metrics and
        returns some comparison of them. This defaults to the difference between
        the two (main - baseline) on a per-variable basis, but in general you
        could use other things like a ratio too.
        Note if you want to use this with statistical inference methods like the
        t-test that only work for means, you will need to limit yourself to a
        linear function here though, and the difference is the main useful
        example of this. It works because a difference of means is a mean of
        differences.
    """
    self.metric = metric
    self.baseline_metric = baseline_metric or metric
    self._comparison = comparison

  @property
  def statistics(self) -> Mapping[str, metrics_base.Statistic]:
    main_stats = {
        # As a reminder, the unique_name is a globally-unique (often lengthy)
        # name used as a key to identify a Statistic when we combine Statistics
        # required by many Metrics in an AggregationState. Whereas `name` is
        # just the short name used locally within a Metric implementation to
        # refer to the statistic. We need to prefix both of them here to avoid
        # clashes.
        f'main_{name}': wrappers.RenamedStatistic(
            stat, f'main_{stat.unique_name}')
        for name, stat in self.metric.statistics.items()
    }
    baseline_stats = {
        f'baseline_{name}': wrappers.RenamedStatistic(
            stat, f'baseline_{stat.unique_name}')
        for name, stat in self.baseline_metric.statistics.items()
    }
    return {**main_stats, **baseline_stats}

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    main_stat_values = {
        name[len('main_'):]: stat_values
        for name, stat_values in statistic_values.items()
        if name.startswith('main_')
    }
    baseline_stat_values = {
        name[len('baseline_'):]: stat_values
        for name, stat_values in statistic_values.items()
        if name.startswith('baseline_')
    }
    main_result = self.metric.values_from_mean_statistics(main_stat_values)
    baseline_result = self.baseline_metric.values_from_mean_statistics(
        baseline_stat_values)
    return self._comparison(main_result, baseline_result)


# Alias for an AggregationState where statistics names have been prefixed with
# 'main_' or 'baseline_' to indicate whether they correspond to statistics from
# an evaluation of the main or the baseline model.
BaselineComparisonAggregationState = aggregation.AggregationState


def combine_aggregation_states(
    aggregation_state: aggregation.AggregationState,
    baseline_aggregation_state: aggregation.AggregationState,
) -> BaselineComparisonAggregationState:
  """Combines main and baseline AggregationStates for a BaselineComparison."""
  main_sum_w = {
      f'main_{k}': v
      for k, v in aggregation_state.sum_weights.items()
  }
  main_sum_ws = {
      f'main_{k}': v
      for k, v in aggregation_state.sum_weighted_statistics.items()
  }
  baseline_sum_w = {
      f'baseline_{k}': v
      for k, v in baseline_aggregation_state.sum_weights.items()
  }
  baseline_sum_ws = {
      f'baseline_{k}': v
      for k, v in baseline_aggregation_state.sum_weighted_statistics.items()
  }
  return aggregation.AggregationState(
      sum_weights={**main_sum_w, **baseline_sum_w},
      sum_weighted_statistics={**main_sum_ws, **baseline_sum_ws},
  )


def for_metrics(
    metrics: Mapping[str, metrics_base.Metric],
    baseline_metrics: Mapping[str, metrics_base.Metric] | None = None,
    comparison: Comparison = difference,
) -> Mapping[str, BaselineComparison]:
  """Forms BaselineComparisons for all metrics present in both mappings."""
  if baseline_metrics is None:
    baseline_metrics = metrics
  return {
      name: BaselineComparison(
          metrics[name], baseline_metrics[name], comparison)
      for name in metrics.keys() & baseline_metrics.keys()
  }
