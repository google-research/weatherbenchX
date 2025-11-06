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
"""Utils for testing metrics."""

from collections.abc import Hashable
import dataclasses
from typing import Mapping
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
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
