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
"""Bootstrap-based statistical inference methods for evaluation metrics."""

from collections.abc import Mapping
from typing import final

import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import base
from weatherbenchX.statistical_inference import utils

import xarray as xr


_REPLICATE_DIM = 'bootstrap_replicate'


class Bootstrap(base.StatisticalInferenceMethod):
  r"""Superclass for bootstrap-based statistical inference methods."""

  # Subclass constructors must set these.
  _resampled_values: base.MetricValues
  _point_estimates: base.MetricValues

  @property
  def resampled_values(self) -> base.MetricValues:
    """Resampled values of the metric, with a _REPLICATE_DIM dimension."""
    return self._resampled_values

  def point_estimates(self) -> base.MetricValues:
    return self._point_estimates

  # Note: we set skipna=True when dealing with bootstrap replicates below,
  # because in some more unusual cases (e.g. finely-binned metric values where
  # relatively little data is available to estimate values in some bins),
  # resampled values may sometimes be NaN due to the resampling process omitting
  # all of the data points used to estimate the value in a particular bin.
  # Skipping NaNs means our confidence intervals etc are implicitly conditioned
  # on there being enough data present to estimate the quantity in question.
  # So e.g. a 95% interval for the value in a particular bin aims to contain the
  # true value 95% of the time in replications of the experiment *in which
  # there happens to be at least one data point available in that bin*.

  @final
  def standard_error_estimates(self) -> base.MetricValues:
    return xarray_tree.map_structure(
        lambda x: x.std(_REPLICATE_DIM, ddof=1, skipna=True),
        self.resampled_values)

  @final
  def confidence_intervals(
      self, alpha: float = 0.05
  ) -> tuple[base.MetricValues, base.MetricValues]:
    # TODO(matthjw): implement BCa intervals.
    return (
        xarray_tree.map_structure(
            lambda x: x.quantile(alpha/2, _REPLICATE_DIM, skipna=True),
            self.resampled_values),
        xarray_tree.map_structure(
            lambda x: x.quantile(1-alpha/2, _REPLICATE_DIM, skipna=True),
            self.resampled_values),
    )

  @final
  def p_values(self, null_value: float = 0.) -> base.MetricValues:
    """p-value for a two-sided test with the given null hypothesis value."""

    # Obtained by inverting the percentile confidence interval above.
    # TODO(matthjw): replace with inverting the BCa interval when implemented.

    def p_value_numpy_1d(resampled: np.ndarray) -> float:
      resampled = resampled[~np.isnan(resampled)]  # Equiv. to skipna=True.
      data = np.sort(resampled)
      q = np.linspace(0, 1, data.shape[0])
      empirical_cdf_at_null = np.interp(null_value, data, q)
      return 2 * min(empirical_cdf_at_null, 1 - empirical_cdf_at_null)

    def p_value(resampled: xr.DataArray) -> xr.DataArray:
      return xr.apply_ufunc(
          p_value_numpy_1d,
          resampled,
          input_core_dims=[[_REPLICATE_DIM]],
          vectorize=True)

    return xarray_tree.map_structure(p_value, self.resampled_values)


class IIDBootstrap(Bootstrap):
  r"""Standard IID bootstrap method."""

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
      n_replicates: int,
      ):
    num_units = utils.get_and_check_experimental_unit_coord(
        aggregated_statistics, experimental_unit_dim).size
    # We optimize the computation of sums of resampled statistics.
    # Rather than first compute bootstrap indices, then slice out values at
    # those indices and then take their sums, we instead sample a matrix of
    # counts for how many times each data point is sampled in each bootstrap
    # replicate, then do a matrix multiply with this count matrix to get all
    # the sums at once.
    counts = np.random.multinomial(
        num_units, np.full(num_units, 1/num_units),
        size=n_replicates)
    counts = xr.DataArray(
        data=counts, dims=[_REPLICATE_DIM, experimental_unit_dim])
    resampled_stats = aggregated_statistics.dot(
        counts, dim=experimental_unit_dim)
    self._point_estimates = metrics_base.compute_metrics_from_statistics(
        metrics, aggregated_statistics.sum_along_dims(
            [experimental_unit_dim]).mean_statistics())
    self._resampled_values = metrics_base.compute_metrics_from_statistics(
        metrics, resampled_stats.mean_statistics())
