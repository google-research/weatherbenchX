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
"""Base class for statistical inference methods."""

import abc
from collections.abc import Hashable, Mapping
from typing import final

from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base
from weatherbenchX.statistical_inference import baseline_comparison
import xarray as xr


# Metric name -> variable name -> DataArray
MetricValues = Mapping[str, Mapping[Hashable, xr.DataArray]]


class StatisticalInferenceMethod(abc.ABC):
  """A statistical inference method.

  Can compute confidence intervals, p-values and other related quantities
  for underlying / population values of Metrics.

  Can also be used to compute confidence intervals for differences of Metrics
  between two models, and p-values for paired-differences significance tests,
  via the `for_baseline_comparison` classmethod.
  """

  @abc.abstractmethod
  def __init__(
      self,
      metrics: Mapping[str, base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      ):
    """Initializer.

    Subclasses should support the arguments here, and may support extra ones
    too.

    Args:
      metrics: The metrics for which you want to perform statistical
        inference (compute confidence intervals etc) using this inference
        method.
      aggregated_statistics: The aggregated statistics to use to compute the
        metric values. These should already be reduced over any dimensions that
        (a) you don't care to preserve in the results, and (b) you wish to
        treat as fixed under any hypothetical resampling of the data, i.e.
        you don't care about generalization to values with other, differently-
        sampled coordinates along these dimensions. An example might be reducing
        over lat/lon dimensions for metrics where you only care about their
        value aggregated over the grid cells of a specific fixed grid. In that
        case you would supply aggregated_statistics that have already been
        reduced over lat/lon dimensions (with appropriate area weighting).
        To perform statistical inference, there should be some dimension(s)
        retained in the aggregated_statistics which you wish to treat as a
        random sample, and estimate confidence intervals etc for the true
        underlying value of the metric under the full population or data
        distribution from which that sample was drawn. Typically this would be a
        dimension corresponding to initialization time. Specific implementations
        of this interface will require you to specify what these dimension(s)
        are and what properties hold for them (e.g. can data points be assumed
        independent, are they temporally dependent etc).
    """

  @classmethod
  def for_baseline_comparison(
      cls,
      metrics: Mapping[str, base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      baseline_aggregated_statistics: aggregation.AggregationState,
      baseline_metrics: Mapping[str, base.Metric] | None = None,
      comparison: baseline_comparison.Comparison = baseline_comparison.difference,
      **init_kwargs
  ):
    """StatisticalInferenceMethod for a baseline comparison / paired test.

    Args:
      metrics: The Metrics whose values we are comparing to a baseline.
      aggregated_statistics: The aggregated statistics to use to compute the
        metric values for the main model.
      baseline_aggregated_statistics: The aggregated statistics to use to
        compute the metric values for the baseline model. These should align
        with `aggregated_statistics` along any dimensions that are treated as
        part of a random sample by the inference method, so that the paired
        test makes sense.
      baseline_metrics: The Metrics used to compute equivalent values for the
        baseline, under the same keys as `metrics`. If None, we assume the exact
        same Metrics are used in both cases. In general it can be different,
        e.g. if the same metric needs to be computed differently for the main vs
        baseline models, although of course if you want a fair apples-to-apples
        comparison then you will need to be careful about this.
      comparison: A function that takes the results of the two metrics and
        returns some comparison of them, defaulting to a difference.
      **init_kwargs: Passed on to __init__.
    """
    return cls(
        metrics=baseline_comparison.for_metrics(
            metrics, baseline_metrics, comparison),
        aggregated_statistics=baseline_comparison.combine_aggregation_states(
            aggregated_statistics, baseline_aggregated_statistics),
        **init_kwargs
    )

  @abc.abstractmethod
  def point_estimates(self) -> MetricValues:
    """Point estimates for the values (or differences in values) of metrics."""

  @abc.abstractmethod
  def confidence_intervals(
      self,
      alpha: float = 0.05,  # TODO(matthjw): support a list of values for alpha.
  ) -> tuple[MetricValues, MetricValues]:
    """Confidence intervals for values (or differences in values) of metrics.

    If we repeatedly resample the dataset from the same underlying distribution
    and compute a confidence interval for each resampled dataset, we would
    ideally expect that these intervals contain the true underlying value with
    limiting frequency 1-alpha.

    Args:
      alpha: Significance level, probability that the interval does not contain
        the true underlying value.

    Returns:
      A tuple of MetricValues containing the lower and upper bounds of the
      confidence intervals.
    """

  @abc.abstractmethod
  def standard_error_estimates(self) -> MetricValues:
    """Estimates of standard errors for our estimator of the true metric values.

    This is an estimate of the standard deviation of the estimator returned by
    `point_estimates`, under repeated resamplings of the data.

    It will depend on the sample size used to estimate the metric. The standard
    error is often used in computing confidence intervals and p-values and
    can sometimes be useful to report directly.
    """

  @abc.abstractmethod
  def p_values(self, null_value: float = 0.) -> MetricValues:
    """p-value for a two-sided test with the given null hypothesis value.

    This is for:
    H_0: true metric == null_value
    H_1: true metric != null_value

    Args:
      null_value: Value under the null hypothesis. For a difference in metrics
        between two models, this would typically be zero.

    Returns:
      MetricsValues containing p-values.
    """

  @final
  def significance_tests(
      self,
      null_value: float = 0,
      alpha: float = 0.05,  # TODO(matthjw): support a list of values for alpha.
      ) -> MetricValues:
    """Significance test for a given null hypothesis and confidence level.

    This is for:
    H_0: true metric == null_value
    H_1: true metric != null_value

    Prefer to report p-values where you have space to report more than a
    binary significance statement.

    Args:
      null_value: Value under the null hypothesis. For a difference in metrics
        between two models, this would typically be zero.
      alpha: Significance level.

    Returns:
      MetricValues containing True for a 'significant' finding (we reject the
        null hypothesis at the given significance level) or False when there is
        insufficient evidence to reject the null hypothesis at the given
        significance level.
    """
    p_values = self.p_values(null_value)
    return xarray_tree.map_structure(
        lambda p_value: p_value <= alpha, p_values)
