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
"""The t-test and associated confidence intervals for evaluation metrics."""

from collections.abc import Mapping
import dataclasses
from typing import cast

import numpy as np
import scipy
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import base
import xarray as xr


def _check_constant(data_array: xr.DataArray, dim: str, error_suffix: str = ''):
  try:
    xr.testing.assert_allclose(
        *xr.broadcast(
            data_array.isel({dim: 0}),
            data_array,
        ),
    )
  except AssertionError as e:
    raise ValueError(
        f'Found non-constant values along dimension {dim} for '
        f'{data_array.name}. {error_suffix}'
    ) from e


def _check_uniform_step(
    data_array: xr.DataArray, dim: str) -> None:
  """Checks that any name coordinate for `dim` has uniform steps."""
  coord = data_array.coords.get(dim)
  if coord is not None:
    _check_constant(coord.diff(dim), dim, 'Non-uniform timestep not supported.')


def _autocorrelation_estimate(
    data_array: xr.DataArray,
    dim: str,
    lag: int = 1,
    skipna: bool = False) -> xr.DataArray:
  mean = data_array.mean(dim, skipna=skipna)
  variance = data_array.var(dim, skipna=skipna)

  # Drop coordinates on `dim` to allow alignment at lagged offsets:
  data_array = data_array.drop_vars(
      [name for name, coord in data_array.coords.items() if dim in coord.dims])
  original = data_array.isel({dim: slice(0, -lag)})
  lagged = data_array.isel({dim: slice(lag, None)})

  return (((original - mean) * (lagged - mean)).mean(dim, skipna=skipna)
          / variance)


def _inflation_factor_from_autocorrelation(
    rho1: xr.DataArray, rho2: xr.DataArray) -> xr.DataArray:
  """The inflation factor k from the Geer paper."""
  # This assumes an AR(2) process.
  # r1 is our estimate of autocorrelation at lag 1.
  # r2 is our estimate of autocorrelation at lag 2.
  denominator = (1 - rho1**2)
  phi1 = (rho1 * (1 - rho2)) / denominator
  phi2 = (rho2 - rho1**2) / denominator
  k_squared = (1 - rho1 * phi1 - rho2 * phi2) / (1 - phi1 - phi2)**2
  return np.sqrt(k_squared)


def _inflation_factor_from_ar2_coeffs(
    phi1: xr.DataArray, phi2: xr.DataArray) -> xr.DataArray:
  rho1 = phi1 / (1 - phi2)
  rho2 = phi2 + phi1**2 / (1 - phi2)
  k_squared = (1 - rho1 * phi1 - rho2 * phi2) / (1 - phi1 - phi2)**2
  return np.sqrt(k_squared)


@dataclasses.dataclass(frozen=True)
class _TTestResults:
  """Results of the t-test, for a single metric DataArray."""

  mean: xr.DataArray
  standard_error: xr.DataArray
  sample_size: int

  def ci_lower(self, alpha: float = 0.05) -> xr.DataArray:
    z_alpha = -scipy.stats.t(df=self.sample_size - 1).ppf(alpha / 2)
    return self.mean - self.standard_error * z_alpha

  def ci_upper(self, alpha: float = 0.05) -> xr.DataArray:
    z_alpha = -scipy.stats.t(df=self.sample_size - 1).ppf(alpha / 2)
    return self.mean + self.standard_error * z_alpha

  def p_value(self, null_value: float = 0.) -> xr.DataArray:
    """p-value for a two-sided test with the given null hypothesis value."""
    z_score = (self.mean - null_value) / self.standard_error
    t_dist = scipy.stats.t(df=self.sample_size - 1)
    return 2 * (1 - xr.apply_ufunc(t_dist.cdf, abs(z_score)))


class TTest(base.StatisticalInferenceMethod):
  r"""t-test for evaluation metrics, with optional autocorrelation adjustment.

  The t-test is used to test hypotheses about (and provide confidence intervals
  for) means / differences of means, and so this only applies to `Statistic`s,
  which represent the mean of a single statistic, and not to other more general
  `Metric`s.

  The main assumptions of the standard t-test are:

  * Gaussianity of the statistic.
    Due to the central limit theorem, the test is relatively robust to failures
    of this assumption especially for larger sample sizes. For smaller
    sample sizes and/or highly non-Gaussian statistics or variables, you may
    want to consider other tests.
  * Independence of the statistic across experimental units. This is important
    and is generally *not* true for the typical case where experimental units
    correspond to different forecast initialization times, unless they are
    sufficiently far apart in time that any temporal dependence has become
    negligible. This can be checked approximately for example by looking at
    autocorrelation plots.

  We support optionally relaxing the independence assumption by correcting for
  temporal autocorrelation between experimental units (which would typically
  correspond to different forecast initialization times). If this is specified,
  we use the temporal autocorrelation inflation factor described in by Geer [1]
  to inflate the t-test standard errors. This inflation factor is estimated
  assuming the statistic timeseries follow a stationary AR(2) process. This can
  be approximately checked by looking at PACF plots.
  Subject to this assumption the correction to the t-test is well-motivated
  asymptotically, but tends to be over-optimistic (confidence intervals too
  narrow) for smaller sample sizes and/or high levels of autocorrelation (which
  reduces the effective sample size).

  See [1] for further details and caveats that may be useful to determine
  whether this is appropriate for your data.

  [1] A. J. Geer, Significance of changes in medium-range forecast scores.
  Tellus A Dyn. Meterol. Oceanogr. 68, 30229 (2016).
  doi:10.3402/tellusa.v68.30229
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
      temporal_autocorrelation: bool,
      baseline_aggregated_statistics:
      aggregation.AggregationState | None = None,
      ):
    r"""Initializer.

    Args:
      metrics: The metrics for which you want to perform statistical
        inference (compute confidence intervals etc) using this inference
        method. Because the t-test only applies to means, these `Metrics` must
        be instances of `Statistic` (Metrics that are just the mean of a
        statistic).
      aggregated_statistics: The aggregated statistics to use to compute the
        metric values. See base.StatisticalInferenceMethod docs for more
        details.
      experimental_unit_dim: The dimension corresponding to experimental
        units sampled randomly from some underlying distribution. We reduce
        over this dimension to estimate the true/population mean of the metric
        in expectation over this underlying distribution.
      temporal_autocorrelation: Whether the statistic has temporal
        autocorrelation between experimental units along
        `experimental_unit_dim`. If so, we apply the autocorrelation inflation
        factor adjustment described in [1] to the t-test. The time coordinate
        along `experimental_unit_dim` must have a uniform timestep.
      baseline_aggregated_statistics: The aggregated statistics to use to
        compute the difference in metric values relative to a baseline.
    """
    self._dim = experimental_unit_dim
    self._temporal_autocorrelation = temporal_autocorrelation

    for metric in metrics.values():
      if not isinstance(metric, metrics_base.Statistic):
        raise TypeError(
            'The t-test only applies to means, and so this inference method '
            'can only be used with Metrics which are just the mean of a '
            'single statistic. You passed a Metric which is not a Statistic: '
            f'{type(metric)}'
        )
    metrics = cast(Mapping[str, metrics_base.Statistic], metrics)

    # We also support only a constant weighting of experimental units, so we
    # can take a mean of the per-unit means without having to apply any
    # further weighting (which the standard t-test is not set up to handle).
    xarray_tree.map_structure(
        self._check_constant_weights, aggregated_statistics.sum_weights)

    # This divides by the constant per-unit sum of weights to get the per-unit
    # means, which we can then take a non-weighted mean of (since weights are
    # constant across units).
    per_unit_means = aggregated_statistics.mean_statistics()
    if baseline_aggregated_statistics is not None:
      # If values are given for a baseline, check their weights are also
      # constant and subtract the baseline values of the per-unit means.
      xarray_tree.map_structure(
          self._check_constant_weights,
          baseline_aggregated_statistics.sum_weights)
      baseline_per_unit_means = baseline_aggregated_statistics.mean_statistics()
      per_unit_means = xarray_tree.map_structure(
          lambda x, y: x - y, per_unit_means, baseline_per_unit_means)

    self._results = xarray_tree.map_structure(
        self._compute_results, per_unit_means)
    # Rename from the Statistic.unique_name which the aggregated_statistics
    # are keyed by, to the metric names used in the supplied `metrics` mapping.
    # Because the metrics are all instances of Statistic, their value is equal
    # to the mean of the statistic and we don't need to apply
    # metric.values_from_mean_statistics here.
    self._results = {metric_name: self._results[metric.unique_name]
                     for metric_name, metric in metrics.items()}

  def _check_constant_weights(self, weights: xr.DataArray) -> None:
    _check_constant(
        weights, self._dim,
        'Must have the same sum_weights for all values along'
        'experimental_unit_dim in order to apply the t-test.')

  def _compute_results(
      self, per_unit_means: xr.DataArray) -> _TTestResults:
    """Computes t-test results for a single DataArray of statistics."""

    mean = per_unit_means.mean(dim=self._dim, skipna=False)
    sample_size = per_unit_means.sizes[self._dim]
    stddev = per_unit_means.std(dim=self._dim, skipna=False, ddof=1)
    stderr = stddev / np.sqrt(sample_size)

    if self._temporal_autocorrelation:
      _check_uniform_step(per_unit_means, self._dim)
      r1 = _autocorrelation_estimate(per_unit_means, self._dim, lag=1)
      r2 = _autocorrelation_estimate(per_unit_means, self._dim, lag=2)
      # This is 'k' from the Geer paper, which is applied as an inflation
      # factor to the standard error in the t-test.
      k = _inflation_factor_from_autocorrelation(r1, r2)
      stderr = stderr * k

      # Note: the Geer paper applies this inflation factor to the standard
      # error, which is similar in effect to using a smaller effective sample
      # size of n/k^2. Note that the degrees-of-freedom used for the variance
      # estimate and the t-distribution is still based on the original sample
      # size n however, not the smaller effective sample size. This will not
      # matter asymptotically but ideally something better would be done here
      # for smaller sample sizes. TODO(matthjw): Find out if there's something
      # better we could do here that doesn't introduce too much complexity.
      #
      # Also Geer doesn't account for the fact that we're plugging in noisy
      # estimates of the autocorrelation coefficients. Not sure there's a simple
      # way to account for this without e.g. going to the bootstrap or similar?

    return _TTestResults(
        mean=mean,
        standard_error=stderr,
        sample_size=sample_size,
    )

  def point_estimates(self):
    return xarray_tree.map_structure(lambda x: x.mean, self._results)

  def standard_error_estimates(self) -> base.MetricValues:
    return xarray_tree.map_structure(lambda x: x.standard_error, self._results)

  def confidence_intervals(
      self, alpha: float = 0.05
  ) -> tuple[base.MetricValues, base.MetricValues]:
    return (
        # TODO(matthjw): Compute (lower, upper) in a single method, once we have
        # a better alternative to using xarray_tree here.
        xarray_tree.map_structure(lambda x: x.ci_lower(alpha), self._results),
        xarray_tree.map_structure(lambda x: x.ci_upper(alpha), self._results),
    )

  def p_values(self, null_value: float = 0.) -> base.MetricValues:
    """p-value for a two-sided test with the given null hypothesis value."""
    return xarray_tree.map_structure(
        lambda x: x.p_value(null_value), self._results)
