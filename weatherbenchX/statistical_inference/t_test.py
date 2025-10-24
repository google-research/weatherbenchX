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
"""The t-test and associated confidence intervals for evaluation metrics.

Includes extensions to handle autocorrelation and non-linear
values_from_mean_statistics.
"""

from collections.abc import Mapping
import dataclasses

import numpy as np
import scipy
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import autodiff
from weatherbenchX.statistical_inference import base

import xarray as xr
import xarray.ufuncs as xu


def _check_constant(data_array: xr.DataArray, dim: str, error_suffix: str = ''):
  if data_array.dtype.kind == 'f':
    equiv = np.allclose
  else:
    equiv = lambda x, y: np.all(x == y)
  if not equiv(
      data_array.isel({dim: [0]}).data,
      data_array.data):
    raise ValueError(
        f'Found non-constant values along dimension {dim} for '
        f'{data_array.name}. {error_suffix}'
    )


def _check_uniform_step(
    data_array: xr.DataArray, dim: str) -> None:
  """Checks that any name coordinate for `dim` has uniform steps."""
  coord = data_array.coords.get(dim)
  if coord is not None:
    _check_constant(coord.diff(dim), dim, 'Non-uniform timestep not supported.')


def  _variance_estimate_from_deviations(
    deviations: xr.DataArray, dim: str, ddof: int = 1) -> xr.DataArray:
  """Computes a variance estimate, given deviations from the mean."""
  sample_size = deviations.sizes[dim]
  return (deviations**2).sum(dim, skipna=False) / (sample_size - ddof)


def _autocorrelation_estimate_from_deviations(
    deviations: xr.DataArray,
    dim: str,
    lag: int = 1
    ) -> xr.DataArray:
  """Computes an autocorrelation estimate, given deviations from the mean."""
  variance = _variance_estimate_from_deviations(deviations, dim)

  # Drop coordinates on `dim` to allow alignment at lagged offsets:
  deviations = deviations.drop_vars(
      [name for name, coord in deviations.coords.items() if dim in coord.dims])
  original = deviations.isel({dim: slice(0, -lag)})
  lagged = deviations.isel({dim: slice(lag, None)})

  return (original * lagged).mean(dim, skipna=False) / variance


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
  for) means. Here we extend it to handle non-linear functions of means too,
  using the (multivariate) Delta Method, which approximates the function with a
  1st-order Taylor series around the mean. This approximation is good if the
  function is close to linear over the range of sampling variation of the mean
  statistics, but might fail for highly nonlinear functions or smaller sample
  sizes.

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
      ):
    r"""Initializer.

    Args:
      metrics: The metrics for which you want to perform statistical
        inference (compute confidence intervals etc) using this inference
        method. Because the t-test only applies to means, these `Metrics` must
        be instances of `Statistic` (Metrics that are just the mean of a
        statistic) or linear functions of the means of multiple Statistics.
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
    """
    self._dim = experimental_unit_dim
    self._temporal_autocorrelation = temporal_autocorrelation

    # When the values_from_mean_statistics is a linear function,
    # per_unit_tangents will just be the per-unit deviations from the mean,
    # and the standard t-test applies.
    #
    # When the values_from_mean_statistics is not a linear function,
    # per_unit_tangents will be stand-ins for the per-unit deviations which take
    # into account the gradient of the function we're going to apply to the mean
    # and its effect on the final variance. This is in effect using the delta
    # method, a common approximation used in statistics for the variance of a
    # nonlinear function. See docs on
    # autodiff.per_unit_values_linearized_around_mean_statistics for more.
    (values, per_unit_tangents
     ) = autodiff.per_unit_values_linearized_around_mean_statistics(
         metrics, aggregated_statistics, experimental_unit_dim)

    self._results = xarray_tree.map_structure(
        self._compute_results, values, per_unit_tangents)

  def _compute_results(
      self,
      mean: xr.DataArray,
      per_unit_deviations: xr.DataArray) -> _TTestResults:
    """Computes t-test results for a single variable of a single metric.

    Args:
      mean: Mean value of the metric over all experimental units.
      per_unit_deviations: Per-unit deviations from the mean. Adding `mean` to
        this will give per-unit values of the metric.

    Returns:
      _TTestResults.
    """
    sample_size = per_unit_deviations.sizes[self._dim]
    variance = _variance_estimate_from_deviations(
        per_unit_deviations, self._dim, ddof=1)
    stderr = xu.sqrt(variance / sample_size)

    if self._temporal_autocorrelation:
      _check_uniform_step(per_unit_deviations, self._dim)
      r1 = _autocorrelation_estimate_from_deviations(
          per_unit_deviations, self._dim, lag=1)
      r2 = _autocorrelation_estimate_from_deviations(
          per_unit_deviations, self._dim, lag=2)
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
      # for smaller sample sizes.
      #
      # Also this method doesn't account for the fact that we're plugging in
      # noisy estimates of the autocorrelation coefficients.
      #
      # TODO(matthjw): Find out if there's something better we could do here.
      # There is a large literature on Heteroskedasticity and Autocorrelation
      # Consistent (HAC) standard error estimators for example, although these
      # have their own issues at small sample sizes and/or high degrees of
      # autocorrelation.

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
