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

Includes extensions to handle autocorrelation (including a HAC estimator
and a method based on an AR(2) parametric model) as well as non-linear
functions of means via the delta method.
"""

import abc
from collections.abc import Mapping
import dataclasses
import functools
from typing import final

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
  if coord is not None and np.issubdtype(coord.dtype, np.number):
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

  result = (original * lagged).mean(dim, skipna=False) / variance
  # If variance is zero, the timeseries is constant and autocorrelation is
  # technically undefined, but we can safely treat it as zero since no
  # autocorrelation correction is required. This avoids NaN values in
  # confidence intervals and standard errors.
  return result.where(variance != 0, 0)


def _inflation_factor_from_ar2_autocorrelation(
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
  degrees_of_freedom: int

  def ci_lower(self, alpha: float = 0.05) -> xr.DataArray:
    z_alpha = -scipy.stats.t(df=self.degrees_of_freedom).ppf(alpha / 2)
    return self.mean - self.standard_error * z_alpha

  def ci_upper(self, alpha: float = 0.05) -> xr.DataArray:
    z_alpha = -scipy.stats.t(df=self.degrees_of_freedom).ppf(alpha / 2)
    return self.mean + self.standard_error * z_alpha

  def p_value(self, null_value: float = 0.) -> xr.DataArray:
    """p-value for a two-sided test with the given null hypothesis value."""
    difference = self.mean - null_value
    # If the difference is zero and the standard error is zero, then the
    # distribution is constant with the null value in the 'center' (or rather,
    # only value) of the distribution, so we set the z-score to zero giving
    # a p-value of 1.
    # If the difference is non-zero but the standard error is zero, the
    # null value is outside the support of the (constant) distribution, so we
    # let the division by zero happen and give +/-inf and a p-value of zero.
    z_score = xr.where((difference == 0) & (self.standard_error == 0),
                       0., difference / self.standard_error)
    t_dist = scipy.stats.t(df=self.degrees_of_freedom)
    return 2 * (1 - xr.apply_ufunc(t_dist.cdf, abs(z_score)))


class _Base(base.StatisticalInferenceMethod):
  r"""Base class for implementations of variants of the t-test.

  The t-test is used to test hypotheses about (and provide confidence intervals
  for) means. Here we extend it to handle non-linear functions of means too,
  using the (multivariate) Delta Method, which approximates the function with a
  1st-order Taylor series around the mean. This approximation is good if the
  function is close to linear over the range of sampling variation of the mean
  statistics, but might fail for highly nonlinear functions or smaller sample
  sizes.

  The main assumption common to all implementations of the standard t-test is
  Gaussianity of the statistics. Due to the central limit theorem, the test is
  relatively robust to failures of this assumption especially for larger sample
  sizes. For smaller sample sizes and/or highly non-Gaussian statistics or
  variables, you may want to consider other tests however.

  The standard t-test also assumes independence of the statistic across
  experimental units. This is important and is often *not* true for the
  typical case where experimental units correspond to different forecast
  initialization times, unless they are sufficiently far apart in time that any
  temporal dependence has become negligible. This can be checked approximately
  for example by looking at autocorrelation plots.

  Luckily some subclasses implement versions of the t-test which are adapted to
  be robust to autocorrelation.
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
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
    """
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
        functools.partial(self._compute_results, experimental_unit_dim),
        values,
        per_unit_tangents)

  @abc.abstractmethod
  def _compute_results(
      self,
      experimental_unit_dim: str,
      mean: xr.DataArray,
      per_unit_deviations: xr.DataArray) -> _TTestResults:
    """Computes t-test results for a single variable of a single metric.

    Args:
      experimental_unit_dim: The dimension corresponding to experimental units.
      mean: Mean value of the metric over all experimental units.
      per_unit_deviations: Deviations of the per-unit metric values from the
        mean (in the case where the metric is a simple mean over evaluation
        units), or more generally, linearized influence values (see
        `autodiff.per_unit_values_linearized_around_mean_statistics`) which
        can be treated in the same way.
        The per_unit_deviations will be zero-mean themselves.

    Returns:
      _TTestResults.
    """

  @final
  def point_estimates(self):
    return xarray_tree.map_structure(lambda x: x.mean, self._results)

  @final
  def standard_error_estimates(self) -> base.MetricValues:
    return xarray_tree.map_structure(lambda x: x.standard_error, self._results)

  @final
  def confidence_intervals(
      self, alpha: float = 0.05
  ) -> tuple[base.MetricValues, base.MetricValues]:
    return (
        # TODO(matthjw): Compute (lower, upper) in a single method, once we have
        # a better alternative to using xarray_tree here.
        xarray_tree.map_structure(lambda x: x.ci_lower(alpha), self._results),
        xarray_tree.map_structure(lambda x: x.ci_upper(alpha), self._results),
    )

  @final
  def p_values(self, null_value: float = 0.) -> base.MetricValues:
    """p-value for a two-sided test with the given null hypothesis value."""
    return xarray_tree.map_structure(
        lambda x: x.p_value(null_value), self._results)


class IID(_Base):
  """The classic t-test assuming i.i.d. evaluation units."""

  def _compute_results(
      self,
      experimental_unit_dim: str,
      mean: xr.DataArray,
      per_unit_deviations: xr.DataArray) -> _TTestResults:
    """Computes the results of the t-test for a single metric."""
    sample_size = per_unit_deviations.sizes[experimental_unit_dim]
    variance = _variance_estimate_from_deviations(
        per_unit_deviations, experimental_unit_dim, ddof=1)
    stderr = xu.sqrt(variance / sample_size)
    degrees_of_freedom = sample_size - 1
    return _TTestResults(mean, stderr, degrees_of_freedom)


class GeerAR2Corrected(_Base):
  """t-test corrected for AR(2) autocorrelation following Geer et al. 2016 [1].

  This correction inflates the standard error to account for AR(2)
  autocorrelation between experimental units, while leaving the degrees of
  freedom of the t-test unchanged. The inflation factor is estimated assuming
  the statistic timeseries follow a stationary AR(2) process. This can
  be approximately checked by looking at PACF plots.

  Subject to this assumption the correction to the t-test is well-motivated
  asymptotically, but tends to be over-optimistic (confidence intervals too
  narrow) for smaller sample sizes and/or high levels of autocorrelation (which
  reduces the effective sample size). Note that it does not adjust the degrees
  of freedom of the t-distribution to account for extra variance due to plugging
  in noisy estimates of AR(2) coefficient, and the sensitive non-linear
  dependence of the long-run variance on these coefficients. This may in part
  explain the over-optimism with finite sample sizes, but the issue is not
  simple to fix.

  Nevertheless it performs well when its AR(2) assumption holds and the
  effective sample size is not too small. See [1] for further details and
  caveats that may be useful to determine whether this is appropriate for your
  data.

  [1] A. J. Geer, Significance of changes in medium-range forecast scores.
  Tellus A Dyn. Meterol. Oceanogr. 68, 30229 (2016).
  doi:10.3402/tellusa.v68.30229
  """

  def _compute_results(
      self,
      experimental_unit_dim: str,
      mean: xr.DataArray,
      per_unit_deviations: xr.DataArray) -> _TTestResults:
    _check_uniform_step(per_unit_deviations, experimental_unit_dim)

    sample_size = per_unit_deviations.sizes[experimental_unit_dim]
    variance = _variance_estimate_from_deviations(
        per_unit_deviations, experimental_unit_dim, ddof=1)

    r1 = _autocorrelation_estimate_from_deviations(
        per_unit_deviations, experimental_unit_dim, lag=1)
    r2 = _autocorrelation_estimate_from_deviations(
        per_unit_deviations, experimental_unit_dim, lag=2)
    # This is 'k' from the Geer paper, which is applied as an inflation
    # factor to the standard error in the t-test.
    k = _inflation_factor_from_ar2_autocorrelation(r1, r2)
    stderr = xu.sqrt(variance / sample_size) * k
    # This would ideally be adjusted downwards, significantly in cases of high
    # autocorrelation, but this is not simple to do in a principled way:
    degrees_of_freedom = sample_size - 1

    return _TTestResults(mean, stderr, degrees_of_freedom)


class LazarusHACEWC(_Base):
  r"""t-test using the EWC-based HAC estimator from Lazarus et al. 2018 [1].

  This is a relatively general-purpose practical recommendation drawn by
  Lazarus et al from a review of a wide literature on Heteroscedasticity and
  Autocorrelation Consistent (HAC, a.k.a. HAR) estimators, with some empirical
  testing and optimization of settings.

  Specifically we use the EWC (equal-weighted cosine) HAC estimator with the
  settings they recommend.

  HAC estimators do not rely on parametric assumptions about the form of
  temporal dependence in the data, and are used widely to handle autocorrelation
  in econometrics and other fields. An HAC estimator also forms the basis of the
  Diebold-Mariano test [2] for comparing the accuracy of forecasting methods,
  which is very relevant to our application here. (We use a more modern
  recommendation for a HAC estimator than the one used in the original
  Diebold-Mariano paper, but `LazarusHACEWC.for_baseline_comparison` is still
  similar in spirit to Diebold-Mariano.)

  The default recommendations of [1], which we adopt here, aim to maintain a
  reasonably accurate size of test up to fairly high degrees of autocorrelation
  (rho=0.7). However this robustness does trade off against some loss of power
  in cases where the true autocorrelation is low. Advanced users are able to
  control this trade-off by adjusting the `v_0` setting, and the paper [1]
  includes a thorough investigation of the trade-offs involved and a table
  (Table 2b) which can guide the choice of v_0. See below for further details.
  The default setting is designed to be a reasonably robust general choice
  however.

  Further details:

  The high-level idea of the EWC estimator is to use only the `v`
  lowest-frequency components of the time-series to estimate the variance of the
  mean, since higher-frequency components will be influenced by autocorrelation
  and will bias the estimate of variance. The trade-off comes in how much of the
  frequency spectrum you discard: throwing away too much (low v) leads to a
  high-variance estimator, but controls any bias due to autocorrelation.
  Retaining too much of the spectrum (high v) leads to a more biased estimator
  which doesn't account well for any autocorrelation present in the data,
  however variance is reduced. (In the extreme case where v is maxed out at T-1,
  the test reduces to the standard IID t-test.)

  The paper recommends a choice v = v_0 T^(2/3). The T^(2/3) scaling is
  chosen for an optimal asymptotic trade-off of bias and variance. Good
  asymptotics don't guarantee good finite-sample behaviour though, for this the
  choice of constant factor v_0 is important too. Their recommended default
  v_0 = 0.4 was chosen to minimize a combination of size distortion and power
  loss for an alpha=0.05 significance test, assuming that the data has
  relatively high autocorrelation of rho=0.7. They focused mainly on minimizing
  size distortion (weight of 0.9 in their loss) with lower priority (weight of
  0.1) given to minimizing power loss.

  They show this leads to good size control (rejection rate close to
  nominal alpha) without too much loss of power, for autocorrelation of 0.7.
  Size control is good for lower degrees of autocorrelation too, although power
  is reduced more in these settings. In a sense this loss of power is the
  price you pay for being robust to relatively high levels of autocorrelation
  for finite sample sizes.

  It is however possible to make different choices of v_0:
  * By setting it higher than the default recommendation, you can obtain a more
    powerful test, at the cost of larger size distortions especially at higher
    degrees of autocorrelation.
  * By setting it lower than the default recommendation you can obtain better
    size control and better robustness to strong autocorrelation, at the expense
    of power.

  In Table 2b of [1] they give optimal values of v_0 for a range of choices both
  of autocorrelation robustness and size-control vs power trade-off. We allow a
  custom v_0 to be passed here, chosen from this table.

  [1] Lazarus, E., Lewis, D. J., Stock, J. H. & Watson, M. W. HAR inference:
  Recommendations for practice. J. Bus. Econ. Stat. 36, 541-559 (2018).

  [2] Diebold, F. X. & Mariano, R. S. Comparing predictive accuracy. J. Bus.
  Econ. Stat. 20, 134-144 (2002).
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
      v_0: float = 0.4,
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
      v_0: The setting v_0 from the Lazarus et al (2018) paper, which can be
        chosen using Table 2b in the paper, and which controls the trade-off
        of power vs size control and auto-correlation robustness of the test.
        The default setting of 0.4 follows the recommendations of the paper and
        is chosen to offer reasonably accurate size of an alpha=0.05 test up to
        fairly high (rho=0.7) degrees of autocorrelation, at the cost of some
        loss of power when autocorrelation is lower than this.
    """
    self._v_0 = v_0
    super().__init__(metrics, aggregated_statistics, experimental_unit_dim)

  def _compute_results(
      self,
      experimental_unit_dim: str,
      mean: xr.DataArray,
      per_unit_deviations: xr.DataArray) -> _TTestResults:
    sample_size = per_unit_deviations.sizes[experimental_unit_dim]

    _check_uniform_step(per_unit_deviations, experimental_unit_dim)
    # v is the number of DCT coefficients to use, and follows the formula of
    # Lazarus et al (2018). The constant factor v_0 can be chosen with reference
    # to their Table 2b and defaults to their default recommendation of 0.4.
    v = int(self._v_0 * (sample_size**(2/3)))
    v = max(1, v)
    v = min(v, sample_size-1)

    def compute_long_run_variance_estimate(x: np.ndarray) -> np.ndarray:
      projections = scipy.fft.dct(x, type=2, axis=-1, norm='ortho')
      # Select required number of coefficients, discarding the zero-frequency
      # / DC component (which should be 0 anyway since the per_unit_deviations
      # are zero-mean.)
      # Note if were to set v=sample_size-1, we would sum over all frequency
      # components in the decomposition of variance, which is equivalent to the
      # standard variance estimator for the IID case. We don't actually do this
      # here, but it perhaps gives some intuition about the trade-off made by
      # the setting of v.
      projections = projections[..., 1:v+1]
      return np.mean(projections**2, axis=-1)

    long_run_variance_estimate = xr.apply_ufunc(
        compute_long_run_variance_estimate,
        per_unit_deviations,
        input_core_dims=[[experimental_unit_dim]],
        output_core_dims=[[]],
    )
    stderr = xu.sqrt(long_run_variance_estimate / sample_size)

    return _TTestResults(
        mean=mean,
        standard_error=stderr,
        degrees_of_freedom=v,
    )
