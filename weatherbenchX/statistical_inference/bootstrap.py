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

from collections.abc import Mapping, Hashable
import functools
from typing import final, Literal

import arch.bootstrap
import numpy as np
import scipy.stats
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import base
from weatherbenchX.statistical_inference import utils

import xarray as xr


_REPLICATE_DIM = 'bootstrap_replicate'
_JACKKNIFE_DIM = 'jackknife'


CIMethod = Literal['percentile', 'bc', 'bca']


def _bca_bias(
    resampled: xr.DataArray, point_estimate: xr.DataArray) -> xr.DataArray:
  """Computes the bias used in BCa confidence intervals."""
  # Compute which proportion of values fall below the point estimate, where
  # values exactly equal are counted as a half. This avoids an arbitrary choice
  # of 0 or 1 for the equal case, and is important to avoid NaNs in some
  # degenerate situations, e.g. where there is no uncertainty (all resamples
  # equal to the point estimate) or where the point estimate lies on the
  # boundary.
  # This is equivalent to (resampled < point_estimate).mean(_REPLICATE_DIM)
  # except that it counts exact equality as a half:
  p = (xr.ufuncs.sign(resampled - point_estimate).mean(_REPLICATE_DIM) + 1)/2
  return xr.apply_ufunc(scipy.stats.norm.ppf, p)


def _bca_acceleration(jackknife_values: xr.DataArray) -> xr.DataArray:
  """Computes the acceleration `a` used in BCa confidence intervals."""
  u = jackknife_values.mean(_JACKKNIFE_DIM) - jackknife_values
  # This is formula 6.2 in Efron (1987):
  numerator = (u**3).sum(_JACKKNIFE_DIM)
  denominator = 6 * (u**2).sum(_JACKKNIFE_DIM)**1.5
  acceleration_estimate = numerator / denominator
  # When the denominator is zero the variance of the jackknife estimates is
  # zero (i.e. they're all the same), we have no evidence of any skewness in
  # this case and so set acceleration to zero.
  return xr.where(denominator == 0, 0, acceleration_estimate)


def _bca_adjust_cdf_values(
    cdf_values: xr.DataArray,
    bias: xr.DataArray,
    acceleration: xr.DataArray,
    ) -> xr.DataArray:
  """Adjusts CDF values `q` used for BCa confidence intervals [1]."""
  z = xr.apply_ufunc(scipy.stats.norm.ppf, cdf_values)
  z_plus_bias = bias + z
  z = bias + z_plus_bias / (1.0 - acceleration * z_plus_bias)
  result = xr.apply_ufunc(scipy.stats.norm.cdf, z)
  # Where the input is equal to 0 or 1, the limiting value of the output is the
  # same but we will encounter a NaN intermediate value without this logic:
  return xr.where(cdf_values.isin([0, 1]), cdf_values, result)


def _bca_inverse_adjust_cdf_values(
    cdf_values: xr.DataArray,
    bias: xr.DataArray,
    acceleration: xr.DataArray,
    ) -> xr.DataArray:
  """Inverse of _bca_adjust_cdf_values, used for p-value computation."""
  z = xr.apply_ufunc(scipy.stats.norm.ppf, cdf_values)
  z_minus_bias = z - bias
  z = z_minus_bias / (1 + acceleration * z_minus_bias) - bias
  result = xr.apply_ufunc(scipy.stats.norm.cdf, z)
  # Where the input is equal to 0 or 1, the limiting value of the output is the
  # same but we will encounter a NaN intermediate value without this logic:
  return xr.where(cdf_values.isin([0, 1]), cdf_values, result)


def _interpolated_empirical_cdf(data: xr.DataArray, dim: str, at_value: float):
  """Computes the interpolated empirical CDF of `data` at `at_value`."""

  def interpolated_empirical_cdf_1d_numpy(array: np.ndarray) -> float:
    assert array.ndim == 1
    array = array[~np.isnan(array)]  # Equiv. to skipna=True.
    # We could do this in O(n) rather than O(n log n) time, but the
    # interpolation would be more fiddly, so using sort and np.interp for now.
    array = np.sort(array)
    q = np.linspace(0, 1, array.shape[0])
    return np.interp(at_value, array, q)

  return xr.apply_ufunc(
      interpolated_empirical_cdf_1d_numpy,
      data,
      input_core_dims=[[dim]],
      vectorize=True)


class Bootstrap(base.StatisticalInferenceMethod):
  r"""Superclass for bootstrap-based statistical inference methods.

  This implements both traditional-but-naive percentile confidence intervals,
  and the more modern/best-practise BCa (and BC) confidence intervals of [1].

  The acceleration value required for BCa intervals is estimated using a
  jackknife procedure. So subclasses are required to provide jackknife values as
  well as bootstrap-resampled values to faciliate this. In the standard IID
  setup of [1] these are leave-one-out values of the metric, each omitting one
  of the evaluation units. In non-IID settings one can use e.g. leave-block-out
  or leave-cluster-out values, see subclasses for more details.

  # General caveats about bootstrap confidence intervals

  While a somewhat general tool, the bootstrap is no magic bullet, it can
  exhibit biases and its confidence intervals can have poor coverage, especially
  when the sample size is not huge. Unfortunately coverage is often less than
  advertised (i.e. the CIs are too narrow). Highly non-linear functions and
  highly skewed distributions can also cause it to perform poorly, although the
  BCa method is sometimes able to mitigate this.

  Ideally we would like a confidence interval for the true underlying value
  f(E[X]) where X are our statistics, E is their expectation (or limiting value
  of the mean under infinite data) and f is our values_from_mean_statistics
  function.

  What bootstrap methods actually aim to give us, is a confidence interval for
  the expectation of the finite-sample estimator: E[f(1/N \sum_n X_n)].

  In cases where f is linear, this estimator is unbiased for f(E[X]) and so
  these two are the same thing, but when f is nonlinear we will have to live
  with the mismatch. In particular this means that bootstrap intervals may not
  be strictly comparable across different sample sizes (they are intervals for
  different quantities), although if doing a paired test this is not a problem.

  While BCa intervals [1] aim to mitigate certain kinds of bias which can arise
  within the bootstrap procedure itself, they are fundamentally unable to detect
  or correct for biases of our estimator itself. Fundamentally, the procedure
  doesn't have any way to know what we intend it to be an estimator of, besides
  its own expectation.

  [1] Efron, B. Better bootstrap confidence intervals. J.A.S.A. 82, 171-185
  (1987).
  """

  # Subclass constructors must set these.
  _resampled_values: base.MetricValues  # Should have _REPLICATE_DIM.
  _jackknife_values: base.MetricValues  # Should have _JACKKNIFE_DIM.
  _point_estimates: base.MetricValues

  @property
  def resampled_values(self) -> base.MetricValues:
    """Resampled values of the metric, with a _REPLICATE_DIM dimension."""
    return self._resampled_values

  def point_estimates(self) -> base.MetricValues:
    return self._point_estimates

  def _adjust_cdf_values(
      self,
      cdf_values: base.MetricValues,
      method: CIMethod,
  ) -> base.MetricValues:
    """Adjusts CDF values for use by the given CI method."""
    if method not in ('bc', 'bca'):
      return cdf_values

    def adjust(cdf_values, resampled, point_estimate, jackknife):
      b = _bca_bias(resampled, point_estimate)
      a = _bca_acceleration(jackknife) if method == 'bca' else 0.
      return _bca_adjust_cdf_values(cdf_values, b, a)

    return xarray_tree.map_structure(
        adjust,
        cdf_values,
        self._resampled_values, self._point_estimates, self._jackknife_values)

  def _inverse_adjust_cdf_values(
      self,
      cdf_values: base.MetricValues,
      method: CIMethod,
  ) -> base.MetricValues:
    """Inverse of _adjust_cdf_values."""
    if method not in ('bc', 'bca'):
      return cdf_values

    def inverse_adjust(cdf_values, resampled, point_estimate, jackknife):
      b = _bca_bias(resampled, point_estimate)
      a = _bca_acceleration(jackknife) if method == 'bca' else 0.
      return _bca_inverse_adjust_cdf_values(cdf_values, b, a)

    return xarray_tree.map_structure(
        inverse_adjust,
        cdf_values,
        self._resampled_values, self._point_estimates, self._jackknife_values)


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
      self, alpha: float = 0.05, method: CIMethod = 'percentile',
  ) -> tuple[base.MetricValues, base.MetricValues]:
    """Two-sided confidence intervals.

    Args:
      alpha: The confidence level to use.
      method: The method to use to compute bootstrap confidence intervals.
        'percentile': This is the commonly-used method which just takes the
          alpha/2 and 1-alpha/2 quantiles of the bootstrap resampled values.
          It is somewhat naive and suboptimal especially at smaller sample
          sizes, see [1] for more details.
        'bca': The bias-corrected and accelerated method of [1], which
          corrects bootstrap confidence intervals for bias and skewness.
        'bc': Like 'bca' but without the correction for skewness
          ('acceleration'), also see [1].

    Returns:
      A tuple of lower and upper bounds of the confidence intervals, each of
      which are computed separately for each component of each variable of each
      metric.
    """
    cdf_values = xr.DataArray(
        [alpha/2, 1-alpha/2],
        dims=['level'],
        coords={'level': ['lower', 'upper']})
    cdf_values = xarray_tree.map_structure(
        lambda _: cdf_values, self.resampled_values)
    cdf_values = self._adjust_cdf_values(cdf_values, method)

    def compute_quantiles(
        resampled: xr.DataArray, cdf_values: xr.DataArray) -> xr.DataArray:
      return xr.apply_ufunc(
          lambda x, q: np.nanquantile(x, q, axis=-1),
          resampled,
          cdf_values,
          input_core_dims=[[_REPLICATE_DIM], ['level']],
          output_core_dims=[['level']],
          vectorize=True,
      )
    quantiles = xarray_tree.map_structure(
        compute_quantiles, self.resampled_values, cdf_values)

    return (
        xarray_tree.map_structure(lambda qs: qs.sel(level='lower'), quantiles),
        xarray_tree.map_structure(lambda qs: qs.sel(level='upper'), quantiles),
    )

  @final
  def p_values(
      self,
      null_value: float = 0.,
      method: CIMethod = 'percentile',
    ) -> base.MetricValues:
    """p-value for a two-sided test with the given null hypothesis value.

    This is computed by inverting the procedure used to compute confidence
    intervals.

    Args:
      null_value: The null hypothesis value to test.
      method: The method for computing confidence intervals, since the p-values
        are obtained by inverting this method. See `confidence_intervals` for
        more details.

    Returns:
      Two-sided p-values against the given null hypothesis, computed separately
      for each component of each variable of each metric.
    """

    empirical_cdf_at_null = xarray_tree.map_structure(
        functools.partial(_interpolated_empirical_cdf,
                          dim=_REPLICATE_DIM, at_value=null_value),
        self.resampled_values)

    # Inverts the transformation applied when computing confidence intervals.
    empirical_cdf_at_null = self._inverse_adjust_cdf_values(
        empirical_cdf_at_null, method)

    # Convert to a two-sided p-value.
    return xarray_tree.map_structure(
        lambda q: 2 * xr.ufuncs.minimum(q, 1-q), empirical_cdf_at_null)


class IIDBootstrap(Bootstrap):
  r"""Standard IID bootstrap method."""

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
      n_replicates: int,
      ):
    summed_stats = aggregated_statistics.sum_along_dims([experimental_unit_dim])
    self._point_estimates = metrics_base.compute_metrics_from_statistics(
        metrics, summed_stats.mean_statistics())

    def jackknife(summed_stats, separate_stats):
      return (summed_stats - separate_stats).rename(
          {experimental_unit_dim: _JACKKNIFE_DIM})
    jackknife_summed_stats = aggregation.AggregationState.map_multi(
        jackknife, summed_stats, aggregated_statistics)
    self._jackknife_values = (
        metrics_base.compute_metrics_from_statistics(
            metrics, jackknife_summed_stats.mean_statistics()))

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
    self._resampled_values = metrics_base.compute_metrics_from_statistics(
        metrics, resampled_stats.mean_statistics())


class ClusterBootstrap(Bootstrap):
  r"""Resamples clusters identified by the distinct values of a coordinate.

  The coordinate need not be unique and need not be an index coordinate, but it
  must be 1-dimensional.

  This method assumes independence between clusters, but allows for arbitrary
  dependence within clusters. It corresponds to the 'Strategy 1' cluster
  bootstrap of [1, pp.100-101], in the equal-cluster-size case anyway, and more
  generally to the 'all block' bootstrap of [2, p.5].

  In the unequal-cluster-size case, the resampled datasets may have a different
  total size to the original one (albeit with the same number of clusters).
  [2, p.5] also propose another option which ensures equal total size. It does
  not seem likely to work well unless you have many clusters of every occurring
  size, and it is also debatable in which situations it is desirable to maintain
  an equal total size (see below), and so we have not implemented it here.

  We can think of the approach implemented here as equivalent to the IID
  bootstrap, but where the IID data points are pairs of (sum-weighted-stats,
  sum-weights) computed within each cluster, not the original data points. From
  this point of view it is not a problem if the clusters' sizes (or summed
  weights) are unequal, provided the sizes are assumed to be drawn IID at random
  and not (say) fixed in the experimental design to have specific unequal
  values.

  This should give us good frequentist properties under replications of the
  experiment in which the number of clusters is the same, but the size of each
  cluster is random. In my view this is reasonable in many situations.

  If instead  we care about properties under replications of the experiment
  in which the total size of the dataset and/or individual cluster sizes are
  also constrained to be the same -- then we would need a more sophisticated
  approach.


  [1] Davison, A. C. & Hinkley, D. V. Bootstrap Methods and their Application
  (Cambridge University Press, 1997).
  [2] Sherman, M. & le Cessie, Saskia, A comparison between bootstrap methods
  and generalized estimating equations for correlated outcomes in generalized
  linear models, Communications in Statistics - Simulation and Computation,
  26:3, 901-925 (1997).
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_coord: str,
      n_replicates: int,
      ):
    coord = utils.get_and_check_experimental_unit_coord(
        aggregated_statistics, experimental_unit_coord, check_is_dim=False)
    experimental_unit_dim = coord.dims[0]

    self._point_estimates = metrics_base.compute_metrics_from_statistics(
        metrics, aggregated_statistics.sum_along_dims(
            [experimental_unit_dim]).mean_statistics())

    unique_cluster_ids, cluster_ids = np.unique(coord.data, return_inverse=True)
    cluster_ids = xr.DataArray(cluster_ids, dims=[experimental_unit_dim])
    num_units = unique_cluster_ids.shape[0]

    counts = np.random.multinomial(
        num_units,
        np.full(num_units, 1/num_units),
        size=n_replicates)
    counts = xr.DataArray(
        data=counts, dims=[_REPLICATE_DIM, 'cluster_id'])
    counts = counts.isel(cluster_id=cluster_ids)
    resampled_stats = aggregated_statistics.dot(
        counts, dim=experimental_unit_dim)
    self._resampled_values = metrics_base.compute_metrics_from_statistics(
        metrics, resampled_stats.mean_statistics())

    unique_cluster_ids = xr.DataArray(
        unique_cluster_ids, dims=[_JACKKNIFE_DIM])
    leave_one_cluster_out_weights = (cluster_ids != unique_cluster_ids)
    leave_one_cluster_out_stats = aggregated_statistics.dot(
        leave_one_cluster_out_weights, dim=experimental_unit_dim)
    self._jackknife_values = (
        metrics_base.compute_metrics_from_statistics(
            metrics, leave_one_cluster_out_stats.mean_statistics()))


def stationary_bootstrap_indices(
    n_data: int,
    mean_block_length: float,
    n_replicates: int,
    dtype: np.typing.DTypeLike = np.int64,
) -> np.ndarray:
  """Samples indices for stationary bootstrap, shape (n_data, n_replicates)."""
  end_block_prob = 1/mean_block_length
  current_indices = np.random.randint(n_data, size=(n_replicates,), dtype=dtype)
  all_indices = [current_indices]
  for _ in range(1, n_data):
    end_block_flags = np.random.rand(n_replicates) < end_block_prob
    new_random_indices = np.random.randint(
        n_data, size=(n_replicates,), dtype=dtype)
    # Blocks wrap around in a periodic fashion. This feature of the stationary
    # bootstrap method exists to avoid endpoint bias / ensure that each data
    # point is equally likely to be sampled.
    next_indices = (current_indices+1) % n_data
    current_indices = np.where(
        end_block_flags, new_random_indices, next_indices)
    all_indices.append(current_indices)
  return np.stack(all_indices, axis=0)


def _stationary_jackknife_block_lengths(
    n_data: int, mean_block_length: float) -> np.ndarray:
  """Sample block lengths for the stationary jackknife [1].

  Two modifications are made to the original method:
  * We limit sampled block lengths to a maximum of 1/2 the sample size, as well
    as imposing the paper's limit of 2 log(n) * mean_block_length.
  * We use circular wraparound to avoid endpoint bias and for further
    consistency with the stationary bootstrap of [2] which we use this in
    combination with.

  [1] Zhou, W. & Lahiri, S. Stationary jackknife. J. Time Ser. Anal. 45,
  333-360 (2024).
  [2] Politis, D. N. & Romano, J. P. The stationary bootstrap. J.A.S.A. 89,
  1303-1313 (1994).

  Args:
    n_data: The number of data points.
    mean_block_length: The mean block length to leave out (before truncation
      of the block length distribution anyway).

  Returns:
    1D array of lengths of the blocks to leave out.
  """
  # This geometric distribution is supported on {1, 2, ...} with mean 1/p.
  block_lengths = scipy.stats.geom.rvs(p=1/mean_block_length, size=n_data)
  # This is the upper bound on sampled block lengths from the paper, which aims
  # to ensure that the "length of the missing part is much smaller than the
  # length of the original observations":
  upper_bound = round(2 * mean_block_length * np.log(n_data))
  # The above achieves its aim in cases where block_length << n/2log(n), which
  # should usually be the case for reasonable sample sizes and
  # mean block lengths that scale e.g. with sqrt(n) or slower and with a
  # smallish constant factor. But it's far from guaranteed. We impose a tighter
  # upper bound here to stop the longest sampled block lengths from exceeding
  # half the sample size. (TODO(matthjw): or should this be lower than 1/2?)
  upper_bound = min(upper_bound, n_data//2)
  # Note this truncation of the geometric distribution means that
  # 'mean_block_length' is not strictly true any more, but it shouldn't affect
  # the mean too much in typical cases.
  return np.minimum(block_lengths, upper_bound)


def _leave_block_out_sums(
    data: np.ndarray, block_lengths: np.ndarray) -> np.ndarray:
  """Computes sums of data excluding blocks of the given lengths.

  The left-out blocks start at each index of the data in turn, have the lengths
  specified and they wrap around at the boundary.

  Args:
    data: Array of data points, to compute leave-block-out sums over final axis.
    block_lengths: Array of same shape as `data`, block lengths to leave
      out for blocks starting at each index of `data`'s final axis.

  Returns:
    Array of same shape as `data`, with sums of `data` excluding blocks of the
    given lengths.
  """
  n_data = data.shape[-1]
  # Add periodic padding for wraparound:
  data = np.concatenate([data, data], axis=-1)
  cumsums = data.cumsum(axis=-1)
  # We want sums that exclude blocks of the given lengths. With wraparound,
  # complements of blocks are themselves just blocks, with length:
  complement_of_block_lengths = n_data - block_lengths[::-1]
  end_indices = np.arange(n_data) + complement_of_block_lengths
  result = cumsums[..., end_indices] - cumsums[..., :n_data]
  return result[::-1]


class StationaryBootstrap(Bootstrap):
  r"""Stationary bootstrap method of Politis and Romano [1].

  This is a block bootstrap resampling method designed to work with stationary
  time series data where there may be some temporal dependence. By default we
  use the optimal block length selection procedure from [2], [3], and this is
  done separately for every metric, variable, and index along any extra
  dimensions present in the metric result.

  The core method isn't limited to Metrics which are simple means or linear
  functions of means and so has broader applicability than the t-test or
  autocorrelation-corrected versions of it. There are still some caveats to
  note however:

  # Optimal block length selection for functions of multiple time-series

  The optimal block length selection algorithm we use was only designed to apply
  to means of univariate time series, but our metrics in general can be computed
  from arbitrary functions of the means of multiple time-series of statistics.

  The compromise we make is to apply the block length selection procedure to
  scalar values of the metric computed on a per-timestep basis. For metrics that
  are a simple mean or a linear function of means this is using the method
  exactly as intended. For non-linear functions of means, essentially we're
  approximating f(mean(X)) as mean(f(X)) for the purposes of the block length
  selection. This is justified when the function f is close to linear over the
  range of variation of *per-timestep* values of X, but if f is very nonlinear
  over this range then block length selection can fail badly and you are
  advised to select an appropriate block length manually instead.
  TODO(matthjw): Provide more options for this, e.g. a way to specify a
  particular statistic to use for the block length selection instead of the
  per-timestep values of the metric itself.

  Other possible heuristic approaches seen in the literature are to base it on
  the average or maximum of the optimal block lengths computed for each separate
  univariate statistics time series, or on a VAR (vector auto-regressive) model
  of the statistics. These may sometimes be too conservative, because they don't
  take into account the potential of the function f to reduce the effect of
  autocorrelation in some cases, for example where f computes something like
  a difference of two positively-correlated time-series. A better solution may
  be to linearize f around the mean, and then apply block length selection to
  the per-timestep values of this linearized function. This would require the
  gradient of f however.

  From what I understand, automatic block-length selection for bootstrap methods
  applied to multivariate time series data is a difficult open problem in
  statistics. If the default approach doesn't work for you, you are free to
  manually specify the block length to use, and you may sometimes need to.

  # Stationarity assumption

  While this method makes few distributional assumptions, one assumption is does
  make is that the time series of statistics is stationary, meaning the
  distribution (including marginal mean and variance, autocorrelation at
  different lags, etc) doesn't change over time. If you have clearly non-random
  trends over time in the distribution of your data, including seasonality --
  then ideally you would detrend or de-seasonalize the data in some way
  beforehand, or use a more tailored method.
  Note that it's not uncommon to apply tests like this to data with mild
  seasonality though, for better or worse. The hope is that when you are
  comparing (e.g.) errors of two models, the errors may be less seasonal or
  trended than the ground-truth data itself, and the difference of errors even
  less so.

  # Weightings

  This method can handle non-constant weights being used (via the
  AggregationState) to compute the means of the statistics. In this case the
  weights are treated as randomly sampled alongside the statistics themselves,
  with the joint distribution of weights and statistics assumed stationary in
  time. So when we think about long-run properties of our confidence intervals
  etc, this in the context of repeated sampling of new weights as well as new
  statistics.

  In particular, if the per-experimental-unit weights are fixed, non-random
  values that are different at different timesteps, this would violate the
  stationarity assumption. It would be more of an issue the more uneven the
  weights are.

  # Finite-sample bias from the bootstrap

  While quite general tools, the bootstrap in general (and the block bootstrap
  in particular) are no magic bullet and can exhibit biases especially when the
  sample size (or here, the effective sample size after taking into account
  temporal dependence) is small.

  # Stationary jackknife for estimating acceleration in BCa intervals

  We support the BCa (bias-corrected and accelerated) intervals of [4]. In the
  original IID setting these use the IID jackknife to estimate the acceleration.
  Here the data is not IID however, and we use the stationary jackknife of [5]
  instead, which is suitable for use with dependent data, and takes a similar
  approach to the stationary bootstrap [1] in sampling block lengths from a
  geometric distribution. We use the same mean block length for this as we
  use for the stationary bootstrap. Our implementation also modifies [5] to use
  circular wraparound, for consistency with the stationary bootstrap.

  [1] Politis, D. N. & Romano, J. P. The stationary bootstrap. J.A.S.A. 89,
  1303-1313 (1994).
  [2] Politis, D. N. & White, H. Automatic Block-Length Selection for the
  Dependent Bootstrap, Econometric Reviews, 23:1, 53-70 (2004).
  [3] Patton, A., Politis, D. N. & White, H. Correction to "Automatic
  Block-Length Selection for the Dependent Bootstrap" by D. Politis and
  H. White, Econometric Reviews, 28:4, 372-375 (2009).
  [4] Efron, B. Better bootstrap confidence intervals. J.A.S.A. 82, 171-185
  (1987).
  [5] Zhou, W. & Lahiri, S. Stationary jackknife. J. Time Ser. Anal. 45,
  333-360 (2024).
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregated_statistics: aggregation.AggregationState,
      experimental_unit_dim: str,
      n_replicates: int,
      mean_block_length: float | None = None,
      block_length_rounding_resolution: float | None = 30.0,
      stationary_bootstrap_indices_cache_size: int = 50,
  ):
    """Initializer.

    Args:
      metrics: The metrics to compute.
      aggregated_statistics: The statistics to use to compute the metrics.
      experimental_unit_dim: The dimension over which to bootstrap, along which
        any serial dependence occurs. Typically this will be a dimension
        corresponding to time.
      n_replicates: The number of bootstrap replicates to use.
      mean_block_length: The mean block length to use. If None, an optimal
        block length will be computed automatically for every time series
        present in the metrics: for each metric, for each variable within that
        metric, and for each index into any dimensions present besides the
        `experimental_unit_dim`.
      block_length_rounding_resolution: As a performance optimization, we round
        off the block length and reuse bootstrap indices when the rounded block
        length is the same. This setting controls how aggressitvely we round the
        block length when doing this. The rounding is done in the log domain and
        the resolution corresponds to the number of distinct rounded values
        between consective powers of 10 (e.g. 1 and 10, 10 and 100 etc).
        You can set it to None to disable rounding altogether.
      stationary_bootstrap_indices_cache_size: The size of the LRU cache used
        to cache bootstrap indices as a function of the rounded block length.
        This is a memory / speed trade-off.
    """
    self._experimental_unit_dim = experimental_unit_dim
    self._mean_block_length = mean_block_length
    self._n_replicates = n_replicates
    self._aggregated_statistics = aggregated_statistics
    self._block_length_rounding_resolution = block_length_rounding_resolution
    self._stationary_bootstrap_indices = functools.lru_cache(
        maxsize=stationary_bootstrap_indices_cache_size)(
            stationary_bootstrap_indices)
    self._point_estimates = {}
    self._resampled_values = {}
    self._jackknife_values = {}
    for metric_name, metric in metrics.items():
      (point_estimates, resampled_values, jackknife_values
       ) = self._bootstrap_results_for_metric(metric)
      self._point_estimates[metric_name] = point_estimates
      self._resampled_values[metric_name] = resampled_values
      self._jackknife_values[metric_name] = jackknife_values

  def _optimal_block_length(self, data_array: xr.DataArray) -> float:
    if self._mean_block_length is not None:
      return self._mean_block_length

    assert self._experimental_unit_dim in data_array.dims
    if data_array.sizes[self._experimental_unit_dim] < 8:
      # At least, arch.bootstrap.optimal_block_length craps out with a very
      # unfriendly error if given a smaller array.
      raise ValueError(
          'Need at least 8 data points along experimental_unit_dim '
          f'{self._experimental_unit_dim} to set mean_block_length '
          'automatically -- and many more than 8 recommended.')
    data_array = data_array.squeeze()
    assert data_array.ndim == 1

    # We use the arch library to compute optimal block length, since it's a
    # somewhat fiddly procedure. (Ideally we would re-use their entire
    # implementation of the stationary bootstrap, but it is quite slow and we
    # would have to patch it awkwardly to fix some issues and extend it to
    # produce p-values.)
    #
    # .stationary gives the mean block length for use with the stationary
    # bootstrap:
    result = arch.bootstrap.optimal_block_length(
        data_array.data).stationary.item()
    # Values <1 can sometimes show up, but 1 is the minimum.
    result = max(1.0, result)
    if self._block_length_rounding_resolution is not None:
      # Rounding this off makes it a useful key for LRU caching of the
      # bootstrap indices. These need to be sampled separately for each mean
      # block length used, and this forms a significant fraction of total
      # running time. The inference of an optimal block length is noisy enough
      # that rounding off to 1 or 2 significant figures (or the similar but
      # smoother logarithmic rounding below) should be perfectly acceptable.
      result = utils.logarithmic_round(
          result, self._block_length_rounding_resolution)
    return result

  def _bootstrap_results_for_metric(
      self, metric: metrics_base.Metric) -> tuple[
          Mapping[Hashable, xr.DataArray],
          Mapping[Hashable, xr.DataArray],
          Mapping[Hashable, xr.DataArray]]:

    point_estimates = metrics_base.compute_metric_from_statistics(
        metric, self._aggregated_statistics.sum_along_dims(
            [self._experimental_unit_dim]).mean_statistics())
    per_unit_values = metrics_base.compute_metric_from_statistics(
        metric, self._aggregated_statistics.mean_statistics())
    sum_weighted_stats = {
        stat_name: self._aggregated_statistics.sum_weighted_statistics[
            stat.unique_name]
        for stat_name, stat in metric.statistics.items()
    }
    sum_weights = {
        stat_name: self._aggregated_statistics.sum_weights[
            stat.unique_name]
        for stat_name, stat in metric.statistics.items()
    }
    resampled_values = {}
    jackknife_values = {}
    for var_name in point_estimates.keys():
      # Results for different variables will need to be computed separately,
      # as the optimal block length will depend on the variable.
      #
      # We try to avoid computing results for *all* variables every time we
      # do a bootstrap resample based on the optimal block length for a single
      # *one* of these variables though, using this logic:
      if (len(point_estimates) > 1 and
          all(var_name in vars for vars in sum_weighted_stats.values())):
        # A corresponding variable is present in each Statistic and we make the
        # assumption that this variable in the result only depends on these
        # corresponding variables in the stats and that we can recompute the
        # Metric with the statistics restricted just to this single variable.
        # This saves us resampling statistics for all the other variables.
        sum_weighted_stats_for_this_var = {
            stat_name: {var_name: vars[var_name]}
            for stat_name, vars in sum_weighted_stats.items()
        }
        sum_weights_for_this_var = {
            stat_name: {var_name: vars[var_name]}
            for stat_name, vars in sum_weights.items()
        }
      else:
        # If there was only a single variable, it's fine to resample all the
        # statistics since this will only be done once.
        # If there are multiple variables and they don't correspond 1:1 to
        # variables in the statistics, then we can't do any better than
        # resampling all the statistics even though this may result in some
        # redundant work. This should be a rare edge case though.
        sum_weighted_stats_for_this_var = sum_weighted_stats
        sum_weights_for_this_var = sum_weights

      # The optimal block length will also depend on the specific index along
      # any extra dimensions present in the metric result, for example suppose a
      # lead_time dimension is present, different degrees of autocorrelation may
      # be observed for forecast metrics at different lead times.
      # And so bootstrap indices will need to be sampled separately for each
      # index along any extra dimensions present in the metric result.
      #
      # We assume that where a dimension of the metric result also occurs in the
      # statistics, that the metrics at index i along that dimension only depend
      # on the statistics at index i along the same dimension, and that we can
      # therefore slice the statistics down to a single index along any such
      # dimensions when computing a single index of the metric result.
      #
      # This assumption isn't strictly guaranteed, but it is true in the vast
      # majority of cases, including:
      # * The common case of a component-wise metric like RMSE, which is a
      #   scalar quantity computed independently for each component.
      # * Metrics which introduce some additional internal dimensions on their
      #   statistics, but reduce them down to a scalar value in their output.
      # * Metrics which introduce some additional dimensions in their output
      #   which aren't present in the statistics, but use a different dimension
      #   name for them to any dimensions used in the statistics.
      (resampled_values[var_name],
       jackknife_values[var_name]) = utils.apply_to_slices(
           functools.partial(self._bootstrap_results_for_metric_scalar,
                             metric, var_name),
           per_unit_values[var_name],
           sum_weighted_stats_for_this_var,
           sum_weights_for_this_var,
           dim=point_estimates[var_name].dims,
       )

    return point_estimates, resampled_values, jackknife_values

  def _bootstrap_results_for_metric_scalar(
      self,
      metric: metrics_base.Metric,
      var_name: str,
      per_unit_values: xr.DataArray,
      sum_weighted_stats: Mapping[str, Mapping[Hashable, xr.DataArray]],
      sum_weights: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> tuple[xr.DataArray, xr.DataArray]:
    n_data = per_unit_values.sizes[self._experimental_unit_dim]
    mean_block_length = self._optimal_block_length(per_unit_values)

    bootstrap_indices = self._stationary_bootstrap_indices(
        n_data=n_data,
        mean_block_length=mean_block_length,
        n_replicates=self._n_replicates,
    )
    bootstrap_indices = xr.DataArray(
        bootstrap_indices, dims=[self._experimental_unit_dim, _REPLICATE_DIM])

    agg_state = aggregation.AggregationState(sum_weighted_stats, sum_weights)
    def sum_of_resampled(data):
      # Note the dimensions of bootstrap_indices (experimental_unit_dim,
      # _REPLICATE_DIM) that we're selecting, will be present in the result of
      # the isel call.
      return data.isel({self._experimental_unit_dim: bootstrap_indices}).sum(
          self._experimental_unit_dim)
    resampled_values = metric.values_from_mean_statistics(
        agg_state.map(sum_of_resampled).mean_statistics())[var_name]

    block_lengths = _stationary_jackknife_block_lengths(
        n_data, mean_block_length)
    def jackknife_sums(data):
      return xr.apply_ufunc(
          lambda x: _leave_block_out_sums(x, block_lengths),
          data,
          input_core_dims=[[self._experimental_unit_dim]],
          output_core_dims=[[_JACKKNIFE_DIM]],
      )
    jackknife_values = metric.values_from_mean_statistics(
        agg_state.map(jackknife_sums).mean_statistics())[var_name]

    return resampled_values, jackknife_values
