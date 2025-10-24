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
from typing import final

import arch.bootstrap
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import autodiff
from weatherbenchX.statistical_inference import base
from weatherbenchX.statistical_inference import utils

import xarray as xr


_REPLICATE_DIM = 'bootstrap_replicate'


class Bootstrap(base.StatisticalInferenceMethod):
  r"""Superclass for bootstrap-based statistical inference methods.

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
  """

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
    self._point_estimates = metrics_base.compute_metrics_from_statistics(
        metrics, aggregated_statistics.sum_along_dims(
            [experimental_unit_dim]).mean_statistics())
    self._resampled_values = metrics_base.compute_metrics_from_statistics(
        metrics, resampled_stats.mean_statistics())


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

  # Block length selection for nonlinear functions of multiple time-series

  The optimal block length selection algorithm we use [2], [3] was designed to
  apply to means of univariate time series, but our metrics in general can be
  nonlinear functions f of the means of multiple time-series of statistics.

  Our solution is to linearize the function f around the means, and then
  apply block length selection to the time-series of per-timestep values of this
  linearized function. We believe this is a reasonable approximation because:
  * The mean of this timeseries is the correct value of the metric
  * The variance estimate for the mean of the linearized values is a good
    approximation of the variance of the actual metric value f(mean(X)),
    provided that the function f is sufficiently linear over the range of
    sampling variation we expect for the *means* of the statistics, which will
    drop off with sample size.
  * It's exactly correct in the common case where f is a linear function.

  In short, with a smooth function f this is valid asymptotically, and for
  middling effective sample sizes we anticipate it will still be a good
  enough approximation for the purposes of block length selection at least.
  While it may seem a bit of a simplistic approximation, note that block
  length selection for this general case of nonlinear functions and multiple
  timeseries is an open research problem and I haven't been able to find a
  better general method in the literature.

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

  [1] Politis, D. N. & Romano, J. P. The stationary bootstrap. J. Am. Stat.
  Assoc. 89, 1303–1313 (1994).
  [2] Politis, D. N. & White, H. Automatic Block-Length Selection for the
  Dependent Bootstrap, Econometric Reviews, 23:1, 53-70 (2004).
  [3] Patton, A., Politis, D. N. & White, H. Correction to "Automatic
  Block-Length Selection for the Dependent Bootstrap" by D. Politis and
  H. White, Econometric Reviews, 28:4, 372-375 (2009).
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

    # We use linearized per-unit values for block length selection:
    (self._point_estimates, self._per_unit_tangents
     ) = autodiff.per_unit_values_linearized_around_mean_statistics(
         metrics, aggregated_statistics, experimental_unit_dim)

    self._resampled_values = {}
    for metric_name, metric in metrics.items():
      self._resampled_values[metric_name] = self._bootstrap_results_for_metric(
          metric,
          self._point_estimates[metric_name],
          self._per_unit_tangents[metric_name])

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
      self,
      metric: metrics_base.Metric,
      point_estimates: Mapping[Hashable, xr.DataArray],
      per_unit_tangents: Mapping[Hashable, xr.DataArray],
      ) -> Mapping[Hashable, xr.DataArray]:

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
      resampled_values[var_name] = utils.apply_to_slices(
          functools.partial(self._bootstrap_results_for_metric_scalar,
                            metric, var_name),
          per_unit_tangents[var_name],
          sum_weighted_stats_for_this_var,
          sum_weights_for_this_var,
          dim=point_estimates[var_name].dims,
      )

    return resampled_values

  def _bootstrap_results_for_metric_scalar(
      self,
      metric: metrics_base.Metric,
      var_name: str,
      per_unit_tangents: xr.DataArray,
      sum_weighted_stats: Mapping[str, Mapping[Hashable, xr.DataArray]],
      sum_weights: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> xr.DataArray:
    n_data = per_unit_tangents.sizes[self._experimental_unit_dim]
    mean_block_length = self._optimal_block_length(per_unit_tangents)

    bootstrap_indices = self._stationary_bootstrap_indices(
        n_data=n_data,
        mean_block_length=mean_block_length,
        n_replicates=self._n_replicates,
    )
    bootstrap_indices = xr.DataArray(
        bootstrap_indices, dims=[self._experimental_unit_dim, _REPLICATE_DIM])

    def sum_of_resampled(data):
      # Note the dimensions of bootstrap_indices (experimental_unit_dim,
      # _REPLICATE_DIM) that we're selecting, will be present in the result of
      # the isel call.
      return data.isel({self._experimental_unit_dim: bootstrap_indices}).sum(
          self._experimental_unit_dim)
    sum_weighted_stats, sum_weights = xarray_tree.map_structure(
        sum_of_resampled, (sum_weighted_stats, sum_weights))
    mean_stats = xarray_tree.map_structure(
        lambda x, y: x / y, sum_weighted_stats, sum_weights)
    del sum_weighted_stats, sum_weights

    return metric.values_from_mean_statistics(mean_stats)[var_name]
