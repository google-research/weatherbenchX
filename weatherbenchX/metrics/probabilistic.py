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
"""Implementation of probabilistic metrics and assiciated statistics."""

from collections.abc import Hashable
from typing import Mapping, Tuple
import numpy as np
import scipy.stats
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base
from weatherbenchX.metrics import categorical
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import wrappers
import xarray as xr
import xarray.ufuncs as xu

### Statistics
# TODO(srasp): NaN mask seem to get lost in some probabilistic metrics.
# Investigate and fix this.


class EnsembleAveragedStatistic(base.Statistic):
  """Statistic that averages a statistic over the ensemble dimension."""

  def __init__(
      self,
      wrapped_statistic: base.Statistic,
      *,
      ensemble_dim: str,
      skipna_ensemble: bool,
  ):
    self._wrapped_statistic = wrapped_statistic
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return self._wrapped_statistic.unique_name + '_each_'  + self._ensemble_dim

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes statistics per predictions/targets chunk."""
    statistics = self._wrapped_statistic.compute(predictions, targets)

    # Enforce that every single array has the ensemble dimension.
    def reduce_over_ensemble(da: xr.DataArray) -> xr.DataArray:
      if self._ensemble_dim not in da.dims:
        raise ValueError(
            f'Dimension {self._ensemble_dim} not found in {da.dims}'
        )
      return da.mean(dim=self._ensemble_dim, skipna=self._skipna_ensemble)

    return xarray_tree.map_structure(reduce_over_ensemble, statistics)


class EnsembleAveragedMetric(base.Metric):
  """Wraps any metric to average its statistics over the ensemble dimension.

  This can be useful to evaluate the statistics of deterministic metrics on
  individual ensemble members (rather than e.g. on the ensemble average). For
  example, can be used to compute the RMSE where the MSE is averaged over the
  individual squared errors of each of the ensemble members.

  Note that this can also be achieved by directly using the underlying metric
  without this wrapper, and then using an Aggregator that includes the
  ensemble_dim in its reduce_dims. However, having the metric reduce the sample
  axis can be convenient in some cases to avoid having to specify a separate
  aggregator.

  """

  def __init__(
      self,
      wrapped_metric: base.Metric,
      *,
      ensemble_dim: str = 'number',
      skipna_ensemble: bool = False
  ):
    self._wrapped_metric = wrapped_metric
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    wrapped_statistics = {}
    for stat_name, stat in self._wrapped_metric.statistics.items():
      wrapped_statistics[stat_name] = EnsembleAveragedStatistic(
          wrapped_statistic=stat,
          ensemble_dim=self._ensemble_dim,
          skipna_ensemble=self._skipna_ensemble,
      )
    return wrapped_statistics

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self._wrapped_metric.values_from_mean_statistics(statistic_values)


class CRPSSkill(base.PerVariableStatistic):
  """The skill measure associated with CRPS, E|X - Y|."""

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'CRPSSkill_{self._ensemble_dim}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    reduce_dims = [self._ensemble_dim]
    if self._ensemble_dim in targets.dims:
      # TODO: shoyer - Add an implementation based on sort (similar to
      # scipy.stats.energy_distance), for which runtime would scale like
      # O((N + M) * log(N + M)) rather than O(N * M) for prediction and
      # target realizations of sizes N and M.
      pseudo_ensemble_dim = f'{self._ensemble_dim}_PSEUDO_FOR_TARGETS'
      reduce_dims += [pseudo_ensemble_dim]
      targets = targets.rename({self._ensemble_dim: pseudo_ensemble_dim})
    return np.abs(predictions - targets).mean(
        reduce_dims, skipna=self._skipna_ensemble
    )


def _rankdata(x: np.ndarray, axis: int) -> np.ndarray:
  """Version of (ordinal) scipy.rankdata from V13."""
  x = np.asarray(x)
  x = np.swapaxes(x, axis, -1)
  j = np.argsort(x, axis=-1)
  ordinal_ranks = np.broadcast_to(
      np.arange(1, x.shape[-1] + 1, dtype=int), x.shape
  )
  ordered_ranks = np.empty(j.shape, dtype=ordinal_ranks.dtype)
  np.put_along_axis(ordered_ranks, j, ordinal_ranks, axis=-1)
  return np.swapaxes(ordered_ranks, axis, -1)


def _rank_da(da: xr.DataArray, dim: str) -> np.ndarray:
  return da.copy(data=_rankdata(da.values, axis=da.dims.index(dim)))


class CRPSSpread(base.PerVariableStatistic):
  """Sample-based estimate of the spread measure used in CRPS, E|X - X`|.

  (This is also referred to in places as Mean Absolute Difference.)

  See the docstring for CRPSEnsemble for more details on what 'fair' means
  and the two different options (use_sort=True vs False) for computing the
  estimate.
  """

  def __init__(
      self,
      ensemble_dim: str = 'number',
      use_sort: bool = False,
      fair: bool = True,
      which: str = 'predictions',
      skipna_ensemble: bool = False,
  ):
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._which = which
    self._fair = fair
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    fair_str = 'fair' if self._fair else 'unfair'
    return f'CRPSSpread_{self._ensemble_dim}_{fair_str}_{self._which}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    if self._which == 'predictions':
      da = predictions
    elif self._which == 'targets':
      da = targets
    else:
      raise ValueError(f'Unhandled {self._which=}')

    if self._skipna_ensemble:
      n_ensemble = da.count(self._ensemble_dim)
    else:
      n_ensemble = da.sizes[self._ensemble_dim]
    if not self._skipna_ensemble:
      if n_ensemble < 2:
        raise ValueError('Cannot estimate CRPS spread with n_ensemble < 2.')

    if self._use_sort:
      if self._skipna_ensemble:
        raise ValueError('skipna_ensemble is not supported with use_sort=True.')
      # one_half_spread is ̂̂λ₂ from Zamo. That is, with n_ensemble = M,
      #   λ₂ = 1 / (2 M (M - 1)) Σ_{i,j=1}^M |Xi - Xj|
      # See the definition of eFAIR and then
      # eqn 3 (appendix B), which shows that this double summation of absolute
      # differences can be written as a sum involving sorted elements multiplied
      # by their index. That is, if X1 < X2 < ... < XM,
      #   λ₂ = 1 / (M(M-1)) Σ_{i,j=1}^M (2*i - M - 1) Xi.
      # The term (2*i - M - 1) is +1 times the number of terms Xi is greater
      # than, and -1 times the number of terms Xi is less than.
      # Here we compute the rank of each element, multiply appropriately, then
      # sum. This second form involves an O(M Log[M]) compute and O(M) memory
      # usage, but with a larger constant factor, whereas the first is O(M²) in
      # compute and memory but with a smaller constant factor due to being
      # easily parallelizable.
      rank = _rank_da(da, self._ensemble_dim)
      return (
          2
          * (
              ((2 * rank - n_ensemble - 1) * da).mean(
                  self._ensemble_dim, skipna=False
              )
          )
          / (n_ensemble - int(self._fair))
      )
    else:
      second_ensemble_dim = 'ensemble_dim_2'
      da_2 = da.rename({self._ensemble_dim: second_ensemble_dim})
      return abs(da - da_2).sum(
          dim=(self._ensemble_dim, second_ensemble_dim),
          skipna=self._skipna_ensemble,
      ) / (n_ensemble * (n_ensemble - int(self._fair)))


class EnsembleVariance(base.PerVariableStatistic):
  """Computes the mean variance in the ensemble dimension.

  This uses the standard unbiased estimator of variance.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'EnsembleVariance_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return predictions.var(
        dim=self._ensemble_dim, ddof=1, skipna=self._skipna_ensemble
    )


class UnbiasedEnsembleMeanSquaredError(base.PerVariableStatistic):
  """Computes the unbiased ensemble mean squared error.

  Let X be the ensemble mean of predictions.
  If targets.dims contains self.ensemble_dim, then let Y be the ensemble mean
  of the targets. Otherwise (the usual case), let Y be the targets.

  This class estimates E(X - Y)² with no finite-ensemble bias. This is done by
  subtracting the sample variance divided by ensemble size. As such, you must
  have ensemble size > 1 or the result will be NaN.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def unique_name(self) -> str:
    return f'UnbiasedEnsembleMeanSquaredError_{self._ensemble_dim}_skipna_ensemble_{self._skipna_ensemble}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    if self._ensemble_dim in predictions.dims:
      predictions_mean = predictions.mean(
          dim=self._ensemble_dim, skipna=self._skipna_ensemble
      )
      predictions_var = predictions.var(
          dim=self._ensemble_dim, ddof=1, skipna=self._skipna_ensemble
      )
      if self._skipna_ensemble:
        num_predictions = predictions.count(self._ensemble_dim)
      else:
        num_predictions = predictions.sizes[self._ensemble_dim]
      predictions_bias = predictions_var / num_predictions
    else:
      raise ValueError(
          f'Dimension {self._ensemble_dim} not found in {predictions.dims}'
      )

    if self._ensemble_dim in targets.dims:
      targets_mean = targets.mean(
          dim=self._ensemble_dim, skipna=self._skipna_ensemble
      )
      targets_var = targets.var(
          dim=self._ensemble_dim, ddof=1, skipna=self._skipna_ensemble
      )
      if self._skipna_ensemble:
        num_targets = targets.count(self._ensemble_dim)
      else:
        num_targets = targets.sizes[self._ensemble_dim]
      targets_bias = targets_var / num_targets
    else:
      targets_mean = targets
      targets_bias = 0.0
    biased_mse = (predictions_mean - targets_mean) ** 2
    return biased_mse - predictions_bias - targets_bias


class EnsembleRankedProbabilityScore(base.PerVariableStatistic):
  """Ranked probability score for an ensemble prediction.

  The RPS implemented here is either biased if fair=False, or unbiased if
  fair=True.

  Given a ground truth scalar random variable Y, a prediction random variable X,
  a sequence of bin boundaries b_0 < b_1 < ... < b_k, where b_0 = -inf and
  b_K = +inf, the Ranked Probability Score is defined as

    RPS = E[ Σk (CDF(Y)(b_k) - CDF(X)(b_k))^2 ]

  where the sum over k is taken over k = 1, 2, ..., K, and CDF(X) and CDF(Y)
  are the cumulative distribution functions of X and Y, respectively.

  In practice the CDFs are estimated by binning samples from targets and
  predictions in the bins defined by the bin boundaries, e.g. given N iid
  prediction samples X1, ..., XN, we estimate

    CDF(X)(b_k) ≈ count(Xn <= b_k) / N,

  where the count is taken over n = 1, 2, ..., N, and similarly for CDF(Y).
  Because we are using a finite-size sample to estimate the CDFs, this
  approximation introduces bias due to the square operation in RPS. We implement
  a 'fair' estimate of RPS that removes this bias, by using the unbiased mean
  squared error estimator. Note that the de-biasing of the fair estimator is
  only applicable (and only applied) for ensembles. So, in the typical case
  where the predictions are ensembles and the targets are not, the fair estimate
  applies debiasing for the predictions, not for the targets.

  For computing the RPS directly from CDFs, see
  `categorical.RankedProbabilityScore`.
  """

  def __init__(
      self,
      prediction_bin_thresholds: xr.Dataset,
      target_bin_thresholds: xr.Dataset,
      bin_dim: str,
      unique_name_suffix: str,
      ensemble_dim: str = 'number',
      skipna_ensemble: bool = False,
      fair: bool = True,
      enforce_monotonicity: bool = True,
  ):
    """Init.

    Args:
      prediction_bin_thresholds: Binning thresholds for the predictions.
      target_bin_thresholds: Binning thresholds for the targets.
      bin_dim: Name of the binning dimension.
      unique_name_suffix: Suffix to add to the unique name.
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, skip NaN values when computing the ensemble
        mean. Default: False.
      fair: If True, use the fair estimate of RPS. If False, use the biased
        estimate. Default: True.
      enforce_monotonicity: If True (default), enforce monotonicity of the
        binning thresholds. If bin thresholds contain NaNs, this will raise an
        error.
    """
    self._prediction_bin_thresholds = prediction_bin_thresholds
    self._target_bin_thresholds = target_bin_thresholds
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble
    self._fair = fair
    self._bin_dim = bin_dim
    self._unique_name_suffix = unique_name_suffix

    binned_prediction_wrapper = wrappers.ContinuousToCDF(
        which='predictions',
        threshold_values=self._prediction_bin_thresholds,
        threshold_dim=bin_dim,
        unique_name_suffix=self._unique_name_suffix,
        enforce_monotonicity=enforce_monotonicity,
    )
    binned_target_wrapper = wrappers.ContinuousToCDF(
        which='targets',
        threshold_values=self._target_bin_thresholds,
        threshold_dim=bin_dim,
        unique_name_suffix=self._unique_name_suffix,
        enforce_monotonicity=enforce_monotonicity,
    )

    if self._fair:
      se_stat = UnbiasedEnsembleMeanSquaredError(
          ensemble_dim=self._ensemble_dim,
          skipna_ensemble=self._skipna_ensemble,
      )

    else:
      # This wrapper will be appled after the binning wrappers below.
      se_stat = wrappers.WrappedStatistic(
          deterministic.SquaredError(),
          wrappers.EnsembleMean(
              which='both',
              ensemble_dim=self._ensemble_dim,
              skipna=self._skipna_ensemble,
              # In the case where the targets are not an ensemble, we want to
              # skip the ensemble mean. We could have done which='predictions'
              # instead, but this would prevent us from using an ensemble as
              # the target, which may be useful functionality.
              skip_if_ensemble_dim_missing=True,
          ),
      )
    # Wrap the statistic with binning wrappers for predictions and targets.
    self._se_stat = wrappers.WrappedStatistic(
        wrappers.WrappedStatistic(se_stat, binned_target_wrapper),
        binned_prediction_wrapper,
    )

  @property
  def unique_name(self) -> str:
    return (
        f'RankedProbabilityScore_{self._ensemble_dim}_'
        f'skipna_ensemble_{self._skipna_ensemble}_'
        f'fair_{self._fair}_'
        f'{self._unique_name_suffix}'
    )

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:

    # TODO(stratismarkou, matthjw): make _compute_per_variable into a public
    # method so we can reuse it in cases like the one below.
    cdf_mse = self._se_stat.compute({'tmp': predictions}, {'tmp': targets})[
        'tmp'
    ]

    # RPS is the sum of squared errors over the bin dimension.
    return cdf_mse.sum(self._bin_dim, skipna=self._skipna_ensemble)


### Metrics


class CRPSEnsemble(base.PerVariableMetric):
  """Continuous ranked probabilisty score for an ensemble prediction.

  Given ground truth scalar random variable Y, and two iid predictions X, X`,
  the Continuously Ranked Probability Score is defined as
    CRPS = E|X - Y| - 0.5 * E|X - X`|
  where `E` is mathematical expectation, and | ⋅ | is the absolute value. CRPS
  has a unique minimum when X is distributed the same as Y.

  We implement a 'fair' sample-based estimate of CRPS based on 2 or more
  ensemble members. 'Fair' means this is an unbiased estimate of the CRPS
  attained by the underlying predictive distribution from which the ensemble
  members are drawn -- equivalently, the CRPS attained in the limit of an
  infinite ensemble.

  [Zamo & Naveau, 2018] derive two equivalent ways to compute the spread term in
  their fair estimator:

  1. By averaging absolute differences of all pairs of distinct ensemble
  members. This is O(M^2) in compute and memory, but easy to parallelize and
  hence generally cheaper for small-to-medium-sized ensembles. This is
  CRPS_{Fair} in their paper, and is the default implementation here.

  2. By sorting the ensemble members and using their ranks. This is O(M log M)
  and will be more efficient for sufficiently large ensembles. This is
  CRPS_{PWM} in their paper. It can be enabled by setting use_sort=True.

  References:

  - [Gneiting & Raftery, 2012], Strictly Proper Scoring Rules, Prediction, and
    Estimation
  - [Zamo & Naveau, 2018], Estimation of the Continuous Ranked Probability Score
    with Limited Information and Applications to Ensemble Weather Forecasts.
  """

  def __init__(
      self,
      ensemble_dim: str = 'number',
      use_sort: bool = False,
      fair: bool = True,
      skipna_ensemble: bool = False,
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      use_sort: If True, use the sorted-rank method for computing the fair
        estimate of CRPS. This may be more efficient for large ensembles, see
        class docstring for more details. Default: False.
      fair: If True, use the fair estimate of CRPS. If False, use the
        conventional estimate. Default: True.
      skipna_ensemble: If True, any NaN values are treated as missing ensemble
        members. The metric is computed using an ensemble size corresponding to
        the number of non-NaN values along the ensemble_dim, which may vary by
        position along any other dims. When fewer than two ensemble members are
        present along the ensemble_dim, the metric is computed but will be NaN.
    """
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._fair = fair
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            use_sort=self._use_sort,
            fair=self._fair,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['CRPSSkill'] - 0.5 * statistic_values['CRPSSpread']


class CRPSEnsembleDistance(base.PerVariableMetric):
  """Unbiased CRPS distance between forecast and targets.

  Given ground truth scalar random variable Y, and two iid predictions X, X`,
  the Continuously Ranked Probability Score distance is defined as
    CRPS = E|X - Y| - 0.5 * E|X - X`| - 0.5 * E|Y - Y`|,
  where `E` is mathematical expectation over X and Y, and | ⋅ | is the absolute
  value. This version of CRPS has a unique minimum, equal to zero, when X is
  distributed the same as Y. This uniqueness holds only for scalar variates. The
  multi-dimensional generalization is the (squared) Energy Distance. See
  [Szekely and Rizzo 2013].

  Using this Metric requires that both forecasts and truth have an ensemble_dim,
  and ensemble size >= 2. The ensemble sizes in forecasts and truth may be
  different.

  [Zamo & Naveau, 2018] derive two equivalent ways to compute the spread term in
  their fair estimator:

  1. By averaging absolute differences of all pairs of distinct ensemble
  members. This is O(M^2) in compute and memory, but easy to parallelize and
  hence generally cheaper for small-to-medium-sized ensembles. This is
  CRPS_{Fair} in their paper, and is the default implementation here.

  2. By sorting the ensemble members and using their ranks. This is O(M log M)
  and will be more efficient for sufficiently large ensembles. This is
  CRPS_{PWM} in their paper. It can be enabled by setting use_sort=True.

  References:

  - [Szekely and Rizzo 2013], Energy statistics: statistics based on
    distances.
  - [Gneiting & Raftery, 2012], Strictly Proper Scoring Rules, Prediction, and
    Estimation
  - [Zamo & Naveau, 2018], Estimation of the Continuous Ranked Probability Score
    with Limited Information and Applications to Ensemble Weather Forecasts.
  """

  def __init__(
      self,
      ensemble_dim: str = 'number',
      use_sort: bool = False,
      fair: bool = True,
      skipna_ensemble: bool = False,
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      use_sort: If True, use the sorted-rank method for computing the spread
        estimates. This may be more efficient for large ensembles, see class
        docstring for more details. Note that this is not used for the skill
        estimate though, which this is O(M*N) in the two ensemble sizes.
        Default: False.
      fair: If True, use the fair estimate of CRPS. If False, use the
        conventional estimate. Default: True.
      skipna_ensemble: If True, skip NaN values when computing the ensemble MAE
        and MAD. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._fair = fair
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(ensemble_dim=self._ensemble_dim),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            use_sort=self._use_sort,
            fair=self._fair,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'CRPSTargetSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            use_sort=self._use_sort,
            fair=self._fair,
            which='targets',
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return (
        statistic_values['CRPSSkill']
        - 0.5 * statistic_values['CRPSSpread']
        - 0.5 * statistic_values['CRPSTargetSpread']
    )


class WassersteinDistance(base.PerVariableStatistic):
  """Computes the 1-Wasserstein distance (Earth Mover's Distance).

  Calculates the distance between the prediction ensemble distribution and the
  target ensemble distribution for each point. Requires the ensemble dimension
  to be present in both predictions and targets.

  Note that unlike the CRPS-based distances (e.g., the squared energy distance),
  there is no "fair" version of the Wasserstein's distance, so it cannot be
  debiased with respect to ensemble size. On average, smaller ensembles will
  tend to have a smaller distance than larger ensembles.
  """

  def __init__(self, ensemble_dim: str = 'number'):
    self._ensemble_dim = ensemble_dim

  @property
  def unique_name(self) -> str:
    return f'WassersteinDistance_{self._ensemble_dim}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    """Computes Wasserstein distance using scipy and xr.apply_ufunc."""
    if self._ensemble_dim not in predictions.dims:
      raise ValueError(
          f'Ensemble dimension {self._ensemble_dim!r} not found in '
          f'predictions: {predictions}'
      )
    if self._ensemble_dim not in targets.dims:
      raise ValueError(
          f'Ensemble dimension {self._ensemble_dim!r} not found in '
          f'targets: {targets}'
      )

    input_core_dims = [[self._ensemble_dim], [self._ensemble_dim]]

    return xr.apply_ufunc(
        scipy.stats.wasserstein_distance,
        predictions,
        targets,
        input_core_dims=input_core_dims,
        exclude_dims={self._ensemble_dim},  # allow different size ensembles
        vectorize=True,  # scipy.stats.wasserstein_distance is not vectorized
        dask='parallelized',
        output_dtypes=[predictions.dtype],
    )


class UnbiasedEnsembleMeanRMSE(base.PerVariableMetric):
  """Square root of the unbiased estimate of the ensemble mean MSE."""

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, NaN values will be ignored along the ensemble
        dimension. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'UnbiasedEnsembleMeanSquaredError': UnbiasedEnsembleMeanSquaredError(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        )
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return xu.sqrt(statistic_values['UnbiasedEnsembleMeanSquaredError'])


def SpreadSkillRatio(**unused_kwargs):  # pylint: disable=invalid-name
  raise ValueError(
      'SpreadSkillRatio is no longer supported as it was not correctly '
      'implemented. Please use UnbiasedSpreadSkillRatio instead and see '
      'the docstring of that class for more details.')


class UnbiasedSpreadSkillRatio(base.PerVariableMetric):
  """Computes a spread-skill ratio based on the unbiased skill estimator.

  Specifically this is the ratio:

  sqrt(unbiased estimate of the mean variance of the predictive distribution) /
  sqrt(unbiased estimate of the MSE of the predictive distribution's mean)

  This has the convenient property that the numerator and denominator are
  equal in expectation for perfect ensemble forecasts, meaning the ratio should
  be close to 1 in this perfect case. (Because of the non-linear division
  operation and square root, the resulting ratio itself is not exactly 1 in
  expectation in the perfect case, but this bias goes away as we average over
  more forecasts.)

  Another way to achieve this property is to apply a correction factor to a
  spread-skill ratio computed using a more standard skill estimator (the MSE of
  the ensemble mean, which is biased). This corrected spread-skill relationship
  is described in [1].
  Here we've decided to standardize on using the unbiased MSE estimator instead,
  because it doesn't require a correction factor, reuses our existing
  implementation of the unbiased MSE estimator, and makes it easier to support
  the case of variable ensemble sizes (via skipna_ensemble) correctly.

  [1] Fortin, V. et al. Why Should Ensemble Spread Match the RMSE of the
  Ensemble Mean? J. Hydrometeorol. 15, 1708-1713 (2014).
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, NaN values will be ignored along the ensemble
        dimension. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'EnsembleVariance': EnsembleVariance(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
        'UnbiasedEnsembleMeanSquaredError': UnbiasedEnsembleMeanSquaredError(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return xu.sqrt(
        statistic_values['EnsembleVariance']
        / statistic_values['UnbiasedEnsembleMeanSquaredError']
    )


class EnsembleRootMeanVariance(base.PerVariableMetric):
  """Square root of the mean ensemble variance.

  Note this is not the same thing as the mean ensemble standard deviation, and
  is generally preferable to it.
  """

  def __init__(
      self, ensemble_dim: str = 'number', skipna_ensemble: bool = False
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      skipna_ensemble: If True, NaN values will be ignored along the ensemble
        dimension. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._skipna_ensemble = skipna_ensemble

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'EnsembleVariance': EnsembleVariance(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      mean_statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    return xu.sqrt(mean_statistic_values['EnsembleVariance'])


def _select_optimal_thresholds(
    values: xr.DataArray,
    optimal_thresholds: xr.DataArray,
    method: str | None = None,
) -> xr.DataArray:
  """Selects the optimal thresholds for each cost/loss ratio."""
  # These are the indices that we can use with a .isel.
  optimal_indices = xr.core.indexing.map_index_queries(
      values,
      dict(threshold=optimal_thresholds),
      method=method,
  ).dim_indexers['threshold']

  # We need some extra work here to support jax arrays for the autodiff
  # functionality used by weatherbenchX.statistical_inference:
  try:
    import jax  # pylint: disable=g-import-not-at-top
  except ImportError:
    jax = None

  if jax and isinstance(values.data, jax.Array):
    # This is equivalent to the .isel below, but for the vectorized case
    # when extra dimensions are present, we have to implement a
    # jax-compatible alternative to avoid a 'Vectorized indexing is not
    # supported' error from xarray, which doesn't know how to do this for
    # jax arrays.

    # We broadcast the values and indices to have compatible shapes except
    # for their last axis (which will respectively contain thresholds, and
    # indexes into the threshold dimension for each cost/loss ratio).
    values, indices = xr.broadcast(
        values, optimal_indices, exclude=['threshold', 'cost_loss_ratio'])

    # We then apply a one-dimensional multiple-index slice, vmap'd once
    # for each of the aligned leading axes to make it vectorize over these
    # other axes.
    select = lambda arr, i: arr[i]
    for _ in range(indices.ndim - 1):
      select = jax.vmap(select, in_axes=0, out_axes=0)

    return xr.apply_ufunc(
        select, values, indices,
        input_core_dims=[['threshold'], ['cost_loss_ratio']],
        output_core_dims=[['cost_loss_ratio']]
        )
  else:
    # Much simpler:
    result = values.isel(threshold=optimal_indices)
    # This extra coordinate isn't needed after selection, and maintaining it in
    # the jax code too is a bit fiddly, so dropping it for now.
    return result.drop_vars('threshold')


class RelativeEconomicValue(base.Metric):
  """Relative economic value.

  This metric assumes that the targets are a binary and the predictions
  are probabilities between in [0, 1].

  Defaults to 51 cost-loss ratios spaced uniformly on a log scale from 0.005 to
  1, unless overridden by setting cost_loss_ratios directly.

  By default, REV is computed across all possible decision thresholds for a
  given ensemble size. However, it is also possible to specify the desired
  probability thresholds. This is useful if you only want to compute REV for a
  subset of all possible probability thresholds.

  TODO(tomandersson, matthjw): Add a helper method to ensure overridden
  probability thresholds are a subset of the between-each-ensemble-member-count
  thresholds, e.g. by snapping-to-nearest then deduplicating.
  """

  def __init__(
      self,
      *,
      ensemble_size: int | None = None,
      probability_thresholds: np.ndarray | None = None,
      cost_loss_ratios: np.ndarray | None = None,
      optimal_thresholds: xr.DataArray | Mapping[Hashable, xr.DataArray] | None
      = None,
      optimal_thresholds_select_nearest: bool = False,
      statistic_suffix: str | None = None,
  ):
    """Init.

    Args:
      ensemble_size: The size of the ensemble; if specified, we compute
        statistics for binary predictions at all possible probability thresholds
        for the given ensemble size. Specifically the possible empirical
        probabilities for an ensemble of size N are 0/N, 1/N, 2/N, ..., N/N,
        and the thresholds between them are 0.5/N, 1.5/N, ..., (N-0.5)/N.
        Please specify either `ensemble_size` or `probability_thresholds`, but
        not both.
      probability_thresholds: If specified, we compute statistics for binary
        predictions at only these specific probability thresholds. This can be
        useful when dealing with large ensembles where we don't need to evaluate
        every possible threshold. Please specify either `ensemble_size` or
        `probability_thresholds`, but not both.
      cost_loss_ratios: The cost/loss ratios to compute REV for. Default:
        geometric progression from 0.005 to 1, not including 1.
      optimal_thresholds: Optionally specifies optimal probability thresholds
        (out of those specified by `probability_thresholds` or `ensemble_size`)
        to use for each of the specified `cost_lost_ratios`.
        Thresholds can be specified as a DataArray with a 'cost_loss_ratio'
        dimension whose coordinate matches the `cost_loss_ratios` argument.
        It may also have dimensions aligning with other dimensions present in
        the statistics, where an optimal threshold has been selected separately
        for each value along those dimensions (e.g. lead_time).
        Thresholds can also be specified as a mapping from variable name to
        DataArray, where different thresholds have been selected for each
        variable.
        If specified, we will evaluate REV only at these optimal thresholds,
        and there will be no `threshold` dimension in the result.
        If not specified, REV is computed for all probability thresholds and all
        cost/loss ratios, and there will be a `threshold` dimension in the
        result.
      optimal_thresholds_select_nearest: Only relevant if optimal_thresholds are
        specified. If True, will select the nearest available thresholds to the
        optimal_thresholds specified. If False (the default), only exact matches
        will be allowed.
      statistic_suffix: Optional suffix to append to the statistic name. If a
        custom set of `probability_thresholds` is specified, this must be
        specified as well and should identify the set of thresholds used.
    """

    if ensemble_size is None and probability_thresholds is None:
      raise ValueError(
          'Either ensemble_size or probability_thresholds must be specified.'
      )
    if probability_thresholds is not None and ensemble_size is not None:
      raise ValueError(
          'Only one of ensemble_size or probability_thresholds must be'
          ' specified.'
      )
    if probability_thresholds is not None and statistic_suffix is None:
      raise ValueError(
          'If probability_thresholds is specified, statistic_suffix must be'
          ' specified.'
      )

    if cost_loss_ratios is None:
      cost_loss_ratios = np.geomspace(0.005, 1, 51)[:-1]

    self._cost_loss_ratio = xr.DataArray(
        cost_loss_ratios,
        dims=['cost_loss_ratio'],
        coords={'cost_loss_ratio': cost_loss_ratios},
    )

    self._thresholds = probability_thresholds
    if self._thresholds is None:
      self._thresholds = (np.arange(ensemble_size) + 0.5) / ensemble_size
      if statistic_suffix is None:
        statistic_suffix = 'all_thresholds_for_ensemble_size'
    if not np.all(self._thresholds >= 0.0) or not np.all(
        self._thresholds <= 1.0
    ):
      raise ValueError(
          'Probability thresholds must be in [0, 1], got'
          f' {self._thresholds=}.'
      )

    self._unique_name_suffix = statistic_suffix or ''

    if optimal_thresholds is not None:
      if isinstance(optimal_thresholds, Mapping):
        data_vars = optimal_thresholds.values()
      else:
        data_vars = [optimal_thresholds]
      for var in data_vars:
        if 'cost_loss_ratio' not in var.dims:
          raise ValueError(
              'optimal_thresholds must have "cost_loss_ratio" dimensions.'
          )
        if not var.coords['cost_loss_ratio'].equals(self._cost_loss_ratio):
          raise ValueError(
              'optimal_thresholds must have cost_loss_ratio coordinates with '
              'the same values as the cost_loss_ratios argument.'
          )

    self._optimal_thresholds = optimal_thresholds
    self._optimal_thresholds_select_nearest = optimal_thresholds_select_nearest

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    binarize = wrappers.ContinuousToBinary(
        which='predictions',
        threshold_value=self._thresholds,
        threshold_dim='threshold',
        unique_name_suffix=self._unique_name_suffix,
    )

    return {
        'TruePositives': wrappers.WrappedStatistic(
            categorical.TruePositives(), binarize
        ),
        'TrueNegatives': wrappers.WrappedStatistic(
            categorical.TrueNegatives(), binarize
        ),
        'FalsePositives': wrappers.WrappedStatistic(
            categorical.FalsePositives(), binarize
        ),
        'FalseNegatives': wrappers.WrappedStatistic(
            categorical.FalseNegatives(), binarize
        ),
    }

  def _add_constant_threshold_results(
      self, tp: xr.DataArray, fp: xr.DataArray, fn: xr.DataArray
  ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    base_rate = tp.isel(threshold=0, drop=True) + fn.isel(
        threshold=0, drop=True
    )
    zero = xr.full_like(base_rate, 0.0)

    def at(x, threshold):
      return x.expand_dims(threshold=[threshold])

    # At probability threshold 0: always positive. fn = 0, tp = base_rate,
    # fp = 1-base_rate. At probability threshold 1: predict positive if p > 1,
    # so always negative. fn = base_rate, tp = 0, fp = 0.
    tp = xr.concat([at(base_rate, 0.0), tp, at(zero, 1.0)], dim='threshold')
    fp = xr.concat(
        [at(1.0 - base_rate, 0.0), fp, at(zero, 1.0)], dim='threshold'
    )
    fn = xr.concat([at(zero, 0.0), fn, at(base_rate, 1.0)], dim='threshold')
    return tp, fp, fn

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    # Like base.PerVariableMetric.values_from_mean_statistics, but passes the
    # name of each variable through as well. TODO(matthjw): Consider refactoring
    # PerVariableMetrics and all subclasses to take this extra var_name
    # argument.
    common_variables = set.intersection(
        *[set(statistic_values[s]) for s in self.statistics])
    values = {}
    for var_name in common_variables:
      stats_per_variable = {s: statistic_values[s][var_name]
                            for s in self.statistics}
      values[var_name] = self._values_from_mean_statistics_per_variable(
          stats_per_variable, var_name
      )
    return values

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
      var_name: str,
  ) -> xr.DataArray:
    """Computes REV from confusion matrices for all c/l ratios & thresholds.

    Note that one still needs to choose which threshold(s) from which to report
    REV. This typically either involves taking the max REV over all thresholds
    for each cost/loss ratio (if reporting "recalibrating" REV), or setting the
    threshold equal to (the closest) threshold to the considered cost/loss
    ratio.

    TODO(pricei): Add an option to specify the desired threshold(s) directly,
    for use when writing out the metrics for all thresholds and all cost/loss
    ratios gets too expensive.

    Args:
      statistic_values: The confusion matrices components for all thresholds.
      var_name: The name of the variable to compute REV for.

    Returns:
      The REV values. If optimal_thresholds is None, REV is computed for all
      thresholds and cost/loss ratios, with dimensions for both in the result.
      If optimal_thresholds is specified, REV is computed only for the optimal
      thresholds for each cost/loss ratio, with a cost_loss_ratio dimension but
      no threshold dimension.
    """

    tp = statistic_values['TruePositives']
    fp = statistic_values['FalsePositives']
    fn = statistic_values['FalseNegatives']
    tp, fp, fn = self._add_constant_threshold_results(tp, fp, fn)

    if self._optimal_thresholds is not None:
      if isinstance(self._optimal_thresholds, Mapping):
        optimal_thresholds = self._optimal_thresholds[var_name]
      else:
        optimal_thresholds = self._optimal_thresholds
      method = 'nearest' if self._optimal_thresholds_select_nearest else None
      tp = _select_optimal_thresholds(tp, optimal_thresholds, method)
      fp = _select_optimal_thresholds(fp, optimal_thresholds, method)
      fn = _select_optimal_thresholds(fn, optimal_thresholds, method)

    pred_cost = self._cost_loss_ratio * (tp + fp) + fn
    perf_cost = self._cost_loss_ratio * (tp + fn)
    clim_cost = xu.minimum(self._cost_loss_ratio, tp + fn)
    return (clim_cost - pred_cost) / (clim_cost - perf_cost)


class RankHistogram(base.PerVariableStatistic):
  r"""Rank histogram of an ensemble forecast.

  This produces a histogram of weighted counts of the rank (position in a sorted
  list) of the target relative to the ensemble of predictions, so there are
  num_samples+1 bins.

  An additional 'rank' dimension is added, along which these counts are stored.
  """

  def __init__(
      self,
      *,
      ensemble_dim: str = 'number',
  ):
    self._ensemble_dim = ensemble_dim

  @property
  def unique_name(self) -> str:
    return f'RankHistogram_{self._ensemble_dim}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    num_bins = predictions.sizes[self._ensemble_dim] + 1
    ranks = (predictions < targets).astype(int).sum(
        self._ensemble_dim, skipna=False)

    categories = xr.DataArray(np.arange(num_bins), dims=['rank'])
    categories.coords['rank'] = categories

    # Transform to floating point for the subsequent aggregation.
    return (ranks == categories).astype(np.float32)

