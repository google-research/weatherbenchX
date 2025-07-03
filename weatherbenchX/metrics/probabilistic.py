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

from typing import Mapping
import numpy as np
import scipy.stats
from weatherbenchX.metrics import base
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import wrappers
import xarray as xr

### Statistics


class CRPSSkill(base.PerVariableStatistic):
  """The skill measure associated with CRPS, E|X - Y|."""

  def __init__(self, ensemble_dim: str = 'number'):
    self._ensemble_dim = ensemble_dim

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
    return np.abs(predictions - targets).mean(reduce_dims, skipna=False)


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
  ):
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._which = which
    self._fair = fair

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

    n_ensemble = da.sizes[self._ensemble_dim]
    if n_ensemble < 2:
      raise ValueError('Cannot estimate CRPS spread with n_ensemble < 2.')

    if self._use_sort:
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
          dim=(self._ensemble_dim, second_ensemble_dim), skipna=False
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
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      use_sort: If True, use the sorted-rank method for computing the fair
        estimate of CRPS. This may be more efficient for large ensembles, see
        class docstring for more details. Default: False.
      fair: If True, use the fair estimate of CRPS. If False, use the
        conventional estimate. Default: True.
    """
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._fair = fair

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(ensemble_dim=self._ensemble_dim),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            use_sort=self._use_sort,
            fair=self._fair,
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
    """
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort
    self._fair = fair

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(ensemble_dim=self._ensemble_dim),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim,
            use_sort=self._use_sort,
            fair=self._fair,
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
    return np.sqrt(statistic_values['UnbiasedEnsembleMeanSquaredError'])


class SpreadSkillRatio(base.PerVariableMetric):
  """Computes the (biased) spread-skill ratio.

  The spread skill ratio is defined as the ensemble standard deviation divided
  by the RMSE of the ensemble mean.
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
        'EnsembleMeanSquaredError': wrappers.WrappedStatistic(
            deterministic.SquaredError(),
            wrappers.EnsembleMean(
                which='predictions',
                ensemble_dim=self._ensemble_dim,
                skipna=self._skipna_ensemble,
            ),
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(
        statistic_values['EnsembleVariance']
        / statistic_values['EnsembleMeanSquaredError']
    )


class UnbiasedSpreadSkillRatio(base.PerVariableMetric):
  """Computes the spread-skill ratio based on the unbiased skill estimator.

  This is analogous to the regular spread skill ratio but using the unbiased
  estimator of the ensemble mean squared error. This is useful for estimating
  the spread skill ratio for differing ensemble sizes.

  Note that the ratio and square root are still biased, however, this is
  negligible if the number of time points is large.
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
    return np.sqrt(
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
    return np.sqrt(mean_statistic_values['EnsembleVariance'])
