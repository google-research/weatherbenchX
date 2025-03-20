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

from typing import Hashable, Mapping
import numpy as np
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
    return np.abs(predictions - targets).mean(self._ensemble_dim, skipna=False)


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
  """Fair sample-based estimate of the spread measure used in CRPS, E|X - X`|.

  (This is also referred to in places as Mean Absolute Difference.)

  See the docstring for CRPSEnsemble for more details on what 'fair' means
  and the two different options (use_sort=True vs False) for computing the
  fair estimate.
  """

  def __init__(
      self,
      ensemble_dim: str = 'number',
      use_sort: bool = False,
  ):
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort

  @property
  def unique_name(self) -> str:
    return f'CRPSSpread_{self._ensemble_dim}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    n_ensemble = predictions.sizes[self._ensemble_dim]
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
      rank = _rank_da(predictions, self._ensemble_dim)
      return (
          2
          * (
              ((2 * rank - n_ensemble - 1) * predictions).mean(
                  self._ensemble_dim, skipna=False
              )
          )
          / (n_ensemble - 1)
      )
    else:
      second_ensemble_dim = 'ensemble_dim_2'
      predictions_2 = predictions.rename(
          {self._ensemble_dim: second_ensemble_dim})
      return abs(predictions - predictions_2).sum(
          dim=(self._ensemble_dim, second_ensemble_dim),
          skipna=False) / (n_ensemble * (n_ensemble - 1))


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
      predictions_bias = predictions_var / predictions.sizes[self._ensemble_dim]
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
      targets_bias = targets_var / targets.sizes[self._ensemble_dim]
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

  [Zamo & Naveau, 2018] derive two equivalent ways to compute the same
  fair estimator:

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
      self, ensemble_dim: str = 'number', use_sort: bool = False,
  ):
    """Init.

    Args:
      ensemble_dim: Name of the ensemble dimension. Default: 'number'.
      use_sort: If True, use the sorted-rank method for computing the fair
        estimate of CRPS. This may be more efficient for large ensembles,
        see class docstring for more details. Default: False.
    """
    self._ensemble_dim = ensemble_dim
    self._use_sort = use_sort

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'CRPSSkill': CRPSSkill(ensemble_dim=self._ensemble_dim),
        'CRPSSpread': CRPSSpread(
            ensemble_dim=self._ensemble_dim, use_sort=self._use_sort),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['CRPSSkill'] - 0.5 * statistic_values['CRPSSpread']


class UnbiasedEnsembleMeanRMSE(base.PerVariableMetric):
  """Unbiased estimate of the ensemble mean RMSE."""

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
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'UnbiasedEnsembleMeanSquaredError': UnbiasedEnsembleMeanSquaredError(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        )
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
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
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
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
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(
        statistic_values['EnsembleMeanSquaredError']
        / statistic_values['EnsembleVariance']
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
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
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
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(
        statistic_values['UnbiasedEnsembleMeanSquaredError']
        / statistic_values['EnsembleVariance']
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
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    return {
        'EnsembleVariance': EnsembleVariance(
            ensemble_dim=self._ensemble_dim,
            skipna_ensemble=self._skipna_ensemble,
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self, mean_statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    return np.sqrt(mean_statistic_values['EnsembleVariance'])
