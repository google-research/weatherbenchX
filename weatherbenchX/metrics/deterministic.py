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
"""Implementation of deterministic metrics and assiciated statistics."""

from typing import Hashable, Mapping, Sequence, Union
import numpy as np
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base
import xarray as xr


### Statistics


class Error(base.PerVariableStatistic):
  """Error between predictions and targets."""

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    error = predictions - targets
    return error


class AbsoluteError(base.PerVariableStatistic):
  """Absolute error between predictions and targets."""

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    error = predictions - targets
    return abs(error)


class SquaredError(base.PerVariableStatistic):
  """Squared error between predictions and targets."""

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    return (predictions - targets) ** 2


class PredictionPassthrough(base.PerVariableStatistic):
  """Simply returns predictions."""

  def __init__(self, copy_nans_from_targets: bool = False):
    """Init.

    Args:
      copy_nans_from_targets: If True, copy any nans from the targets to the
        predictions.
    """
    self._copy_nans_from_targets = copy_nans_from_targets

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    # Make sure potential coordinates from targets are preserved.
    result = predictions + xr.zeros_like(targets)
    if self._copy_nans_from_targets:
      result = result.where(~targets.isnull())
    return result


class TargetPassthrough(base.PerVariableStatistic):
  """Simply returns targets."""

  def __init__(self, copy_nans_from_predictions: bool = False):
    """Init.

    Args:
      copy_nans_from_predictions: If True, copy any nans from the predictions to
        the predictions.
    """
    self._copy_nans_from_predictions = copy_nans_from_predictions

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    # Make sure potential coordinates from predictions are preserved.
    result = targets + xr.zeros_like(predictions)
    if self._copy_nans_from_predictions:
      result = result.where(~predictions.isnull())
    return result


class WindVectorSquaredError(base.Statistic):
  """Computes squared error between two wind components.

  SE = (u_pred - u_target) ** 2 + (v_pred - v_target) ** 2
  """

  def __init__(
      self,
      u_name: Sequence[str],
      v_name: Sequence[str],
      vector_name: Sequence[str],
  ):
    """Init.

    Args:
      u_name: Name of the u wind component, e.g. [`u_component_of_wind`].
      v_name: Name of the v wind component, e.g. [`v_component_of_wind`].
      vector_name: Name to give output variable, e.g. [`wind`].
    """
    self._u_name = u_name
    self._v_name = v_name
    self._vector_name = vector_name
    if not len(self._u_name) == len(self._v_name) == len(self._vector_name):
      raise ValueError(
          'u_name, v_name, and vector_name must have the same length'
      )

  @property
  def unique_name(self) -> str:
    suffix = '_'.join(self._vector_name)
    return 'WindVectorSquaredError_' + suffix

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    out = {}
    for u, v, vector in zip(self._u_name, self._v_name, self._vector_name):
      predictions_u = predictions[u]
      predictions_v = predictions[v]
      targets_u = targets[u]
      targets_v = targets[v]
      se = (predictions_u - targets_u) ** 2 + (predictions_v - targets_v) ** 2
      out[vector] = se
    return out


class SquaredPredictionAnomaly(base.PerVariableStatisticWithClimatology):
  """Computes (predictions - climatology)**2."""

  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    prediction_anom = predictions - aligned_climatology
    return prediction_anom**2


class SquaredTargetAnomaly(base.PerVariableStatisticWithClimatology):
  """Computes (targets - climatology)**2."""

  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    target_anom = targets - aligned_climatology
    return target_anom**2


class AnomalyCovariance(base.PerVariableStatisticWithClimatology):
  """Computes (predictions - climatology) * (targets - climatology)."""

  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    prediction_anom = predictions - aligned_climatology
    target_anom = targets - aligned_climatology
    return prediction_anom * target_anom


### Metrics


# The following metrics are just the mean of a Statistic defined above, and
# so we can just use the Statistic directly as the Metric. We provide
# convenience aliases here, however:

Bias = Error  # Bias is the mean Error.
MAE = AbsoluteError  # MAE is the Mean Absolute Error.
MSE = SquaredError  # MSE is the Mean Squared Error.
PredictionAverage = PredictionPassthrough
TargetAverage = TargetPassthrough


class RMSE(base.PerVariableMetric):
  """Root mean squared error."""

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {'SquaredError': SquaredError()}

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(statistic_values['SquaredError'])


class WindVectorRMSE(base.Metric):
  """Computes vector RMSE between two wind components."""

  def __init__(
      self,
      u_name: Union[str, list[str]],
      v_name: Union[str, list[str]],
      vector_name: Union[str, list[str]],
  ):
    """Init.

    Args can be a single string or a list, in which case the statistic will be
    computed separately for the different elements in the list. For example,
    `u_name=['u_component_of_wind', '10m_u_component_of_wind_10m']`.

    Args:
      u_name: Name of the u wind component, e.g. `u_component_of_wind`.
      v_name: Name of the v wind component, e.g. `v_component_of_wind`.
      vector_name: Name to give output variable, e.g. `wind`.
    """
    self._u_name = [u_name] if isinstance(u_name, str) else u_name
    self._v_name = [v_name] if isinstance(v_name, str) else v_name
    self._vector_name = (
        [vector_name] if isinstance(vector_name, str) else vector_name
    )
    if not len(self._u_name) == len(self._v_name) == len(self._vector_name):
      raise ValueError(
          'u_name, v_name, and vector_name must have the same length'
      )

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'WindVectorSquaredError': WindVectorSquaredError(
            self._u_name, self._v_name, self._vector_name
        )
    }

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return xarray_tree.map_structure(
        np.sqrt, statistic_values['WindVectorSquaredError']
    )


class ACC(base.PerVariableMetric):
  """Anomaly correlation coefficient."""

  def __init__(self, climatology: xr.Dataset):
    self._climatology = climatology

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'SquaredPredictionAnomaly': SquaredPredictionAnomaly(
            climatology=self._climatology
        ),
        'SquaredTargetAnomaly': SquaredTargetAnomaly(
            climatology=self._climatology
        ),
        'AnomalyCovariance': AnomalyCovariance(climatology=self._climatology),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['AnomalyCovariance'] / (
        np.sqrt(statistic_values['SquaredPredictionAnomaly'])
        * np.sqrt(statistic_values['SquaredTargetAnomaly'])
    )


class PredictionActivity(base.PerVariableMetric):
  """Activity in predictions defined as the std dev of the prediction anomalies.

  This is used e.g. by ECMWF: https://arxiv.org/abs/2307.10128
  """

  def __init__(self, climatology: xr.Dataset):
    self._climatology = climatology

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    return {
        'SquaredPredictionAnomaly': SquaredPredictionAnomaly(
            climatology=self._climatology
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return np.sqrt(statistic_values['SquaredPredictionAnomaly'])
