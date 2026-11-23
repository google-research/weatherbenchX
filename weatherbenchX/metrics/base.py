# Copyright 2026 Google LLC
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
"""Base metrics class."""

import abc
from collections.abc import Mapping
from typing import Hashable, final
from weatherbenchX import computation
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import interfaces
import xarray as xr

# Backwards-compatibility aliases for interfaces moved to
# weatherbenchX.metrics.interfaces.
Metric = interfaces.Metric
Statistic = interfaces.Statistic


class PerVariableStatistic(Statistic):
  """Abstract base class for statistics that are computed per variable.

  The statistic will be computed independently for each variable that is
  present in both predictions and targets.
  """

  @final
  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Maps computation over all variables."""
    result = {}
    for var_name in predictions.keys():
      if var_name in targets.keys():
        per_var_result = self._compute_per_variable(
            predictions[var_name], targets[var_name])
        if per_var_result is not None:
          result[var_name] = per_var_result
    return result

  @abc.abstractmethod
  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray | None:
    """Computes statistic for a variable, or None if it's not defined."""


class PerVariableMetric(Metric):
  """Abstract base class for metrics that are computed per variable."""

  @final
  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    # Get list of common variables present for all statistics.
    common_variables = set.intersection(
        *[set(statistic_values[s]) for s in self.statistics]
    )
    values = {}
    # Compute values for all common variables.
    for v in common_variables:
      stats_per_variable = {s: statistic_values[s][v] for s in self.statistics}
      values[v] = self._values_from_mean_statistics_per_variable(
          stats_per_variable
      )
    return values

  @abc.abstractmethod
  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Compute metric values for a single variable.

    Args:
      statistic_values: Mapping from your internal statistic names (the keys
        of your self.statistics) to the mean values of the statistics for a
        single specific variable.

    Returns:
      The value of the metric for the given variable.
    """


# Deprecated backwards-compatibility shim. NoOpMetric used to wrap a Statistic
# as a Metric, but this is no longer necessary as Statistic implements Metric
# directly.
NoOpMetric = lambda statistic: statistic


# Backwards-compatibility aliases for helpers moved to weatherbenchX.computation
generate_unique_statistics_for_all_metrics = (
    computation.generate_unique_statistics_for_all_metrics
)
compute_unique_statistics_for_all_metrics = (
    computation.compute_unique_statistics_for_all_metrics
)
compute_metric_from_statistics = computation.compute_metric_from_statistics
compute_metrics_from_statistics = computation.compute_metrics_from_statistics


class PerVariableStatisticWithClimatology(Statistic):
  """Base class for per-variable statistics with climatology.

  This class provides a convenient way to compute statistics that are a function
  of both the prediction/target and the climatology. The climatology is aligned
  with the prediction/target based on the prediction's valid_time.

  Subclasses must implement the `_compute_per_variable_with_aligned_climatology`
  method, which takes the predictions, targets, and aligned climatology as
  arguments.
  """

  def __init__(self, climatology: xr.Dataset):
    """Init.

    Args:
      climatology: The climatology dataset.
    """
    self._climatology = climatology

  @final
  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    # Ensure both inputs are dictionaries.
    # This is because sometimes mask coordinates can get lost if xarray_tree
    # combines variables into a Dataset.
    predictions = dict(predictions)
    targets = dict(targets)
    climatology = dict(self._climatology[list(predictions.keys())])
    return xarray_tree.map_structure(
        self._compute_per_variable, predictions, targets, climatology
    )

  @final
  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      climatology: xr.DataArray,
  ) -> xr.DataArray:
    """Compute statistics per variable."""
    # Predictions/targets can either have a single time dimension: valid_time
    if hasattr(predictions, 'valid_time'):
      valid_time = predictions.valid_time
    # Or init and lead time dimensions.
    elif hasattr(predictions, 'init_time') and hasattr(
        predictions, 'lead_time'
    ):
      valid_time = predictions.init_time + predictions.lead_time
    else:
      raise ValueError(
          'Predictions should have either valid_time or init/lead_time'
          ' dimensions.'
      )

    # Climatology either has time, dayofyear or dayofyear/hour dimensions
    if hasattr(climatology, 'time'):
      sel_kwargs = {'time': valid_time}
    else:
      sel_kwargs = {'dayofyear': valid_time.dt.dayofyear}
      if hasattr(climatology, 'hour'):
        sel_kwargs['hour'] = valid_time.dt.hour
    aligned_climatology = climatology.sel(**sel_kwargs).compute()
    return self._compute_per_variable_with_aligned_climatology(
        predictions, targets, aligned_climatology
    )

  @abc.abstractmethod
  def _compute_per_variable_with_aligned_climatology(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
      aligned_climatology: xr.DataArray,
  ) -> xr.DataArray:
    """Computes statistics per variable."""
