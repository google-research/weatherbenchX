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
"""Base metrics class."""

import abc
from typing import Hashable, Mapping
from weatherbenchX import xarray_tree
import xarray as xr


class Statistic(abc.ABC):
  """Abstract base class for statistics.

  Statistics are computed for a pair of predictions/targets chunks. The
  resulting statistics chunks will then be averaged (potentially weighted)
  across chunks.

  The incoming predictions/targets chunks can either be a dictionary of
  DataArrays or a Dataset.

  For univariate metrics, a PerVariableStatistic should be implemented.
  Multivariate metrics have access to all variables. The output should also be a
  Mapping from str to xr.DataArray. In other words, the DataArray has to be
  named.

  Statistics are required to assign their own unique name. In the case of
  additional parameters, these should be in self.unique_name.

  Statistics should preserve dimensions that are a) required to compute binnings
  or weights on and b) over which the (weighted) mean is computed. These will
  typically be the time dimensions (if chunking is done in time) and/or the
  spatial/observation dimensions (if these are needed for binning or weighting).
  Other dimensions can be reduced.

  Typically, one or more statistics are assiciated with a metric which then
  uses the averaged statistic(s) to compute the final metric values.
  """

  @property
  def unique_name(self) -> str:
    """Name of the statistic.

    Defaults to class name. Remember to change to a unique identifier in case
    statistic has additional parameters.
    """
    return type(self).__name__

  @abc.abstractmethod
  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes statistics per predictions/targets chunk.

    Args:
      predictions: Xarray Dataset or DataArray.
      targets: Xarray Dataset or DataArray.

    Returns:
      statistic: Corresponding statistic
    """


class PerVariableStatistic(Statistic):
  """Abstract base class for statistics that are computed per variable."""

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Maps computation over all variables."""
    # Ensure both inputs are dictionaries.
    # This is because sometimes mask coordinates can get lost if xarray_tree
    # combines variables into a Dataset.
    predictions = dict(predictions)
    targets = dict(targets)
    return xarray_tree.map_structure(
        self._compute_per_variable, predictions, targets
    )

  @abc.abstractmethod
  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    """Computes statistics per variable."""


class Metric(abc.ABC):
  """Abstract base class for metrics.

  Metrics define one or more statistics to be computed. Their names can be
  chosen freely inside the metric. Before the computation of the metrics from
  the aggregated statistics, the unique statistic names will be renamed to the
  internal names. Metrics computed for each variable independently should
  be implemented as PerVariableMetric classes.
  """

  @property
  @abc.abstractmethod
  def statistics(self) -> Mapping[str, Statistic]:
    """Dictionary of required statistics."""

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes metrics from averaged statistics."""
    # Rename statistics from unique to internal names.
    statistic_values = {
        k: statistic_values[v.unique_name] for k, v in self.statistics.items()
    }
    return self._values_from_mean_statistics_with_internal_names(
        statistic_values
    )

  @abc.abstractmethod
  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes metric values from statistics after renaming to internal names."""


class PerVariableMetric(Metric):
  """Abstract base class for metrics that are computed per variable."""

  def _values_from_mean_statistics_with_internal_names(
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
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Compute metric values for a single variable."""


class NoOpMetric(PerVariableMetric):
  """General metric wrapper that simply returns the mean statistics."""

  def __init__(self, statistic: Statistic):
    """Init.

    Args:
      statistic: Statistic to be wrapped.
    """
    self._statistic = statistic

  @property
  def statistics(self) -> Mapping[str, Statistic]:
    return {'statistic': self._statistic}

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[Hashable, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return statistic_values['statistic']


def compute_unique_statistics_for_all_metrics(
    metrics: Mapping[str, Metric],
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
) -> Mapping[str, Mapping[Hashable, xr.DataArray]]:
  """Computes unique statistics for all metrics.

  Args:
    metrics: Dictionary of metrics instances.
    predictions: Xarray Dataset or dictionary of DataArrays.
    targets: Xarray Dataset or dictionary of DataArrays.

  Returns:
    statistic_values: Unique statistics computed for each input element. If
      inputs are Datasets, returns a dict of statistic_name to statistic
      Dataset; if inputs are dictionaries, returns a nested dictionary of
      statistic_name to variable to statistic DataArrays.
  """
  unique_statistics = {}
  for m in metrics.values():
    for _, stat in m.statistics.items():
      unique_statistics[stat.unique_name] = stat
  statistic_values = {
      k: stat.compute(predictions, targets)
      for k, stat in unique_statistics.items()
  }
  return statistic_values


def compute_metrics_from_statistics(
    metrics: Mapping[str, Metric],
    statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
) -> Mapping[str, Mapping[Hashable, xr.DataArray]]:
  """Computes metrics from averaged statistics."""
  return {
      metric_name: metric.values_from_mean_statistics(statistic_values)
      for metric_name, metric in metrics.items()
  }


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

    # Climatology either has dayofyear or dayofyear/hour dimensions
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
