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
"""Core Metric and Statistic abstract interfaces."""

import abc
from collections.abc import Mapping
from typing import Hashable, final
import xarray as xr


class Metric(abc.ABC):
  """Abstract base class for metrics.

  A `Metric` is defined by specifying:

  * One or more `Statistic`s, which are functions of prediction/target pairs.
    These are specified by implementing the `statistics` property, but are
    implemented separately to the `Metric` to allow them to be reused across
    multiple `Metric`s.
  * A function to compute the metric's final value from (weighted) *means* of
    the statistics, computed in aggregate over multiple prediction/target pairs.
    This is specified by implementing `values_from_mean_statistics`.

  As an example, the `RMSE` metric is defined by specifying the
  `SquaredError` statistic, which returns squared errors of prediction/target
  pairs, and a function which takes the square root of the mean of the
  `SquaredError` statistic.

  The form of weighted mean(s) used to aggregate the statistics is not
  determined by the Metric and can be chosen independently, for example to
  achieve different types of disaggregation and weighting. See
  `aggregation.Aggregator` for details.

  `Metric`s computed for each variable independently should be implemented by
  subclassing `PerVariableMetric`.
  """

  @property
  @abc.abstractmethod
  def statistics(self) -> Mapping[str, 'Statistic']:
    """Statistics whose mean values are required to compute the metric.

    The keys of this Mapping are internal names for the statistics which will
    be used to pass the mean values of the requested statistics to you in your
    `values_from_mean_statistics` method. They are not required to be unique
    outside of a specific Metric instance; externally the .unique_name of the
    Statistic will be used instead, the internal names can be chosen to be more
    convenient for the Metric.
    """

  @abc.abstractmethod
  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Computes metrics from mean statistics, given by their internal names.

    Args:
      statistic_values: Mapping from your internal statistic names (the keys
        of your self.statistics) to the mean values of the corresponding
        statistics. These mean values consist of a mapping from variable
        name to a DataArray of values.

    Returns:
      A Mapping from variable name to metric value DataArray. As with
      statistics, the variable names don't have to correspond to variables in
      the original predictions and targets, although they should where it makes
      sense. If your Metric is defined on a per-variable basis, consider
      subclassing PerVariableMetric.
    """


class Statistic(Metric):
  """Abstract base class for statistics.

  Statistics are functions of a pair of predictions/targets chunks, which are
  intended to be aggregated by taking a (potentially weighted) mean over
  multiple prediction/target pairs, and then used in the computation of a
  Metric.

  A Statistic can be used in two ways:
  * It can be used directly as a Metric, since it implements the Metric
    interface itself by passing through the mean of the statistic's values.
  * One or more statistics can be wrapped as a Metric which performs some
    additional computation (via `values_from_mean_statistics`) on the mean
    statistics.

  The incoming predictions/targets chunks can either be a dictionary of
  DataArrays or a Dataset.

  For univariate metrics, a PerVariableStatistic should be implemented.
  Multivariate metrics have access to all variables. The output should also be a
  Mapping from str to xr.DataArray. In other words, the DataArray has to be
  named.

  Statistics are required to assign their own `unique_name`, which is used to
  deduplicate the computation of statistics that are used by multiple metrics.
  Any additional parameters of the statistic which affect the result of the
  computation should be captured in self.unique_name.

  Statistics should preserve dimensions that are a) required to compute binnings
  or weights on and b) over which the (weighted) mean is computed. These will
  typically be the time dimensions (if chunking is done in time) and/or the
  spatial/observation dimensions (if these are needed for binning or weighting).
  Other dimensions can be reduced.

  A note on mask propagation:

  Predictions and targets received by the `compute` method may have a special
  boolean 'mask' coordinate indicating which locations are valid, along some
  subset of their dimensions. This coordinate will be consistent between
  predictions and targets if present at all. Where present, it is the
  responsibility of the statistic to ensure that this mask is propagated
  correctly onto its return value, as it will be used during aggregation of the
  statistic. In many simple cases xarray's automatic coordinate propagation will
  be sufficient, but if your computation involves interactions between multiple
  values that may be masked differently, then you may need to manually compute
  an appropriate mask coordinate for the output.
  """

  @property
  def unique_name(self) -> str:
    """Unique name of the statistic.

    When computing multiple metrics, statistics with the same unique_name will
    be assumed to be duplicates and computed only once, hence it is important
    that this is unique across all statistics that are likely to be used
    together.

    Defaults to class name. Remember to change to a unique identifier in case
    the statistic has additional parameters which affect the result of the
    computation.
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
      statistic: Corresponding statistic, as a mapping of variable name to
        DataArray.
        For statistics whose values correspond to specific variables in the
        predictions and targets, the variable names should be the same as the
        relevant keys in the predictions and targets, and you should consider
        subclassing from PerVariableStatistic if your metric can be computed
        one variable at a time in a generic way.
        For statistics whose values don't correspond to specific variables in
        the predictions and targets, you'll need to make up new variable name(s)
        to use here.
    """

  # Trivial implementation of the Metric interface as a Metric which outputs
  # just the mean of the statistic:

  @final
  @property
  def statistics(self) -> Mapping[str, 'Statistic']:
    return {'self': self}

  @final
  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return statistic_values['self']
