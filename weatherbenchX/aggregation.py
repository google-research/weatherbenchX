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
"""Definition of aggregation methods and AggregationState."""

import dataclasses
from typing import Any, Collection, Hashable, Mapping, Sequence

from weatherbenchX import binning
from weatherbenchX import weighting
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


@dataclasses.dataclass
class AggregationState:
  """An object that contains a sum of weighted statistics and a sum of weights.

  Allows for aggregation over multiple chunks before computing a final weighted
  mean.

  Attributes:
    sum_weighted_statistics: Structure containing summed/aggregated statistics,
      as a DataArray or nested dictionary of DataArrays, or None.
    sum_weights: Similar structure containing the corresponding summed weights.
  """

  sum_weighted_statistics: Any
  sum_weights: Any

  @classmethod
  def zero(cls) -> 'AggregationState':
    """An initial/'zero' aggregation state."""
    return cls(sum_weighted_statistics=None, sum_weights=None)

  def __add__(self, other: 'AggregationState') -> 'AggregationState':
    return self.sum([self, other])

  @classmethod
  def sum(
      cls, aggregation_states: list['AggregationState']
  ) -> 'AggregationState':
    """Sum of aggregation states."""
    sum_weighted_statistics_and_sum_weights_tuples = [
        (a.sum_weighted_statistics, a.sum_weights)
        for a in aggregation_states
        if a.sum_weighted_statistics is not None
    ]

    # Sometimes beam does a reduction with only Zero states. In this case, we
    # end up with an empty collection. In these cases, we need to return a zero
    # state.
    if not sum_weighted_statistics_and_sum_weights_tuples:
      return cls.zero()

    # Sum over each element in the nested dictionaries
    sum_weighted_statistics, sum_weights = xarray_tree.map_structure(
        lambda *a: sum(a),
        *sum_weighted_statistics_and_sum_weights_tuples,
    )

    return cls(sum_weighted_statistics, sum_weights)

  def mean_statistics(self) -> Any:
    """Returns the statistics normalized by their corresponding weights."""

    def normalize(sum_weighted_statistics, sum_weights):
      return sum_weighted_statistics / sum_weights

    return xarray_tree.map_structure(
        normalize, self.sum_weighted_statistics, self.sum_weights
    )

  def metric_values(
      self, metrics: Mapping[str, metrics_base.Metric]
  ) -> xr.Dataset:
    """Returns metrics computed from the normalized statistics.

    This requires sum_weighted_statistics and sum_weights to be nested mappings
    of statistic_name -> variable_name -> DataArray, which is a stronger
    assumption than the rest of this class. (TODO(matthjw): split it off as a
    helper function instead.)

    Args:
      metrics: Dictionary of metric names and instances.

    Returns:
      values: Combined dataset with naming convention <metric>.<variable>
    """

    mean_statistics = self.mean_statistics()
    metric_values = metrics_base.compute_metrics_from_statistics(
        metrics, mean_statistics
    )
    values = xr.Dataset()
    for metric_name in metric_values:
      for var_name in metric_values[metric_name]:
        da = metric_values[metric_name][var_name]
        values[f'{metric_name}.{var_name}'] = da
    return values


@dataclasses.dataclass
class Aggregator:
  """Defines aggregation over set of dataset dimensions.

  Note on NaNs: By default, all reductions are performed with skipna=False,
  meaning that the aggregated statistics will be NaN if any of the input
  statistics are NaN. Currently, there is one awkward use case, where even if
  the input NaNs are outside the binning mask, e.g. if NaNs appear in a
  different region from the binning region, the aggregated statistics will
  still be NaN. Use the masking option to avoid this.

  Attributes:
    reduce_dims: Dimensions to average over. Any variables that don't have these
      dimensions will be filtered out during aggregation.
    bin_by: List of binning instances. All bins will be multiplied.
    weigh_by: List of weighting instance. All weights will be multiplied.
    masked: If True, aggregation will only be performed for non-masked (True on
      the mask) values. This requires a 'mask' coordinate on the statistics
      passed to aggregate_statistics.
    skipna: If True, NaNs will be omitted in the aggregation. This option is not
      recommended, as it won't catch unexpected NaNs.
  """

  reduce_dims: Collection[str]
  bin_by: Sequence[binning.Binning] | None = None
  weigh_by: Sequence[weighting.Weighting] | None = None
  masked: bool = False
  skipna: bool = False

  def aggregation_fn(
      self,
      stat: xr.DataArray,
  ) -> xr.DataArray | None:
    """Returns the aggregation function."""
    # Recall that masked out values have already been set to zero in
    # aggregate_statistics. The logic below has to respect this.

    reduce_dims_set = set(self.reduce_dims)
    eval_unit_dims = set(stat.dims)
    if not reduce_dims_set.issubset(eval_unit_dims):
      # Can't reduce over dims that aren't present as evaluation unit dims.
      return None

    weights = [
        weighting_method.weights(stat)
        for weighting_method in self.weigh_by or []
    ]

    bin_dim_names = {binning.bin_dim_name for binning in self.bin_by or []}
    if len(bin_dim_names) != len(self.bin_by or []):
      raise ValueError('Bin dimension names must be unique.')

    bin_masks = []
    for binning_method in self.bin_by or []:
      bin_mask = binning_method.create_bin_mask(stat)
      # bin_masks_dims are all of the dims the mask operate with on the input
      # data (e.g. the actual bin dimension does not count).
      bin_masks_dims = set(bin_mask.dims) - {binning_method.bin_dim_name}
      if bin_masks_dims.issubset(eval_unit_dims):
        bin_masks.append(bin_mask)
      else:
        # Can't bin based on dims that aren't present as evaluation unit dims:
        return None

    return xr.dot(stat, *weights, *bin_masks, dims=reduce_dims_set)

  def aggregate_stat_var(self, stat: xr.DataArray) -> AggregationState | None:
    """Aggregate one statistic DataArray for one variable."""
    if self.masked and hasattr(stat, 'mask'):
      mask = stat.mask
      if self.skipna:
        mask = mask & ~stat.isnull()

      # Set masked values to Zero for stat and weights, which will therefore
      # be ignored in mean_statistics(). this is equivalent to multiplying by
      # the mask, but avoids NaN * 0 -> NaN in cases where there are NaNs in
      # masked positions. Only for variables with a mask attribute.
      stat = stat.where(mask, 0)

      # We need to broadcast the mask to the same shape as the stat, so that
      # reductions over it behave the same as reductions over the full stat.
      mask = mask.broadcast_like(stat)
    elif self.skipna:
      mask = ~stat.isnull()
      stat = stat.where(mask, 0)
    else:
      mask = xr.ones_like(stat)

    assert mask.sizes == stat.sizes

    sum_weighted_statistics = self.aggregation_fn(stat)
    sum_weights = self.aggregation_fn(mask.astype(stat.dtype))
    if sum_weighted_statistics is None or sum_weights is None:
      return None
    else:
      return AggregationState(sum_weighted_statistics, sum_weights)

  def aggregate_stat_vars(
      self, stats: Mapping[Hashable, xr.DataArray]) -> AggregationState:
    """Aggregate per-variable DataArrays of a single statistic."""
    per_var = {var_name: self.aggregate_stat_var(stat)
               for var_name, stat in stats.items() if stat is not None}
    return AggregationState(
        sum_weighted_statistics={
            var_name: agg_state.sum_weighted_statistics
            for var_name, agg_state in per_var.items()
            if agg_state is not None},
        sum_weights={
            var_name: agg_state.sum_weights
            for var_name, agg_state in per_var.items()
            if agg_state is not None},
    )

  def aggregate_statistics(
      self,
      statistics: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> AggregationState:
    """Aggregate multiple statistics, each defined for multiple variables.

    Args:
      statistics: Full statistics for a batch.

    Returns:
      AggregationState instance with a sum of weighted statistics and a sum of
      weights for the current batch. These can be summed over multiple batches,
      and then used to compute weighted mean statistics, and from these the
      final values of the metrics.
    """
    per_stat = {stat_name: self.aggregate_stat_vars(stats)
                for stat_name, stats in statistics.items()}
    return AggregationState(
        sum_weighted_statistics={
            stat_name: agg_state.sum_weighted_statistics
            for stat_name, agg_state in per_stat.items()},
        sum_weights={
            stat_name: agg_state.sum_weights
            for stat_name, agg_state in per_stat.items()},
    )


def compute_metric_values_for_single_chunk(
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: Aggregator,
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
  """Convenience function to compute metric results for a given predictions/targets pair.

  This is not intended to accumulate over multiple chunks.

  Args:
    metrics: Dictionary of metrics instances.
    aggregator: Aggregator instance.
    predictions: Xarray Dataset or dictionary of DataArrays.
    targets: Xarray Dataset or dictionary of DataArrays.

  Returns:
    results: Xarray Dataset of metric values.
  """
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions, targets
  )
  aggregation_state = aggregator.aggregate_statistics(statistics)
  results = aggregation_state.metric_values(metrics)
  return results
