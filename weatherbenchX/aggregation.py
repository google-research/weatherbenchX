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

import collections
import dataclasses
from typing import Any, Callable, Collection, Hashable, Iterable, Mapping, Sequence

from weatherbenchX import binning
from weatherbenchX import weighting
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


def combining_sum(data_arrays: Sequence[xr.DataArray]) -> xr.DataArray:
  """Sum of DataArrays, with zero-filled outer join for non-aligned coordinates.

  Outer join means that no coordinates will be dropped during the sum, like they
  would be under the default inner join behavior for arithmetic ops. This is
  important when dealing with sparse data for which coordinates may not be
  exactly the same for each chunk.

  Zero-filling makes sense here as zero is the identity for addition; the
  NaN-filling behaviour of arithmetic_join='outer' doesn't work here and we
  can't replace NaNs with zero afterwards as the NaNs propagate in the sum.

  If your DataArrays have non-overlapping coordinates, this will have the effect
  of combining or concatenating them, similarly to `xr.combine_by_coords`,
  but will be inefficient and memory-hungry since it will zero-fill every input
  to the size of the full result before summing them, which is quadratic in the
  number of arrays being combined in the non-overlapping case. Please prefer to
  use a more a more tailored approach like `xr.combine_by_coords` in cases like
  this, an example of this is in beam_pipeline.py.

  Args:
    data_arrays: Sequence of DataArrays to sum.

  Returns:
    The results as a DataArray.
  """
  return sum(xr.align(*data_arrays, join='outer', fill_value=0))


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
      cls, aggregation_states: Iterable['AggregationState']
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
        lambda *a: combining_sum(a),
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

  def sum_along_dims(self, dims: Collection[str]) -> 'AggregationState':
    """Further sum aggregated statistics over the given dimensions.

    This can be useful in cases where we want a two-stage reduction, e.g.
    when computing confidence intervals we want to postpone a final reduction
    over separate experimental units (typically initialization times) but
    perform all other reductions (e.g. over lat/lon) beforehand.

    The first reduction would generally be done via an Aggregator (perhaps
    followed by further aggregation over multiple batches/chunks). Further
    reduction can then be done here. Note we continue to use the weights and
    mask specified in the original aggregation, by further summing both the
    sum_weighted_statistics and sum_weights. The result should be the same as
    if the specified `dims` had been reduced over in the original aggregation.

    Args:
      dims: Dimensions to sum over.

    Returns:
      A new AggregationState with the given dimensions summed over.
    """
    if self.sum_weighted_statistics is None:
      # Further reduction of a generic zero state is also a zero state.
      return self
    else:
      return self.map(lambda x: x.sum(dims, skipna=False))

  def dot(
      self, *arrays: xr.DataArray, dim: Hashable | Sequence[Hashable]
      ) -> 'AggregationState':
    """Dot product of all stats with other arrays, over the given dimensions."""
    return self.map(lambda x: xr.dot(x, *arrays, dim=dim))

  @classmethod
  def map_multi(
      cls,
      func: Callable[..., xr.DataArray],
      *agg_states: 'AggregationState',
  ) -> 'AggregationState':
    """Like `map` but takes a multi-arg func and multiple AggregationStates."""
    if any(a.sum_weighted_statistics is None for a in agg_states):
      raise ValueError('Cannot map a zero AggregationState.')
    sum_weighted_statistics = xarray_tree.map_structure(
        func, *[a.sum_weighted_statistics for a in agg_states])
    sum_weights = xarray_tree.map_structure(
        func, *[a.sum_weights for a in agg_states])
    return AggregationState(sum_weighted_statistics, sum_weights)

  def map(
      self, func: Callable[[xr.DataArray], xr.DataArray]) -> 'AggregationState':
    """Map a function over the DataArrays in the AggregationState."""
    return self.map_multi(func, self)

  def to_data_tree(self) -> xr.DataTree:
    """Returns a DataTree representation of the AggregationState."""
    if isinstance(self.sum_weighted_statistics, xr.DataArray):
      return xr.DataTree(dataset=xr.Dataset({
          'sum_weighted_statistics': self.sum_weighted_statistics,
          'sum_weights': self.sum_weights}))
    elif isinstance(self.sum_weighted_statistics, Mapping):
      return xr.DataTree(children={
          k: AggregationState(self.sum_weighted_statistics[k],
                              self.sum_weights[k]).to_data_tree()
          for k in self.sum_weighted_statistics.keys()})
    else:
      raise TypeError('Bad type for AggregationState.sum_weighted_statistics.')

  @classmethod
  def from_data_tree(cls, data_tree: xr.DataTree) -> 'AggregationState':
    """Returns an AggregationState from a DataTree representation."""
    if data_tree.dataset:
      return cls(
          data_tree.dataset['sum_weighted_statistics'].rename(data_tree.name),
          data_tree.dataset['sum_weights'].rename(data_tree.name))
    else:
      children = {
          k: cls.from_data_tree(v) for k, v in data_tree.children.items()}
      return cls(
          sum_weighted_statistics={
              k: v.sum_weighted_statistics for k, v in children.items()},
          sum_weights={
              k: v.sum_weights for k, v in children.items()},
      )

  def to_dataset(self, separator='#') -> xr.Dataset:
    """A Dataset representation of the AggregationState.

    This won't work (or at least won't round-trip correctly) if DataArrays have
    incompatible coordinates, or if any statistic or variable names contain the
    `separator`. Prefer to_data_tree() where possible.

    Args:
      separator: Separator to use between path components in the variable names
        of the Dataset. '#' is used by default since '.' may occur in
        statistics' unique_names, and '/' is reserved for netCDF groups.

    Returns:
      A Dataset representation of the AggregationState.
    """
    result = {}
    for path, dataset in self.to_data_tree().to_dict().items():
      path = str(path).lstrip('/').replace('/', separator)
      for var_name, data_array in dataset.items():
        result[f'{path}{separator}{var_name}'] = data_array
    return xr.Dataset(result)

  @classmethod
  def from_dataset(
      cls, dataset: xr.Dataset, separator='#') -> 'AggregationState':
    """Returns an AggregationState from a Dataset representation."""
    dataset_dict = collections.defaultdict(xr.Dataset)
    for path, data_array in dataset.items():
      path, var_name = str(path).rsplit(separator, 1)
      path = '/' + path.replace(separator, '/')
      dataset_dict[path][var_name] = data_array
    return cls.from_data_tree(xr.DataTree.from_dict(dataset_dict))


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

    # Some downstream code relies on attrs on statistics being preserved, which
    # xr.dot will not do by default.
    with xr.set_options(keep_attrs=True):
      return xr.dot(stat, *weights, *bin_masks, dim=reduce_dims_set)

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
