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
"""Defines the beam pipeline for evaluation."""

from collections.abc import Hashable
import dataclasses
import os
import time
import typing
from typing import Callable, Iterable, Iterator, Literal, Mapping, Never, Optional, Union

from absl import logging
import apache_beam as beam
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import beam_utils
from weatherbenchX import time_chunks
from weatherbenchX.data_loaders import base as data_loaders_base
from weatherbenchX.metrics import base as metrics_base
import xarray as xr
import xarray_beam as xbeam


class LoadPredictionsAndTargets(beam.DoFn):
  """Loads prediction and target chunks."""

  def __init__(
      self,
      predictions_loader: data_loaders_base.DataLoader,
      targets_loader: data_loaders_base.DataLoader,
      setup_fn: Optional[Callable[[], None]] = None,
      ignore_missing_variables: bool = False,
  ):
    """Init.

    Args:
      predictions_loader: The data loader for the predictions.
      targets_loader: The data loader for the targets.
      setup_fn: (Optional) A function to call once per worker.
      ignore_missing_variables: (Optional) If True, filter targets and
        predictions chunks to their common variables, logging a warning for any
        missing variables. If False (default), keep loaded variables as-is.
    """
    self.predictions_loader = predictions_loader
    self.targets_loader = targets_loader
    self.setup_fn = setup_fn
    self.ignore_missing_variables = ignore_missing_variables
    self.is_initialized = False
    self.target_load_time = beam.metrics.Metrics.distribution(
        'LoadPredictionsAndTargets', 'target_load_time'
    )
    self.prediction_load_time = beam.metrics.Metrics.distribution(
        'LoadPredictionsAndTargets', 'prediction_load_time'
    )

  def setup(self):
    # Call this function once per process.
    if self.setup_fn is not None:
      if not self.is_initialized:
        self.setup_fn()
        self.is_initialized = True

  def process(
      self,
      all_inputs: tuple[
          time_chunks.TimeChunkOffsets,
          tuple[np.ndarray, Union[np.ndarray, slice]],
      ],
  ) -> Iterable[
      tuple[
          time_chunks.TimeChunkOffsets,
          tuple[
              Mapping[Hashable, xr.DataArray], Mapping[Hashable, xr.DataArray]
          ],
      ]
  ]:
    """Returns prediction and target chunks for a chunk of init/lead times.

    Args:
      all_inputs: (time_chunk_offsets, (init_times, lead_times))

    Returns:
      (time_chunk_offsets, (predictions_chunk, targets_chunk))
    """
    logging.log_first_n(
        logging.INFO, 'LoadPredictionsAndTargets inputs: %s', 10, all_inputs
    )
    time_chunk_offsets, (init_times, lead_times) = all_inputs

    start_time = time.time()
    targets_chunk = self.targets_loader.load_chunk(init_times, lead_times)
    self.target_load_time.update(
        (time.time() - start_time) * 1000
    )  # In milliseconds because beam counters use longs.

    start_time = time.time()
    predictions_chunk = self.predictions_loader.load_chunk(
        init_times, lead_times, targets_chunk
    )
    self.prediction_load_time.update(
        (time.time() - start_time) * 1000
    )  # In milliseconds because beam counters use longs.

    if self.ignore_missing_variables:
      common_vars = [v for v in targets_chunk.keys() if v in predictions_chunk]
      if not common_vars:
        raise ValueError(
            'No common variables found between targets and predictions.'
            f' Targets: {targets_chunk.keys()}, Predictions:'
            f' {predictions_chunk.keys()}'
        )
      if len(common_vars) < len(targets_chunk) or len(common_vars) < len(
          predictions_chunk
      ):
        missing_in_preds = set(targets_chunk.keys()) - set(
            predictions_chunk.keys()
        )
        missing_in_targets = set(predictions_chunk.keys()) - set(
            targets_chunk.keys()
        )
        if missing_in_preds:
          logging.warning(
              'Targets chunk has variables not present in predictions'
              ' chunk: %s',
              missing_in_preds,
          )
        if missing_in_targets:
          logging.warning(
              'Predictions chunk has variables not present in targets'
              ' chunk: %s',
              missing_in_targets,
          )
        targets_chunk = {v: targets_chunk[v] for v in common_vars}
        predictions_chunk = {v: predictions_chunk[v] for v in common_vars}

    logging.log_first_n(
        logging.INFO,
        'LoadPredictionsAndTargets outputs: %s',
        10,
        (time_chunk_offsets, (predictions_chunk, targets_chunk)),
    )
    return [(time_chunk_offsets, (predictions_chunk, targets_chunk))]


# TODO(matthjw): Consider whether we could reuse xarray_beam.Key here and
# use more of the xarray_beam API to do the aggregation.
@dataclasses.dataclass(frozen=True)
class _AggregationKey:
  """Key under which statistics are aggregated (summed or combine_by_coords)."""

  type: Literal['sum_weighted_statistics', 'sum_weights']
  statistic_name: str
  variable_name: str
  # Offsets for the chunk in the result of the aggregation. Should be None if
  # the relevant dimension is being aggregated over.
  init_time_offset: int | None
  lead_time_offset: int | None
  aggregator_name: str | None = None

  def drop_offsets(self, preserve_lead_time: bool = False) -> '_AggregationKey':
    lead_time_offset = self.lead_time_offset if preserve_lead_time else None
    return dataclasses.replace(
        self, init_time_offset=None, lead_time_offset=lead_time_offset
    )


class ComputeStatisticsAggregateAndPrepareForCombine(beam.DoFn):
  """Computes statistics needed for our metrics, for a chunk of init/lead times.

  Then performs the initial per-chunk aggregation on them using the Aggregator,
  then prepares them for further aggregation by breaking the AggregationState
  up into separate DataArrays for each statistic, variable, type (sum_weights or
  sum_weighted_statistics) and chunk offset, keyed by _AggregationKey.
  """
  _aggregators: Mapping[str | None, aggregation.Aggregator]

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregator: aggregation.Aggregator | Mapping[str, aggregation.Aggregator],
  ):
    self.metrics = metrics
    if isinstance(aggregator, aggregation.Aggregator):
      self._aggregators = {None: aggregator}
    else:
      self._aggregators = aggregator  # pyrefly: ignore[bad-assignment]

  def process(
      self,
      all_inputs: tuple[
          time_chunks.TimeChunkOffsets,
          tuple[
              Mapping[Hashable, xr.DataArray],
              Mapping[Hashable, xr.DataArray],
          ],
      ],
  ) -> Iterator[tuple[_AggregationKey, xr.DataArray]]:
    """Yields statistics for further aggregation.

    Args:
      all_inputs: (time_chunk_offsets, (predictions_chunk, targets_chunk))

    Yields:
      Multiple key/value pairs (aggregation_key, data_array), where the
      aggregation_key identifying the scope for further aggregation.
    """
    logging.log_first_n(
        logging.INFO,
        'ComputeStatisticsAggregateAndPrepareForCombine inputs: %s',
        10,
        all_inputs,
    )
    time_chunk_offsets, (predictions_chunk, targets_chunk) = all_inputs

    # We use a generator below and yield one at a time, to avoid holding all
    # unaggregated statistics in memory all at once in case of large statistics.
    stats_iter = metrics_base.generate_unique_statistics_for_all_metrics(
        self.metrics, predictions_chunk, targets_chunk
    )

    while True:
      try:
        # Compute stats.
        start_time = time.time()
        stat_name, stats = next(stats_iter)

        # Create a short name so that it's more readable in dashboards.
        short_stat_name = (
            stat_name[:30] + '...' if len(stat_name) > 30 else stat_name
        )

        beam.metrics.Metrics.distribution(
            'ComputeStatistics', f'compute_{short_stat_name}'
        ).update(
            (time.time() - start_time) * 1000
        )  # In milliseconds because beam counters use longs.

        for var_name, stat in stats.items():
          for agg_name, aggregator in self._aggregators.items():
            start_time = time.time()
            aggregation_state = aggregator.aggregate_stat_var(stat)
            if aggregation_state is None:
              continue
            beam.metrics.Metrics.distribution(
                'ComputeStatistics',
                f'agg_{agg_name}_{var_name}_{short_stat_name}'
            ).update(
                (time.time() - start_time) * 1000
            )  # In milliseconds because beam counters use longs.
            if 'init_time' in aggregation_state.sum_weighted_statistics.dims:
              init_time_offset = time_chunk_offsets.init_time
            else:
              init_time_offset = None
            if 'lead_time' in aggregation_state.sum_weighted_statistics.dims:
              lead_time_offset = time_chunk_offsets.lead_time
            else:
              lead_time_offset = None
            aggregation_key = _AggregationKey(
                type='sum_weighted_statistics',
                statistic_name=stat_name,
                variable_name=str(var_name),
                init_time_offset=init_time_offset,
                lead_time_offset=lead_time_offset,
                aggregator_name=agg_name,
            )
            yield aggregation_key, aggregation_state.sum_weighted_statistics
            aggregation_key = _AggregationKey(
                type='sum_weights',
                statistic_name=stat_name,
                variable_name=str(var_name),
                init_time_offset=init_time_offset,
                lead_time_offset=lead_time_offset,
                aggregator_name=agg_name,
            )
            yield aggregation_key, aggregation_state.sum_weights
      except StopIteration:
        break


class ConcatPerStatisticPerVariable(beam.PTransform):
  """Concatenates DataArrays on a per-statistic, per-variable basis.

  The DataArrays correspond to chunks along whichever of the {lead_time,
  init_time} dimensions are being preserved in the result. They arrive keyed
  by _AggregationKey.
  """

  def __init__(self, chunk_metrics_by_lead_time: bool = False):
    super().__init__()
    self.chunk_metrics_by_lead_time = chunk_metrics_by_lead_time

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]
  ):

    def drop_offsets_from_key(
        key: _AggregationKey, data_array: xr.DataArray
    ) -> tuple[_AggregationKey, xr.DataArray]:
      return (
          key.drop_offsets(preserve_lead_time=self.chunk_metrics_by_lead_time),
          data_array,
      )

    def combine_data_arrays_by_coords(
        key: _AggregationKey, data_arrays: Iterable[xr.DataArray]
    ) -> tuple[_AggregationKey, xr.DataArray]:

      # To deal with overlapping coordinates to be combined other than init_time
      # and lead_time, we align them here first.
      data_arrays = xr.align(
          *data_arrays,
          join='outer',
          fill_value=0,
          exclude=['init_time', 'lead_time'],
      )
      # combine_by_coords will return a Dataset if there are any names on the
      # input DataArrays, so we remove the names before calling it.
      # We also drop zero-sized arrays since combine_by_coords cannot deal with
      # them.
      data_arrays = [d.rename(None) for d in data_arrays if d.size > 0]
      # Drop non-dimension coordinates that are not present in all arrays,
      # since combine_by_coords cannot handle mismatched coordinates.
      if data_arrays:
        shared_non_dim_coords = set.intersection(
            *[set(d.coords) - set(d.dims) for d in data_arrays]
        )
        data_arrays = [
            d.drop_vars([
                c
                for c in set(d.coords) - set(d.dims)
                if c not in shared_non_dim_coords
            ])
            for d in data_arrays
        ]
      # If all arrays are empty, we need to manually return an empty DataArray,
      # since combine_by_coords will return a Dataset in this case.
      if not data_arrays:
        return key, xr.DataArray()
      return key, xr.combine_by_coords(data_arrays)  # pyrefly: ignore[bad-return]

    return (
        pcoll
        # Drop the chunk offsets from the key, so that we group by statistic
        # name, variable name, type (sum_weighted_statistics or sum_weights),
        # and lead time (if streaming lead times).
        | 'DropOffsetsFromKey' >> beam.MapTuple(drop_offsets_from_key)
        # We use GroupByKey instead of CombinePerKey because the data all needs
        # to be in memory at once to concatenate it, there is no saving from
        # doing this incrementally via a CombineFn.
        | 'GroupByStatAndVariable' >> beam.GroupByKey()
        | 'CombineDataArraysByCoords'
        >> beam.MapTuple(combine_data_arrays_by_coords)
    )


def reconstruct_aggregation_state(
    key_value_pairs: Iterable[tuple[_AggregationKey, xr.DataArray]],
) -> aggregation.AggregationState:
  """Reconstructs an AggregationState from (_AggregationKey, DataArray) pairs.

  Args:
    key_value_pairs: Component DataArrays of the AggregationState keyed by
      _AggregationKey, as generated by
      ComputeStatisticsAggregateAndPrepareForCombine above except that all
      chunks over the lead_time and init_time dimensions have been combined
      before we reach this stage.

  Returns:
    The reconstituted AggregationState containing all statistics and all
    variables.
  """
  sum_weighted_statistics = {}
  sum_weights = {}
  for key, stat in key_value_pairs:
    if key.type == 'sum_weighted_statistics':
      add_to = sum_weighted_statistics
    elif key.type == 'sum_weights':
      add_to = sum_weights
    else:
      assert False
    variables = add_to.setdefault(key.statistic_name, {})
    variables[key.variable_name] = stat
  return aggregation.AggregationState(sum_weighted_statistics, sum_weights)


class ReconstructAggregationState(beam.PTransform):
  """Reconstructs AggregationState from all (_AggregationKey, DataArray)."""

  def __init__(self, chunk_metrics_by_lead_time: bool = False):
    super().__init__()
    self.chunk_metrics_by_lead_time = chunk_metrics_by_lead_time

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]
  ) -> beam.PCollection[tuple[typing.Any, aggregation.AggregationState]]:
    if not self.chunk_metrics_by_lead_time:
      group_key = lambda x: x[0].aggregator_name
    else:
      group_key = lambda x: (x[0].aggregator_name, x[0].lead_time_offset)
    return (
        pcoll
        | 'GroupByAggregatorAndMaybeLeadTime' >> beam.GroupBy(group_key)
        | 'Reconstruct'
        >> beam.MapTuple(
            lambda group_key, xs: (group_key, reconstruct_aggregation_state(xs))
        )
    )


class ComputeMetrics(beam.DoFn):
  """Computes the metrics from the aggregated statistics."""

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      chunk_metrics_by_lead_time: bool = False,
  ):
    self.metrics = metrics
    self.chunk_metrics_by_lead_time = chunk_metrics_by_lead_time

  def process(
      self, element: tuple[typing.Any, aggregation.AggregationState]
  ) -> Iterable[tuple[typing.Any, typing.Any]]:
    """Computes a metrics Dataset from the final AggregationState."""
    key, aggregation_state = element
    logging.log_first_n(
        logging.INFO,
        'ComputeMetrics inputs: %s, %s',
        10,
        key,
        aggregation_state,
    )
    if not self.chunk_metrics_by_lead_time:
      agg_name = key
      yield agg_name, aggregation_state.metric_values(self.metrics)
    else:
      agg_name, lead_time_offset = key
      metrics_ds = aggregation_state.metric_values(self.metrics)
      metrics_ds = _transpose_time_dims_first(metrics_ds)
      chunk_key = xbeam.Key({'lead_time': lead_time_offset})
      yield agg_name, (chunk_key, metrics_ds)


def _resolve_out_path(
    out_path: str | Mapping[str, str],
    agg_name: str | None,
) -> str:
  if isinstance(out_path, str):
    if agg_name is None:
      return out_path
    else:
      base, ext = os.path.splitext(out_path)
      return f'{base}_{agg_name}{ext}'
  else:
    return out_path[agg_name]  # pyrefly: ignore[bad-index]


def write_dataset(
    ds: xr.Dataset,
    target_path: str,
    zarr_chunks: Mapping[str, int] | None = None,
    drop_attrs: bool = True,
) -> None:
  """Atomically writes a dataset to NetCDF or to a Zarr file with chunking.

  Args:
    ds: The dataset to write.
    target_path: The path to write the dataset to.
    zarr_chunks: Optional chunking specification for Zarr output files.
    drop_attrs: Whether to remove attributes that may have been propagated from
      the targets or predictions.

  Raises:
    ValueError: If the target path is not a Zarr or NetCDF file.
  """
  if drop_attrs:
    # Remove attributes that may have been propagated from the targets or
    # predictions.
    ds = ds.drop_attrs(deep=True)

  if target_path.endswith('.zarr'):
    encoding = None
    if zarr_chunks:
      # Use the smaller of the chunk spec and the dimension size.
      encoding = {}
      for var_name, da in ds.data_vars.items():
        encoding[var_name] = {
            'chunks': tuple(
                min(
                    da.sizes[dim],
                    zarr_chunks.get(str(dim), da.sizes[dim]),
                )
                for dim in da.dims
            )
        }
    with beam_utils.atomic_write_dir(target_path) as tmp_output_path:
      ds.to_zarr(tmp_output_path, mode='w', encoding=encoding)
  else:
    beam_utils.atomic_write(
        target_path,
        ds.to_netcdf(),  # pyrefly: ignore[bad-argument-type]
    )


class WriteMetrics(beam.DoFn):
  """Writes the metrics to a NetCDF or Zarr file."""

  def __init__(
      self,
      out_path: str | Mapping[str, str],
      zarr_chunks: Mapping[str, int] | None = None,
  ):
    self.out_path = out_path
    self.zarr_chunks = zarr_chunks

  def process(self, element: tuple[str | None, xr.Dataset]) -> Iterable[Never]:
    agg_name, metrics = element
    logging.log_first_n(
        logging.INFO, 'WriteMetrics inputs: %s, %s', 10, agg_name, metrics
    )
    target_path = _resolve_out_path(self.out_path, agg_name)
    write_dataset(metrics, target_path, self.zarr_chunks)
    return []


class WriteAggregationState(beam.DoFn):
  """Writes the final AggregationState to a NetCDF or Zarr file."""

  def __init__(
      self,
      out_path: str | Mapping[str, str],
      zarr_chunks: Mapping[str, int] | None = None,
  ):
    self.out_path = out_path
    self.zarr_chunks = zarr_chunks

  def process(
      self, element: tuple[str | None, aggregation.AggregationState]
  ) -> Iterable[Never]:
    agg_name, aggregation_state = element
    aggregation_state_ds = aggregation_state.to_dataset()
    target_path = _resolve_out_path(self.out_path, agg_name)
    write_dataset(aggregation_state_ds, target_path, self.zarr_chunks)
    return []


def _get_template_metrics_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    aggregator: aggregation.Aggregator,
    setup_fn: Optional[Callable[[], None]] = None,
    ignore_missing_variables: bool = False,
) -> xr.Dataset:
  """Computes metrics for first chunk to create a template dataset."""
  logging.info('Building metrics template with data from first chunk')
  # TODO(tomandersson, matthjw): This requires computing the first chunk
  # on the controller. If this causes controller memory issues that aren't
  # easily resolved by upsizing the controller, consider getting xbeam to
  # automatically infer the template.
  predictions_chunk, targets_chunk = _load_first_chunk(
      predictions_loader,
      targets_loader,
      times,
      setup_fn=setup_fn,
      ignore_missing_variables=ignore_missing_variables,
  )
  agg_state = _compute_aggregation_state(
      metrics, aggregator, predictions_chunk, targets_chunk
  )
  first_metrics = agg_state.metric_values(metrics)
  template = _expand_template_time_dimensions(first_metrics, times)
  template = _transpose_time_dims_first(template)
  logging.info('Metrics template: %s', template)
  return template


def _drop_non_template_coords(
    key_and_chunk: tuple[xbeam.Key, xr.Dataset],
    template_coords: frozenset[Hashable],
) -> tuple[xbeam.Key, xr.Dataset]:
  """Drops chunk coordinates that were stripped from the expanded template."""
  key, chunk_ds = key_and_chunk
  extra_coords = set(chunk_ds.coords) - template_coords
  if extra_coords:
    logging.log_first_n(
        logging.INFO,
        'Dropping non-template coordinates before Zarr write: %s',
        10,
        extra_coords,
    )
    chunk_ds = chunk_ds.drop_vars(extra_coords, errors='ignore')
  return key, chunk_ds


class WriteMetricsChunksToZarr(beam.PTransform):
  """Writes lead-time-chunked metrics to a Zarr store using xarray-beam."""

  def __init__(
      self,
      out_path: str | Mapping[str, str],
      metrics: Mapping[str, metrics_base.Metric],
      predictions_loader: data_loaders_base.DataLoader,
      targets_loader: data_loaders_base.DataLoader,
      times: time_chunks.TimeChunks,
      aggregator: aggregation.Aggregator | Mapping[str, aggregation.Aggregator],
      setup_fn: Optional[Callable[[], None]] = None,
      ignore_missing_variables: bool = False,
      zarr_chunks: Mapping[str, int] | None = None,
  ):
    super().__init__()
    self.out_path = out_path
    self.metrics = metrics
    self.predictions_loader = predictions_loader
    self.targets_loader = targets_loader
    self.times = times
    self.aggregator = aggregator
    self.setup_fn = setup_fn
    self.ignore_missing_variables = ignore_missing_variables
    self.zarr_chunks = zarr_chunks

  def expand(
      self,
      pcoll: beam.PCollection[tuple[str | None, tuple[xbeam.Key, xr.Dataset]]],
  ):
    if isinstance(self.aggregator, Mapping):
      aggregators = self.aggregator
    else:
      aggregators = {None: self.aggregator}

    # Used to find the partition index to partition the PCollection containing
    # (potentially) multiple aggregators' data into separate PCollections, one
    # per aggregator.
    agg_name_list = list(aggregators.keys())

    def by_aggregator_name(
        element: tuple[str | None, tuple[xbeam.Key, xr.Dataset]],
        num_partitions: int,
    ) -> int:
      """Partition function to partition PCollection by aggregator name."""
      del num_partitions  # Required by beam.Partition, but not used.
      agg_name, _ = element
      return agg_name_list.index(agg_name)

    aggregator_partitioned_pcolls = (
        pcoll
        # The metrics chunks for each aggregator are emitted to a single
        # PCollection (keyed by aggregator_name), so we partition the elements
        # into each aggregator that we want to write out the metrics for.
        | 'PartitionByAggregatorName'
        >> beam.Partition(
            by_aggregator_name,
            len(aggregators),
        )
    )

    results = []
    for agg_name, agg, partitioned_pcoll in zip(
        aggregators.keys(), aggregators.values(), aggregator_partitioned_pcolls
    ):
      target_path = _resolve_out_path(self.out_path, agg_name)
      if not target_path.endswith('.zarr'):
        raise ValueError(
            'Output path with chunk_metrics_by_lead_time must end with .zarr,'
            f' got {target_path}'
        )

      template = _get_template_metrics_dataset(
          self.metrics,
          self.predictions_loader,
          self.targets_loader,
          self.times,
          agg,
          setup_fn=self.setup_fn,
          ignore_missing_variables=self.ignore_missing_variables,
      )
      dim_sizes = typing.cast(Mapping[str, int], template.sizes)
      if 'lead_time' not in dim_sizes:
        raise ValueError(
            'Cannot use chunk_metrics_by_lead_time=True when lead_time is not a'
            ' dimension in the output (e.g. it was reduced).'
        )

      in_chunks = {}
      for dim, size in dim_sizes.items():
        if dim == 'init_time':
          in_chunks[dim] = self.times.init_time_chunk_size or -1
        elif dim == 'lead_time':
          in_chunks[dim] = self.times.lead_time_chunk_size or -1
        else:
          in_chunks[dim] = size
      out_chunks = in_chunks.copy()
      if self.zarr_chunks:
        out_chunks.update(self.zarr_chunks)

      template_coords = frozenset(template.coords)
      label_suffix = f'_{agg_name}' if agg_name else ''
      res = (
          partitioned_pcoll
          | f'ExtractChunk{label_suffix}'
          >> beam.Map(
              lambda x, coords=template_coords: _drop_non_template_coords(
                  x[1], coords
              )
          )
          | f'Rechunk{label_suffix}'
          >> xbeam.Rechunk(
              dim_sizes=dim_sizes,
              source_chunks=in_chunks,
              target_chunks=out_chunks,
              itemsize=4,
          )
          | f'WriteMetricsToZarr{label_suffix}'
          >> xbeam.ChunksToZarr(
              target_path,
              template=template,
              zarr_chunks=out_chunks,
          )
      )
      results.append(res)
    return results


def _format_aggregation_state_chunk(
    element: tuple[_AggregationKey, xr.DataArray],
    coords_to_keep: frozenset[Hashable],
) -> tuple[xbeam.Key, xr.Dataset]:
  """Formats a single (_AggregationKey, DataArray) pair into an xarray-beam chunk."""
  key, da = element
  var_name = f'{key.statistic_name}#{key.variable_name}#{key.type}'
  var_ds = xr.Dataset({var_name: da})
  offsets = {}
  if 'init_time' in var_ds.dims and key.init_time_offset is not None:
    offsets['init_time'] = key.init_time_offset
  if 'lead_time' in var_ds.dims and key.lead_time_offset is not None:
    offsets['lead_time'] = key.lead_time_offset
  if (
      'valid_time' in coords_to_keep
      and 'valid_time' not in var_ds.coords
      and 'init_time' in var_ds.dims
      and 'lead_time' in var_ds.dims
  ):
    var_ds.coords['valid_time'] = var_ds.init_time + var_ds.lead_time
  var_ds = _transpose_time_dims_first(var_ds)
  return _drop_non_template_coords(
      (xbeam.Key(offsets, vars={var_name}), var_ds), coords_to_keep
  )


class WriteAggregationStateChunksToZarr(beam.PTransform):
  """Writes chunked AggregationState to a Zarr store using xarray-beam."""

  def __init__(
      self,
      aggregation_state_out_path: str | Mapping[str, str],
      metrics: Mapping[str, metrics_base.Metric],
      predictions_loader: data_loaders_base.DataLoader,
      targets_loader: data_loaders_base.DataLoader,
      times: time_chunks.TimeChunks,
      aggregator: aggregation.Aggregator | Mapping[str, aggregation.Aggregator],
      setup_fn: Optional[Callable[[], None]] = None,
      ignore_missing_variables: bool = False,
      zarr_chunks: Mapping[str, int] | None = None,
  ):
    super().__init__()
    self.aggregation_state_out_path = aggregation_state_out_path
    self.metrics = metrics
    self.predictions_loader = predictions_loader
    self.targets_loader = targets_loader
    self.times = times
    self.aggregator = aggregator
    self.setup_fn = setup_fn
    self.ignore_missing_variables = ignore_missing_variables
    self.zarr_chunks = zarr_chunks

  def expand(
      self,
      pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]],
  ):
    if isinstance(self.aggregator, Mapping):
      aggregators = self.aggregator
    else:
      aggregators = {None: self.aggregator}

    # Used to find the partition index to partition the PCollection containing
    # (potentially) multiple aggregators' data into separate PCollections, one
    # per aggregator.
    agg_name_list = list(aggregators.keys())

    def by_aggregator_name(
        element: tuple[_AggregationKey, xr.DataArray],
        num_partitions: int,
    ) -> int:
      """Partition function to partition PCollection by aggregator name."""
      del num_partitions  # Required by beam.Partition, but not used.
      key, _ = element
      return agg_name_list.index(key.aggregator_name)

    aggregator_partitioned_pcolls = (
        pcoll
        # The summed statistics for each aggregator are emitted to a single
        # PCollection (whose name is stored in the _AggregationKey), so we
        # partition the elements into each aggregator that we want to write
        # out the aggregation state for.
        | 'PartitionByAggregatorName'
        >> beam.Partition(
            by_aggregator_name,
            len(aggregators),
        )
    )

    results = []
    for agg_name, agg, partitioned_pcoll in zip(
        aggregators.keys(), aggregators.values(), aggregator_partitioned_pcolls
    ):
      target_path = _resolve_out_path(
          self.aggregation_state_out_path, agg_name
      )
      # We should have already checked this in define_pipeline, so this is just
      # a sanity check.
      assert target_path.endswith('.zarr'), (
          'Aggregation state output path must end with .zarr for chunked'
          f' output, got {target_path}'
      )

      # Whereas the input pcoll calls aggregate_stat_var on a single statistic
      # variable at a time, here we compute the full aggregation state for all
      # variables and statistics at once from the first chunk of predictions and
      # targets. This constructs the template used by xbeam to know the data
      # variables, dims, and coords that will be written to.
      template = _get_template_aggregation_state_dataset(
          self.metrics,
          self.predictions_loader,
          self.targets_loader,
          self.times,
          agg,
          setup_fn=self.setup_fn,
          ignore_missing_variables=self.ignore_missing_variables,
      )
      dim_sizes = typing.cast(Mapping[str, int], template.sizes)

      in_chunks = {}
      for dim, size in dim_sizes.items():
        if dim == 'init_time':
          in_chunks[dim] = self.times.init_time_chunk_size or -1
        elif dim == 'lead_time':
          in_chunks[dim] = self.times.lead_time_chunk_size or -1
        else:
          in_chunks[dim] = size
      out_chunks = in_chunks.copy()
      if self.zarr_chunks:
        out_chunks.update(self.zarr_chunks)

      template_coords = frozenset(template.coords)
      label_suffix = f'_{agg_name}' if agg_name else ''
      res = (
          partitioned_pcoll
          | f'FormatAggStateChunk{label_suffix}'
          >> beam.Map(
              _format_aggregation_state_chunk,
              coords_to_keep=template_coords,
          )
          | f'RechunkAggState{label_suffix}'
          >> xbeam.Rechunk(
              dim_sizes=dim_sizes,
              source_chunks=in_chunks,
              target_chunks=out_chunks,
              itemsize=4,
          )
          | f'WriteAggregationStateToZarr{label_suffix}'
          >> xbeam.ChunksToZarr(
              target_path,
              template=template,
              zarr_chunks=out_chunks,
          )
      )
      results.append(res)
    return results


def _is_zarr_path(path: str | Mapping[str, str] | None) -> bool:
  if path is None:
    return False
  if isinstance(path, str):
    return path.endswith('.zarr')
  return all(p.endswith('.zarr') for p in path.values())


def define_pipeline(
    root: beam.Pipeline,
    times: time_chunks.TimeChunks,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator | Mapping[str, aggregation.Aggregator],
    out_path: str | Mapping[str, str] | None = None,
    aggregation_state_out_path: str | Mapping[str, str] | None = None,
    setup_fn: Optional[Callable[[], None]] = None,
    zarr_chunks: Mapping[str, int] | None = None,
    ignore_missing_variables: bool = False,
    chunk_metrics_by_lead_time: bool = False,
):
  """Defines a beam pipeline for calculating aggregated metrics.

  Args:
    root: Pipeline root.
    times: TimeChunks instance.
    predictions_loader: DataLoader instance.
    targets_loader: DataLoader instance.
    metrics: A dictionary of metrics to compute.
    aggregator: Aggregator instance or mapping of aggregator name to Aggregator
      instance.
    out_path: The full path to write the metrics to (or mapping of aggregator
      name to path). If you specify multiple aggregators but only a single
      out_path, the aggregator name will be appended to the filename to get a
      path for each aggregator.
    aggregation_state_out_path: The full path to write the final aggregation
      state to (or mapping of aggregator name to path). This can be useful if
      you want to compute further metrics from it later, and if you are
      preserving init_time, it can be useful to compute confidence intervals
      from later too. Behaviour is the same as for out_path if multiple
      aggregators are specified.
    setup_fn: (Optional) A function to call once per worker in
      LoadPredictionsAndTargets.
    zarr_chunks: Optional chunking specification for Zarr output files.
    ignore_missing_variables: (Optional) If True, filter targets and predictions
      chunks to their common variables. Default: False.
    chunk_metrics_by_lead_time: (Optional) If True, compute metrics and write
      per lead_time chunk to Zarr to eliminate single-worker memory bottlenecks.
      Note that this will not be appropriate for metrics that rely on having
      multiple lead times available at once, and may produce unexpected
      behaviour in such cases.
  """

  if isinstance(aggregator, Mapping):
    if isinstance(out_path, Mapping) and out_path.keys() != aggregator.keys():
      raise ValueError("Keys of out_path don't match aggregator names.")
    if (isinstance(aggregation_state_out_path, Mapping) and
        aggregation_state_out_path.keys() != aggregator.keys()):
      raise ValueError(
          "Keys of aggregation_state_out_path don't match aggregator names.")

  if out_path is None and aggregation_state_out_path is None:
    raise ValueError(
        'At least one of (metrics) out_path or aggregation_state_out_path must '
        'be specified.'
    )

  summed_stat_chunks = (
      root
      | 'CreateTimeChunks' >> beam.Create(times.iter_with_chunk_offsets())
      | beam.ParDo(
          LoadPredictionsAndTargets(
              predictions_loader,
              targets_loader,
              setup_fn=setup_fn,
              ignore_missing_variables=ignore_missing_variables,
          )
      )
      # Compute statistics for each chunk, perform the initial per-chunk
      # aggregation on them using the Aggregator, then prepare them for further
      # aggregation by breaking the AggregationState up into separate
      # DataArrays for each statistic, variable, type (sum_weights or
      # sum_weighted_statistics) and chunk offset.
      | beam.ParDo(
          ComputeStatisticsAggregateAndPrepareForCombine(metrics, aggregator)
      )
      # Sum up the statistic DataArrays over dimensions of the TimeChunks that
      # we are reducing over, typically just init_time but can also be
      # lead_time. This is done separately for each statistic, each variable,
      # and each chunk offset along dimensions not being reduced over (e.g.
      # typically lead_time is not reduced over).
      # If reduce_dims=[], this should be an identity pass-through, because
      # the keys include init_time and lead_time offsets, so there should only
      # be one DataArray element per key.
      | 'SumPerStatisticPerVariableAndPerUnreducedOffset'
      >> beam.CombinePerKey(beam_utils.CombiningSum())
  )

  write_agg_state_as_zarr = _is_zarr_path(aggregation_state_out_path)
  if aggregation_state_out_path is not None and write_agg_state_as_zarr:
    _ = (
        summed_stat_chunks
        | 'WriteAggregationStateChunksToZarr'
        >> WriteAggregationStateChunksToZarr(
            aggregation_state_out_path=aggregation_state_out_path,
            metrics=metrics,
            predictions_loader=predictions_loader,
            targets_loader=targets_loader,
            times=times,
            aggregator=aggregator,
            setup_fn=setup_fn,
            ignore_missing_variables=ignore_missing_variables,
            zarr_chunks=zarr_chunks,
        )
    )

  if out_path is not None or (
      aggregation_state_out_path is not None and not write_agg_state_as_zarr
  ):
    agg_state_pipeline = (
        summed_stat_chunks
        # Now we've reduced the size of the data as much as we can by summing,
        # we concatenate the resulting chunks along any remaining dimensions
        # where we know that coordinates will not overlap across chunks.
        | ConcatPerStatisticPerVariable(
            chunk_metrics_by_lead_time=chunk_metrics_by_lead_time
        )
        # Finally we gather together all the concatenated chunks for all
        # statistics and variables and reconstitute the AggregationState
        # from them, which we can use to compute the final values of metrics.
        | ReconstructAggregationState(
            chunk_metrics_by_lead_time=chunk_metrics_by_lead_time
        )
    )

    if out_path is not None:
      metrics_pcoll = agg_state_pipeline | 'ComputeMetrics' >> beam.ParDo(
          ComputeMetrics(
              metrics, chunk_metrics_by_lead_time=chunk_metrics_by_lead_time
          )
      )

      if not chunk_metrics_by_lead_time:
        _ = metrics_pcoll | 'WriteMetrics' >> beam.ParDo(
            WriteMetrics(
                out_path,
                zarr_chunks=zarr_chunks,
            )
        )
      else:
        _ = (
            metrics_pcoll
            | 'WriteMetricsChunksToZarr'
            >> WriteMetricsChunksToZarr(
                out_path=out_path,
                metrics=metrics,
                predictions_loader=predictions_loader,
                targets_loader=targets_loader,
                times=times,
                aggregator=aggregator,
                setup_fn=setup_fn,
                ignore_missing_variables=ignore_missing_variables,
                zarr_chunks=zarr_chunks,
            )
        )

    if aggregation_state_out_path is not None and not write_agg_state_as_zarr:
      if not chunk_metrics_by_lead_time:
        _ = agg_state_pipeline | beam.ParDo(
            WriteAggregationState(
                aggregation_state_out_path,
                zarr_chunks=zarr_chunks,
            )
        )
      else:
        raise ValueError(
            'Non-Zarr aggregation_state_out_path can only be written if'
            ' chunk_metrics_by_lead_time=False.'
        )


def _transpose_time_dims_first(ds: xr.Dataset) -> xr.Dataset:
  """Transposes dataset to put ('init_time', 'lead_time') first."""
  time_dims = [d for d in ('init_time', 'lead_time') if d in ds.dims]
  return ds.transpose(*time_dims, ...)


class ComputeAndFormatStatistics(beam.DoFn):
  """Computes statistics and formats them for xarray-beam."""

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      times: time_chunks.TimeChunks,
  ):
    """Init.

    Args:
      metrics: A dictionary of metrics to compute statistics for.
      times: TimeChunks instance providing chunk key logic.
    """
    self.metrics = metrics
    self.times = times

  def process(
      self,
      element: tuple[
          time_chunks.TimeChunkOffsets,
          tuple[
              Mapping[Hashable, xr.DataArray],
              Mapping[Hashable, xr.DataArray],
          ],
      ],
  ) -> Iterable[tuple[xbeam.Key, xr.Dataset]]:
    """Computes statistics and yields (chunk_key, dataset) tuples."""
    time_chunk_offsets, (predictions_chunk, targets_chunk) = element

    statistics_dict = metrics_base.compute_unique_statistics_for_all_metrics(
        self.metrics, predictions_chunk, targets_chunk
    )

    for stat_name, var_dict in statistics_dict.items():
      for var_name, da in var_dict.items():
        name = f'{stat_name}.{var_name}'

        chunk_ds = xr.Dataset({name: da})

        offsets = {}
        if 'init_time' in chunk_ds.dims:
          offsets['init_time'] = time_chunk_offsets.init_time
        if 'lead_time' in chunk_ds.dims:
          offsets['lead_time'] = time_chunk_offsets.lead_time

        chunk_ds = _transpose_time_dims_first(chunk_ds)
        chunk_key = xbeam.Key(offsets, vars={name})

        yield chunk_key, chunk_ds


def _get_template_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    setup_fn: Optional[Callable[[], None]] = None,
    ignore_missing_variables: bool = False,
) -> xr.Dataset:
  """Computes statistics for the first chunk to create a template dataset."""
  logging.info('Building template with data from first chunk')

  predictions_chunk, targets_chunk = _load_first_chunk(
      predictions_loader,
      targets_loader,
      times,
      setup_fn=setup_fn,
      ignore_missing_variables=ignore_missing_variables,
  )
  statistics_dict = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )
  first_chunk = xr.Dataset()
  for stat_name, var_dict in statistics_dict.items():
    for var_name, da in var_dict.items():
      first_chunk[f'{stat_name}.{var_name}'] = da

  return _expand_template_time_dimensions(first_chunk, times)


def _expand_template_time_dimensions(
    first_chunk: xr.Dataset,
    times: time_chunks.TimeChunks,
) -> xr.Dataset:
  """Convert first chunk to template, expanding time dimensions if necessary."""
  template = xbeam.make_template(first_chunk)

  if 'mask' in template.coords:
    raise ValueError(
        'mask coordinate found in template. add_nan_mask=True on data loaders '
        'is not supported for unaggregated pipelines.'
    )

  if 'lead_time' in template.dims:
    vars_to_expand = [k for k, v in template.items() if 'lead_time' in v.dims]
    template = template.isel(lead_time=0, drop=True)
    lead_times = times.lead_times
    if isinstance(lead_times, slice):
      lead_times = np.arange(
          lead_times.start, lead_times.stop + lead_times.step, lead_times.step
      )
    for k in vars_to_expand:
      template[k] = template[k].expand_dims(lead_time=lead_times)

  if 'init_time' in template.dims:
    vars_to_expand = [k for k, v in template.items() if 'init_time' in v.dims]
    template = template.isel(init_time=0, drop=True)
    for k in vars_to_expand:
      template[k] = template[k].expand_dims(init_time=times.init_times)

  if 'init_time' in template.dims and 'lead_time' in template.dims:
    template.coords['valid_time'] = template.init_time + template.lead_time

  return template


# TOOD: shoyer - consider renaming this function to refer to "statistics" (vs
# the metrics calculated by define_pipeline)
def define_unaggregated_pipeline(
    root: beam.Pipeline,
    times: time_chunks.TimeChunks,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    out_path: str,
    zarr_chunks: Mapping[str, int] | None = None,
    setup_fn: Optional[Callable[[], None]] = None,
    ignore_missing_variables: bool = False,
):
  """Defines a Beam pipeline that calculates statistics without aggregation.

  Outputs statistics for all predictions and targets to a single Zarr store,
  which assumes that all statistics have compatible coordinates. If this is not
  the case, you'll need to run separate pipelines for incompatible statistics.

  Args:
    root: Pipeline root.
    times: TimeChunks instance. Must implement `get_chunk_key(index)` returning
      a Dict[str, slice] and `get_zarr_chunks()` returning Dict[str, int].
    predictions_loader: DataLoader instance for predictions.
    targets_loader: DataLoader instance for targets.
    metrics: A dictionary of metrics to compute statistics for.
    out_path: The full path to write the output Zarr store to.
    zarr_chunks: (Optional) A dictionary of chunks to use for the output Zarr
      store. If None, the chunks will match those of TimeChunks.
    setup_fn: (Optional) A function to call once per worker in
      LoadPredictionsAndTargets.
    ignore_missing_variables: (Optional) If True, filter targets and predictions
      chunks to their common variables. Default: False.
  """
  template = _get_template_dataset(
      metrics,
      predictions_loader,
      targets_loader,
      times,
      setup_fn=setup_fn,
      ignore_missing_variables=ignore_missing_variables,
  )
  dim_sizes = typing.cast(Mapping[str, int], template.sizes)

  stat_chunks = {}
  for dim, size in dim_sizes.items():
    if dim == 'init_time':
      stat_chunks[dim] = times.init_time_chunk_size or -1
    elif dim == 'lead_time':
      stat_chunks[dim] = times.lead_time_chunk_size or -1
    else:
      stat_chunks[dim] = size  # unchunked

  if zarr_chunks is None:
    zarr_chunks = {}

  # Use any entries in stat_chunks as defaults for zarr_chunks.
  # Consider raising an error for missing dimensions instead?
  zarr_chunks = stat_chunks | zarr_chunks  # pyrefly: ignore[unsupported-operation]

  _ = (
      root
      | 'CreateTimeChunks' >> beam.Create(times.iter_with_chunk_offsets())
      | 'LoadPredictionsAndTargets'
      >> beam.ParDo(
          LoadPredictionsAndTargets(
              predictions_loader,
              targets_loader,
              setup_fn=setup_fn,
              ignore_missing_variables=ignore_missing_variables,
          )
      )
      | 'ComputeAndFormatStatistics'
      >> beam.ParDo(ComputeAndFormatStatistics(metrics, times))
      | 'Rechunk'
      >> xbeam.Rechunk(
          dim_sizes,
          stat_chunks,
          zarr_chunks,
          itemsize=4,  # assumes float32
      )
      | 'WriteStatisticsToZarr'
      >> xbeam.ChunksToZarr(
          out_path, template=template, zarr_chunks=zarr_chunks
      )
  )


def _load_first_chunk(
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    setup_fn: Optional[Callable[[], None]] = None,
    ignore_missing_variables: bool = False,
) -> tuple[Mapping[Hashable, xr.DataArray], Mapping[Hashable, xr.DataArray]]:
  """Loads the first chunk of preds and targets for template generation."""
  if setup_fn is not None:
    setup_fn()

  first_chunk_index = 0
  try:
    first_init_times, first_lead_times = times[first_chunk_index]
  except IndexError:
    raise ValueError('Cannot generate template: TimeChunks is empty') from None

  targets_chunk = targets_loader.load_chunk(first_init_times, first_lead_times)
  predictions_chunk = predictions_loader.load_chunk(
      first_init_times, first_lead_times, targets_chunk
  )
  if ignore_missing_variables:
    common_vars = [v for v in targets_chunk.keys() if v in predictions_chunk]
    targets_chunk = {v: targets_chunk[v] for v in common_vars}
    predictions_chunk = {v: predictions_chunk[v] for v in common_vars}
  return predictions_chunk, targets_chunk


def _compute_aggregation_state(
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
    predictions_chunk: Mapping[Hashable, xr.DataArray],
    targets_chunk: Mapping[Hashable, xr.DataArray],
) -> aggregation.AggregationState:
  """Computes AggregationState for a single chunk."""
  statistics = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )
  return aggregator.aggregate_statistics(statistics)


def _compute_aggregation_state_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
    predictions_chunk: Mapping[Hashable, xr.DataArray],
    targets_chunk: Mapping[Hashable, xr.DataArray],
) -> xr.Dataset:
  """Computes the AggregationState dataset for a single chunk."""
  aggregation_state = _compute_aggregation_state(
      metrics, aggregator, predictions_chunk, targets_chunk
  )
  return aggregation_state.to_dataset()


def _get_template_aggregation_state_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    aggregator: aggregation.Aggregator,
    setup_fn: Optional[Callable[[], None]] = None,
    ignore_missing_variables: bool = False,
) -> xr.Dataset:
  """Computes AggregationState dataset for the first chunk to create a template dataset."""
  logging.info('Building AggregationState template with data from first chunk.')
  predictions_chunk, targets_chunk = _load_first_chunk(
      predictions_loader,
      targets_loader,
      times,
      setup_fn=setup_fn,
      ignore_missing_variables=ignore_missing_variables,
  )
  first_chunk = _compute_aggregation_state_dataset(
      metrics,
      aggregator,
      predictions_chunk,
      targets_chunk,
  )
  template = _expand_template_time_dimensions(first_chunk, times)
  template = _transpose_time_dims_first(template)
  logging.info('AggregationState template: %s', template)
  return template
