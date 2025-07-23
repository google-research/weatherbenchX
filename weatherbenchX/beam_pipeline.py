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
"""Defines the beam pipeline for evaluation."""

from collections.abc import Hashable
import dataclasses
import logging
import typing
from typing import Callable, Iterable, Iterator, Literal, Mapping, Optional, Union

import apache_beam as beam
import fsspec
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
  ):
    """Init.

    Args:
      predictions_loader: The data loader for the predictions.
      targets_loader: The data loader for the targets.
      setup_fn: (Optional) A function to call once per worker.
    """
    self.predictions_loader = predictions_loader
    self.targets_loader = targets_loader
    self.setup_fn = setup_fn
    self.is_initialized = False

  def setup(self):
    # Call this function once per process.
    if self.setup_fn is not None:
      if not self.is_initialized:
        self.setup_fn()
        self.is_initialized = True

  def process(
      self, all_inputs: tuple[time_chunks.TimeChunkOffsets,
                              tuple[np.ndarray, Union[np.ndarray, slice]]]
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
    logging.info('LoadPredictionsAndTargets inputs: %s', all_inputs)
    time_chunk_offsets, (init_times, lead_times) = all_inputs
    targets_chunk = self.targets_loader.load_chunk(init_times, lead_times)
    predictions_chunk = self.predictions_loader.load_chunk(
        init_times, lead_times, targets_chunk
    )
    logging.info(
        'LoadPredictionsAndTargets outputs: %s',
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
  variable_name: Hashable
  # Offsets for the chunk in the result of the aggregation. Should be None if
  # the relevant dimension is being aggregated over.
  init_time_offset: int | None
  lead_time_offset: int | None

  def drop_offsets(self) -> '_AggregationKey':
    return dataclasses.replace(
        self, init_time_offset=None, lead_time_offset=None)


class ComputeStatisticsAggregateAndPrepareForCombine(beam.DoFn):
  """Computes statistics needed for our metrics, for a chunk of init/lead times.

  Then performs the initial per-chunk aggregation on them using the Aggregator,
  then prepares them for further aggregation by breaking the AggregationState
  up into separate DataArrays for each statistic, variable, type (sum_weights or
  sum_weighted_statistics) and chunk offset, keyed by _AggregationKey.
  """

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregator: aggregation.Aggregator,
  ):
    self.metrics = metrics
    self.aggregator = aggregator

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
    logging.info('ComputeStatisticsAggregateAndPrepareForCombine inputs: %s',
                 all_inputs)
    time_chunk_offsets, (predictions_chunk, targets_chunk) = all_inputs
    for stat_name, stats in (
        metrics_base.generate_unique_statistics_for_all_metrics(
            self.metrics, predictions_chunk, targets_chunk)):
      # We use a generator above and yield one at a time, to avoid holding all
      # statistics in memory all at once in case of large statistics.
      for var_name, stat in stats.items():
        aggregation_state = self.aggregator.aggregate_stat_var(stat)
        if aggregation_state is None:
          continue
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
            variable_name=var_name,
            init_time_offset=init_time_offset,
            lead_time_offset=lead_time_offset,
        )
        yield aggregation_key, aggregation_state.sum_weighted_statistics
        aggregation_key = _AggregationKey(
            type='sum_weights',
            statistic_name=stat_name,
            variable_name=var_name,
            init_time_offset=init_time_offset,
            lead_time_offset=lead_time_offset,
        )
        yield aggregation_key, aggregation_state.sum_weights


class ConcatPerStatisticPerVariable(beam.PTransform):
  """Concatenates DataArrays on a per-statistic, per-variable basis.

  The DataArrays correspond to chunks along whichever of the {lead_time,
  init_time} dimensions are being preserved in the result. They arrive keyed
  by _AggregationKey.
  """

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]):

    def drop_offsets_from_key(
        key: _AggregationKey, data_array: xr.DataArray
    ) -> tuple[_AggregationKey, xr.DataArray]:
      return (key.drop_offsets(), data_array)

    def combine_data_arrays_by_coords(
        key: _AggregationKey, data_arrays: Iterable[xr.DataArray]
    ) -> tuple[_AggregationKey, xr.DataArray]:
      # combine_by_coords will return a Dataset if there are any names on the
      # input DataArrays, so we remove the names before calling it.
      return key, xr.combine_by_coords([d.rename(None) for d in data_arrays])

    return (
        pcoll
        # Drop the chunk offsets from the key, so that we group by statistic
        # name, variable name and type (sum_weighted_statistics or sum_weights)
        # alone.
        | 'DropOffsetsFromKey'
        >> beam.MapTuple(drop_offsets_from_key)
        # We use GroupByKey instead of CombinePerKey because the data all needs
        # to be in memory at once to concatenate it, there is no saving from
        # doing this incrementally via a CombineFn.
        | 'GroupByStatAndVariable'
        >> beam.GroupByKey()
        | 'CombineDataArraysByCoords'
        >> beam.MapTuple(combine_data_arrays_by_coords))


def reconstruct_aggregation_state(
    key_value_pairs: Iterable[tuple[_AggregationKey, xr.DataArray]]
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

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]
  ) -> beam.PCollection[aggregation.AggregationState]:
    return (
        pcoll
        | beam_utils.GroupAll()
        | beam.Map(reconstruct_aggregation_state)
    )


class ComputeMetrics(beam.DoFn):
  """Computes the metrics from the aggregated statistics."""

  def __init__(self, metrics: Mapping[str, metrics_base.Metric]):
    """Init.

    Args:
      metrics: A dictionary of metrics to compute. Same as passed to the
        ComputeStatisticsAndAggregateChunks.
    """
    self.metrics = metrics

  def process(
      self, aggregation_state: aggregation.AggregationState
  ) -> Iterable[xr.Dataset]:
    """Returns results Dataset from AggregationState.

    Args:
      aggregation_state: The AggregationState to compute the metrics from.

    Returns:
      A Dataset with the metrics (in a list for Beam).
    """
    logging.info('ComputeMetrics inputs: %s', aggregation_state)
    return [aggregation_state.metric_values(self.metrics)]


class WriteMetrics(beam.DoFn):
  """Writes the metrics to a file."""

  def __init__(self, out_path: str):
    """Init.

    Args:
      out_path: The full path to write the metrics to.
    """
    self.out_path = out_path

  def process(self, metrics: xr.Dataset) -> None:
    """Writes the metrics to a NetCDF file.

    Args:
      metrics: Metrics dataset to write to disc.
    """
    logging.info('WriteMetrics inputs: %s', metrics)
    with fsspec.open(self.out_path, 'wb', auto_mkdir=True) as f:
      f.write(metrics.to_netcdf())
    return None


def define_pipeline(
    root: beam.Pipeline,
    times: time_chunks.TimeChunks,
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    metrics: Mapping[str, metrics_base.Metric],
    aggregator: aggregation.Aggregator,
    out_path: str,
    setup_fn: Optional[Callable[[], None]] = None,
):
  """Defines a beam pipeline for calculating aggregated metrics.

  Args:
    root: Pipeline root.
    times: TimeChunks instance.
    predictions_loader: DataLoader instance.
    targets_loader: DataLoader instance.
    metrics: A dictionary of metrics to compute.
    aggregator: Aggregation instance.
    out_path: The full path to write the metrics to.
    setup_fn: (Optional) A function to call once per worker in
      LoadPredictionsAndTargets.
  """

  _ = (
      root
      | 'CreateTimeChunks' >> beam.Create(times.iter_with_chunk_offsets())
      | beam.ParDo(
          LoadPredictionsAndTargets(
              predictions_loader, targets_loader, setup_fn=setup_fn
          )
      )
      # Compute statistics for each chunk, perform the initial per-chunk
      # aggregation on them using the Aggregator, then prepare them for further
      # aggregation by breaking the AggregationState up into separate
      # DataArrays for each statistic, variable, type (sum_weights or
      # sum_weighted_statistics) and chunk offset.
      | beam.ParDo(ComputeStatisticsAggregateAndPrepareForCombine(
          metrics, aggregator))
      # Sum up the statistic DataArrays over dimensions of the TimeChunks that
      # we are reducing over, typically just init_time but can also be
      # lead_time. This is done separately for each statistic, each variable,
      # and each chunk offset along dimensions not being reduced over (e.g.
      # typically lead_time is not reduced over)
      | 'SumPerStatisticPerVariableAndPerUnreducedOffset'
      >> beam.CombinePerKey(beam_utils.Sum())
      # Now we've reduced the size of the data as much as we can by summing,
      # we concatenate the resulting chunks along any dimensions that we are not
      # summing over (e.g. concatenating lead_time chunks)
      | ConcatPerStatisticPerVariable()
      # Finally we gather together all the concatenated chunks for all
      # statistics and variables and reconstitute the full AggregationState
      # from them, which we can use to compute the final values of metrics.
      | ReconstructAggregationState()
      | 'ComputeMetrics' >> beam.ParDo(ComputeMetrics(metrics))
      | 'WriteMetrics' >> beam.ParDo(WriteMetrics(out_path))
  )


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

        dim_order = []
        offsets = {}
        if 'init_time' in chunk_ds.dims:
          dim_order.append('init_time')
          offsets['init_time'] = time_chunk_offsets.init_time
        if 'lead_time' in chunk_ds.dims:
          dim_order.append('lead_time')
          offsets['lead_time'] = time_chunk_offsets.lead_time

        chunk_ds = chunk_ds.transpose(*dim_order, ...)
        chunk_key = xbeam.Key(offsets, vars={name})

        yield chunk_key, chunk_ds


def _get_template_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    setup_fn: Optional[Callable[[], None]] = None,
) -> xr.Dataset:
  """Computes statistics for the first chunk to create a template dataset."""
  if setup_fn is not None:
    setup_fn()

  logging.info('Building template with data from first chunk')

  # Evaluate statistics on the first chunk
  first_chunk_index = 0
  try:
    first_init_times, first_lead_times = times[first_chunk_index]
  except IndexError:
    raise ValueError('Cannot generate template: TimeChunks is empty') from None

  targets_chunk = targets_loader.load_chunk(first_init_times, first_lead_times)
  predictions_chunk = predictions_loader.load_chunk(
      first_init_times, first_lead_times, targets_chunk
  )
  statistics_dict = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )
  first_chunk = xr.Dataset()
  for stat_name, var_dict in statistics_dict.items():
    for var_name, da in var_dict.items():
      first_chunk[f'{stat_name}.{var_name}'] = da

  # Convert the first chunk into a template, with the proper init_time and
  # lead_time dimensions
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
  """
  template = _get_template_dataset(
      metrics, predictions_loader, targets_loader, times, setup_fn
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
  zarr_chunks = stat_chunks | zarr_chunks

  _ = (
      root
      | 'CreateTimeChunks' >> beam.Create(times.iter_with_chunk_offsets())
      | 'LoadPredictionsAndTargets'
      >> beam.ParDo(
          LoadPredictionsAndTargets(
              predictions_loader, targets_loader, setup_fn=setup_fn
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
