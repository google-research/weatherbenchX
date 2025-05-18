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

import logging
import typing
from typing import Callable, Hashable, Iterable, Mapping, Optional, Tuple, Union
import apache_beam as beam
import fsspec
import numpy as np
from weatherbenchAfrica import aggregation
from weatherbenchAfrica import beam_utils
from weatherbenchAfrica import time_chunks
from weatherbenchAfrica.data_loaders import base as data_loaders_base
from weatherbenchAfrica.metrics import base as metrics_base
import xarray as xr
import xarray_beam as xbeam


class LoadPredictionsAndTargets(beam.DoFn):
  """Loads prediction and target chunks from their respective data loaders."""

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
      self, all_inputs: Tuple[int, Tuple[np.ndarray, Union[np.ndarray, slice]]]
  ) -> Iterable[
      Tuple[
          int,
          Tuple[
              Mapping[Hashable, xr.DataArray],
              Mapping[Hashable, xr.DataArray],
          ],
      ]
  ]:
    """Returns the predictions and targets chunks for a given init/lead time.

    Args:
      all_inputs: (chunk_index, (init_times, lead_times))

    Returns:
      (chunk_index, (predictions_chunk, targets_chunk))
    """
    logging.info('LoadPredictionsAndTargets inputs: %s', all_inputs)
    chunk_index, (init_times, lead_times) = all_inputs
    targets_chunk = self.targets_loader.load_chunk(init_times, lead_times)
    predictions_chunk = self.predictions_loader.load_chunk(
        init_times, lead_times, targets_chunk
    )
    logging.info(
        'LoadPredictionsAndTargets outputs: %s',
        (chunk_index, (predictions_chunk, targets_chunk)),
    )
    return [(chunk_index, (predictions_chunk, targets_chunk))]


class ComputeStatisticsAndAggregateChunks(beam.DoFn):
  """Computes the statistics for each metric and aggregates chunks."""

  def __init__(
      self,
      metrics: Mapping[str, metrics_base.Metric],
      aggregator: aggregation.Aggregator,
  ):
    """Init.

    Args:
      metrics: A dictionary of metrics to compute.
      aggregator: Aggregation instance.
    """
    self.metrics = metrics
    self.aggregator = aggregator

  def process(
      self,
      all_inputs: Tuple[
          int,
          Tuple[
              Mapping[Hashable, xr.DataArray],
              Mapping[Hashable, xr.DataArray],
          ],
      ],
  ) -> Iterable[Tuple[int, aggregation.AggregationState]]:
    """Returns AggregationState for given predictions and targets chunks.

    Args:
      all_inputs: (chunk_index, (predictions_chunk, targets_chunk))

    Returns:
      (chunk_index, aggregation_state)
    """
    logging.info('ComputeStatisticsAndAggregateChunks inputs: %s', all_inputs)
    chunk_index, (predictions_chunk, targets_chunk) = all_inputs
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        self.metrics, predictions_chunk, targets_chunk
    )
    aggregation_state = self.aggregator.aggregate_statistics(statistics)
    logging.info(
        'ComputeStatisticsAndAggregateChunks outputs: %s',
        (chunk_index, aggregation_state),
    )
    return [(chunk_index, aggregation_state)]


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
    max_chunks_per_aggregation_stage: Optional[int] = 10,
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
    max_chunks_per_aggregation_stage: The maximum number of chunks to aggregate
      in a single worker. If None, does aggregation in a single step. Default:
      10
    setup_fn: (Optional) A function to call once per worker in
      LoadPredictionsAndTargets.
  """
  if max_chunks_per_aggregation_stage is None:
    max_chunks_per_aggregation_stage = len(times)

  _ = (
      root
      | 'CreateTimeChunks' >> beam.Create(enumerate(times))  # pytype: disable=wrong-arg-types
      | 'LoadPredictionsAndTargets'
      >> beam.ParDo(
          LoadPredictionsAndTargets(
              predictions_loader, targets_loader, setup_fn=setup_fn
          )
      )
      | 'ComputeStatisticsAndAggregateChunks'
      >> beam.ParDo(ComputeStatisticsAndAggregateChunks(metrics, aggregator))
      | 'AggregateStates'
      >> beam_utils.CombineMultiStage(
          total_num_elements=len(times),
          max_bin_size=max_chunks_per_aggregation_stage,
          combine_fn=beam_utils.SumAggregationStates(),
      )
      | 'ComputeMetrics' >> beam.ParDo(ComputeMetrics(metrics))
      | 'WriteMetrics' >> beam.ParDo(WriteMetrics(out_path))
  )


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
      self, all_inputs: Tuple[int, Tuple[np.ndarray, Union[np.ndarray, slice]]]
  ) -> Iterable[
      Tuple[
          int,
          Tuple[
              Mapping[Hashable, xr.DataArray], Mapping[Hashable, xr.DataArray]
          ],
      ]
  ]:
    """Returns prediction and target chunks for a given init/lead time.

    Args:
      all_inputs: (chunk_index, (init_times, lead_times))

    Returns:
      (chunk_index, (predictions_chunk, targets_chunk))
    """
    chunk_index, (init_times, lead_times) = all_inputs
    targets_chunk = self.targets_loader.load_chunk(init_times, lead_times)
    predictions_chunk = self.predictions_loader.load_chunk(
        init_times, lead_times, targets_chunk
    )
    return [(chunk_index, (predictions_chunk, targets_chunk))]


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
      element: Tuple[
          int,
          Tuple[
              Mapping[Hashable, xr.DataArray],
              Mapping[Hashable, xr.DataArray],
          ],
      ],
  ) -> Iterable[Tuple[xbeam.Key, xr.Dataset]]:
    """Computes statistics and yields (chunk_key, dataset) tuples."""
    chunk_index, (predictions_chunk, targets_chunk) = element

    init_index, lead_index = self.times.get_init_and_lead_chunk_starts(
        chunk_index
    )

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
          offsets['init_time'] = init_index
        if 'lead_time' in chunk_ds.dims:
          dim_order.append('lead_time')
          offsets['lead_time'] = lead_index

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
      | 'CreateTimeChunks' >> beam.Create(enumerate(times))
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
