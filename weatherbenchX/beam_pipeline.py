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
from typing import Callable, Hashable, Iterable, Mapping, Optional, Tuple, Union
import apache_beam as beam
import fsspec
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import beam_utils
from weatherbenchX import time_chunks
from weatherbenchX.data_loaders import base as data_loaders_base
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


class LoadChunksAndAggregateStatistics(beam.DoFn):
  """Loads prediction and target chunks, computes and aggregates statistics."""

  def __init__(
      self,
      predictions_loader: data_loaders_base.DataLoader,
      targets_loader: data_loaders_base.DataLoader,
      metrics: Mapping[str, metrics_base.Metric],
      aggregator: aggregation.Aggregator,
      setup_fn: Optional[Callable[[], None]] = None,
  ):
    """Init.

    Args:
      predictions_loader: The data loader for the predictions.
      targets_loader: The data loader for the targets.
      metrics: A dictionary of metrics to compute.
      aggregator: Aggregation instance.
      setup_fn: (Optional) A function to call once per worker.
    """
    self.predictions_loader = predictions_loader
    self.targets_loader = targets_loader
    self.metrics = metrics
    self.aggregator = aggregator
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
  ) -> Iterable[Tuple[int, aggregation.AggregationState]]:
    """Returns AggregationState for a given init/lead time.

    Args:
      all_inputs: (chunk_index, (init_times, lead_times))

    Returns:
      (chunk_index, aggregation_state)
    """
    logging.info('LoadChunksAndAggregateStatistics inputs: %s', all_inputs)
    chunk_index, (init_times, lead_times) = all_inputs
    targets_chunk = self.targets_loader.load_chunk(init_times, lead_times)
    predictions_chunk = self.predictions_loader.load_chunk(
        init_times, lead_times, targets_chunk
    )
    logging.info(
        'LoadChunksAndAggregateStatistics chunks: %s',
        (chunk_index, (predictions_chunk, targets_chunk)),
    )
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        self.metrics, predictions_chunk, targets_chunk
    )
    aggregation_state = self.aggregator.aggregate_statistics(statistics)
    logging.info(
        'LoadChunksAndAggregateStatistics outputs: %s',
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
  """Defines the beam pipeline.

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
      LoadChunksAndAggregateStatistics.
  """
  if max_chunks_per_aggregation_stage is None:
    max_chunks_per_aggregation_stage = len(times)

  _ = (
      root
      | 'CreateTimeChunks' >> beam.Create(enumerate(times))  # pytype: disable=wrong-arg-types
      | 'LoadChunksAndAggregateStatistics'
      >> beam.ParDo(
          LoadChunksAndAggregateStatistics(
              predictions_loader,
              targets_loader,
              metrics,
              aggregator,
              setup_fn=setup_fn,
          )
      )
      | 'AggregateStates'
      >> beam_utils.CombineMultiStage(
          total_num_elements=len(times),
          max_bin_size=max_chunks_per_aggregation_stage,
          combine_fn=beam_utils.SumAggregationStates(),
      )
      | 'ComputeMetrics' >> beam.ParDo(ComputeMetrics(metrics))
      | 'WriteMetrics' >> beam.ParDo(WriteMetrics(out_path))
  )
