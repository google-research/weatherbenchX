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
import os

os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache')
import time
import typing
from typing import Callable, Iterable, Iterator, Literal, Mapping, Never, Optional, Union
import uuid

from absl import logging
import apache_beam as beam
from google3.pyglib import gfile
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
    time_chunk_offsets, (init_times, lead_times) = all_inputs
    logging.info(
        'Worker loading chunk %s (init_times count: %d, lead_times: %s)...',
        time_chunk_offsets,
        len(init_times),
        len(lead_times) if hasattr(lead_times, '__len__') else lead_times,
    )

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

    common_vars = [v for v in targets_chunk.keys() if v in predictions_chunk]
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
            'Targets chunk has variables not present in predictions chunk: %s',
            missing_in_preds,
        )
      if missing_in_targets:
        logging.warning(
            'Predictions chunk has variables not present in targets chunk: %s',
            missing_in_targets,
        )
      targets_chunk = {v: targets_chunk[v] for v in common_vars}
      predictions_chunk = {v: predictions_chunk[v] for v in common_vars}

    logging.info(
        'Worker finished loading chunk %s (common_vars: %s, pred shapes: %s)',
        time_chunk_offsets,
        common_vars,
        {k: v.shape for k, v in predictions_chunk.items()},
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

  def drop_offsets(self) -> '_AggregationKey':
    return dataclasses.replace(
        self, init_time_offset=None, lead_time_offset=None
    )


class FormatAggregationStateForXarrayBeam(beam.DoFn):
  """Converts (_AggregationKey, xr.DataArray) to (xbeam.Key, xr.Dataset) for xarray-beam."""

  def process(
      self,
      element: tuple[_AggregationKey, xr.DataArray],
  ) -> Iterable[tuple[xbeam.Key, xr.Dataset]]:
    key, da = element
    var_name = f'{key.statistic_name}#{key.variable_name}#{key.type}'

    offsets = {}
    dim_order = []
    if key.init_time_offset is not None:
      offsets['init_time'] = key.init_time_offset
      dim_order.append('init_time')
      if 'init_time' not in da.dims:
        da = da.expand_dims('init_time')
    if key.lead_time_offset is not None:
      offsets['lead_time'] = key.lead_time_offset
      dim_order.append('lead_time')
      if 'lead_time' not in da.dims:
        da = da.expand_dims('lead_time')

    # Transpose dimensions to match expected template ordering
    remaining_dims = [d for d in da.dims if d not in dim_order]
    da = da.transpose(*(dim_order + remaining_dims))

    # Add valid_time coordinate if both init_time and lead_time are present
    if 'init_time' in da.dims and 'lead_time' in da.dims:
      da = da.assign_coords(valid_time=da.init_time + da.lead_time)

    # Drop non-dimension coords (e.g. lead_time_secs) that aren't in the
    # Zarr template. Keep dimension coords and valid_time coordinate.
    non_dim_coords = [
        c for c in da.coords if c not in da.dims and c != 'valid_time'
    ]
    if non_dim_coords:
      da = da.drop_vars(non_dim_coords)

    chunk_ds = xr.Dataset({var_name: da}).drop_attrs(deep=True)
    chunk_key = xbeam.Key(offsets, vars={var_name})
    yield chunk_key, chunk_ds


def _get_aggregation_state_template_dataset(
    metrics: Mapping[str, metrics_base.Metric],
    predictions_loader: data_loaders_base.DataLoader,
    targets_loader: data_loaders_base.DataLoader,
    times: time_chunks.TimeChunks,
    aggregator: aggregation.Aggregator,
    setup_fn: Optional[Callable[[], None]] = None,
) -> xr.Dataset:
  """Computes AggregationState for the first chunk to create a template dataset."""
  if setup_fn is not None:
    setup_fn()

  logging.info('Building AggregationState template with data from first chunk')

  first_chunk_index = 0
  try:
    first_init_times, first_lead_times = times[first_chunk_index]
  except IndexError:
    raise ValueError('Cannot generate template: TimeChunks is empty') from None

  targets_chunk = targets_loader.load_chunk(first_init_times, first_lead_times)
  predictions_chunk = predictions_loader.load_chunk(
      first_init_times, first_lead_times, targets_chunk
  )
  stats_iter = metrics_base.generate_unique_statistics_for_all_metrics(
      metrics, predictions_chunk, targets_chunk
  )

  first_chunk = xr.Dataset()
  for stat_name, stats in stats_iter:
    for var_name, stat in stats.items():
      agg_state = aggregator.aggregate_stat_var(stat)
      if agg_state is None:
        continue
      key_stat = f'{stat_name}#{var_name}#sum_weighted_statistics'
      key_weights = f'{stat_name}#{var_name}#sum_weights'
      first_chunk[key_stat] = agg_state.sum_weighted_statistics
      first_chunk[key_weights] = agg_state.sum_weights

  template = xbeam.make_template(first_chunk)
  template = template.drop_attrs(deep=True)
  for k in template.keys():
    template[k].encoding.clear()

  if 'lead_time' in template.dims:
    vars_to_expand = [k for k, v in template.items() if 'lead_time' in v.dims]
    template = template.isel(lead_time=0, drop=True)
    lead_times = times.lead_times
    if isinstance(lead_times, slice):
      if 'lead_time' in first_chunk.coords:
        lead_times = first_chunk.lead_time.values
      elif 'prediction_timedelta' in first_chunk.coords:
        lead_times = first_chunk.prediction_timedelta.values
      else:
        raise ValueError(
            'Cannot infer lead_time coordinates from first_chunk for slice.'
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

  zarr_chunks = {}
  if 'init_time' in template.dims:
    zarr_chunks['init_time'] = times.init_time_chunk_size
  if 'lead_time' in template.dims:
    if isinstance(times.lead_times, slice):
      zarr_chunks['lead_time'] = template.sizes['lead_time']
    else:
      zarr_chunks['lead_time'] = times.lead_time_chunk_size

  template = template.chunk(zarr_chunks)
  return template


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
    time_chunk_offsets, (predictions_chunk, targets_chunk) = all_inputs
    logging.info(
        'Worker computing statistics and per-chunk aggregation for chunk %s',
        time_chunk_offsets,
    )

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

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]
  ):

    def drop_offsets_from_key(
        key: _AggregationKey, data_array: xr.DataArray
    ) -> tuple[_AggregationKey, xr.DataArray]:
      return (key.drop_offsets(), data_array)

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
        # name, variable name and type (sum_weighted_statistics or sum_weights)
        # alone.
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

  def expand(
      self, pcoll: beam.PCollection[tuple[_AggregationKey, xr.DataArray]]
  ) -> beam.PCollection[tuple[str | None, aggregation.AggregationState]]:
    return (
        pcoll
        | 'GroupByAggregator' >> beam.GroupBy(lambda x: x[0].aggregator_name)
        | 'Reconstruct' >> beam.MapTuple(
            lambda agg_name, xs: (agg_name, reconstruct_aggregation_state(xs))
        )
    )


class ComputeMetrics(beam.DoFn):
  """Computes the metrics from the aggregated statistics."""

  def __init__(self, metrics: Mapping[str, metrics_base.Metric]):
    self.metrics = metrics

  def process(
      self, element: tuple[str | None, aggregation.AggregationState]
  ) -> Iterable[tuple[str | None, xr.Dataset]]:
    """Computes a metrics Dataset from the final AggregationState."""
    agg_name, aggregation_state = element
    logging.log_first_n(
        logging.INFO,
        'ComputeMetrics inputs: %s, %s',
        10,
        agg_name,
        aggregation_state,
    )
    yield agg_name, aggregation_state.metric_values(self.metrics)


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


class WriteMetrics(beam.DoFn):
  """Writes the metrics to a NetCDF or Zarr file."""

  def __init__(self, out_path: str | Mapping[str, str]):
    self.out_path = out_path

  def process(self, element: tuple[str | None, xr.Dataset]) -> Iterable[Never]:
    agg_name, metrics = element
    target_path = _resolve_out_path(self.out_path, agg_name)
    logging.info(
        'Worker writing metrics for aggregator %s to %s (data_vars: %s, dims: %s)',
        agg_name,
        target_path,
        list(metrics.data_vars),
        dict(metrics.sizes),
    )
    # Remove attributes that may have been propogated from the targets or
    # predictions.
    metrics = metrics.drop_attrs(deep=True)
    if target_path.endswith('.zarr'):
      temp_path = f'{target_path}.tmp_{uuid.uuid4().hex}'
      if {'latitude', 'longitude'}.issubset(metrics.dims):
        chunk_spec = {'latitude': 181, 'longitude': 360}
        encoding = {
            var: {
                'chunks': tuple(
                    min(
                        metrics[var].sizes[dim],
                        chunk_spec.get(dim, metrics[var].sizes[dim]),
                    )
                    for dim in metrics[var].dims
                )
            }
            for var in metrics.data_vars
        }
        metrics.to_zarr(temp_path, mode='w', encoding=encoding)
      else:
        metrics.to_zarr(temp_path, mode='w')
      if gfile.Exists(target_path):
        gfile.DeleteRecursively(target_path)
      gfile.Rename(temp_path, target_path)
    else:
      beam_utils.atomic_write(
          target_path,
          metrics.to_netcdf(),  # pyrefly: ignore[bad-argument-type]
      )
    logging.info('Worker finished writing metrics to %s', target_path)
    return []


class WriteAggregationState(beam.DoFn):
  """Writes the final AggregationState to a NetCDF or Zarr file."""

  def __init__(self, out_path: str | Mapping[str, str]):
    self.out_path = out_path

  def process(
      self, element: tuple[str | None, aggregation.AggregationState]
  ) -> Iterable[Never]:
    agg_name, aggregation_state = element
    aggregation_state_ds = aggregation_state.to_dataset()
    # Remove attributes that may have been propogated from the targets or
    # predictions.
    aggregation_state_ds = aggregation_state_ds.drop_attrs(deep=True)
    target_path = _resolve_out_path(self.out_path, agg_name)
    logging.info(
        'Worker writing aggregation state for aggregator %s to %s (data_vars: %s, dims: %s)',
        agg_name,
        target_path,
        list(aggregation_state_ds.data_vars),
        dict(aggregation_state_ds.sizes),
    )
    if target_path.endswith('.zarr'):
      temp_path = f'{target_path}.tmp_{uuid.uuid4().hex}'
      if {'latitude', 'longitude'}.issubset(aggregation_state_ds.dims):
        chunk_spec = {'latitude': 181, 'longitude': 360}
        encoding = {
            var: {
                'chunks': tuple(
                    min(
                        aggregation_state_ds[var].sizes[dim],
                        chunk_spec.get(
                            dim, aggregation_state_ds[var].sizes[dim]
                        ),
                    )
                    for dim in aggregation_state_ds[var].dims
                )
            }
            for var in aggregation_state_ds.data_vars
        }
        aggregation_state_ds.to_zarr(temp_path, mode='w', encoding=encoding)
      else:
        aggregation_state_ds.to_zarr(temp_path, mode='w')
      if gfile.Exists(target_path):
        gfile.DeleteRecursively(target_path)
      gfile.Rename(temp_path, target_path)
    else:
      beam_utils.atomic_write(
          target_path,
          aggregation_state_ds.to_netcdf(),  # pyrefly: ignore[bad-argument-type]
      )
    return []


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
    short_circuit_aggregation_state: bool = False,
    overwrite_if_exists: bool = False,
) -> None:
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
      out_path, the aggregator name will be appended to the filename to get
      a path for each aggregator.
    aggregation_state_out_path: The full path to write the final aggregation
      state to (or mapping of aggregator name to path). This can be useful if
      you want to compute further metrics from it later, and if you are
      preserving init_time, it can be useful to compute confidence intervals
      from later too. Behaviour is the same as for out_path if multiple
      aggregators are specified.
    setup_fn: (Optional) A function to call once per worker in
      LoadPredictionsAndTargets.
    short_circuit_aggregation_state: Whether to stream per-chunk AggregationState
      datasets directly to Zarr using xarray_beam.ChunksToZarr, bypassing
      ConcatPerStatisticPerVariable and ReconstructAggregationState.
    overwrite_if_exists: Whether to delete pre-existing output Zarr store directories.
  """
  if isinstance(aggregator, Mapping):
    if isinstance(out_path, Mapping) and out_path.keys() != aggregator.keys():
      raise ValueError("Keys of out_path don't match aggregator names.")
    if (isinstance(aggregation_state_out_path, Mapping) and
        aggregation_state_out_path.keys() != aggregator.keys()):
      raise ValueError(
          "Keys of aggregation_state_out_path don't match aggregator names.")

  if short_circuit_aggregation_state:
    if aggregation_state_out_path is None:
      raise ValueError(
          'aggregation_state_out_path must be specified when'
          ' short_circuit_aggregation_state=True.'
      )

    if isinstance(aggregator, aggregation.Aggregator):
      aggregators = {None: aggregator}
    else:
      aggregators = aggregator

    # Controller-process cleanup if target paths exist and overwrite is True
    for agg_name in aggregators.keys():
      target_path = _resolve_out_path(aggregation_state_out_path, agg_name)
      if overwrite_if_exists and gfile.Exists(target_path):
        logging.info('Deleting existing Zarr store at %s', target_path)
        gfile.DeleteRecursively(target_path)

    chunk_pairs = (
        root
        | 'CreateTimeChunks' >> beam.Create(times.iter_with_chunk_offsets())
        | beam.ParDo(
            LoadPredictionsAndTargets(
                predictions_loader, targets_loader, setup_fn=setup_fn
            )
        )
        | beam.ParDo(
            ComputeStatisticsAggregateAndPrepareForCombine(metrics, aggregator)
        )
    )

    for agg_name, agg_obj in aggregators.items():
      target_path = _resolve_out_path(aggregation_state_out_path, agg_name)
      template = _get_aggregation_state_template_dataset(
          metrics=metrics,
          predictions_loader=predictions_loader,
          targets_loader=targets_loader,
          times=times,
          aggregator=agg_obj,
          setup_fn=setup_fn,
      )

      if len(aggregators) > 1:
        agg_chunk_pairs = (
            chunk_pairs
            | f'FilterAgg_{agg_name}'
            >> beam.Filter(
                lambda kv, name=agg_name: kv[0].aggregator_name == name
            )
        )
      else:
        agg_chunk_pairs = chunk_pairs

      formatted_chunks = (
          agg_chunk_pairs
          | f'FormatAggregationStateChunks_{agg_name}'
          >> beam.ParDo(FormatAggregationStateForXarrayBeam())
      )

      _ = (
          formatted_chunks
          | f'WriteAggregationStateToZarr_{agg_name}'
          >> xbeam.ChunksToZarr(
              target_path,
              template=template,
              zarr_chunks=None,
          )
      )

    return

  agg_state_pipeline = (
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
      | beam.ParDo(
          ComputeStatisticsAggregateAndPrepareForCombine(metrics, aggregator)
      )
      # Sum up the statistic DataArrays over dimensions of the TimeChunks that
      # we are reducing over, typically just init_time but can also be
      # lead_time. This is done separately for each statistic, each variable,
      # and each chunk offset along dimensions not being reduced over (e.g.
      # typically lead_time is not reduced over)
      | 'SumPerStatisticPerVariableAndPerUnreducedOffset'
      >> beam.CombinePerKey(beam_utils.CombiningSum())
      # Now we've reduced the size of the data as much as we can by summing,
      # we concatenate the resulting chunks along any remaining dimensions where
      # we know that coordinates will not overlap across chunks.
      | ConcatPerStatisticPerVariable()
      # Finally we gather together all the concatenated chunks for all
      # statistics and variables and reconstitute the full AggregationState
      # from them, which we can use to compute the final values of metrics.
      | ReconstructAggregationState()
  )

  if out_path is None and aggregation_state_out_path is None:
    raise ValueError(
        'At least one of (metrics) out_path or aggregation_state_out_path must '
        'be specified.'
    )

  if out_path is not None:
    _ = (
        agg_state_pipeline
        | 'ComputeMetrics' >> beam.ParDo(ComputeMetrics(metrics))
        | 'WriteMetrics' >> beam.ParDo(WriteMetrics(out_path))
    )

  if aggregation_state_out_path is not None:
    _ = agg_state_pipeline | beam.ParDo(
        WriteAggregationState(aggregation_state_out_path)
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
      if lead_times.step is not None:
        lead_times = np.arange(
            lead_times.start, lead_times.stop + lead_times.step, lead_times.step
        )
      elif 'lead_time' in first_chunk.coords:
        lead_times = first_chunk.lead_time.values
      else:
        lead_times = first_lead_times
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
  zarr_chunks = stat_chunks | zarr_chunks  # pyrefly: ignore[unsupported-operation]

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
