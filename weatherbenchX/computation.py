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
"""Helper functions for metric and statistic computation."""

from collections.abc import Iterator, Mapping
from typing import Hashable
from weatherbenchX.metrics import interfaces
import xarray as xr


def update_masks_for_targets_and_predictions(
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = True,
) -> tuple[Mapping[Hashable, xr.DataArray], Mapping[Hashable, xr.DataArray]]:
  """Aligns prediction and target masks so they are consistent.

  It is strongly recommended to use allow_prediction_mask=False. To encourage
  comparability across different predictions, the same mask should be used
  across different predictions when evaluating against a given set of targets.

  Assumes that all target mask dimensions are present in the corresponding
  prediction. If neither prediction nor target has a mask, we do not introduce
  one. If only target has a mask, we assign it directly to prediction without
  broadcasting to extra prediction dimensions (so reductions over extra
  prediction dimensions, such as ensemble members, preserve the mask). If
  prediction has a mask, we verify it is constant along any extra dimensions
  not in target, reduce those dimensions using .all(), intersect with any
  target mask using a logical AND, and assign the resulting mask to both
  prediction and target.

  Args:
    predictions: Mapping from variable name to prediction DataArray.
    targets: Mapping from variable name to target DataArray.
    allow_prediction_mask: If True, allow the predictions to contain a mask
      coordinate. If False, only allow the targets to contain a mask
      coordinate.

  Returns:
    A tuple of (predictions, targets).

  Raises:
    ValueError: If target mask has dimensions not in prediction, if a
      prediction has a 'mask' coordinate and allow_prediction_mask is False, or
      if a prediction mask varies along extra dimensions not in target.
  """

  def _align_masks_for_target_and_prediction(
      prediction: xr.DataArray,
      target: xr.DataArray,
  ) -> tuple[xr.DataArray, xr.DataArray]:
    prediction_has_mask = 'mask' in prediction.coords
    target_has_mask = 'mask' in target.coords
    if not prediction_has_mask and not target_has_mask:
      return prediction, target

    if prediction_has_mask and not allow_prediction_mask:
      raise ValueError(
          f'prediction {prediction.name!r} has a "mask" coordinate, '
          f'but {allow_prediction_mask=}.'
      )

    if target_has_mask:
      extra_target_dims = tuple(
          d for d in target['mask'].dims if d not in prediction.dims
      )
      if extra_target_dims:
        raise ValueError(
            f'target mask dims must be a subset of prediction dims for '
            f'variable {prediction.name!r}, got extra target mask dims '
            f'{extra_target_dims} not in {prediction.dims=}.'
        )

    def _reduce_extra_prediction_dims(mask: xr.DataArray) -> xr.DataArray:
      extra_dims = tuple(d for d in mask.dims if d not in target.dims)
      if extra_dims:
        reduced = mask.all(dim=extra_dims, keep_attrs=True)
        if not reduced.variable.broadcast_equals(mask.variable):
          raise ValueError(
              'prediction mask must be constant along its extra dimensions.'
          )
        return reduced
      return mask

    if target_has_mask and not prediction_has_mask:
      mask = target['mask']
    elif prediction_has_mask and not target_has_mask:
      mask = _reduce_extra_prediction_dims(prediction['mask'])
    else:
      prediction_mask = _reduce_extra_prediction_dims(prediction['mask'])
      target_mask = target['mask']
      mask_attrs = prediction_mask.attrs | target_mask.attrs
      mask = (prediction_mask & target_mask).assign_attrs(mask_attrs)

    # Reorder mask to match data's coordinate order without filling.
    def _assign_mask(
        data: xr.DataArray,
        mask_to_assign: xr.DataArray,
    ) -> xr.DataArray:
      for dim in mask_to_assign.dims:
        if dim in data.indexes and dim in mask_to_assign.indexes:
          if not data.indexes[dim].equals(mask_to_assign.indexes[dim]):
            mask_to_assign = mask_to_assign.sel({dim: data.indexes[dim]})
      return data.assign_coords(
          xr.Coordinates(coords={'mask': mask_to_assign}, indexes={})
      )

    return _assign_mask(prediction, mask), _assign_mask(target, mask)

  predictions = dict(predictions)
  targets = dict(targets)
  for k in predictions.keys() & targets.keys():
    predictions[k], targets[k] = _align_masks_for_target_and_prediction(
        predictions[k], targets[k]
    )
  return predictions, targets


def generate_unique_statistics_for_all_metrics(
    metrics: Mapping[str, interfaces.Metric],
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = True,
) -> Iterator[tuple[str, Mapping[Hashable, xr.DataArray]]]:
  """Like compute_unique_statistics_for_all_metrics, but yields k/v pairs.

  It is strongly recommended to use allow_prediction_mask=False. This is because
  in order to ensure comparability across different predictions, the same mask
  should be used across different predictions when evaluating against a given
  set of targets.

  Note on masking: This function ensures that the masks of the targets and
  predictions are consistent (if present) before passing them to the statistics.
  By consistent, we mean we use the same mask for both predictions and targets.
  If only one of the predictions or targets has a mask, we add a mask to the
  other. It is the responsibility of the statistics implementors to ensure
  the masks are correctly handled downstream.

  Args:
    metrics: Dictionary of metrics instances.
    predictions: Xarray Dataset or dictionary of DataArrays.
    targets: Xarray Dataset or dictionary of DataArrays.
    allow_prediction_mask: If True, allow the predictions to contain a mask
      coordinate, and update the mask for *both* the predictions and targets,
      to be the logical AND of the two. If False, only allow the targets to
      contain a mask coordinate.

  Yields:
    Pairs of statistic name and statistic values.
  """
  predictions, targets = update_masks_for_targets_and_predictions(
      predictions,
      targets,
      allow_prediction_mask,
  )
  unique_statistics = {}
  for m in metrics.values():
    for _, stat in m.statistics.items():
      unique_statistics[stat.unique_name] = stat
  for k, stat in unique_statistics.items():
    try:
      yield k, stat.compute(predictions, targets)
    except Exception as e:
      raise ValueError(
          'Failed to compute statistic'
          f' {k}={stat} from:\n{predictions=}\n{targets=}'
      ) from e


def compute_unique_statistics_for_all_metrics(
    metrics: Mapping[str, interfaces.Metric],
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = True,
) -> Mapping[str, Mapping[Hashable, xr.DataArray]]:
  """Computes unique statistics for all metrics.

  Args:
    metrics: Dictionary of metrics instances.
    predictions: Xarray Dataset or dictionary of DataArrays.
    targets: Xarray Dataset or dictionary of DataArrays.
    allow_prediction_mask: If True, allow the predictions to contain a mask
      coordinate, and update the mask for *both* the predictions and targets,
      to be the logical AND of the two. If False, only allow the targets to
      contain a mask coordinate.

  Returns:
    statistic_values: Unique statistics computed for each input element. If
      inputs are Datasets, returns a dict of statistic_name to statistic
      Dataset; if inputs are dictionaries, returns a nested dictionary of
      statistic_name to variable to statistic DataArrays.
  """
  return {
      k: stat
      for k, stat in generate_unique_statistics_for_all_metrics(
          metrics, predictions, targets, allow_prediction_mask
      )
  }


def compute_metric_from_statistics(
    metric: interfaces.Metric,
    statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
) -> Mapping[Hashable, xr.DataArray]:
  """Computes values of a metric from mean statistics keyed by .unique_name.

  This handles re-keying the statistics by the internal names used by the
  `metric`.

  Args:
    metric: A Metric.
    statistic_values: Values of statistics keyed by their .unique_name,
      for example as returned by compute_unique_statistics_for_all_metrics.

  Returns:
    The resulting values of the `metric`.
  """
  # Rename statistics from unique to internal names.
  statistic_values = {
      k: statistic_values[v.unique_name] for k, v in metric.statistics.items()
  }
  return metric.values_from_mean_statistics(statistic_values)


def compute_metrics_from_statistics(
    metrics: Mapping[str, interfaces.Metric],
    statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
) -> Mapping[str, Mapping[Hashable, xr.DataArray]]:
  """Computes multiple metrics from mean statistics keyed by .unique_name.

  Args:
    metrics: A mapping of multiple `Metric`s.
    statistic_values: Values of statistics keyed by their .unique_name,
      for example as returned by compute_unique_statistics_for_all_metrics.

  Returns:
    The resulting values of the `metrics`.
  """
  return {
      metric_name: compute_metric_from_statistics(metric, statistic_values)
      for metric_name, metric in metrics.items()
  }
