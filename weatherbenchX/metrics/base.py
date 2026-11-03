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
"""Base metrics class."""

import abc
from collections.abc import Iterator, Mapping
from typing import Hashable, final
from weatherbenchX import xarray_tree
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


class PerVariableStatistic(Statistic):
  """Abstract base class for statistics that are computed per variable.

  The statistic will be computed independently for each variable that is
  present in both predictions and targets.
  """

  @final
  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    """Maps computation over all variables."""
    result = {}
    for var_name in predictions.keys():
      if var_name in targets.keys():
        per_var_result = self._compute_per_variable(
            predictions[var_name], targets[var_name])
        if per_var_result is not None:
          result[var_name] = per_var_result
    return result

  @abc.abstractmethod
  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray | None:
    """Computes statistic for a variable, or None if it's not defined."""


class PerVariableMetric(Metric):
  """Abstract base class for metrics that are computed per variable."""

  @final
  def values_from_mean_statistics(
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
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Compute metric values for a single variable.

    Args:
      statistic_values: Mapping from your internal statistic names (the keys
        of your self.statistics) to the mean values of the statistics for a
        single specific variable.

    Returns:
      The value of the metric for the given variable.
    """


# Deprecated backwards-compatibility shim. NoOpMetric used to wrap a Statistic
# as a Metric, but this is no longer necessary as Statistic implements Metric
# directly.
NoOpMetric = lambda statistic: statistic


def update_masks_for_targets_and_predictions(
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = False,
) -> tuple[Mapping[Hashable, xr.DataArray], Mapping[Hashable, xr.DataArray]]:
  """Aligns prediction and target masks so they are consistent.

  It is strongly recommended to use allow_prediction_mask=False. To encourage
  comparability across different predictions, the same mask should be used
  across different predictions when evaluating against a given set of targets.

  Assumes that all target dimensions are present in the corresponding
  prediction. If neither prediction nor target has a mask, we do not introduce
  one. If only target has a mask, we assign it directly to prediction without
  broadcasting to extra prediction dimensions (so reductions over extra
  prediction dimensions, such as ensemble members, preserve the mask). If
  prediction has a mask, we intersect any target mask with a logical AND and
  reduce extra prediction dimensions not in target using .any().

  Args:
    predictions: Mapping from variable name to prediction DataArray.
    targets: Mapping from variable name to target DataArray.
    allow_prediction_mask: If True, allow the predictions to contain a mask
      coordinate. If False, only allow the targets to contain a mask
      coordinate.

  Returns:
    A tuple of (prediction, target).

  Raises:
    ValueError: If target has dimensions not in prediction when aligning masks,
      or if a prediction has a 'mask' coordinate and allow_prediction_mask is
      False.
  """

  def _align_masks_for_target_and_prediction(
      prediction: xr.DataArray,
      target: xr.DataArray,
  ) -> tuple[xr.DataArray, xr.DataArray]:
    prediction_has_mask = 'mask' in prediction.coords
    target_has_mask = 'mask' in target.coords
    if not prediction_has_mask and not target_has_mask:
      return prediction, target

    if target_has_mask:
      if prediction_has_mask:
        prediction_mask_dims = prediction['mask'].dims
      else:
        prediction_mask_dims = prediction.dims
      extra_target_dims = tuple(
          d for d in target['mask'].dims if d not in prediction_mask_dims
      )
      if extra_target_dims:
        if prediction_has_mask:
          expected_dims_label = 'prediction mask dims'
        else:
          expected_dims_label = 'prediction dims'
        raise ValueError(
            f'target mask dims must be a subset of {expected_dims_label}. '
            f'for variable {prediction.name!r}, got extra target dims '
            f'{extra_target_dims} not in {prediction_mask_dims=}.'
        )

    if prediction_has_mask and not allow_prediction_mask:
      raise ValueError(
          f'prediction {prediction.name!r} has a "mask" coordinate, '
          f'but {allow_prediction_mask=}.'
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
      prediction_mask = target['mask']
      target_mask = target['mask']
    elif prediction_has_mask and not target_has_mask:
      prediction_mask = prediction['mask']
      target_mask = _reduce_extra_prediction_dims(prediction_mask)
    else:
      prediction_mask = prediction['mask']
      target_mask = target['mask']
      mask_attrs = prediction_mask.attrs | target_mask.attrs
      prediction_mask = (prediction_mask & target_mask).assign_attrs(mask_attrs)
      target_mask = _reduce_extra_prediction_dims(prediction_mask)

    return (
        prediction.assign_coords(
            xr.Coordinates(coords={'mask': prediction_mask}, indexes={})
        ),
        target.assign_coords(
            xr.Coordinates(coords={'mask': target_mask}, indexes={})
        ),
    )

  predictions = dict(predictions)
  targets = dict(targets)
  for k in predictions.keys() & targets.keys():
    predictions[k], targets[k] = _align_masks_for_target_and_prediction(
        predictions[k], targets[k]
    )
  return predictions, targets


def generate_unique_statistics_for_all_metrics(
    metrics: Mapping[str, Metric],
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = False,
) -> Iterator[tuple[str, Mapping[Hashable, xr.DataArray]]]:
  """Like compute_unique_statistics_for_all_metrics, but yields k/v pairs.

  It is highly recommended to use allow_prediction_mask=False. This is because
  in order to ensure comparability across different predictions, the same mask
  should be used across different predictions when evaluating against a given
  set of targets.

  Note on masking: This function ensures that the masks of the targets the
  predictions are consistent (if present) before passing them to the statistics.
  By consistent, we mean we use the same mask for both predictions and targets.
  If only one of the predictions or targets has a mask, we add a mask to the
  other. It is the responsibility of the statistics user to ensure the masks are
  correctly handled downstream.

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
      predictions, targets, allow_prediction_mask,
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
    metrics: Mapping[str, Metric],
    predictions: Mapping[Hashable, xr.DataArray],
    targets: Mapping[Hashable, xr.DataArray],
    allow_prediction_mask: bool = False,
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
  return {k: stat for k, stat in generate_unique_statistics_for_all_metrics(
      metrics, predictions, targets, allow_prediction_mask)}


def compute_metric_from_statistics(
    metric: Metric,
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
    metrics: Mapping[str, Metric],
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

  @final
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

  @final
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

    # Climatology either has time, dayofyear or dayofyear/hour dimensions
    if hasattr(climatology, 'time'):
      sel_kwargs = {'time': valid_time}
    else:
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
