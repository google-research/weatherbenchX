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
"""Wrappers for statistics that transform the inputs.

Example to compute binary metrics from a continuous ensemble prediction and
a continuous ground truth:

wrappers.WrappedMetric(
    categorical.CSI(),
    [
        wrappers.ContinuousToBinary(
            which='both',
            threshold_value=[0, 50],
            threshold_dim='threshold_value'
        ),
        wrappers.EnsembleMean(
            which='predictions', ensemble_dim='realization'
        ),
        wrappers.ContinuousToBinary(
            which='predictions',
            threshold_value=[0.25, 0.75],
            threshold_dim='threshold_probability'
        ),
    ],
)
"""

import abc
from typing import Hashable, Iterable, Mapping, Union
import numpy as np
from weatherbenchX import xarray_tree
from weatherbenchX.metrics import base
import xarray as xr


def binarize_thresholds(
    x: xr.DataArray,
    thresholds: Iterable[float],
    threshold_dim: str,
) -> xr.DataArray:
  """Binarizes a continuous array using a threshold value or a list of values.

  Note that this retains NaNs in the input array. If NaNs are present, the
  output will be of type float otherwise bool.

  Args:
    x: Input DataArray.
    thresholds: List of threshold values.
    threshold_dim: Name of dimension to use for threshold values.

  Returns:
    binary_x: Binarized DataArray.
  """
  threshold = xr.DataArray(
      thresholds, dims=[threshold_dim], coords={threshold_dim: thresholds}
  )
  return (x > threshold).where(~np.isnan(x))


# Transforms
class InputTransform(abc.ABC):
  """Base class for input transformations."""

  def __init__(self, which):
    """Init.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
    """
    if which not in ['predictions', 'targets', 'both']:
      raise ValueError(f'Invalid value for `which`: {which}')
    self.which = which

  @property
  @abc.abstractmethod
  def unique_name_suffix(self) -> str:
    """Add a suffix to unique statistics name."""

  @abc.abstractmethod
  def tranform_fn(self, da: xr.DataArray) -> xr.DataArray:
    """Function to apply to predictions and/or targets."""


class EnsembleMean(InputTransform):
  """Compute ensemble mean."""

  def __init__(self, which, ensemble_dim='number', skipna=False):
    """Init.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      ensemble_dim: Name of ensemble dimension. Default: 'number'.
      skipna: If True, skip NaNs in the ensemble mean. Default: False.
    """
    super().__init__(which)
    self._ensemble_dim = ensemble_dim
    self._skipna = skipna

  @property
  def unique_name_suffix(self) -> str:
    return 'ensemble_mean'

  def tranform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return da.mean(self._ensemble_dim, skipna=self._skipna)


class ContinuousToBinary(InputTransform):
  """Converts a continuous input to a binary one.

  Applies x > threshold for all threholds and concatenates along a new dimension
  of name `threshold_dim`.
  """

  def __init__(
      self,
      which: str,
      threshold_value: Union[float, Iterable[float]],
      threshold_dim: str,
  ):
    """Init.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      threshold_value: Threshold value or list of values.
      threshold_dim: Name of dimension to use for threshold values.
    """
    super().__init__(which)
    # Convert to list if it isn't already.
    self._threshold_value = (
        threshold_value
        if isinstance(threshold_value, Iterable)
        else [threshold_value]
    )
    self._threshold_dim = threshold_dim

  @property
  def unique_name_suffix(self) -> str:
    threshold_value_str = ','.join([str(t) for t in self._threshold_value])
    return f'{self._threshold_dim}={threshold_value_str}'

  def tranform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return binarize_thresholds(da, self._threshold_value, self._threshold_dim)


class WrappedStatistic(base.Statistic):
  """Wraps a statistic with an input transform.

  Also adds suffix to unique name.
  """

  def __init__(self, statistic: base.Statistic, transform: InputTransform):
    """Init.

    Args:
      statistic: Statistic object to wrap.
      transform: Transform to apply to inputs.
    """
    self.statistic = statistic
    self.transform = transform

  @property
  def unique_name(self) -> str:
    return f'{self.statistic.unique_name}_{self.transform.which}_{self.transform.unique_name_suffix}'

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    if self.transform.which in ('predictions', 'both'):
      predictions = xarray_tree.map_structure(
          self.transform.tranform_fn,
          predictions,
      )
    if self.transform.which in ('targets', 'both'):
      targets = xarray_tree.map_structure(
          self.transform.tranform_fn,
          targets,
      )
    return self.statistic.compute(predictions, targets)


class WrappedMetric(base.Metric):
  """Wraps all statistics of a metric with input transforms."""

  def __init__(self, metric: base.Metric, transforms: list[InputTransform]):
    """Init.

    Args:
      metric: Metric to wrap.
      transforms: List of input transforms to apply. The transforms will be
        applied in the order they are listed, i.e. the first transform in the
        list will be applied first.
    """
    self.metric = metric
    self.transforms = transforms

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    stats = {}
    for name, stat in self.metric.statistics.items():
      # Apply wrappers in reverse order since the last one will be called first.
      for wrapper in self.transforms[::-1]:
        stat = WrappedStatistic(stat, wrapper)
      stats[name] = stat
    return stats

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self.metric._values_from_mean_statistics_with_internal_names(  # pylint: disable=protected-access
        statistic_values
    )
