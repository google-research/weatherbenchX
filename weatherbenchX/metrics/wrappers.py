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
from collections.abc import Sequence
from typing import Any, Callable, Hashable, Iterable, Mapping, Union
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
  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    """Function to apply to predictions and/or targets."""


class EnsembleMean(InputTransform):
  """Compute ensemble mean."""

  def __init__(self, which: str, ensemble_dim='number', skipna=False):
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
    return f'ensemble_mean_{self._ensemble_dim=}_{self._skipna=}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
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

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return binarize_thresholds(da, self._threshold_value, self._threshold_dim)


class Inline(InputTransform):
  """Transform data with a provided function.

  Example:
    # Negate values
    x = xr.DataArray(...)
    y = Inline('both', lambda da: -da, 'negate').transform_fn(x)
    xr.testing.assert_equal(y, -x)
  """

  def __init__(
      self,
      which: str,
      transform_fn: Callable[[xr.DataArray], xr.DataArray],
      unique_name_suffix: str,
  ):
    """Initializes an Inline transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      transform_fn: Function to transform a dataarray.
      unique_name_suffix: Name to give this transform. Should be different than
        any other transform used. Should uniquely identify this class and all
        init args (except `which`).
    """
    super().__init__(which)
    self._transform_fn = transform_fn
    self._unique_name_suffix = unique_name_suffix

  @property
  def unique_name_suffix(self) -> str:
    return f'{self._unique_name_suffix}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return self._transform_fn(da)


class ReLU(InputTransform):
  """Transform data by passing through a rectified linear unit."""

  def __init__(  # pylint: disable=useless-parent-delegation
      self,
      which: str,
  ):
    """Initializes a ReLU transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
    """
    super().__init__(which)

  @property
  def unique_name_suffix(self) -> str:
    return 'relu'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return xr.where(da > 0, da, 0).where(~np.isnan(da))


class ShiftAlongNewDim(InputTransform):
  """Transforms by shifting along a new dimension, possibly more than once.

  This transform adds a new dimension `shift_dim` to the input DataArray, and
  shifts the values along this dimension. The `shift_value` can be a constant,
  a list of constants, or an xarray.Dataset.

  Examples:

    x = xr.DataArray(...)
    x.sizes
    ==> Frozen({'time': 366, 'latitude': 19, 'longitude': 36})

    # Shift every value by a constant (=1), add a singleton dim.
    y = ShiftAlongNewDim('both', shift_value=1, shift_dim='threshold')
    y.sizes
    ==> Frozen({'time': 366, 'latitude': 19, 'longitude': 36, 'threshold': 1})
    y.threshold
    ==> <xarray.DataArray 'threshold' (threshold: 1)> Size: 8B
        array([0.5])
        Coordinates:
          * threshold  (threshold) float64 8B 0.5

    # Shift by two different values, indexing the shift by dim 'threshold'.
    y = ShiftAlongNewDim('both', shift_value=[10., 20.], shift_dim='threshold')
    y.sizes
    ==> Frozen({'time': 366, 'latitude': 19, 'longitude': 36, 'threshold': 2})
    y.threshold
    ==> <xarray.DataArray 'threshold' (threshold: 2)> Size: 16B
        array([10., 20.])
        Coordinates:
          * threshold  (threshold) float64 8B 10.0 20.0

    # Shift by adding a DataArray. In this case the quantiles of another Dataset
    quantiles = xr.Dataset(...).quantile([0.2, 0.8])
    y = ShiftAlongNewDim('both', shift_value=quantiles, shift_dim='threshold')
    y.sizes
    ==> Frozen({'time': 366, 'latitude': 19, 'longitude': 36, 'quantile': 2})
    y['quantile']
    ==> <xarray.DataArray 'quantile' (quantile: 2)> Size: 16B
        array([0.2, 0.8])
        Coordinates:
          * quantile  (quantile) float64 8B 0.2 0.8
  """

  def __init__(
      self,
      which: str,
      shift_value: Union[float, Iterable[float], xr.Dataset],
      shift_dim: str,
      unique_name_suffix: str,
  ):
    """Initializes a ShiftAlongNewDim transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      shift_value: Constant value, list of values or xarray.Dataset.
      shift_dim: Name of dimension to use for the shift. The output will have a
        new dimension, `shift_dim`.
      unique_name_suffix: Name to give this transform. Should be different than
        any other transform used. Should uniquely identify this class and all
        init args (except `which`).
    """
    super().__init__(which)
    self._shift_value = (
        shift_value
        if isinstance(shift_value, (Iterable, xr.Dataset))
        else [shift_value]
    )
    self._shift_dim = shift_dim
    self._unique_name_suffix = unique_name_suffix

  @property
  def unique_name_suffix(self) -> str:
    return self._unique_name_suffix

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    # Convert self._shifts to a DataArray in all cases.
    if isinstance(self._shift_value, xr.Dataset):
      shifts = self._shift_value[da.name]
      if self._shift_dim not in shifts.dims:
        raise RuntimeError(
            f'Expected to find {self._shift_dim=} in {shifts.dims=} but did'
            ' not. This is probably an error.'
        )
      shifts = self._shift_value[da.name]
    else:
      shifts = xr.DataArray(
          self._shift_value,
          dims=[self._shift_dim],
          coords={self._shift_dim: self._shift_value},
      )

    return da + shifts


class Rename(InputTransform):
  """Renames variables, coordinates and dimensions with xr.rename."""

  def __init__(
      self,
      which: str,
      renames: Mapping[Hashable, Hashable],
  ):
    """Initializes a Rename transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      renames: Map from OLD to NEW name.
    """
    super().__init__(which)
    self._renames = renames

  @property
  def unique_name_suffix(self) -> str:
    return f'rename_{self._renames}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return da.rename(self._renames)


class Select(InputTransform):
  """Selects data by applying xr.sel and/or xr.isel."""

  def __init__(
      self,
      which: str,
      sel: Mapping[Hashable, Any] | None = None,
      isel: Mapping[Hashable, Any] | None = None,
      sel_kwargs: Mapping[Hashable, Any] | None = None,
      isel_kwargs: Mapping[Hashable, Any] | None = None,
  ):
    """Initializes a Select transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      sel: Selector to pass to xr.sel.
      isel: Selector to pass to xr.isel.
      sel_kwargs: Additional kwargs to pass to xr.sel.
      isel_kwargs: Additional kwargs to pass to xr.isel.
    """
    super().__init__(which)
    self._isel = isel
    self._sel = sel
    self._isel_kwargs = isel_kwargs or {}
    self._sel_kwargs = sel_kwargs or {}

  @property
  def unique_name_suffix(self) -> str:
    return f'select_{self._isel=}_{self._isel_kwargs=}_{self._sel=}_{self._sel_kwargs=}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    da = da.copy()
    if self._sel is not None:
      da = da.sel(self._sel, **self._sel_kwargs)
    if self._isel is not None:
      da = da.isel(self._isel, **self._isel_kwargs)
    return da


class StackToNewDimension(InputTransform):
  """Stacks any number of existing dimensions into a "range" new dimension.

  Similar to xr.stack, except StackToNewDimension...
  * allows the new dimension name to be the same as one of the stacked dims.
  * results in a dimension with values [0, 1, ...,]
  """

  def __init__(
      self,
      which: str,
      dims_to_stack: Sequence[Hashable],
      new_dim_name: Hashable,
  ):
    """Initializes a StackToNewDimension Transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      dims_to_stack: Sequence of dimensions to stack.
      new_dim_name: Name of stacked dimension.
    """
    super().__init__(which)
    self._dims_to_stack = dims_to_stack
    self._temporary_new_dim_name = 'STACK_OF_' + '_AND_'.join(
        (str(d) for d in self._dims_to_stack)
    )
    self._new_dim_name = new_dim_name

  @property
  def unique_name_suffix(self) -> str:
    return f'stack_{self._dims_to_stack}_to_{self._new_dim_name}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    stacked = da.stack({self._temporary_new_dim_name: self._dims_to_stack})
    return stacked.drop_vars(self._dims_to_stack).rename(
        {self._temporary_new_dim_name: self._new_dim_name}
    )


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
          self.transform.transform_fn,
          predictions,
      )
    if self.transform.which in ('targets', 'both'):
      targets = xarray_tree.map_structure(
          self.transform.transform_fn,
          targets,
      )
    return self.statistic.compute(predictions, targets)


class RenamedStatistic(base.Statistic):
  """Wraps a statistic with a new unique name."""

  def __init__(self, statistic: base.Statistic, unique_name: str):
    self._statistic = statistic
    self._unique_name = unique_name

  @property
  def unique_name(self) -> str:
    return self._unique_name

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self._statistic.compute(predictions, targets)


class WrappedMetric(base.Metric):
  """Wraps all statistics of a metric with input transforms."""

  def __init__(
      self,
      metric: base.Metric,
      transforms: list[InputTransform],
      unique_name_suffix: str | None = None,
  ):
    """Init.

    Args:
      metric: Metric to wrap.
      transforms: List of input transforms to apply. The transforms will be
        applied in the order they are added to the list. I.e. transforms [f, g,
        h], transform x as h(g(f(x))).
      unique_name_suffix: Optional suffix to use for uniquely naming all
        associated statistics. By default, this is constructed automatically
        from the transforms, which may be overly verbose.
    """
    self.metric = metric
    self.transforms = transforms
    self.unique_name_suffix = unique_name_suffix

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    stats = {}
    for name, stat in self.metric.statistics.items():
      # Apply wrappers in reverse order since the last one will be called first
      # in subsequent code...i.e., if stat = W(V(stat)), then the final stat is
      # computed as x --> V(W(x)).
      original_name = stat.unique_name
      for wrapper in self.transforms[::-1]:
        stat = WrappedStatistic(stat, wrapper)
      if self.unique_name_suffix is not None:
        unique_name = f'{original_name}_{self.unique_name_suffix}'
        stat = RenamedStatistic(stat, unique_name)
      stats[name] = stat
    return stats

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self.metric._values_from_mean_statistics_with_internal_names(  # pylint: disable=protected-access
        statistic_values
    )


class SubselectVariablesForStatistic(base.Statistic):
  """Only compute variables for a subset of variables."""

  def __init__(self, statistic: base.Statistic, variables: Sequence[str]):
    """Init.

    Args:
      statistic: Statistic object to wrap.
      variables: Variables to compute the statistic for.
    """
    self.statistic = statistic
    self.variables = variables

  @property
  def unique_name(self) -> str:
    # Make sure to change unique name in case there is another, non-subsetted
    # statistic with the same name.
    variables_str = '_'.join(self.variables)
    return f'{self.statistic.unique_name}_{variables_str}'

  def compute(
      self,
      predictions: Mapping[Hashable, xr.DataArray],
      targets: Mapping[Hashable, xr.DataArray],
  ) -> Mapping[Hashable, xr.DataArray]:
    predictions = {k: v for k, v in predictions.items() if k in self.variables}
    targets = {k: v for k, v in targets.items() if k in self.variables}
    return self.statistic.compute(predictions, targets)


class SubselectVariables(base.Metric):
  """Only compute metric for a subset of variables."""

  def __init__(self, metric: base.Metric, variables: Sequence[str]):
    """Init.

    Args:
      metric: Metric to wrap.
      variables: Variables to compute the metric for.
    """
    self.metric = metric
    self.variables = variables

  @property
  def statistics(self) -> Mapping[Hashable, base.Statistic]:
    stats = {}
    for name, stat in self.metric.statistics.items():
      stat = SubselectVariablesForStatistic(stat, self.variables)
      stats[name] = stat
    return stats

  def _values_from_mean_statistics_with_internal_names(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self.metric._values_from_mean_statistics_with_internal_names(  # pylint: disable=protected-access
        statistic_values
    )
