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
    thresholds: Union[Iterable[float], xr.DataArray, xr.Dataset],
    threshold_dim: str,
) -> xr.DataArray:
  """Binarizes a continuous array using a threshold value or a list of values.

  Note that this retains NaNs in the input array. If NaNs are present, the
  output will be of type float otherwise bool.

  Args:
    x: Input DataArray.
    thresholds: List, xarray.DataArray or xarray.Dataset of threshold values.
    threshold_dim: Name of dimension to use for threshold values.

  Returns:
    binary_x: Binarized DataArray.
  """
  if isinstance(thresholds, xr.Dataset):
    assert threshold_dim in thresholds.dims, (
        f'threshold_dim ({threshold_dim}) not found in thresholds'
        f' ({thresholds.dims})'
    )
    assert x.name in thresholds.data_vars, (
        f'Input DataArray name ({x.name}) not found in thresholds'
        f' ({thresholds.data_vars})'
    )
    threshold = thresholds[x.name]
  elif isinstance(thresholds, xr.DataArray):
    assert threshold_dim in thresholds.dims, (
        f'threshold_dim ({threshold_dim}) not found in thresholds'
        f' ({thresholds.dims})'
    )
    threshold = thresholds
  else:
    threshold = xr.DataArray(
        thresholds, dims=[threshold_dim], coords={threshold_dim: thresholds}
    )
  return (x > threshold).where(~np.isnan(x)).astype(np.float32)


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
      threshold_value: Union[float, Iterable[float], xr.DataArray, xr.Dataset],
      threshold_dim: str,
  ):
    """Init.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      threshold_value: Threshold value, list of values, xarray.DataArray or
        xarray.Dataset.
      threshold_dim: Name of dimension to use for threshold values.
    """
    super().__init__(which)
    # Convert to list if it isn't already.
    self._threshold_value = (
        threshold_value
        if isinstance(threshold_value, (Iterable, xr.DataArray, xr.Dataset))
        else [threshold_value]
    )
    self._threshold_dim = threshold_dim

  @property
  def unique_name_suffix(self) -> str:
    threshold_value_str = ','.join([str(t) for t in self._threshold_value])
    return f'{self._threshold_dim}={threshold_value_str}'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    return binarize_thresholds(da, self._threshold_value, self._threshold_dim)


class ContinuousToBins(InputTransform):
  """Converts a continuous input to a binned one.

  Bins data into open-closed intervals of the form (left, right] as defined by
  `bin_thresholds`. The bins are right-inclusive, i.e.,
  `threshold[i-1] < x <= threshold[i]`. For example, if `bin_thresholds` is
  [0.5, 1.0, 1.5], then the bins will be (-np.inf, 0.5], (0.5, 1.0], (1.0, 1.5],
  and (1.5, np.inf).

  The output DataArray will have two new coordinates: `{bin_dim}_left` and
  `{bin_dim}_right`, which will contain the left and rin bin edges,
  respectively.
  """

  def __init__(
      self,
      which: str,
      bin_thresholds: Union[Iterable[float], xr.DataArray, xr.Dataset],
      bin_dim: str,
  ):
    """Initialize the transform.

    Args:
      which: Which input to apply the wrapper to. Must be one of 'predictions',
        'targets', or 'both'.
      bin_thresholds: Iterable of threshold values, xarray.DataArray or
        xarray.Dataset. If it a list or DataArray, the same bin thresholds are
        applied regardless of the input to transform_fn. If it is a Dataset,
        then transform_fn uses the bins in the data variable of bin_thresholds
        corresponding to the name of the DataArray that is passed as input to
        transform_fn, e.g. if the input to trasform_fn has name 'temperature'
        then transform_fn will look for a DataArray called 'temperature' in the
        bin_thresholds dataset, and raise an error if it is not found.
      bin_dim: Name of dimension to use for threshold values.
    """
    super().__init__(which)

    if isinstance(bin_thresholds, (xr.DataArray, xr.Dataset)):
      self._bin_thresholds = bin_thresholds

    elif isinstance(bin_thresholds, Iterable):
      # In this case, we can check that the bin_thresholds are monotonic
      # increasing up front, since they are small. We can't do this in __init__
      # in the case where bin_thresholds is a DataArray or Dataset, since the
      # size of the data may actually be large. For those cases, we check
      # monotonicity at runtime.
      if tuple(sorted(bin_thresholds)) != tuple(bin_thresholds):
        raise ValueError('bin_thresholds must be monotonic increasing')
      self._bin_thresholds = xr.DataArray(
          bin_thresholds,
          dims=[bin_dim],
          coords={bin_dim: bin_thresholds},
      )

    else:
      raise ValueError(
          'bin_thresholds must be an Iterable, DataArray, or Dataset, but found'
          f' {type(bin_thresholds)}.'
      )

    self._bin_dim = bin_dim

    # Shift integer-coordinate of the bin values by one. This is because later
    # we prepend the left (-inf) bin edge to the bin values so the coordinates
    # need to be shifted by one to account for this.
    self._bin_thresholds = self._bin_thresholds.assign_coords(
        {bin_dim: self._bin_thresholds[self._bin_dim].values + 1}
    )

    self._num_bins = self._bin_thresholds.sizes[bin_dim] + 1
    self._bin_indices = xr.DataArray(
        np.arange(self._num_bins), {bin_dim: np.arange(self._num_bins)}
    )

  @property
  def unique_name_suffix(self) -> str:
    return self._bin_dim

  def _check_bin_edges_non_nan_and_increasing(
      self, bin_thresholds: xr.DataArray
  ):
    """Check that bin edges are non-nan and increasing."""
    if bin_thresholds.isnull().any():
      raise ValueError('Found nan bin edges, which is not allowed')

    # Check that, if we have more than one bin threshold, the bin thresholds are
    # monotonic increasing.
    if bin_thresholds.size > 1:

      # Slice out left and right edges
      left_edges = bin_thresholds.isel(
          {self._bin_dim: slice(None, -1)}
      ).drop_vars(self._bin_dim)

      right_edges = bin_thresholds.isel(
          {self._bin_dim: slice(1, None)}
      ).drop_vars(self._bin_dim)

      if (left_edges >= right_edges).any():
        raise ValueError('Found non-increasing bin edges, which is not allowed')

  def _get_positive_and_negative_inf_bin_edges(
      self, bin_thresholds: xr.DataArray
  ):

    pos_inf_edge = np.inf * xr.ones_like(
        bin_thresholds.isel({self._bin_dim: 0})
    )
    pos_inf_edge = pos_inf_edge.assign_coords({self._bin_dim: self._num_bins})

    neg_inf_edge = -np.inf * xr.ones_like(
        bin_thresholds.isel({self._bin_dim: 0})
    )
    neg_inf_edge = neg_inf_edge.assign_coords({self._bin_dim: 0})

    return pos_inf_edge, neg_inf_edge

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:

    if isinstance(self._bin_thresholds, xr.Dataset):
      bin_thresholds = self._bin_thresholds[da.name]
    else:
      bin_thresholds = self._bin_thresholds.rename(da.name)

    if self._bin_dim in da:
      raise ValueError(
          f'ContinuousToBins has {self._bin_dim=} from __init__ which should '
          'which was found as a variable in the input of transform_fn, which '
          'is not allowed.'
      )
    for bin_coord in self._bin_thresholds.dims:
      if bin_coord != self._bin_dim and bin_coord not in da.coords:
        raise ValueError(
            f'{bin_coord} is a coordinate in bin_thresholds, but not in the '
            f'input of transform_fn, which has {da.coords=}.'
        )

    self._check_bin_edges_non_nan_and_increasing(bin_thresholds)

    # Get positive and negative inf bin edges.
    pos_inf_edge, neg_inf_edge = self._get_positive_and_negative_inf_bin_edges(
        bin_thresholds
    )

    # Add right inf bin edges to bin values.
    bins_with_inf_edges: xr.DataArray = xr.concat(
        [neg_inf_edge, bin_thresholds, pos_inf_edge],
        dim=self._bin_dim,
    )

    is_less_than_left_bin_edge = (da <= bins_with_inf_edges).astype(np.float32)
    binned = is_less_than_left_bin_edge.diff(dim=self._bin_dim)

    # Add positive and negative inf bin edges to bin_thresholds to get the left
    # and right edges of each bin
    left_edges = xr.concat([neg_inf_edge, bin_thresholds], dim=self._bin_dim)
    left_edges = left_edges.drop_vars(self._bin_dim)

    right_edges = xr.concat([bin_thresholds, pos_inf_edge], dim=self._bin_dim)
    right_edges = right_edges.drop_vars(self._bin_dim)

    # Add left and right edges of the bins to binned value dataset as new coords
    binned = binned.assign_coords({f'{self._bin_dim}_left': left_edges})
    binned = binned.assign_coords({f'{self._bin_dim}_right': right_edges})

    return binned.where(~np.isnan(da))


class WeibullEnsembleToProbabilistic(InputTransform):
  """Convert ensemble forecasts into probabilitic forecast using the Weibull’s plotting position (Makkonen, 2006).

  The forecasts should be binarized before applying this wrapper and you can
  wrap the metric with the ContinuousToBinary firstly.

  Makkonen, L.: Plotting Positions in Extreme Value Analysis, Journal of Applied
  Meteorology and Climatology,
      45, 334–340, https://doi.org/10.1175/JAM2349.1, 2006.
  """

  def __init__(self, which, ensemble_dim='number', skipna=False):
    """Init.

    Args:
      which: Which input to apply the wrapper to. Must be 'predictions'.
      ensemble_dim: Name of ensemble dimension. Default: 'number'.
    """
    assert (
        which == 'predictions'
    ), 'Only predictions can be converted to probabilities'
    super().__init__(which)
    self._ensemble_dim = ensemble_dim
    self._skipna = skipna

  @property
  def unique_name_suffix(self) -> str:
    return 'ensemble_to_probabilistic_by_weibull_plotting_position'

  def transform_fn(self, da: xr.DataArray) -> xr.DataArray:
    ensemble_members = da.sizes[self._ensemble_dim]
    return da.sum(self._ensemble_dim, skipna=self._skipna) / (
        ensemble_members + 1
    )


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

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self.metric.values_from_mean_statistics(statistic_values)


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

  def values_from_mean_statistics(
      self,
      statistic_values: Mapping[str, Mapping[Hashable, xr.DataArray]],
  ) -> Mapping[Hashable, xr.DataArray]:
    return self.metric.values_from_mean_statistics(statistic_values)
