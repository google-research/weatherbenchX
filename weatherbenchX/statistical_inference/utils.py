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
"""Utility functions for statistical inference method implementations."""

from collections.abc import Hashable, Sequence
from typing import Any, Callable

import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import xarray_tree
import xarray as xr


def get_and_check_experimental_unit_coord(
    aggregated_statistics: aggregation.AggregationState,
    name: str,
    check_is_dim: bool = True,
) -> xr.DataArray:
  """Returns the coordinate used to identify experimental units.

  Checks that this is consistent across all statistics.

  Args:
    aggregated_statistics: Statistics to obtain coordinate from.
    name: Name of the coordinate to get and check for consistency.
    check_is_dim: Whether to check that the coordinate always corresponds to a
      dimension too.

  Returns:
    The coordinate.
  """
  coord = None
  for stat_name, stat_vars in (
      aggregated_statistics.sum_weighted_statistics.items()):
    for var_name, var in stat_vars.items():
      try:
        var_coord = var.coords[name]
      except KeyError as exc:
        raise ValueError(
            f'No experimental unit coordinate {name} found for {stat_name=} '
            f'{var_name=}.') from exc
      if var_coord.ndim != 1:
        raise ValueError(
            f'Experimental unit coordinate {name} has multiple dimensions.')
      if check_is_dim and var_coord.dims[0] != name:
        raise ValueError(f'Coordinate {name} is not a dimension coordinate.')
      if coord is None:
        coord = var_coord
      elif var_coord.size != coord.size:
        raise ValueError(
            f'Inconsistent sizes for coordinate {name}: {var_coord.size} and '
            f'{coord.size}.')
      elif not np.all(var_coord.data == coord.data):
        raise ValueError(f'Inconsistent coordinate values for {name}.')
  if coord is None:
    raise ValueError('No statistics found.')
  return coord


DataArrayTree = Any


def apply_to_slices(
    func: Callable[[DataArrayTree], DataArrayTree],
    *args: DataArrayTree,
    dim: Hashable | Sequence[Hashable],
    ) -> DataArrayTree:
  """Apply a function vectorized over slices of the arguments.

  This is a little bit similar to xr.apply_ufunc with vectorize=True, but takes
  a function operating on DataArrays for the slices, not just on raw numpy data.

  Args:
    func: The function to apply to each slice.
    *args: The arguments which you want to pass slices of to the function.
      These can be any tree of DataArrays.
    dim: The dimension along which to vectorize. This can be a single dimension
      or a sequence of dimensions (e.g. ['init_time', 'lead_time']).
      The function will be called once for each combination of indices along
      these dimensions. The dimensions will be retained but with size 1.

  Returns:
    The result of combining the results of calling func on each slice.
    We use combine_by_coords for this. You are not required to retain the
    original `dim`s in your result, but outputs of `func` should be
    non-overlapping slices of the final result and should have coordinates on
    them that combine_by_coords can use to combine them.
  """

  dims = (dim,) if isinstance(dim, str) else tuple(dim)
  sizes = {}
  def check_arg_sizes_and_maybe_add_missing_coords(arg):
    for dim in dims:
      if dim not in arg.dims:
        continue
      if dim not in arg.coords:
        # xr.combine_by_coords later will expect a coordinate to be present on
        # any dimensions we're combining slices along. Setting a default
        # coordinate here saves the user's fn having to do it.
        arg = arg.assign_coords({dim: np.arange(arg.sizes[dim])})
      if dim not in sizes:
        sizes[dim] = arg.sizes[dim]
      if sizes[dim] != arg.sizes[dim]:
        raise ValueError(
            f'Different sizes {sizes[dim]}, {arg.sizes[dim]} for {dim=}.')
    return arg
  args = xarray_tree.map_structure(
      check_arg_sizes_and_maybe_add_missing_coords, args)
  for dim in dims:
    if dim not in sizes:
      raise ValueError(f'Dimension {dim=} not found in any arguments.')

  results = []
  for indexes in np.ndindex(*[sizes[d] for d in dims]):
    def slice_arg(arg, indexes=indexes):
      return arg.isel(
          {dim: [i] for dim, i in zip(dims, indexes) if dim in arg.dims})
    arg_slices = xarray_tree.map_structure(slice_arg, args)
    results.append(func(*arg_slices))

  return xarray_tree.map_structure(
      lambda *args: xr.combine_by_coords(args), *results)
