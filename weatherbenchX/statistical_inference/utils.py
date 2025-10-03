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

import numpy as np
from weatherbenchX import aggregation
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
