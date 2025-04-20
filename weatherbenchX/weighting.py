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

"""Weighting classes."""

import abc
import dataclasses
import numpy as np
import xarray as xr


class Weighting(abc.ABC):
  """Abstract class for weighting."""

  @abc.abstractmethod
  def weights(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Return weights for a given statistic.

    For now the implementation assumes that all information necessary to
    calculate the weights is contained in the statistic.

    Args:
      statistic: Individual DataArray with statistic values.

    Returns:
      weights: Weights that should broadcast against statistic dimensions.
    """


def _is_strictly_monotonic(vector):
  diff = np.diff(vector)
  return np.all(diff > 0) or np.all(diff < 0)


def _is_increasing(vector):
  diff = np.diff(vector)
  return np.all(diff > 0)


def _is_uniformly_spaced(vector):
  diff = np.diff(vector)
  expected_diff = diff[0]
  # rtol=1e-5 sometimes failed due to rounding errors.
  return np.all(np.isclose(expected_diff, diff, rtol=1e-4))


def _latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  assert _is_increasing(x), 'Points must be increasing.'
  diff = np.diff(x)

  # A reasonable guess for the left bound is x[0] - diff[0] / 2.
  # Of course, if this is -90, then a better guess is -90.
  # Similar for the upper bound.
  left_bound = x[0] - diff[0] / 2
  right_bound = x[-1] + diff[-1] / 2
  pi_over_2 = np.pi / 2
  left_bound = np.max([left_bound, -pi_over_2])
  right_bound = np.min([right_bound, pi_over_2])
  return np.concatenate([
      np.array([left_bound], dtype=x.dtype),
      (x[:-1] + x[1:]) / 2,
      np.array([right_bound], dtype=x.dtype),
  ])


def _cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = _latitude_cell_bounds(points)
  upper = bounds[1:]
  lower = bounds[:-1]
  # Normalized cell area: integral from lower to upper of cos(latitude).
  return np.sin(upper) - np.sin(lower)


@dataclasses.dataclass
class GridAreaWeighting(Weighting):
  """Return normalized weights proportional to area of rectangular grid box.

  Attributes:
    latitude_name: Name of latitude dimension on statistic data array. Default:
      'latitude'
    return_normalized: Whether to return weights normalized to a mean of 1. This
      should not matter for the aggregation. Default: True.
  """

  latitude_name: str = 'latitude'
  return_normalized: bool = True

  def weights(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    # If latitude is not a dimension, do not apply any weighting.
    if self.latitude_name not in statistic.dims:
      return xr.DataArray(1)

    latitude = statistic[self.latitude_name].data

    assert _is_strictly_monotonic(
        latitude
    ), f'Points must be strictly monotonic: {latitude}'
    if latitude[0] > latitude[1]:
      needs_reversing = True
      latitude = latitude[::-1]
    else:
      needs_reversing = False

    weights = _cell_area_from_latitude(np.deg2rad(latitude))
    if needs_reversing:
      weights = weights[::-1]
    if self.return_normalized:
      weights /= np.mean(weights)
    weights = statistic[self.latitude_name].copy(data=weights)
    return weights
