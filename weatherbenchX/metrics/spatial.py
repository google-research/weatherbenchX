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
"""Spatial verification metrics."""

import dataclasses
from typing import Iterable, Mapping, Union
import numpy as np
from scipy import ndimage
from weatherbenchX.metrics import base
import xarray as xr


def convolve2d_wrap_longitude(
    x: np.ndarray,
    neighborhood_size: int,
    wrap_longitude: bool = False,
) -> np.ndarray:
  # TODO(srasp): There is potentially faster way to compute the neighborhood
  # averaging, see Faggian, et. al. "Fast calculation of the fractions skill
  # score."
  # TODO(srasp): For the full globe convolutions could also be in spherical
  # harmonics space, which would be faster.
  """Convolve2d with optional wrap around longitude."""
  if neighborhood_size == 1:
    return x
  if neighborhood_size % 2 != 1:
    raise ValueError('neighborhood_size must be odd.')
  half_n = (neighborhood_size - 1) // 2
  # Implement 2d uniform convolution as two 1d convolutions which is much
  # faster. Need to convert to float since convolve will return bool for
  # bool inputs. Use wrap by default.
  kernel = np.ones(neighborhood_size, dtype=np.float32) / neighborhood_size
  out = ndimage.convolve1d(x.astype(np.float32), kernel, mode='wrap', axis=0)
  out = ndimage.convolve1d(out, kernel, mode='wrap', axis=1)

  # Set non-valid regions, i.e. outermost half_n pixels, to zero.
  # In FSS, zeros (after neigbrhood averaging) will be ignored.
  # First, set latitude edgs to zero.
  out[:half_n] = 0
  out[-half_n:] = 0
  # If wrap_longitude is False, set longitude edges to zero.
  if not wrap_longitude:
    out[:, :half_n] = 0
    out[:, -half_n:] = 0
  return out


def neighborhood_averaging_for_single_size(
    da: xr.DataArray, neighborhood_size: int, wrap_longitude: bool = False
) -> xr.DataArray:
  """Neighborhood averaging for a single neighborhood size."""
  out = xr.apply_ufunc(
      lambda x: convolve2d_wrap_longitude(x, neighborhood_size, wrap_longitude),
      da.copy(deep=True),
      input_core_dims=[['latitude', 'longitude']],
      output_core_dims=[['latitude', 'longitude']],
      vectorize=True,
  )
  # Also need to do this to a potential NaN mask.
  if 'mask' in da.coords:
    new_mask = neighborhood_averaging(
        da.mask.drop('mask'), neighborhood_size, wrap_longitude
    )
    # This should be new_mask == 1 but there are some rounding errors in
    # convolve2d, so that this fails. Therefore, we use isclose.
    new_mask = np.isclose(new_mask, True)
    out.coords['mask'][:] = new_mask.astype(bool)
  return out


def neighborhood_averaging(
    da: xr.DataArray,
    neighborhood_size: Union[int, Iterable[int]],
    wrap_longitude: bool = False,
):
  """Performs the neighborhood averaging, potentially over a range of sizes."""
  if isinstance(neighborhood_size, Iterable):
    return xr.concat(
        [
            neighborhood_averaging_for_single_size(da, n, wrap_longitude)
            for n in neighborhood_size
        ],
        dim=xr.DataArray(neighborhood_size, dims=['neighborhood_size']),
    )
  else:
    return neighborhood_averaging_for_single_size(
        da, neighborhood_size, wrap_longitude
    )


def get_suffix(
    neighborhood_size: Union[int, Iterable[int]],
    wrap_longitude: bool = False,
):
  if isinstance(neighborhood_size, Iterable):
    suffix = ','.join([str(t) for t in neighborhood_size])
  else:
    suffix = str(neighborhood_size)
  if wrap_longitude:
    suffix += '_wrap_longitude'
  return suffix


@dataclasses.dataclass
class SquaredFractionsError(base.PerVariableStatistic):
  """Numerator of the FSS."""

  neighborhood_size_in_pixels: Union[int, Iterable[int]]
  wrap_longitude: bool = False

  @property
  def unique_name(self) -> str:
    suffix = get_suffix(self.neighborhood_size_in_pixels, self.wrap_longitude)
    return f'SquaredFractionsError_{suffix}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    predictions = neighborhood_averaging(
        predictions, self.neighborhood_size_in_pixels, self.wrap_longitude
    )
    targets = neighborhood_averaging(
        targets, self.neighborhood_size_in_pixels, self.wrap_longitude
    )
    return (predictions - targets) ** 2


@dataclasses.dataclass
class SquaredPredictionFraction(base.PerVariableStatistic):
  """One part of the denominator of the FSS."""

  neighborhood_size_in_pixels: Union[int, Iterable[int]]
  wrap_longitude: bool = False

  @property
  def unique_name(self) -> str:
    suffix = get_suffix(self.neighborhood_size_in_pixels, self.wrap_longitude)
    return f'SquaredPredictionFraction_{suffix}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    predictions = neighborhood_averaging(
        predictions, self.neighborhood_size_in_pixels, self.wrap_longitude
    )
    return predictions**2 + xr.zeros_like(targets)


@dataclasses.dataclass
class SquaredTargetFraction(base.PerVariableStatistic):
  """One part of the denominator of the FSS."""

  neighborhood_size_in_pixels: Union[int, Iterable[int]]
  wrap_longitude: bool = False

  @property
  def unique_name(self) -> str:
    suffix = get_suffix(self.neighborhood_size_in_pixels, self.wrap_longitude)
    return f'SquaredTargetFraction_{suffix}'

  def _compute_per_variable(
      self,
      predictions: xr.DataArray,
      targets: xr.DataArray,
  ) -> xr.DataArray:
    targets = neighborhood_averaging(
        targets, self.neighborhood_size_in_pixels, self.wrap_longitude
    )
    return targets**2 + xr.zeros_like(predictions)


@dataclasses.dataclass
class FSS(base.PerVariableMetric):
  """Implementation of the Fractions Skill Score (FSS).

  Assumes the input data is already binary. The FSS is defined by a square
  neighborhood size in pixels. On a lat-lon grid this can lead to distorted
  neighborhoods towards the poles.

  Original paper: Roberts and Lean, 2008. https://doi.org/10.1175/2007MWR2123.1

  More recent overvew paper, including discussion of how to compute the FSS
  over multiple forecasts:
  https://journals.ametsoc.org/view/journals/mwre/149/10/MWR-D-18-0106.1.xml

  Note that if there is no rain in the aggregated targets and predictions, the
  FSS is undfined (NaN).

  Attributes:
    neighborhood_size_in_pixels: The size of the neighborhood to use for
      averaging in pixels. Must be odd. Can be an integer or a list, in which
      case the result will have an additional dimension 'neighborhood_size'.
    wrap_longitude: If True, averaging operation wraps around longitude.
      Default: False.
  """

  neighborhood_size_in_pixels: Union[int, Iterable[int]]
  wrap_longitude: bool = False

  @property
  def statistics(self) -> Mapping[str, base.Statistic]:
    # TODO(srasp): Currently this computes the target and prediction averages
    # twice. Since this is quite a large computation, we would ideally avoid
    # this. However, that would require some refactoring of the statistics
    # computation code.
    return {
        'SquaredFractionsError': SquaredFractionsError(
            self.neighborhood_size_in_pixels, self.wrap_longitude
        ),
        'SquaredPredictionFraction': SquaredPredictionFraction(
            self.neighborhood_size_in_pixels, self.wrap_longitude
        ),
        'SquaredTargetFraction': SquaredTargetFraction(
            self.neighborhood_size_in_pixels, self.wrap_longitude
        ),
    }

  def _values_from_mean_statistics_per_variable(
      self,
      statistic_values: Mapping[str, xr.DataArray],
  ) -> xr.DataArray:
    """Computes metrics from aggregated statistics."""
    return 1 - (
        statistic_values['SquaredFractionsError']
        / (
            statistic_values['SquaredPredictionFraction']
            + statistic_values['SquaredTargetFraction']
        )
    )
