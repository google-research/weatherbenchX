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
from collections.abc import Sequence
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


def latitude_cell_bounds(x: np.ndarray) -> np.ndarray:
  """Bounds for latitude cells, given increasing cell centers in radians."""
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


def cell_area_from_latitude(points: np.ndarray) -> np.ndarray:
  """Calculate the area overlap as a function of latitude."""
  bounds = latitude_cell_bounds(points)
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

    weights = cell_area_from_latitude(np.deg2rad(latitude))
    if needs_reversing:
      weights = weights[::-1]
    if self.return_normalized:
      weights /= np.mean(weights)
    weights = statistic[self.latitude_name].copy(data=weights)
    return weights


def _haversine(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
  """Computes great-circle angle in radians between points using haversine.

  Args:
    lat1: Latitude of first point or array of points in radians.
    lon1: Longitude of first point or array of points in radians.
    lat2: Latitude of second point or array of points in radians.
    lon2: Longitude of second point or array of points in radians.

  Returns:
    Great-circle angle in radians.
  """
  dlat = lat1 - lat2
  dlon = lon1 - lon2
  a = (
      np.sin(dlat / 2) ** 2
      + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
  )
  return 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclasses.dataclass
class StationDensityWeighting(Weighting):
  """Weighting by inverse station density using a Gaussian kernel.

  The station density for station k is:
    ρ_k = Σ_l exp(-(α_kl / α_0)²)
  where α_kl is the great-circle angle between stations k and l.

  The weight is w_k = 1 / ρ_k, normalized so that the mean weight is 1.

  Reference:
    SEEPS paper: Rodwell, M.J., Richardson, D.S., Hewson, T.D. and Haiden, T.
    (2010), A new equitable score suitable for verifying precipitation in
    numerical weather prediction. Q.J.R. Meteorol. Soc., 136: 1344-1363.
    https://doi.org/10.1002/qj.656 (Eq. 22-23).

  Warning:
    When used with binning (e.g., temporal binning or binning into
    train/holdout stations), density weights are computed before binning is
    applied. This means the density estimate itself is based on the global
    sample of stations.

    Additionally, this weighting accounts strictly for spatial density. If
    duplicate station entries exist in the input array (e.g., across
    timestamps or lead times when using chunk sizes larger than 1), each
    occurrence contributes to the density estimate, causing duplicated
    stations to receive higher local density and proportionally lower weights.

    Finally, this weighting computes pairwise angles between all stations,
    resulting in O(N²) time and memory complexity.

    TODO(srasp): Implement per-bin computation of density weightings (e.g.,
    by implementing this as a combined binning+weighting).

  Attributes:
    alpha_0_degrees: Reference angle in degrees, or a 1D sequence of reference
      angles. If multiple angles are provided, returned weights include an
      additional 'weighting_alpha_0' dimension. Stations further than ~4 *
      alpha_0 have negligible contribution. Default: 0.75° (≈83 km).
    latitude_name: Name of latitude coordinate. Default: 'latitude'.
    longitude_name: Name of longitude coordinate. Default: 'longitude'.
    return_normalized: Whether to normalize weights to have a mean of 1.
      Default: True.
    max_weight: If set, clip the normalized weights to this maximum value.
      Applied after normalization, so the mean weight may be less than 1 when
      clipping is active. Default: None (no clipping).
  """

  alpha_0_degrees: float | Sequence[float] | np.ndarray = 0.75
  latitude_name: str = 'latitude'
  longitude_name: str = 'longitude'
  return_normalized: bool = True
  max_weight: float | None = None

  def weights(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    alpha_0 = np.atleast_1d(np.asarray(self.alpha_0_degrees, dtype=np.float64))
    scalar_alpha = np.ndim(self.alpha_0_degrees) == 0
    if alpha_0.ndim > 1:
      raise ValueError('alpha_0_degrees must be a scalar or 1D sequence.')

    # Only apply weighting to sparse 1D point data sharing a single dimension.
    if (
        self.latitude_name not in statistic.coords
        or self.longitude_name not in statistic.coords
    ):
      return xr.DataArray(1)

    lat = statistic[self.latitude_name]
    lon = statistic[self.longitude_name]

    if lat.ndim != 1 or lon.ndim != 1 or lat.dims != lon.dims:
      return xr.DataArray(1)

    # Convert to radians.
    lat_rad = np.deg2rad(lat.values)
    lon_rad = np.deg2rad(lon.values)

    # Compute pairwise great-circle angles using haversine formula.
    # Shape: (N, 1) vs (1, N) for broadcasting → (N, N).
    alpha_kl = _haversine(
        lat_rad[:, None],
        lon_rad[:, None],
        lat_rad[None, :],
        lon_rad[None, :],
    )
    # Pre-compute squared angles once; avoids redundant work per alpha.
    alpha_kl_sq = alpha_kl ** 2

    # Compute density for each alpha_0 value by looping rather than
    # broadcasting into a 3D (N, N, K) tensor.  This keeps peak memory
    # at O(N²) regardless of how many alpha values are requested.
    n = len(lat_rad)
    k = len(alpha_0)
    w = np.empty((n, k), dtype=np.float64)
    alpha_0_rad = np.deg2rad(alpha_0)
    for i, a0 in enumerate(alpha_0_rad):
      density = np.sum(np.exp(-alpha_kl_sq / (a0 ** 2)), axis=1)
      w[:, i] = 1.0 / density

    if self.return_normalized:
      w /= np.mean(w, axis=0, keepdims=True)
    if self.max_weight is not None:
      w = np.clip(w, None, self.max_weight)

    if scalar_alpha:
      return xr.DataArray(
          w[:, 0],
          dims=lat.dims,
          coords=lat.coords,
      )

    coords = dict(lat.coords)
    coords['weighting_alpha_0'] = alpha_0
    return xr.DataArray(
        w,
        dims=(*lat.dims, 'weighting_alpha_0'),
        coords=coords,
    )
