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
"""Utility functions for unit tests."""

from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr


DEFAULT_2D_VARIABLES = ('2m_temperature',)
DEFAULT_3D_VARIABLES = ('geopotential',)


def mock_target_data(
    *,
    variables_3d: Sequence[str] = DEFAULT_3D_VARIABLES,
    variables_2d: Sequence[str] = DEFAULT_2D_VARIABLES,
    levels: Sequence[int] = (500, 700, 850),
    spatial_resolution_in_degrees: float = 10.0,
    time_start: str = '2020-01-01',
    time_stop: str = '2021-01-01',
    time_resolution: str = '1 day',
    dtype: npt.DTypeLike = np.float32,
    ensemble_size: Optional[int] = None,
    random: bool = False,
    seed: Optional[int] = None,
) -> xr.Dataset:
  """Create a mock truth dataset with all zeros for testing."""

  rng = np.random.default_rng(seed)
  def val_fn(shape):
    if random:
      return rng.random(size=shape)
    else:
      return np.zeros(shape, dtype=dtype)

  num_latitudes = round(180 / spatial_resolution_in_degrees) + 1
  num_longitudes = round(360 / spatial_resolution_in_degrees)
  freq = pd.Timedelta(time_resolution)
  coords = {
      'time': pd.date_range(time_start, time_stop, freq=freq, inclusive='left'),
      'latitude': np.linspace(-90, 90, num_latitudes),
      'longitude': np.linspace(0, 360, num_longitudes, endpoint=False),
      'level': np.array(levels),
  }
  if ensemble_size is not None:
    coords['realization'] = np.arange(ensemble_size)
  dims_3d = coords.keys()
  shape_3d = tuple(coords[dim].size for dim in dims_3d)
  data_vars_3d = {k: (dims_3d, val_fn(shape_3d)) for k in variables_3d}
  if not data_vars_3d:
    del coords['level']

  dims_2d = set(coords.keys()) - {'level'}
  shape_2d = tuple(coords[dim].size for dim in dims_2d)
  data_vars_2d = {k: (dims_2d, val_fn(shape_2d)) for k in variables_2d}

  data_vars = {**data_vars_3d, **data_vars_2d}
  return xr.Dataset(data_vars, coords)


def mock_prediction_data(
    *,
    lead_start: str = '0 day',
    lead_stop: str = '10 day',
    lead_resolution: str = '1 day',
    **kwargs,
):
  """Create a mock forecast dataset with all zeros for testing."""
  lead_time = pd.timedelta_range(
      pd.Timedelta(lead_start),
      pd.Timedelta(lead_stop),
      freq=pd.Timedelta(lead_resolution),
  )
  ds = mock_target_data(**kwargs)
  ds = ds.expand_dims(prediction_timedelta=lead_time)
  return ds


def compute_rps_xarray(
    predictions: xr.Dataset,
    targets: xr.Dataset,
    bin_values: list[float],
    fair: bool,
    ensemble_dim: str = 'sample',
) -> xr.Dataset:
  """Computes RPS using xarray operations."""

  # Helper to bin the data entries.
  digitize = lambda x, bin_values: np.digitize(x, bin_values, right=True)

  y_bin_idx = xr.apply_ufunc(digitize, predictions, bin_values)
  t_bin_idx = xr.apply_ufunc(digitize, targets, bin_values)

  # Convert data to one-hot representation.
  bin_idx = np.arange(len(bin_values) + 1)
  y_one_hot = (y_bin_idx == xr.DataArray(bin_idx, dims=['bins'])).astype(int)
  t_one_hot = (t_bin_idx == xr.DataArray(bin_idx, dims=['bins'])).astype(int)

  y_cmf = y_one_hot.cumsum(dim='bins')
  t_cmf = t_one_hot.cumsum(dim='bins')

  y_mean_per_bin = y_cmf.mean(dim=ensemble_dim)
  rps = ((y_mean_per_bin - t_cmf) ** 2).sum(dim='bins')

  if fair:

    # Variance of predictions per bin
    y_var_per_bin = y_cmf.var(dim=ensemble_dim, ddof=1)

    # Sum variances across bins and apply correction term
    summed_prediction_var = y_var_per_bin.sum(dim='bins')
    rps -= summed_prediction_var / predictions.sizes[ensemble_dim]

  return rps


def make_small_datasets_for_hand_calculated_rps_test() -> (
    tuple[xr.Dataset, xr.Dataset]
):
  """Creates a small dataset that makes it easy to hand-calculate metrics."""

  # Create dummy data for variables
  pred_temp = [0.1, 0.3, 0.3, 0.4, 0.9]
  num_samples = len(pred_temp)

  # Observed temperature value
  targ_temp = 0.7

  # Predictions dataset with a single "sample" dimension.
  ds_pred_temp = xr.Dataset(
      {'temperature': (('sample',), pred_temp)},
      coords={'sample': np.arange(num_samples)},
  )

  # Targets dataset is just a scalar
  ds_targ_temp = xr.Dataset({'temperature': ((), targ_temp)})

  return ds_pred_temp, ds_targ_temp
