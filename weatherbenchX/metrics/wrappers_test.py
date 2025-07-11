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
"""Unit tests for Wrappers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchX import test_utils
from weatherbenchX.metrics import wrappers
import xarray as xr


class ContinuousToBinaryTest(parameterized.TestCase):

  def test_constant_threshold(self):
    target = test_utils.mock_target_data(random=True)
    ctb = wrappers.ContinuousToBinary(
        which='both', threshold_value=0.5, threshold_dim='threshold'
    )

    x = target.geopotential
    y = ctb.transform_fn(x)
    xr.testing.assert_equal(
        y.threshold,
        xr.DataArray(
            0.5,
            dims=['threshold'],
            coords={'threshold': [0.5]},
        ),
    )
    xr.testing.assert_equal(y.sel(threshold=0.5, drop=True), x > 0.5)

  def test_iterable_threshold(self):
    target = test_utils.mock_target_data(random=True)
    threshold_value = [0.2, 0.7]
    ctb = wrappers.ContinuousToBinary(
        which='both', threshold_value=threshold_value, threshold_dim='threshold'
    )

    x = target.geopotential
    y = ctb.transform_fn(x)
    xr.testing.assert_equal(
        y.threshold,
        xr.DataArray(
            threshold_value,
            dims=['threshold'],
            coords={'threshold': threshold_value},
        ),
    )

    for thresh in threshold_value:
      expected = x > thresh
      xr.testing.assert_equal(y.sel(threshold=thresh, drop=True), expected)

  def test_datarray_threshold(self):
    target = test_utils.mock_target_data(random=True)
    threshold_percentiles = [0.25, 0.75]
    threshold_dataarray = target.geopotential.quantile(threshold_percentiles, dim='time')
    threshold_dataarray = threshold_dataarray.rename({"quantile": "threshold"})
    ctb = wrappers.ContinuousToBinary(
        which='both', threshold_value=threshold_dataarray, threshold_dim='threshold'
    )

    x = target.geopotential
    y = ctb.transform_fn(x)
    xr.testing.assert_equal(
        y.threshold,
        threshold_dataarray.threshold,
    )
    for thresh in threshold_percentiles:
      expected = x > threshold_dataarray.sel(threshold=thresh)
      xr.testing.assert_equal(
          y.sel(threshold=thresh),
          expected,
      )

  def test_dataset_threshold(self):
    target = test_utils.mock_target_data(random=True)
    threshold_percentiles = [0.25, 0.75]
    threshold_dataset = target.quantile(threshold_percentiles, dim='time')
    threshold_dataset = threshold_dataset.rename({"quantile": "threshold"})

    ctb = wrappers.ContinuousToBinary(
        which='both', threshold_value=threshold_dataset, threshold_dim='threshold'
    )

    for var in ["geopotential", "2m_temperature"]:
      x = target[var]
      y = ctb.transform_fn(x)
      xr.testing.assert_equal(
          y.threshold,
          threshold_dataset.threshold,
      )
      for thresh in threshold_percentiles:
        expected = x > threshold_dataset[var].sel(threshold=thresh)
        xr.testing.assert_equal(
            y.sel(threshold=thresh),
            expected,
        )


class EnsembleMeanTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(skipna=True),
      dict(skipna=False),
  )
  def test_mean_over_realization_dim(self, skipna):
    forecast = test_utils.mock_target_data(random=True, ensemble_size=3)

    # Set one single realization to nan
    forecast = xr.where(
        forecast.level == forecast.realization[0],
        np.nan,
        forecast,
    )
    em = wrappers.EnsembleMean(
        which='both', ensemble_dim='realization', skipna=skipna
    )

    x = forecast.geopotential
    y = em.transform_fn(x)

    xr.testing.assert_equal(x.mean('realization', skipna=skipna), y)


class WeibullEnsembleToProbabilisticTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(skipna=True),
      dict(skipna=False),
  )
  def test_mean_over_realization_dim(self, skipna):
    forecast = test_utils.mock_target_data(random=True, ensemble_size=3)

    # Set one single realization to nan
    forecast = xr.where(
        forecast.level == forecast.realization[0],
        np.nan,
        forecast,
    )
    ctb = wrappers.ContinuousToBinary(
        which='both', threshold_value=0.5, threshold_dim='threshold'
    )

    em = wrappers.WeibullEnsembleToProbabilistic(
        which='predictions', ensemble_dim='realization'
    )

    x = forecast.geopotential
    binary_y = ctb.transform_fn(x)
    y = em.transform_fn(binary_y)
    ensemble_members = x.sizes['realization']
    xr.testing.assert_equal(
      (x > 0.5).sum('realization', skipna=skipna)/(ensemble_members+1),
      y.sel(threshold=0.5, drop=True))


class InlineTest(parameterized.TestCase):

  def test_negation(self):
    x = test_utils.mock_target_data(random=True).geopotential
    y = wrappers.Inline('both', lambda da: -da, 'negate_both').transform_fn(x)
    xr.testing.assert_equal(y, -x)


class ReLUTest(parameterized.TestCase):

  def test_on_data(self):
    target = test_utils.mock_target_data(random=True)
    relu = wrappers.ReLU(which='both')

    x = target.geopotential
    y = relu.transform_fn(x)
    expected = xr.where(x > 0, x, 0)
    xr.testing.assert_equal(y, expected)


class ShiftAlongNewDimTest(parameterized.TestCase):

  def test_constant_shift(self):
    target = test_utils.mock_target_data(random=True)
    shift = wrappers.ShiftAlongNewDim(
        which='both', shift_value=0.5, shift_dim='threshold',
        unique_name_suffix='shift_along_threshold_0.5',
    )

    x = target.geopotential
    y = shift.transform_fn(x)
    expected = (x + 0.5).expand_dims(threshold=[0.5]).transpose(*y.dims)
    xr.testing.assert_equal(y, expected)

  def test_iterable_shift(self):
    target = test_utils.mock_target_data(random=True)
    shift_value = [0.2, 0.7]
    shift = wrappers.ShiftAlongNewDim(
        which='both', shift_value=shift_value, shift_dim='threshold',
        unique_name_suffix='shift_along_threshold_[0.2,0.7]',
    )

    x = target.geopotential
    y = shift.transform_fn(x)
    xr.testing.assert_equal(
        y.threshold,
        xr.DataArray(
            shift_value,
            dims=['threshold'],
            coords={'threshold': shift_value},
        ),
    )

    for thresh in shift_value:
      expected = (x + thresh).expand_dims(threshold=[thresh]).transpose(*y.dims)
      xr.testing.assert_equal(y.sel(threshold=[thresh]), expected)

  def test_dataset_shift(self):
    target = test_utils.mock_target_data(random=True)

    quantiles = [0.25, 0.75]
    shift_value = target.quantile(q=quantiles, dim='time')

    shift = wrappers.ShiftAlongNewDim(
        which='both', shift_value=shift_value, shift_dim='quantile',
        unique_name_suffix='shift_along_quantile_[0.25, 0.75]',
    )

    x = target.geopotential
    y = shift.transform_fn(x)

    for q in quantiles:
      thresh = shift_value.geopotential.sel(quantile=[q])
      expected = x + thresh
      xr.testing.assert_equal(y.sel(quantile=[q]), expected)


class ContinuousToBinsTest(parameterized.TestCase):

  def test_iterable_threshold(self):
    target = test_utils.mock_target_data(random=True)
    bin_values = [0.2, 0.7]
    ctb = wrappers.ContinuousToBins(
        which='both',
        bin_values=bin_values,
        bin_dim='bin_values'
    )
    x = target.geopotential
    y = ctb.transform_fn(x)

    np.testing.assert_array_equal(
        y.bin_values_left.data,
        np.array([-np.inf, 0.2, 0.7])
    )
    np.testing.assert_array_equal(
        y.bin_values_right.data,
        np.array([0.2, 0.7, np.inf])
    )

    # Bin 0: x <= 0.2 (maps to bin_values coord 0.2)
    expected_bin_0 = x <= 0.2
    xr.testing.assert_equal(
        y.isel(bin_values=0, drop=True),
        expected_bin_0
    )

    # Bin 1: 0.2 < x <= 0.7 (maps to bin_values coord 0.7)
    expected_bin_1 = (x > 0.2) & (x <= 0.7)
    xr.testing.assert_equal(
        y.isel(bin_values=1, drop=True),
        expected_bin_1
    )

    # Bin 2: x > 0.7 (maps to bin_values coord np.inf)
    expected_bin_2 = x > 0.7
    xr.testing.assert_equal(
        y.isel(bin_values=2, drop=True),
        expected_bin_2
    )

  def test_non_monotonic_thresholds(self):
    bin_values = [0.7, 0.2] # Non-monotonic
    with self.assertRaises(ValueError):
      wrappers.ContinuousToBins(
          which='both', bin_values=bin_values, bin_dim='t'
      )


if __name__ == '__main__':
  absltest.main()
