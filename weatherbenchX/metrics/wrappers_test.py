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


if __name__ == '__main__':
  absltest.main()
