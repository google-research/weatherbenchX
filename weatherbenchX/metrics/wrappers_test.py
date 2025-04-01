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
"""Unit tests for wrappers."""

from absl.testing import absltest
from absl.testing import parameterized
from weatherbenchX import test_utils
from weatherbenchX.metrics import wrappers
import xarray as xr


class ReLUTest(parameterized.TestCase):

  def test_constant_threshold(self):
    target = test_utils.mock_target_data(random=True)
    relu = wrappers.ReLU(
        which='both', threshold_value=0.5, threshold_dim='threshold'
    )

    x = target.geopotential
    y = relu.transform_fn(x)
    expected = (
        xr.where(x > 0.5, x - 0.5, 0)
        .expand_dims(threshold=[0.5])
        .transpose(*y.dims)
    )
    xr.testing.assert_equal(y, expected)

  def test_iterable_threshold(self):
    target = test_utils.mock_target_data(random=True)
    threshold_value = [0.2, 0.7]
    relu = wrappers.ReLU(
        which='both', threshold_value=threshold_value, threshold_dim='threshold'
    )

    x = target.geopotential
    y = relu.transform_fn(x)
    xr.testing.assert_equal(
        y.threshold,
        xr.DataArray(
            threshold_value,
            dims=['threshold'],
            coords={'threshold': threshold_value},
        ),
    )

    for thresh in threshold_value:
      expected = (
          xr.where(x > thresh, x - thresh, 0)
      )
      xr.testing.assert_equal(y.sel(threshold=thresh, drop=True), expected)

  def test_dataset_threshold(self):
    target = test_utils.mock_target_data(random=True)

    quantiles = [0.25, 0.75]
    threshold_value = target.quantile(q=quantiles, dim='time')

    relu = wrappers.ReLU(
        which='both', threshold_value=threshold_value, threshold_dim='quantile'
    )

    x = target.geopotential
    y = relu.transform_fn(x)

    for q in quantiles:
      thresh = threshold_value.geopotential.sel(quantile=q, drop=True)
      expected = (
          xr.where(x > thresh, x - thresh, 0)
      )
      xr.testing.assert_equal(y.sel(quantile=q, drop=True), expected)


if __name__ == '__main__':
  absltest.main()
