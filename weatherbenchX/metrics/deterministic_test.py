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
"""Unit tests for deterministic metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import xarray as xr
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import metrics_test_utils
from weatherbenchX.metrics import wrappers


class RelativeIntensityTest(parameterized.TestCase):

  def test_regular(self):
    predictions = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    # Mean pred: 25. Mean target: 10. Ratio: 2.5. Metric: |2.5 - 1| = 1.5.
    ri = deterministic.RelativeIntensity(spatial_dims=['latitude', 'longitude'])
    metric = wrappers.WrappedMetric(ri, [])
    result = metrics_test_utils.compute_all_metrics(
        {'metric': metric},
        {'var': predictions},
        {'var': targets},
        reduce_dims=[],
    )
    np.testing.assert_allclose(result['metric.var'].values, 1.5, atol=1e-5)

  def test_with_mask_and_nans_masked(self):
    # Case where NaNs are present but masked out.
    predictions = xr.DataArray(
        [[10.0, np.nan], [30.0, 40.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = xr.DataArray(
        [[10.0, np.nan], [10.0, 10.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    mask = xr.DataArray(
        [[1, 0], [1, 1]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = targets.assign_coords(mask=mask)

    # Masked region values:
    # Preds: 10, 30, 40. Mean: 80/3 = 26.666...
    # Targets: 10, 10, 10. Mean: 10.
    # Ratio: 2.666...
    # Metric: 1.666...

    ri = deterministic.RelativeIntensity(spatial_dims=['latitude', 'longitude'])
    metric = wrappers.WrappedMetric(ri, [])
    result = metrics_test_utils.compute_all_metrics(
        {'metric': metric},
        {'var': predictions},
        {'var': targets},
        reduce_dims=[],
    )

    expected_ratio = (80 / 3) / 10
    expected_result = abs(expected_ratio - 1)

    np.testing.assert_allclose(
        result['metric.var'].values, expected_result, atol=1e-5
    )
    self.assertIn('mask', result['metric.var'].coords)
    self.assertEqual(result['metric.var'].coords['mask'].item(), 1)

  def test_with_mask_and_time_dimension(self):
    # Case where there is an additional dimension 'time' that is not reduced.
    # Time 0: valid data, mask=1.
    # Time 1: all masked out (mask=0).
    predictions = xr.DataArray(
        np.array([
            [[10.0, 20.0], [30.0, 40.0]],  # Time 0
            [[100.0, 200.0], [300.0, 400.0]],  # Time 1 (values don't matter)
        ]),
        dims=('time', 'latitude', 'longitude'),
        coords={'time': [0, 1], 'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = xr.DataArray(
        np.array([
            [[10.0, 10.0], [10.0, 10.0]],  # Time 0
            [[10.0, 10.0], [10.0, 10.0]],  # Time 1
        ]),
        dims=('time', 'latitude', 'longitude'),
        coords={'time': [0, 1], 'latitude': [0, 1], 'longitude': [0, 1]},
    )
    mask = xr.DataArray(
        np.array([
            [[1, 1], [1, 1]],  # Time 0: all valid
            [[0, 0], [0, 0]],  # Time 1: all masked
        ]),
        dims=('time', 'latitude', 'longitude'),
        coords={'time': [0, 1], 'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = targets.assign_coords(mask=mask)

    ri = deterministic.RelativeIntensity(spatial_dims=['latitude', 'longitude'])
    metric = wrappers.WrappedMetric(ri, [])
    result = metrics_test_utils.compute_all_metrics(
        {'metric': metric},
        {'var': predictions},
        {'var': targets},
        reduce_dims=[],
    )

    # Time 0 calculation:
    # Pred mean: 25. Target mean: 10. Ratio: 2.5. Metric: 1.5.
    # Mask should be 1.

    # Time 1 calculation:
    # Mask is 0 everywhere. Count is 0.
    # Prediction mean: 0 (masked value) -> 0 (count=0 logic).
    # Target mean: 0.
    # Ratio: (0+eps)/(0+eps) = 1. Metric: 0.
    # Mask should be 0 (count=0).

    expected_values = [1.5, 0.0]
    expected_mask = [1, 0]

    np.testing.assert_allclose(
        result['metric.var'].values, expected_values, atol=1e-5
    )
    self.assertIn('mask', result['metric.var'].coords)
    np.testing.assert_array_equal(
        result['metric.var'].coords['mask'].values, expected_mask
    )

  def test_all_nans_masked(self):
    # Case where all values are NaN and there is a corresponding mask.
    predictions = xr.DataArray(
        [[np.nan, np.nan], [np.nan, np.nan]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = xr.DataArray(
        [[np.nan, np.nan], [np.nan, np.nan]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    mask = xr.DataArray(
        [[0, 0], [0, 0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = targets.assign_coords(mask=mask)

    ri = deterministic.RelativeIntensity(spatial_dims=['latitude', 'longitude'])
    metric = wrappers.WrappedMetric(ri, [])
    result = metrics_test_utils.compute_all_metrics(
        {'metric': metric},
        {'var': predictions},
        {'var': targets},
        reduce_dims=[],
    )

    np.testing.assert_allclose(result['metric.var'].values, 0)
    self.assertIn('mask', result['metric.var'].coords)
    self.assertEqual(result['metric.var'].coords['mask'].item(), 0)

  def test_nans_not_covered_by_mask(self):
    # Case where NaNs are present and NOT masked out (i.e. in the valid region).
    predictions = xr.DataArray(
        [[10.0, np.nan], [30.0, 40.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = xr.DataArray(
        [[10.0, 10.0], [10.0, 10.0]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    mask = xr.DataArray(
        [[1, 1], [1, 1]],
        dims=('latitude', 'longitude'),
        coords={'latitude': [0, 1], 'longitude': [0, 1]},
    )
    targets = targets.assign_coords(mask=mask)

    # Valid region includes a NaN. skipna=False so the sum should be NaN.
    # Result should be NaN. Mask should be 1.

    ri = deterministic.RelativeIntensity(spatial_dims=['latitude', 'longitude'])
    metric = wrappers.WrappedMetric(ri, [])
    result = metrics_test_utils.compute_all_metrics(
        {'metric': metric},
        {'var': predictions},
        {'var': targets},
        reduce_dims=[],
    )

    self.assertTrue(np.isnan(result['metric.var'].values))
    self.assertIn('mask', result['metric.var'].coords)
    self.assertEqual(result['metric.var'].coords['mask'].item(), 1)


if __name__ == '__main__':
  absltest.main()
