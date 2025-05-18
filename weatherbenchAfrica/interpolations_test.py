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

from absl.testing import absltest
import numpy as np
from weatherbenchAfrica import interpolations
from weatherbenchAfrica import test_utils
import xarray as xr


class InterpolationsTest(absltest.TestCase):

  def test_interpolate_to_reference_coords(self):
    # For now just a simple test.
    # TODO(srasp): Test edge cases
    # TODO(srasp): Test with sparse data
    reference = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='6 hours',
        spatial_resolution_in_degrees=10,
    )

    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='6 hours',
        spatial_resolution_in_degrees=25,
    )

    interpolation = interpolations.InterpolateToReferenceCoords(
        method='linear',
        dims=['latitude', 'longitude'],
        wrap_longitude=True,
    )

    interpolated_predictions = interpolation.interpolate(predictions, reference)

    xr.testing.assert_equal(interpolated_predictions, reference)

  def test_interpolate_to_fixed_coords(self):
    # For now just a simple test.
    # TODO(srasp): Test edge cases

    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='6 hours',
        spatial_resolution_in_degrees=25,
    )

    coords = {
        'latitude': np.arange(-90, 90, 10),
        'longitude': np.arange(0, 360, 10),
    }
    interpolation = interpolations.InterpolateToFixedCoords(
        method='linear',
        coords=coords,
        wrap_longitude=True,
    )

    interpolated_predictions = interpolation.interpolate(predictions)

    np.testing.assert_equal(
        interpolated_predictions.latitude.values, coords['latitude']
    )
    np.testing.assert_equal(
        interpolated_predictions.longitude.values, coords['longitude']
    )

  def test_multiple_interpolation(self):
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='6 hours',
        spatial_resolution_in_degrees=25,
    )

    coords = {
        'latitude': np.arange(-90, 90, 10),
        'longitude': np.arange(0, 360, 10),
    }
    interpolation1 = interpolations.InterpolateToFixedCoords(
        method='linear',
        coords=coords,
        wrap_longitude=True,
    )
    interpolation2 = interpolations.InterpolateToReferenceCoords(
        method='linear',
        dims=['latitude', 'longitude'],
        wrap_longitude=True,
    )
    interpolation = interpolations.MultipleInterpolation(
        [interpolation1, interpolation2]
    )

    interpolated_predictions = interpolation.interpolate(
        predictions, reference=predictions
    )

    # Should be back to original grid.
    np.testing.assert_allclose(
        interpolated_predictions.latitude, predictions.latitude
    )
    np.testing.assert_allclose(
        interpolated_predictions.longitude, predictions.longitude
    )

  def test_neighborhood_threshold_probabilities(self):
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='6 hours',
        spatial_resolution_in_degrees=15,
        random=True,
    )
    interpolation = interpolations.NeighborhoodThresholdProbabilities(
        neighborhood_sizes=[1, 3, 5],
        thresholds=[0.1, 0.9],
        wrap_longitude=True,
    )
    interpolated_predictions = interpolation.interpolate(predictions)
    self.assertLessEqual(interpolated_predictions.max(), 1.0)
    self.assertGreaterEqual(interpolated_predictions.min(), 0.0)


if __name__ == '__main__':
  absltest.main()
