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
from weatherbenchX import interpolations
from weatherbenchX import test_utils
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

  def test_interpolate_to_reference_coords_empty_reference(self):
    gridded_da = xr.DataArray(
        name='t2m',
        data=np.ones((2, 10, 20)),
        dims=['sample', 'latitude', 'longitude'],
        coords={
            'sample': [1, 2],
            'latitude': np.arange(10),
            'longitude': np.arange(20),
        },
    )
    sparse_reference = xr.DataArray(
        name='t2m',
        data=[],
        dims=['index'],
        coords={
            'latitude': ('index', []),
            'longitude': ('index', []),
            'index': [],
        },
    )

    interpolation = interpolations.InterpolateToReferenceCoords(
        method='linear',
        dims=['latitude', 'longitude'],
    )

    interpolated_da = interpolation.interpolate_data_array(
        gridded_da, sparse_reference
    )

    self.assertIn('sample', interpolated_da.dims)
    self.assertEqual(interpolated_da.sizes['sample'], 2)
    self.assertIn('index', interpolated_da.dims)
    self.assertEqual(interpolated_da.sizes['index'], 0)
    self.assertSequenceEqual(interpolated_da.dims, ('sample', 'index'))
    np.testing.assert_equal(interpolated_da['sample'].values, [1, 2])


class CropToBoxTest(absltest.TestCase):

  def test_crop_to_box_with_0_360_input(self):
    lats = np.arange(-85, 86, 10)  # 18 elements
    lons = np.arange(0, 359, 18)  # 20 elements
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(len(lats), len(lons)),
        coords={
            'latitude': lats,
            'longitude': lons,
        },
        dims=['latitude', 'longitude'],
    )
    cropper = interpolations.CropToBox(
        lat_min=-30, lat_max=30, lon_min=60, lon_max=180
    )
    cropped_da = cropper.interpolate_data_array(da)
    np.testing.assert_array_less(cropped_da.latitude.values, 30.1)
    np.testing.assert_array_less(-30.1, cropped_da.latitude.values)
    np.testing.assert_array_less(cropped_da.longitude.values, 180.1)
    np.testing.assert_array_less(59.9, cropped_da.longitude.values)

  def test_crop_to_box_wrap_invalid_lon(self):
    with self.assertRaisesRegex(ValueError, 'Invalid longitudes.*'):
      interpolations.CropToBox(lat_min=-90, lat_max=90, lon_min=300, lon_max=60)

  def test_crop_to_box_invalid_lat(self):
    with self.assertRaisesRegex(ValueError, 'Invalid latitudes.*'):
      interpolations.CropToBox(lat_min=10, lat_max=-10, lon_min=0, lon_max=10)


class SubsampleTest(absltest.TestCase):

  def test_subsample_basic(self):
    lats = np.arange(0, 100, 1.0)
    lons = np.arange(0, 200, 1.0)
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(len(lats), len(lons)),
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
    )
    subsampler = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=10
    )
    result = subsampler.interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 10)
    self.assertEqual(result.sizes['longitude'], 20)
    np.testing.assert_equal(result.latitude.values, lats[::10])
    np.testing.assert_equal(result.longitude.values, lons[::10])

  def test_subsample_stride_1_is_noop(self):
    lats = np.arange(0, 10, 1.0)
    lons = np.arange(0, 20, 1.0)
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(len(lats), len(lons)),
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
    )
    subsampler = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=1
    )
    result = subsampler.interpolate_data_array(da)
    xr.testing.assert_equal(result, da)

  def test_subsample_missing_dim_is_skipped(self):
    lats = np.arange(0, 10, 1.0)
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(len(lats)),
        coords={'latitude': lats},
        dims=['latitude'],
    )
    subsampler = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=2
    )
    result = subsampler.interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 5)
    self.assertNotIn('longitude', result.dims)

  def test_subsample_single_dim(self):
    lats = np.arange(0, 12, 1.0)
    lons = np.arange(0, 20, 1.0)
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(len(lats), len(lons)),
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
    )
    subsampler = interpolations.Subsample(dims=['latitude'], stride=3)
    result = subsampler.interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 4)
    self.assertEqual(result.sizes['longitude'], 20)

  def test_subsample_invalid_stride(self):
    with self.assertRaisesRegex(ValueError, 'stride must be >= 1'):
      interpolations.Subsample(dims=['latitude'], stride=0)

  def test_subsample_via_interpolate(self):
    lats = np.arange(0, 10, 1.0)
    lons = np.arange(0, 20, 1.0)
    ds = {
        't2m': xr.DataArray(
            data=np.random.rand(len(lats), len(lons)),
            coords={'latitude': lats, 'longitude': lons},
            dims=['latitude', 'longitude'],
        ),
    }
    subsampler = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=2
    )
    result = subsampler.interpolate(ds)
    self.assertEqual(result['t2m'].sizes['latitude'], 5)
    self.assertEqual(result['t2m'].sizes['longitude'], 10)


if __name__ == '__main__':
  absltest.main()
