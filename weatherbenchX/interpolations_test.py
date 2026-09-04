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

  def _make_da(self, ny: int = 10, nx: int = 20) -> xr.DataArray:
    lats, lons = np.arange(ny, dtype=float), np.arange(nx, dtype=float)
    return xr.DataArray(
        data=np.random.rand(ny, nx),
        coords={'latitude': lats, 'longitude': lons},
        dims=['latitude', 'longitude'],
        name='t2m',
    )

  def test_subsample_basic(self):
    da = self._make_da(ny=100, nx=200)
    result = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=10
    ).interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 10)
    self.assertEqual(result.sizes['longitude'], 20)
    xr.testing.assert_equal(
        result,
        da.isel(
            latitude=slice(None, None, 10), longitude=slice(None, None, 10)
        ),
    )

  def test_subsample_stride_1_is_noop(self):
    da = self._make_da()
    result = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=1
    ).interpolate_data_array(da)
    xr.testing.assert_equal(result, da)

  def test_subsample_missing_dim_is_skipped(self):
    da = xr.DataArray(
        name='t2m',
        data=np.random.rand(10),
        coords={'latitude': np.arange(10, dtype=float)},
        dims=['latitude'],
    )
    result = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=2
    ).interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 5)
    self.assertNotIn('longitude', result.dims)

  def test_subsample_single_dim(self):
    da = self._make_da(ny=12, nx=20)
    result = interpolations.Subsample(
        dims=['latitude'], stride=3
    ).interpolate_data_array(da)
    self.assertEqual(result.sizes['latitude'], 4)
    self.assertEqual(result.sizes['longitude'], 20)
    xr.testing.assert_equal(result, da.isel(latitude=slice(None, None, 3)))

  def test_subsample_invalid_stride(self):
    with self.assertRaisesRegex(ValueError, 'stride must be >= 1'):
      interpolations.Subsample(dims=['latitude'], stride=0)

  def test_subsample_via_interpolate(self):
    da = self._make_da()
    result = interpolations.Subsample(
        dims=['latitude', 'longitude'], stride=2
    ).interpolate({'t2m': da})
    self.assertEqual(result['t2m'].sizes['latitude'], 5)
    self.assertEqual(result['t2m'].sizes['longitude'], 10)


class GridToSparseInterpolationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    ny, nx = 20, 30
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    self.grid_lat = 35.0 + 0.5 * y_idx + 0.1 * x_idx
    self.grid_lon = -105.0 + 0.1 * y_idx + 0.5 * x_idx

  def _make_grid_da(self, data, dims=('y', 'x'), coords=None):
    if coords is None:
      coords = {
          'latitude': (('y', 'x'), self.grid_lat),
          'longitude': (('y', 'x'), self.grid_lon),
      }
    return xr.DataArray(
        data=np.asarray(data, dtype=np.float32),
        dims=dims,
        coords=coords,
        name='t2m',
    )

  def _make_ref(self, lats, lons, names=None, dim='point'):
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    n = len(lats)
    if names is None:
      names = [f'p{i}' for i in range(n)]
    return xr.DataArray(
        data=np.zeros(n, dtype=np.float32),
        dims=[dim],
        coords={
            dim: names,
            'latitude': (dim, lats),
            'longitude': (dim, lons),
        },
    )

  def test_curvilinear_exact_linear_field(self):
    ny, nx = self.grid_lat.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    field_vals = 2.0 * y_idx + 3.0 * x_idx + 5.0
    da = self._make_grid_da(field_vals)

    frac_y = np.array([5.25, 12.75])
    frac_x = np.array([10.5, 18.25])
    target_lats = 35.0 + 0.5 * frac_y + 0.1 * frac_x
    target_lons = -105.0 + 0.1 * frac_y + 0.5 * frac_x
    expected_vals = 2.0 * frac_y + 3.0 * frac_x + 5.0

    reference = self._make_ref(target_lats, target_lons)
    interpolator = interpolations.GridToSparseInterpolation(method='linear')
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertEqual(interpolated.dims, ('point',))
    np.testing.assert_allclose(interpolated.values, expected_vals, atol=1e-2)

  def test_curvilinear_nearest(self):
    ny, nx = self.grid_lat.shape
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    field_vals = y_idx * 100 + x_idx
    da = self._make_grid_da(field_vals)

    target_lat = [self.grid_lat[5, 10]]
    target_lon = [self.grid_lon[5, 10]]
    reference = self._make_ref(target_lat, target_lon, names=['stn_exact'])

    interpolator = interpolations.GridToSparseInterpolation(method='nearest')
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertEqual(interpolated.values[0], field_vals[5, 10])

  def test_curvilinear_out_of_bounds_nan(self):
    da = self._make_grid_da(np.ones(self.grid_lat.shape))
    reference = self._make_ref(
        [self.grid_lat[5, 5], 0.0],
        [self.grid_lon[5, 5], 0.0],
        names=['inside', 'outside'],
    )

    interpolator = interpolations.GridToSparseInterpolation(
        method='linear', extrapolate_out_of_bounds=False
    )
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertFalse(np.isnan(interpolated.values[0]))
    self.assertTrue(np.isnan(interpolated.values[1]))

  def test_multidim_batch_dims(self):
    ny, nx = self.grid_lat.shape
    coords = {
        'lead_time': [1, 2, 3],
        'time': [10, 20],
        'latitude': (('y', 'x'), self.grid_lat),
        'longitude': (('y', 'x'), self.grid_lon),
    }
    da = self._make_grid_da(
        np.ones((3, 2, ny, nx)),
        dims=['lead_time', 'time', 'y', 'x'],
        coords=coords,
    )

    reference = self._make_ref(
        [
            self.grid_lat[2, 2],
            self.grid_lat[3, 3],
            self.grid_lat[4, 4],
            self.grid_lat[5, 5],
        ],
        [
            self.grid_lon[2, 2],
            self.grid_lon[3, 3],
            self.grid_lon[4, 4],
            self.grid_lon[5, 5],
        ],
    )

    interpolator = interpolations.GridToSparseInterpolation(method='linear')
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertEqual(interpolated.dims, ('lead_time', 'time', 'point'))
    self.assertEqual(interpolated.shape, (3, 2, 4))
    np.testing.assert_allclose(interpolated.values, 1.0)

  def test_1d_regular_grid_compatibility(self):
    lats = np.arange(30, 50, 1.0)
    lons = np.arange(-110, -90, 1.0)
    da = xr.DataArray(
        data=np.ones((len(lats), len(lons)), dtype=np.float32),
        dims=['latitude', 'longitude'],
        coords={'latitude': lats, 'longitude': lons},
        name='t2m',
    )
    reference = self._make_ref([35.5, 42.5], [-105.5, -95.5])

    interpolator = interpolations.GridToSparseInterpolation(method='linear')
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertEqual(interpolated.dims, ('point',))
    np.testing.assert_allclose(interpolated.values, [1.0, 1.0])

  def test_observation_table_alignment(self):
    lats = np.arange(30, 50, 1.0)
    lons = np.arange(-110, -90, 1.0)
    data = np.arange(2 * 3 * len(lats) * len(lons), dtype=np.float32).reshape(
        (2, 3, len(lats), len(lons))
    )
    da = xr.DataArray(
        data=data,
        dims=['init_time', 'lead_time', 'latitude', 'longitude'],
        coords={
            'init_time': [0, 1],
            'lead_time': [10, 20, 30],
            'latitude': lats,
            'longitude': lons,
        },
        name='t2m',
    )
    reference = xr.DataArray(
        data=np.zeros(3),
        dims=['index'],
        coords={
            'init_time': ('index', [0, 1, 0]),
            'lead_time': ('index', [10, 20, 30]),
            'latitude': ('index', [35.0, 40.0, 45.0]),
            'longitude': ('index', [-105.0, -100.0, -95.0]),
            'pointName': ('index', ['P1', 'P2', 'P3']),
        },
    )
    interpolator = interpolations.GridToSparseInterpolation(method='linear')
    interpolated = interpolator.interpolate_data_array(da, reference)

    self.assertEqual(interpolated.dims, ('index',))
    self.assertEqual(interpolated.shape, (3,))
    self.assertIn('lead_time', interpolated.coords)
    self.assertIn('pointName', interpolated.coords)
    expected = [
        float(
            da.sel(init_time=0, lead_time=10, latitude=35.0, longitude=-105.0)
        ),
        float(
            da.sel(init_time=1, lead_time=20, latitude=40.0, longitude=-100.0)
        ),
        float(
            da.sel(init_time=0, lead_time=30, latitude=45.0, longitude=-95.0)
        ),
    ]
    np.testing.assert_allclose(interpolated.values, expected)


if __name__ == '__main__':
  absltest.main()
