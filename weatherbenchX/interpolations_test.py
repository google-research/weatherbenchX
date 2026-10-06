# Copyright 2026 Google LLC
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


class GridToSparseWithAltitudeAdjustmentTest(absltest.TestCase):

  def test_altitude_adjustment_inverted_latitude(self):
    # Grid elevation oriented North -> South (90 to -90) with asymmetric
    # topography (mountain at lat=45, flat plain elsewhere).
    grid_lats = np.array([90.0, 45.0, 0.0, -45.0, -90.0])
    grid_lons = np.array([0.0, 90.0, 180.0, 270.0])
    elev_data = np.zeros((len(grid_lats), len(grid_lons)))
    elev_data[grid_lats == 45.0, :] = 1000.0
    grid_elev = xr.DataArray(
        elev_data,
        coords={'latitude': grid_lats, 'longitude': grid_lons},
        dims=['latitude', 'longitude'],
    )

    # Forecast DataArray oriented South -> North (-90 to 90)
    da_lats = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])
    da = xr.DataArray(
        np.full((len(da_lats), len(grid_lons)), 280.0),
        coords={'latitude': da_lats, 'longitude': grid_lons},
        dims=['latitude', 'longitude'],
        name='2m_temperature',
    )

    # Sparse reference station in Northern Hemisphere at lat=45, lon=0.
    # Station elevation is 1100m (100m above local 1000m grid terrain).
    # If the elevation grid is correctly reindexed, local terrain is 1000m,
    # diff is +100m -> adjustment is -0.65 K -> 279.35 K.
    # If elevation is silently flipped or assigned without re-ordering, local
    # terrain would be 0m -> diff is +1100m -> -7.15 K -> test fails.
    reference = xr.DataArray(
        [280.0],
        coords={
            'index': [0],
            'latitude': ('index', [45.0]),
            'longitude': ('index', [0.0]),
            'elevation': ('index', [1100.0]),
        },
        dims=['index'],
        name='2m_temperature',
    )

    interpolator = interpolations.GridToSparseWithAltitudeAdjustment(
        method='linear',
        grid_elevation=grid_elev,
    )
    result = interpolator.interpolate_data_array(da, reference)
    self.assertEqual(result.shape, (1,))
    expected_temp = 280.0 - 0.65
    np.testing.assert_allclose(result.values, [expected_temp], rtol=1e-4)


class CoarsenTest(absltest.TestCase):

  def test_coarsen_0p1_to_0p25_global_grid_weights(self):
    # Global 0.1 deg fine grid (1801 x 3600) to 0.25 deg coarse grid
    # (721 x 1440).
    fine_lat = np.linspace(-90.0, 90.0, 1801, dtype=np.float32)
    fine_lon = np.linspace(0.0, 359.9, 3600, dtype=np.float32)
    coarse_lat = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
    coarse_lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)

    # Linear function of lat + periodic function of lon: conservative box
    # average over symmetric interior cells must reproduce exact cell center
    # values for linear functions.
    lat_2d, lon_2d = np.meshgrid(fine_lat, fine_lon, indexing='ij')
    data = (2.0 * lat_2d + 100.0 + 0.0 * lon_2d).astype(np.float32)
    da = xr.DataArray(
        data[None, :, :],
        dims=['sample', 'latitude', 'longitude'],
        coords={'sample': [0], 'latitude': fine_lat, 'longitude': fine_lon},
        name='2m_temperature',
    )
    ref = xr.DataArray(
        np.zeros((721, 1440), dtype=np.float32),
        dims=['latitude', 'longitude'],
        coords={'latitude': coarse_lat, 'longitude': coarse_lon},
    )

    coarsener = interpolations.Coarsen(
        dims=['latitude', 'longitude'], wrap_longitude=True
    )
    out = coarsener.interpolate_data_array(da, ref)

    self.assertEqual(out.dims, ('sample', 'latitude', 'longitude'))
    self.assertEqual(out.shape, (1, 721, 1440))
    self.assertEqual(out.dtype, np.float32)
    np.testing.assert_allclose(out.latitude.values, coarse_lat)
    np.testing.assert_allclose(out.longitude.values, coarse_lon)
    # Interior latitudes (excluding the two polar half-cells at -90 and +90)
    # have symmetric boxes around each coarse_lat center, so a linear field in
    # latitude is preserved exactly.
    expected_interior = 2.0 * coarse_lat[1:-1, None] + 100.0
    np.testing.assert_allclose(
        out.values[0, 1:-1, :],
        np.broadcast_to(expected_interior, (719, 1440)),
        atol=1e-4,
    )

  def test_coarsen_0p1_to_0p25_impulse_weights(self):
    # Verify exact 1D box-overlap weights for 0.1 -> 0.25:
    # - At even multiple of 0.25 (e.g. lon=0.0, box [-0.125, +0.125]):
    #   overlaps lon=359.9 (0.075/0.25=0.3), lon=0.0 (0.1/0.25=0.4),
    #   lon=0.1 (0.075/0.25=0.3).
    # - At odd multiple of 0.25 (e.g. lon=0.25, box [0.125, 0.375]):
    #   overlaps lon=0.1 (0.025/0.25=0.1), lon=0.2 (0.1/0.25=0.4),
    #   lon=0.3 (0.1/0.25=0.4), lon=0.4 (0.025/0.25=0.1).
    fine_lon = np.linspace(0.0, 359.9, 3600, dtype=np.float32)
    coarse_lon = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    eye = xr.DataArray(
        np.eye(3600, dtype=np.float32),
        dims=['impulse', 'longitude'],
        coords={'impulse': np.arange(3600), 'longitude': fine_lon},
    )
    ref = xr.DataArray(
        np.zeros(1440, dtype=np.float32),
        dims=['longitude'],
        coords={'longitude': coarse_lon},
    )
    coarsener = interpolations.Coarsen(dims=['longitude'], wrap_longitude=True)
    out = coarsener.interpolate_data_array(eye, ref)

    # Target j=0 (lon=0.0): impulses at 3599 (lon=359.9), 0 (lon=0.0),
    # and 1 (lon=0.1).
    w_j0 = out.isel(longitude=0).values
    np.testing.assert_allclose(w_j0[[3599, 0, 1]], [0.3, 0.4, 0.3], atol=1e-6)
    np.testing.assert_allclose(w_j0.sum(), 1.0, atol=1e-6)

    # Target j=1 (lon=0.25): impulses at 1, 2, 3, 4 (lon=0.1, 0.2, 0.3, 0.4)
    w_j1 = out.isel(longitude=1).values
    np.testing.assert_allclose(
        w_j1[[1, 2, 3, 4]], [0.1, 0.4, 0.4, 0.1], atol=1e-6
    )
    np.testing.assert_allclose(w_j1.sum(), 1.0, atol=1e-6)

  def test_coarsen_cell_centered_integer_ratio_matches_xarray_coarsen(self):
    # For cell-centered grids with an integer ratio (e.g. 2x2), Coarsen matches
    # xarray's DataArray.coarsen(...).mean() to floating-point precision.
    fine_lat = np.arange(-89.5, 90.0, 1.0, dtype=np.float32)
    fine_lon = np.arange(0.5, 360.0, 1.0, dtype=np.float32)
    da = xr.DataArray(
        np.random.RandomState(0)
        .randn(len(fine_lat), len(fine_lon))
        .astype(np.float32),
        dims=['latitude', 'longitude'],
        coords={'latitude': fine_lat, 'longitude': fine_lon},
    )
    expected = da.coarsen(latitude=2, longitude=2).mean()
    coarsener = interpolations.Coarsen(
        dims=['latitude', 'longitude'], wrap_longitude=True
    )
    actual = coarsener.interpolate_data_array(da, expected)
    xr.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

  def test_coarsen_raises_when_target_is_finer(self):
    coarse_lat = np.linspace(-90.0, 90.0, 721, dtype=np.float32)
    fine_lat = np.linspace(-90.0, 90.0, 1801, dtype=np.float32)
    da = xr.DataArray(
        np.zeros(len(coarse_lat), dtype=np.float32),
        dims=['latitude'],
        coords={'latitude': coarse_lat},
    )
    ref = xr.DataArray(
        np.zeros(len(fine_lat), dtype=np.float32),
        dims=['latitude'],
        coords={'latitude': fine_lat},
    )
    coarsener = interpolations.Coarsen(dims=['latitude'])
    with self.assertRaisesRegex(ValueError, 'Cannot coarsen'):
      coarsener.interpolate_data_array(da, ref)

  def test_coarsen_dataset_with_mismatched_reference_variables(self):
    fine_lat = np.arange(-89.5, 90.0, 1.0, dtype=np.float32)
    fine_lon = np.arange(0.5, 360.0, 1.0, dtype=np.float32)
    coarse_lat = np.arange(-89.0, 90.0, 2.0, dtype=np.float32)
    coarse_lon = np.arange(1.0, 360.0, 2.0, dtype=np.float32)
    ds = xr.Dataset(
        {
            '10m_u_component_of_wind': (
                ('latitude', 'longitude'),
                np.full((len(fine_lat), len(fine_lon)), 3.0, dtype=np.float32),
            ),
            '10m_v_component_of_wind': (
                ('latitude', 'longitude'),
                np.full((len(fine_lat), len(fine_lon)), 4.0, dtype=np.float32),
            ),
        },
        coords={'latitude': fine_lat, 'longitude': fine_lon},
    )
    reference = xr.Dataset(
        {
            '10m_wind_speed': (
                ('latitude', 'longitude'),
                np.full(
                    (len(coarse_lat), len(coarse_lon)), 5.0, dtype=np.float32
                ),
            ),
        },
        coords={'latitude': coarse_lat, 'longitude': coarse_lon},
    )
    coarsener = interpolations.Coarsen(
        dims=['latitude', 'longitude'], wrap_longitude=True
    )
    out = coarsener.interpolate(ds, reference)
    self.assertIn('10m_u_component_of_wind', out)
    self.assertIn('10m_v_component_of_wind', out)
    self.assertEqual(out['10m_u_component_of_wind'].shape, (90, 180))
    np.testing.assert_allclose(out['10m_u_component_of_wind'].values, 3.0)
    np.testing.assert_allclose(out['10m_v_component_of_wind'].values, 4.0)


if __name__ == '__main__':
  absltest.main()
