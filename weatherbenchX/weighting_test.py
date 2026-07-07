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
from weatherbenchX import test_utils
from weatherbenchX import weighting
import xarray as xr


class WeightingTest(absltest.TestCase):

  def test_latitude_weights(self):
    statistic_values = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00'
    )
    latitude_weighting = weighting.GridAreaWeighting()
    weights = latitude_weighting.weights(statistic_values['2m_temperature'])

    # 1. Test for normalization
    self.assertAlmostEqual(weights.mean().values, 1.0)
    # 2. Test for shape
    self.assertEqual(weights.shape, statistic_values.latitude.shape)

    # Test non global weights.
    regional_statistic_values = statistic_values.sel(latitude=slice(-30, 30))
    latitude_weighting = weighting.GridAreaWeighting(return_normalized=False)
    weights = latitude_weighting.weights(statistic_values['2m_temperature'])
    regional_weights = latitude_weighting.weights(
        regional_statistic_values['2m_temperature']
    )

    xr.testing.assert_allclose(
        regional_weights, weights.sel(latitude=slice(-30, 30))
    )


class StationDensityWeightingTest(absltest.TestCase):

  def test_haversine(self):
    """Test haversine angle computation between known points."""
    # Distance between identical points should be 0.
    angle = weighting._haversine(
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])
    )
    self.assertAlmostEqual(float(angle[0]), 0.0)
    # Quarter circumference along equator (90 deg lon diff) is π/2 radians.
    angle_90 = weighting._haversine(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([np.pi / 2]),
    )
    self.assertAlmostEqual(float(angle_90[0]), np.pi / 2)

  def _make_sparse_statistic(self, lats, lons):
    """Create a mock sparse statistic with lat/lon coordinates."""
    n = len(lats)
    return xr.DataArray(
        np.ones(n),
        dims=['index'],
        coords={
            'latitude': ('index', np.array(lats)),
            'longitude': ('index', np.array(lons)),
        },
    )

  def test_uniform_stations_equal_weights(self):
    """Equally spaced stations should have approximately equal weights."""
    # 4 stations at corners of a large square (well separated).
    stat = self._make_sparse_statistic(
        lats=[0, 0, 10, 10],
        lons=[0, 10, 0, 10],
    )
    w = weighting.StationDensityWeighting().weights(stat)
    # All weights should be close to 1 (normalized mean=1).
    np.testing.assert_allclose(w.values, 1.0, atol=0.01)

  def test_dense_cluster_gets_lower_weight(self):
    """Stations in a dense cluster should get lower weight than isolated."""
    # 3 stations very close together + 1 far away.
    stat = self._make_sparse_statistic(
        lats=[0.0, 0.01, 0.02, 10.0],
        lons=[0.0, 0.01, 0.02, 10.0],
    )
    w = weighting.StationDensityWeighting().weights(stat)
    # The isolated station (index 3) should have highest weight.
    self.assertGreater(w.values[3], w.values[0])
    self.assertGreater(w.values[3], w.values[1])
    self.assertGreater(w.values[3], w.values[2])

  def test_normalization(self):
    """Weights should have mean=1 when return_normalized=True."""
    stat = self._make_sparse_statistic(
        lats=[0, 0.5, 1.0, 50.0],
        lons=[0, 0.5, 1.0, 50.0],
    )
    w = weighting.StationDensityWeighting(return_normalized=True).weights(stat)
    self.assertAlmostEqual(float(w.mean()), 1.0, places=5)

  def test_no_normalization(self):
    """Without normalization, weights should be 1/density."""
    stat = self._make_sparse_statistic(
        lats=[0, 0, 10],
        lons=[0, 0, 10],
    )
    w = weighting.StationDensityWeighting(return_normalized=False).weights(stat)
    # Two co-located stations: density ≈ 2 (each contributes exp(0)=1 + exp(-large)≈0)
    # so weight ≈ 0.5 for co-located, ~1.0 for isolated
    self.assertLess(w.values[0], w.values[2])

  def test_single_station(self):
    """Single station should get weight 1."""
    stat = self._make_sparse_statistic(lats=[45.0], lons=[10.0])
    w = weighting.StationDensityWeighting().weights(stat)
    self.assertAlmostEqual(float(w.values[0]), 1.0)

  def test_gridded_data_returns_identity(self):
    """When coordinates have separate dimensions (gridded), should return 1."""
    stat = xr.DataArray(
        np.ones((3, 4)),
        dims=['latitude', 'longitude'],
        coords={'latitude': [0, 1, 2], 'longitude': [0, 1, 2, 3]},
    )
    w = weighting.StationDensityWeighting().weights(stat)
    self.assertEqual(float(w), 1.0)

  def test_missing_coords_returns_identity(self):
    """When lat or lon is missing from coords, should return 1."""
    stat = xr.DataArray([1.0, 2.0], dims=['station'])
    w = weighting.StationDensityWeighting().weights(stat)
    self.assertEqual(float(w), 1.0)

  def test_2d_coords_returns_identity(self):
    """When lat or lon coordinates are 2-dimensional, should return 1."""
    lats = xr.DataArray([[0.0, 1.0], [2.0, 3.0]], dims=['x', 'y'])
    lons = xr.DataArray([[0.0, 1.0], [2.0, 3.0]], dims=['x', 'y'])
    stat = xr.DataArray(
        np.ones((2, 2)),
        dims=['x', 'y'],
        coords={'latitude': lats, 'longitude': lons},
    )
    w = weighting.StationDensityWeighting().weights(stat)
    self.assertEqual(float(w), 1.0)

  def test_custom_alpha_0(self):
    """Larger α₀ should make nearby stations contribute more to density."""
    stat = self._make_sparse_statistic(
        lats=[0, 1.0, 50.0],
        lons=[0, 1.0, 50.0],
    )
    w_small = weighting.StationDensityWeighting(alpha_0_degrees=0.5).weights(
        stat
    )
    w_large = weighting.StationDensityWeighting(alpha_0_degrees=2.0).weights(
        stat
    )
    # With larger α₀, the clustering of the two nearby stations is recognized,
    # so the isolated station gets significantly higher weight relative to them.
    ratio_small = float(w_small.values[2] / w_small.values[0])
    ratio_large = float(w_large.values[2] / w_large.values[0])
    self.assertLess(ratio_small, ratio_large)

  def test_max_weight_clips(self):
    """max_weight should clip weights to the specified maximum."""
    stat = self._make_sparse_statistic(
        lats=[0, 0.01, 0.02, 50.0],
        lons=[0, 0.01, 0.02, 50.0],
    )
    # Without clipping, isolated station gets a high weight.
    w_unclipped = weighting.StationDensityWeighting().weights(stat)
    max_unclipped = float(w_unclipped.max())
    self.assertGreater(max_unclipped, 1.5)

    # With clipping at 1.5
    w_clipped = weighting.StationDensityWeighting(max_weight=1.5).weights(stat)
    self.assertAlmostEqual(float(w_clipped.max()), 1.5)

  def test_max_weight_none_no_clipping(self):
    """max_weight=None should not clip any weights."""
    stat = self._make_sparse_statistic(
        lats=[0, 0.01, 50.0],
        lons=[0, 0.01, 50.0],
    )
    w_default = weighting.StationDensityWeighting().weights(stat)
    w_none = weighting.StationDensityWeighting(max_weight=None).weights(stat)
    np.testing.assert_array_equal(w_default.values, w_none.values)

  def test_array_alpha_0_degrees(self):
    """Sequence alpha_0_degrees returns weights with weighting_alpha_0 dim."""
    stat = self._make_sparse_statistic(
        lats=[0, 1.0, 50.0],
        lons=[0, 1.0, 50.0],
    )
    alphas = [0.5, 2.0]
    w_multi = weighting.StationDensityWeighting(alpha_0_degrees=alphas).weights(
        stat
    )
    self.assertIn('weighting_alpha_0', w_multi.dims)
    np.testing.assert_array_equal(w_multi['weighting_alpha_0'].values, alphas)

    w_small = weighting.StationDensityWeighting(alpha_0_degrees=0.5).weights(
        stat
    )
    w_large = weighting.StationDensityWeighting(alpha_0_degrees=2.0).weights(
        stat
    )
    np.testing.assert_allclose(w_multi.sel(weighting_alpha_0=0.5), w_small)
    np.testing.assert_allclose(w_multi.sel(weighting_alpha_0=2.0), w_large)

  def test_identity_weights_has_no_alpha_dimension(self):
    """Fallback identity weights should not include weighting_alpha_0 dim even with sequence alpha_0_degrees."""
    stat = xr.DataArray([1.0, 2.0], dims=['station'])  # Missing lat/lon coords
    w = weighting.StationDensityWeighting(alpha_0_degrees=[0.5, 1.0]).weights(
        stat
    )
    self.assertEqual(float(w), 1.0)
    self.assertNotIn('weighting_alpha_0', w.dims)



if __name__ == '__main__':
  absltest.main()
