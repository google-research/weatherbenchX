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
from weatherbenchAfrica import test_utils
from weatherbenchAfrica import weighting
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


if __name__ == '__main__':
  absltest.main()
