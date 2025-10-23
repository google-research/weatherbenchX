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
from weatherbenchX.statistical_inference import autodiff
from weatherbenchX.statistical_inference import test_utils
import xarray as xr


class AutodiffTest(absltest.TestCase):

  def test_linearized_per_unit_values_of_ratio(self):
    # We'll evaluate this for a ratio of means defined separately at two
    # different 'levels' (could be anything, just to test broadcasting).
    numerator_values = xr.DataArray(
        data=[
            [2., 4., 6.],  # Mean 4.
            [1., 2., 3.],  # Mean 2.
        ],
        dims=["level", "unit"],
        coords={
            "level": ["a", "b"],
            "unit": [0, 1, 2]})
    denominator_values = xr.DataArray(
        data=[
            [1.4, 2.4, 2.2],  # Mean 2.
            [5., 4., 3.],  # Mean 4.
        ],
        dims=["level", "unit"],
        coords={
            "level": ["a", "b"],
            "unit": [0, 1, 2]}
    )
    # No assumptions should be made about dimension order:
    denominator_values = denominator_values.transpose("unit", "level")

    metrics, agg_state = test_utils.metrics_and_agg_state_for_ratio_of_means(
        numerator_values, denominator_values)

    (value, per_unit_tangents
    ) = autodiff.per_unit_values_linearized_around_mean_statistics(
        metrics=metrics,
        aggregation_state=agg_state,
        experimental_unit_dim="unit",
    )

    xr.testing.assert_allclose(
        value["ratio_of_means"]["variable"],
        numerator_values.mean("unit") / denominator_values.mean("unit"))

    # The function is f(x, y) = x/y.
    # Grad f is (1/y, -x/y**2).
    #
    # For level "a":
    # Grad f at the mean values x=4, y=2, is (1/2, -1)
    # Tangents at x=4, y=2 are (1/2, -1)^T (x-4, y-2) = x/2 - y
    def f_tangent_at_mean_for_level_a(x, y):
      return x/2 - y

    # For level "b":
    # Grad f at the mean values x=2, y=4, is (1/4, -1/8)
    # Tangents at x=2, y=4 are (1/4, -1/8)^T (x-2, y-4) = x/4 - y/8
    def f_tangent_at_mean_for_level_b(x, y):
      return x/4 - y/8

    expected_tangents = xr.concat(
        [
            f_tangent_at_mean_for_level_a(
                numerator_values.sel(level=["a"]),
                denominator_values.sel(level=["a"])),
            f_tangent_at_mean_for_level_b(
                numerator_values.sel(level=["b"]),
                denominator_values.sel(level=["b"])),
        ],
        dim="level"
    )

    xr.testing.assert_allclose(
        per_unit_tangents["ratio_of_means"]["variable"],
        expected_tangents)

    # Tangents should have mean zero.
    xr.testing.assert_allclose(
        per_unit_tangents["ratio_of_means"]["variable"].mean("unit"),
        xr.zeros_like(value["ratio_of_means"]["variable"]),
        atol=1e-7)


if __name__ == "__main__":
  absltest.main()
