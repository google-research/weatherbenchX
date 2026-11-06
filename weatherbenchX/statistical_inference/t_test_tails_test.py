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
"""Regression tests for representable extreme-tail p-values."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from scipy import stats
from weatherbenchX.statistical_inference import t_test
from weatherbenchX.statistical_inference import test_utils
import xarray as xr


class TTestTailTest(parameterized.TestCase):

  @parameterized.parameters(1e5, 1e10, 1e20)
  def test_cauchy_tail_matches_independent_analytic_probability(
      self, statistic
  ):
    mean = xr.DataArray(
        [statistic, -statistic],
        dims="location",
        coords={"location": ["a", "b"]},
    )
    results = t_test._TTestResults(mean, xr.ones_like(mean), 1)
    actual = results.p_value()
    expected = 2 * np.arctan(1 / statistic) / np.pi
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)
    self.assertTrue(bool((actual > 0).all()))
    xr.testing.assert_identical(actual.location, mean.location)

  @parameterized.parameters(5, 30, 100)
  def test_extreme_tails_preserve_small_nonzero_probabilities(self, df):
    values = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
    mean = xr.DataArray(values, dims="case")
    actual = t_test._TTestResults(mean, xr.ones_like(mean), df).p_value()
    expected = 2 * stats.t.sf(np.abs(values), df)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0)

  @parameterized.parameters(-1.0, 1.0)
  def test_public_iid_path_matches_scipy(self, sign):
    data = xr.DataArray(
        sign * (10.0 + np.linspace(-0.5, 0.5, 32)), dims="samples"
    )
    metrics, state = test_utils.metrics_and_agg_state_for_mean(data)
    inference = t_test.IID(metrics, state, experimental_unit_dim="samples")
    actual = inference.p_values()["mean"]["variable"]
    expected = stats.ttest_1samp(data.values, popmean=0.0).pvalue
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=0)
    self.assertGreater(float(actual), 0.0)

  def test_degenerate_and_missing_inputs_keep_existing_semantics(self):
    means = xr.DataArray([0.0, 1.0, -1.0, np.nan, 0.0], dims="case")
    errors = xr.DataArray([0.0, 0.0, 0.0, 1.0, 1.0], dims="case")
    result = t_test._TTestResults(means, errors, 10).p_value()
    np.testing.assert_array_equal(result, [1.0, 0.0, 0.0, np.nan, 1.0])

  def test_nonzero_null_hypothesis_is_preserved(self):
    result = t_test._TTestResults(xr.DataArray(103.0), xr.DataArray(1.0), 30)
    np.testing.assert_allclose(
        result.p_value(null_value=3.0),
        2 * stats.t.sf(100.0, 30),
        rtol=2e-14,
        atol=0,
    )


if __name__ == "__main__":
  absltest.main()
