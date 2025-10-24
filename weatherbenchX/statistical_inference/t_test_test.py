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
from weatherbenchX.statistical_inference import t_test
from weatherbenchX.statistical_inference import test_utils
import xarray as xr


class TTestTest(absltest.TestCase):

  def test_plain_t_test(self):
    # Here we test the plain t-test with all its assumptions met. We check
    # that we see estimated coverage probabilities consistent with the specified
    # significance level alpha for the confidence interval, even for small
    # sample sizes. We check this by computing CIs using many replicates of the
    # data.
    np.random.seed(0)
    true_mean = 10.
    sample_size = 10
    replicates = 100000
    data = np.random.randn(sample_size, replicates) + true_mean
    data = xr.DataArray(data=data, dims=("samples", "replicates"))

    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)
    inference = t_test.TTest(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="samples",
        temporal_autocorrelation=False,
    )
    for alpha in [0.2, 0.1, 0.05]:
      test_utils.assert_coverage_probability_estimate_plausible(
          inference,
          true_value=true_mean,
          metric_name="mean",
          alpha=alpha,
          rtol=0,
          coverage_prob_significance_level=0.1,
      )
      test_utils.assert_p_value_consistent_with_confidence_interval(
          inference, null_value=true_mean, metric_name="mean")

  def test_t_test_with_baseline_comparison(self):
    # Here we test the t-test for a baseline comparison (i.e. a paired
    # t-test). We check that we see the coverage probabilities equal to the
    # specified coverage level alpha for the confidence interval, even for small
    # sample sizes. We check this by computing CIs using many replicates of the
    # data.
    np.random.seed(0)
    true_mean_diff = 0
    sample_size = 10
    replicates = 100000

    # We use data from a baseline model, and from a main model, where the
    # per-sample errors are correlated.
    baseline_data = np.random.randn(sample_size, replicates)
    baseline_data = xr.DataArray(
        data=baseline_data, dims=("samples", "replicates")
    )
    (metrics, baseline_aggregated_stats
     ) = test_utils.metrics_and_agg_state_for_mean(baseline_data)

    main_data = (
        baseline_data
        + np.random.randn(sample_size, replicates) * 0.5
        + true_mean_diff
    )
    main_data = xr.DataArray(data=main_data, dims=("samples", "replicates"))
    (_, main_aggregated_stats
     ) = test_utils.metrics_and_agg_state_for_mean(main_data)

    inference = t_test.TTest.for_baseline_comparison(
        metrics=metrics,
        aggregated_statistics=main_aggregated_stats,
        baseline_aggregated_statistics=baseline_aggregated_stats,
        experimental_unit_dim="samples",
        temporal_autocorrelation=False,
    )
    for alpha in [0.2, 0.1, 0.05]:
      test_utils.assert_coverage_probability_estimate_plausible(
          inference,
          true_value=true_mean_diff,
          metric_name="mean",
          alpha=alpha,
          rtol=0,
          coverage_prob_significance_level=0.1,
      )
      test_utils.assert_p_value_consistent_with_confidence_interval(
          inference, null_value=true_mean_diff, metric_name="mean")

  def test_t_test_with_ar2_correction(self):
    # We'll compute confidence intervals for the mean of many different
    # AR(2) processes all drawn from the same distribution. We'll check the true
    # mean lies within the 95% CI approximately 95% of the time.
    np.random.seed(0)
    true_mean = 10.
    data = test_utils.simulate_ar2(
        mean=true_mean,
        sigma=0.1,
        # This is a decent amount of autocorrelation, but not extreme,
        # equivalent to ~4.3x reduction in effective sample size
        # (_inflation_factor_from_ar_coeffs(0.5, 0.1)**2 is around 4.3)
        # so 1000 steps becomes more like 230.
        phi1=0.5, phi2=0.1,
        # We need a decent number of steps (sample size) here to get
        # confidence intervals close to their correct coverage probability.
        # This is a known shortcoming of the method.
        steps=1000,
        # Adding more replicates just means we can estimate the coverage
        # probabilities more accurately.
        replicates=20000)
    data = xr.DataArray(data=data, dims=("steps", "replicates"))

    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)

    inference = t_test.TTest(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
        temporal_autocorrelation=True,
    )
    for alpha in [0.2, 0.1, 0.05]:
      test_utils.assert_coverage_probability_estimate_plausible(
          inference,
          true_value=true_mean,
          metric_name="mean",
          alpha=alpha,
          rtol=0.01,
          coverage_prob_significance_level=0.1,
      )
      test_utils.assert_p_value_consistent_with_confidence_interval(
          inference, null_value=true_mean, metric_name="mean")

if __name__ == "__main__":
  absltest.main()
