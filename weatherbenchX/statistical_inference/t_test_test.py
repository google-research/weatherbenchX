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
import functools

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchX.statistical_inference import t_test
from weatherbenchX.statistical_inference import test_utils
import xarray as xr


class TTestTest(parameterized.TestCase):

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
    inference = t_test.IID(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="samples",
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

    inference = t_test.IID.for_baseline_comparison(
        metrics=metrics,
        aggregated_statistics=main_aggregated_stats,
        baseline_aggregated_statistics=baseline_aggregated_stats,
        experimental_unit_dim="samples",
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

  @parameterized.named_parameters(
      # All tests are with a fairly high level of autocorrelation, equivalent
      # to a ~4.3x reduction in effective sample size, and with data from an
      # AR(2) process (so consistent with the assumptions of the Geer method).
      #
      # Understand that accurate size control at small sample sizes is hard for
      # a general-purpose method here, so we are not testing for perfect
      # results in this sense at small sample sizes.
      #
      # We also don't check the power of the tests here. We're not aiming to be
      # a comprehensive evaluation of these statistical tests (see the
      # associated papers for that), just checking that some of the most
      # important properties hold within reasonable tolerances for sample sizes
      # we might encounter in practice.
      dict(
          # With a small-ish sample size of 100 (effective sample size more like
          # 23), the Geer method shows significant size distortion (rtol=0.75
          # below, meaning for an alpha=0.05 test we tolerate an actual size of
          # 0.0125 - 0.0875). This is very poor.
          testcase_name="GeerAR2Corrected_sample_100",
          t_test_class=t_test.GeerAR2Corrected,
          sample_size=100,
          rtol=0.75,
      ),
      dict(
          # The HAC method performs better here. There is still some size
          # distortion (rtol=0.35 below, meaning an alpha=0.05 test we tolerate
          # an actual size of 0.0325 - 0.0675) but it's still a lot better than
          # the Geer method at this small sample size.
          testcase_name="LazarusHACEWC_sample_100",
          t_test_class=t_test.LazarusHACEWC,
          sample_size=100,
          rtol=0.35,
      ),
      dict(
          # Further, by selecting a smaller value of v_0 than the default (0.4),
          # we can reduce the size distortion significantly, although it will
          # trade off against power of the test (not evaluated here).
          # v_0 = 0.27 was chosen from table 2b in the Lazarus et al paper to
          # be optimal for rho=0.7 and kappa=0.99 (0.99 weight on minimizing
          # size distortion, 0.01 weight on minimizing power loss).
          # We then get away with rtol=0.15 (tolerating a size of 0.0425-0.0575
          # for alpha=0.05).
          testcase_name="LazarusHACEWC_reduced_v0_sample_100",
          t_test_class=functools.partial(t_test.LazarusHACEWC, v_0=0.27),
          sample_size=100,
          rtol=0.15,
      ),
      dict(
          # With a larger sample size of 1000, and given that its AR(2)
          # assumption holds in this test, the Geer method performs very well,
          # with only minimal size distortion (rtol=0.01 below, meaning for an
          # alpha=0.05 test we tolerate an actual size of 0.0495 - 0.0505).
          # It may not perform as well when the AR(2) assumption does not hold
          # however.
          testcase_name="GeerAR2Corrected_sample_1000",
          t_test_class=t_test.GeerAR2Corrected,
          sample_size=1000,
          rtol=0.01,
      ),
      dict(
          # With this larger sample size the Lazarus method performs well too,
          # size distortion is still very small (rtol=0.03 below, meaning for
          # an alpha=0.05 test we tolerate an actual size of 0.0485 - 0.0515),
          # although not quite as good as the Geer method. It doesn't rely on
          # the AR(2) assumption made by the Geer test though, and may compare
          # better when this assumption doesn't hold.
          testcase_name="LazarusHACEWC_sample_1000",
          t_test_class=t_test.LazarusHACEWC,
          sample_size=1000,
          rtol=0.03,
      ),
  )
  def test_t_test_with_autocorrelation(
      self,
      t_test_class,
      sample_size: int,
      rtol: float,
      ):
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
        steps=sample_size,
        # Adding more replicates just means we can estimate the coverage
        # probabilities more accurately.
        replicates=50000)
    data = xr.DataArray(data=data, dims=("steps", "replicates"))

    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)

    inference = t_test_class(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
    )

    for alpha in [0.2, 0.1, 0.05]:
      test_utils.assert_coverage_probability_estimate_plausible(
          inference,
          true_value=true_mean,
          metric_name="mean",
          alpha=alpha,
          rtol=rtol,
          coverage_prob_significance_level=0.1,
      )
      test_utils.assert_p_value_consistent_with_confidence_interval(
          inference, null_value=true_mean, metric_name="mean")

  def test_t_test_for_constant_sequence(self):
    # We see constant sequences in some corner cases. When there's zero
    # variation we want a zero-width confidence interval, rather than NaNs, say.
    data = xr.DataArray(data=np.ones(100), dims=("steps",))
    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)
    inference = t_test.GeerAR2Corrected(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
    )
    point_estimate = inference.point_estimates()["mean"]["variable"]
    np.testing.assert_allclose(point_estimate.data, 1.)

    stderr_estimate = inference.standard_error_estimates()
    np.testing.assert_allclose(stderr_estimate["mean"]["variable"].data, 0.)

    lower, upper = inference.confidence_intervals(alpha=0.05)
    lower = lower["mean"]["variable"]
    upper = upper["mean"]["variable"]
    np.testing.assert_allclose(lower.data, 1.)
    np.testing.assert_allclose(upper.data, 1.)

    p = inference.p_values(null_value=1.)
    np.testing.assert_allclose(p["mean"]["variable"].data, 1.)
    p = inference.p_values(null_value=2.)
    np.testing.assert_allclose(p["mean"]["variable"].data, 0.)


if __name__ == "__main__":
  absltest.main()
