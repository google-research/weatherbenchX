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
from weatherbenchX import aggregation
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import t_test
import xarray as xr


class PredictionsPassthrough(metrics_base.Statistic):
  def compute(self, predictions, targets):
    return predictions


def simulate_ar2(mean, sigma, phi1, phi2, steps=10, replicates=1000):
  """Simulates a stationary AR(2) process with the given parameters."""
  # https://rf.mokslasplius.lt/stationary-variance-of-ar-2-process/
  denom = (1 + phi2) * (1 - phi1**2 + phi2**2 - 2*phi2)
  gamma_0 = sigma**2 * (1 - phi2) / denom  # Stationary variance.
  gamma_1 = sigma**2 * phi1 / denom  # Lag-1 covariance.
  rho_1 = gamma_1 / gamma_0  # Lag-1 correlation.
  # Sample initial values (y_0, y_1) from the stationary distribution over
  # consecutive pairs, which is Normal with covariance:
  # [[gamma_0, gamma_1],
  #  [gamma_1, gamma_0]]
  # Because we do this, no warm-up is needed to reach the stationary
  # distribution and start generating samples from it.
  x_0 = np.random.randn(replicates)
  y_0 = np.sqrt(gamma_0) * x_0
  x_1 = np.random.randn(replicates)
  y_1 = np.sqrt(gamma_0) * (rho_1 * x_0 + np.sqrt(1 - rho_1**2) * x_1)
  results = [y_0, y_1]
  # Now just need to simulate from the AR(2) process following its definition:
  for _ in range(steps-2):
    y_nm2, y_nm1 = results[-2], results[-1]
    x_n = np.random.randn(replicates)
    y_n = phi1 * y_nm1 + phi2 * y_nm2 + x_n * sigma
    results.append(y_n)
  return np.stack(results, axis=0) + mean


class TTestTest(absltest.TestCase):

  def test_plain_t_test(self):
    # Here we test the plain t-test with all its assumptions met. We check
    # that we see the coverage probabilities equal to the specified coverage
    # level alpha for the confidence interval, even for small sample sizes.
    # We check this by computing CIs using many replicates of the data.
    np.random.seed(0)
    true_mean = 10.
    sample_size = 10
    replicates = 100000
    data = np.random.randn(sample_size, replicates) + true_mean
    data = xr.DataArray(data=data, dims=("samples", "replicates"))

    # We go through a bit of boilerplate to dress this data up as a metric and
    # do a no-op aggregation to get an AggregationState which we can use with
    # the statistical inference method. In more realistic situations we would be
    # using partially-aggregated values of some real weather metric here.
    metrics = {"metric": PredictionsPassthrough()}
    stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics=metrics,
        predictions={"variable": data},
        targets={},
    )
    aggregator = aggregation.Aggregator(reduce_dims=())
    aggregated_stats = aggregator.aggregate_statistics(stats)

    statistical_inference_method = t_test.TTest(
        metrics={"metric": PredictionsPassthrough()},
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="samples",
        temporal_autocorrelation=False,
    )
    for alpha in [0.2, 0.1, 0.05]:
      lower, upper = statistical_inference_method.confidence_intervals(alpha)
      lower = lower["metric"]["variable"]
      upper = upper["metric"]["variable"]
      coverage_probability = (
          (lower <= true_mean) & (true_mean <= upper)).mean("replicates").data
      np.testing.assert_allclose(coverage_probability, 1-alpha, rtol=0.01)

      significance = statistical_inference_method.significance_tests(
          null_value=true_mean, alpha=alpha)["metric"]["variable"]
      type_1_error_probability = significance.mean("replicates").data
      np.testing.assert_allclose(type_1_error_probability, alpha, rtol=0.01)

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
    main_data = (
        baseline_data
        + np.random.randn(sample_size, replicates) * 0.5
        + true_mean_diff
    )
    main_data = xr.DataArray(data=main_data, dims=("samples", "replicates"))

    # We go through a bit of boilerplate to dress this data up as a metric and
    # do a no-op aggregation to get an AggregationState which we can use with
    # the statistical inference method. In more realistic situations we would be
    # using partially-aggregated values of some real weather metric here.
    metrics = {"metric": PredictionsPassthrough()}
    aggregator = aggregation.Aggregator(reduce_dims=())

    baseline_stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics=metrics,
        predictions={"variable": baseline_data},
        targets={},
    )
    baseline_aggregated_stats = aggregator.aggregate_statistics(baseline_stats)
    main_stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics=metrics,
        predictions={"variable": main_data},
        targets={},
    )
    main_aggregated_stats = aggregator.aggregate_statistics(main_stats)

    statistical_inference_method = t_test.TTest.for_baseline_comparison(
        metrics=metrics,
        aggregated_statistics=main_aggregated_stats,
        baseline_aggregated_statistics=baseline_aggregated_stats,
        experimental_unit_dim="samples",
        temporal_autocorrelation=False,
    )
    for alpha in [0.2, 0.1, 0.05]:
      lower, upper = statistical_inference_method.confidence_intervals(alpha)
      lower = lower["metric"]["variable"]
      upper = upper["metric"]["variable"]
      coverage_probability = (
          ((lower <= true_mean_diff) & (true_mean_diff <= upper))
          .mean("replicates")
          .data
      )
      np.testing.assert_allclose(coverage_probability, 1 - alpha, rtol=0.01)

      significance = statistical_inference_method.significance_tests(
          null_value=true_mean_diff, alpha=alpha
      )["metric"]["variable"]
      type_1_error_probability = significance.mean("replicates").data
      np.testing.assert_allclose(type_1_error_probability, alpha, rtol=0.01)

  def test_t_test_with_ar2_correction(self):
    # We'll compute confidence intervals for the mean of many different
    # AR(2) processes all drawn from the same distribution. We'll check the true
    # mean lies within the 95% CI approximately 95% of the time.
    np.random.seed(0)
    true_mean = 10.
    data = simulate_ar2(
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

    metrics = {"metric": PredictionsPassthrough()}
    stats = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics=metrics,
        predictions={"variable": data},
        targets={},
    )
    aggregator = aggregation.Aggregator(reduce_dims=())
    aggregated_stats = aggregator.aggregate_statistics(stats)

    statistical_inference_method = t_test.TTest(
        metrics={"metric": PredictionsPassthrough()},
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
        temporal_autocorrelation=True,
    )
    for alpha in [0.2, 0.1, 0.05]:
      lower, upper = statistical_inference_method.confidence_intervals(alpha)
      lower = lower["metric"]["variable"]
      upper = upper["metric"]["variable"]
      coverage_probability = (
          (lower <= true_mean) & (true_mean <= upper)).mean("replicates").data
      # The tolerance here is somewhat loose because we'd need a lot of
      # replicates to estimate the coverage probability very accurately,
      # and also the coverage may be slightly off due to the finite sample
      # size (steps above) used. Still it's a useful check.
      np.testing.assert_allclose(coverage_probability, 1-alpha, rtol=0.1)


if __name__ == "__main__":
  absltest.main()
