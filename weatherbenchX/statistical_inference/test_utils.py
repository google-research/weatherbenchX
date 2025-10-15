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
"""Utilities for use in tests of statistical inference methods."""

from collections.abc import Mapping

import numpy as np
import scipy.stats
from weatherbenchX import aggregation
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.statistical_inference import base
import xarray as xr


class MeanPrediction(metrics_base.Statistic):
  def compute(self, predictions, targets):
    return predictions


class MeanTarget(metrics_base.Statistic):
  def compute(self, predictions, targets):
    return targets


def metrics_and_agg_state_for_mean(data: xr.DataArray) -> tuple[
    Mapping[str, metrics_base.Metric], aggregation.AggregationState]:
  """Boilerplate setup needed to test statistical inference of the mean.

  Args:
    data: DataArray which contains (along a specific experimental unit
      dimension) values corresponding to samples from a distribution whose
      mean we want to infer. Other dimensions may also be present, e.g. if you
      want to do this for multiple samples from the same distribution in
      order to test for properties of the sampling distirbution.

  Returns:
    metrics: A mapping containing a single metric, 'mean', which when
      further aggregated will yield the sample mean.
    aggregation_state: An AggregationState for use with `metrics` and a
      StatisticalInferenceMethod, where the data are under a variable called
      just 'variable', and no reduction has (yet) been done over any dimensions.
  """
  metrics = {"mean": MeanPrediction()}
  stats = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics=metrics,
      predictions={"variable": data},
      targets={},
  )
  # No-op aggregation to get an AggregationState which we can use with
  # the statistical inference method. In more realistic situations we would be
  # using some real weather metric here whose statistics have been aggregated
  # over some dimensions (e.g. lat/lon grid).
  aggregator = aggregation.Aggregator(reduce_dims=())
  aggregation_state = aggregator.aggregate_statistics(stats)
  return metrics, aggregation_state


class RatioOfPredictionAndTargetMeans(metrics_base.PerVariableMetric):
  """Example of a nonlinear function of the means of two statistics."""

  @property
  def statistics(self):
    return {
        "mean_prediction": MeanPrediction(),
        "mean_target": MeanTarget(),
    }

  def _values_from_mean_statistics_per_variable(
      self, statistic_values: Mapping[str, xr.DataArray]) -> xr.DataArray:
    return statistic_values["mean_prediction"] / statistic_values["mean_target"]


def metrics_and_agg_state_for_ratio_of_means(
    numerator: xr.DataArray, denominator: xr.DataArray) -> tuple[
        Mapping[str, metrics_base.Metric], aggregation.AggregationState]:
  """Like metrics_and_agg_state_for_mean but for the ratio of two means."""
  metrics = {"ratio_of_means": RatioOfPredictionAndTargetMeans()}
  stats = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics=metrics,
      predictions={"variable": numerator},
      targets={"variable": denominator},
  )
  aggregator = aggregation.Aggregator(reduce_dims=())
  aggregation_state = aggregator.aggregate_statistics(stats)
  return metrics, aggregation_state


class ExpMeanPrediction(metrics_base.PerVariableMetric):
  """Example of a nonlinear function (exp) of the mean.

  This can be skewed and non-Gaussian so can be useful to test methods which
  don't rely on normality assumptions.
  """

  @property
  def statistics(self):
    return {"mean_prediction": MeanPrediction()}

  def _values_from_mean_statistics_per_variable(
      self, statistic_values: Mapping[str, xr.DataArray]) -> xr.DataArray:
    return np.exp(statistic_values["mean_prediction"])


def metrics_and_agg_state_for_exp_of_mean(data: xr.DataArray) -> tuple[
    Mapping[str, metrics_base.Metric], aggregation.AggregationState]:
  """Like metrics_and_agg_state_for_mean but for the exp of the mean."""
  metrics = {"exp_mean": ExpMeanPrediction()}
  stats = metrics_base.compute_unique_statistics_for_all_metrics(
      metrics=metrics,
      predictions={"variable": data},
      targets={},
  )
  aggregator = aggregation.Aggregator(reduce_dims=())
  aggregation_state = aggregator.aggregate_statistics(stats)
  return metrics, aggregation_state


def simulate_ar2(mean, sigma, phi1, phi2, steps=10, replicates=1000):
  """Simulates a stationary Gaussian AR(2) process with the given parameters."""
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


def simulate_ar1(mean, sigma_marginal, phi, steps=10, replicates=1000):
  """Simulates a stationary Gaussian AR(1) process with the given parameters."""
  sigma = sigma_marginal * np.sqrt(1 - phi**2)
  # Sample initial value y_0 from the stationary distribution
  # Because we do this, no warm-up is needed to reach the stationary
  # distribution and start generating samples from it.
  x_0 = np.random.randn(replicates)
  y_0 = sigma_marginal * x_0
  results = [y_0]
  # Now just need to simulate from the AR(1) process following its definition:
  for _ in range(steps-1):
    y_nm1 = results[-1]
    x_n = np.random.randn(replicates)
    y_n = phi * y_nm1 + x_n * sigma
    results.append(y_n)
  return np.stack(results, axis=0) + mean


def gaussian_ar1_true_stderr_of_sample_mean(
    sigma_marginal: float, phi: float, n: int):
  """True std.err. for sample mean of a stationary Gaussian AR(1) process."""
  correction_factor = 1 + 2*phi/(1-phi) * (
      1 - (1 - phi**n)/(1-phi)/n)
  effective_sample_size = n / correction_factor
  return sigma_marginal / np.sqrt(effective_sample_size)


def assert_probability_estimate_plausible(
    n_successes: int,
    n_trials: int,
    hypothesized_p: float,
    rtol: float = 1e-2,
    significance_level: float = 0.1,
):
  """Asserts that an estimate of a probability is plausible.

  By this we mean that a confidence interval with a given significance level
  for the probability estimate, overlaps with a specified tolerance window
  for the true/hypothesized probability.

  Args:
    n_successes: Number of successes.
    n_trials: Number of trials.
    hypothesized_p: The hypothesized / expected / true probability that we hope
      our estimate is consistent with.
    rtol: The relative tolerance around the hypothesized probability, relative
      to the minimum of p and 1-p.
      So e.g. if rtol=0.1, and hypothesized_p=0.9, then we're willing to
      tolerate estimates that are consistent a true value between 0.89 and 0.91.
      Likewise if rtol=0.1 and hypothesized_p=0.1, then we're willing to
      tolerate estimates that are consistent with a true value between 0.09 and
      0.11.
      This kind of tolerance can be useful when we accept that the true
      probability (not just our estimate of it) may be off a bit from what it
      should be.
    significance_level: The significance level for the confidence interval
      derived from our estimate. The assertion will fail if this confidence
      interval doesn't overlap with the range of true probabilities specified
      by rtol above.
      Under different random seeds, expect the test to fail with a false
      negative this proportion of times on average. To avoid flaky tests we
      recommend to fix the seed to a known-working value, but not to cherry-pick
      the seed aggressively.
  """

  lower_estimate, upper_estimate = scipy.stats.binomtest(
      k=n_successes, n=n_trials).proportion_ci(1-significance_level)

  atol = rtol * min(hypothesized_p, 1-hypothesized_p)
  upper_true_p = min(1, hypothesized_p + atol)
  lower_true_p = max(0, hypothesized_p - atol)

  estimate = float(n_successes / n_trials)
  if lower_true_p > upper_estimate or upper_true_p < lower_estimate:
    raise AssertionError(
        f"{n_successes}/{n_trials} = {estimate:g} is not close enough to "
        f"{hypothesized_p:g}. A {(1-significance_level)*100}% CI for the "
        f"true value based on our estimate is [{lower_estimate:g}, "
        f"{upper_estimate:g}], and doesn't overlap with the range of true "
        f"probabilities [{lower_true_p:g}, {upper_true_p:g}] that are within "
        f"our specified tolerance around {hypothesized_p:g}.")


def assert_coverage_probability_estimate_plausible(
    inference: base.StatisticalInferenceMethod,
    true_value: float,
    metric_name: str = "mean",
    variable_name: str = "variable",
    replicates_dim: str = "replicates",
    alpha: float = 0.05,
    rtol: float = 0.,
    coverage_prob_significance_level: float = 0.05,
    **confidence_intervals_kwargs
):
  """Asserts that the estimated coverage of confidence intervals is plausible.

  Given the number of replicates used to estimate it, and some relative
  tolerance `rtol` on acceptable values for the underlying coverage probability.
  See `assert_probability_estimate_plausible` for more details.

  Args:
    inference: The StatisticalInferenceMethod to test.
    true_value: The true value of the metric that we want to test.
    metric_name: The name of the metric to test.
    variable_name: The name of the variable to test.
    replicates_dim: The name of the dimension containing multiple replicates
      of the inference procedure.
    alpha: The significance level alpha to use for the confidence intervals
      whose coverage we want to estimate and test.
    rtol: The relative tolerance around the true value that we're willing to
      tolerate, see `assert_probability_estimate_plausible`.
    coverage_prob_significance_level: The significance level to use for our
      confidence interval for the coverage probability itself, see
      `assert_probability_estimate_plausible`.
    **confidence_intervals_kwargs: Keyword arguments to pass to the
      `confidence_intervals` method of the StatisticalInferenceMethod.
  """
  lower, upper = inference.confidence_intervals(
      alpha, **confidence_intervals_kwargs)
  lower = lower[metric_name][variable_name]
  upper = upper[metric_name][variable_name]
  covered = (lower <= true_value) & (true_value <= upper)
  assert_probability_estimate_plausible(
      n_successes=covered.sum(replicates_dim).data,
      n_trials=covered.sizes[replicates_dim],
      hypothesized_p=1-alpha,
      significance_level=coverage_prob_significance_level,
      rtol=rtol
  )


def assert_p_value_consistent_with_confidence_interval(
    inference: base.StatisticalInferenceMethod,
    null_value: float,
    metric_name: str = "mean",
    variable_name: str = "variable",
    **kwargs
):
  """Asserts that p-values are consistent with the CIs.

  Specifically, if a confidence level is requested at significance level
  equal to the p-value for a particular null-hypothesis value, then this null
  value will be on the boundary of the interval.

  Args:
    inference: The StatisticalInferenceMethod to test.
    null_value: The null hypothesis value to test.
    metric_name: The name of the metric to test.
    variable_name: The name of the variable to test.
    **kwargs: Keyword arguments to pass to the `p_values` and
      `confidence_intervals` methods of the StatisticalInferenceMethod.
  """
  p_values = inference.p_values(null_value, **kwargs)
  # There may be p-values for multiple replicates or multiple components of the
  # metric, we just pick the first one to test here as the interface doesn't
  # support specifying the significance level alpha on a per-metric/
  # per-variable/per-component basis.
  p_value = p_values[metric_name][variable_name].data.flatten()[0]
  lower, upper = inference.confidence_intervals(alpha=p_value, **kwargs)
  lower = lower[metric_name][variable_name].data.flatten()[0]
  upper = upper[metric_name][variable_name].data.flatten()[0]
  if not (np.allclose(lower, null_value) or np.allclose(upper, null_value)):
    raise AssertionError(
        f"Confidence interval [{lower:g}, {upper:g}] constructed with "
        f"significance level equal to p-value {p_value:g} does not have the "
        f"null value {null_value:g} on its boundary.")
