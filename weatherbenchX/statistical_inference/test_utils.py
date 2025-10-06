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
from weatherbenchX import aggregation
from weatherbenchX.metrics import base as metrics_base
import xarray as xr


class MeanPrediction(metrics_base.Statistic):
  def compute(self, predictions, targets):
    return predictions


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


def simulate_ar1(mean, sigma, phi, steps=10, replicates=1000):
  """Simulates a stationary Gaussian AR(1) process with the given parameters."""
  gamma_0 = sigma**2 / (1 - phi**2)  # Stationary variance.
  # Sample initial value y_0 from the stationary distribution
  # Because we do this, no warm-up is needed to reach the stationary
  # distribution and start generating samples from it.
  x_0 = np.random.randn(replicates)
  y_0 = np.sqrt(gamma_0) * x_0
  results = [y_0]
  # Now just need to simulate from the AR(1) process following its definition:
  for _ in range(steps-1):
    y_nm1 = results[-1]
    x_n = np.random.randn(replicates)
    y_n = phi * y_nm1 + x_n * sigma
    results.append(y_n)
  return np.stack(results, axis=0) + mean
