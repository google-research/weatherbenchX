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
import scipy.stats
from weatherbenchX.statistical_inference import bootstrap
from weatherbenchX.statistical_inference import test_utils
import xarray as xr


class BootstrapTest(absltest.TestCase):

  def test_iid_bootstrap_inference_of_exp_of_mean(self):
    # For this test we use the exp of the sample mean of i.i.d. Normals. This is
    # going to be highly skewed / non-Gaussian even though the mean itself is
    # Gaussian.
    np.random.seed(0)

    sample_size = 100
    n_replicates = 10000

    sigma = 2.
    mean = 0.
    dist = scipy.stats.norm(loc=mean, scale=sigma)
    true_exp_mean = np.exp(mean)
    # The sample mean is N(mu, sigma^2/n), its exp is then lognormal.
    true_sampling_dist = scipy.stats.lognorm(
        s=sigma/np.sqrt(sample_size), scale=np.exp(mean))

    data = dist.rvs(size=(sample_size, n_replicates))
    data = xr.DataArray(data=data, dims=("samples", "replicates"))
    metrics, aggregated_stats = (
        test_utils.metrics_and_agg_state_for_exp_of_mean(data))

    inference = bootstrap.IIDBootstrap(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="samples",
        n_replicates=n_replicates,
    )

    # Check the point estimates center approximately around the true value.
    # These aren't actually computed using the bootstrap, nor is the
    # exp-of-sample-mean estimator actually unbiased for the exp-of-true-mean.
    # But just to sanity-check that point_estimates is implemented.
    mean_point_estimates = inference.point_estimates()[
        "exp_mean"]["variable"].mean("replicates")
    np.testing.assert_allclose(mean_point_estimates, true_exp_mean, rtol=0.03)

    # We want the standard error estimates (or here their squares, the
    # variance estimates) to be close to unbiased estimates of the true
    # sampling variance of the exp-of-sample-mean estimator.
    stderr_estimates = inference.standard_error_estimates()
    mean_variance_estimate = (
        stderr_estimates["exp_mean"]["variable"]**2).mean("replicates")
    np.testing.assert_allclose(
        np.sqrt(mean_variance_estimate),
        true_sampling_dist.std(), rtol=0.05)

    for alpha in [0.05, 0.2, 0.1]:
      # Now let's check the coverage of the confidence intervals.
      lower, upper = inference.confidence_intervals(alpha)
      lower = lower["exp_mean"]["variable"]
      upper = upper["exp_mean"]["variable"]
      coverage_probability = (
          (lower <= true_exp_mean) & (true_exp_mean <= upper)
      ).mean("replicates").data
      np.testing.assert_allclose(coverage_probability, 1-alpha, rtol=0.05)

      # And we'll check that p-values under the null hypothesis are <alpha in
      # around alpha proportion of replicates.
      p_less_than_alpha = (
          inference.p_values(null_value=true_exp_mean)["exp_mean"]["variable"]
          <= alpha)
      np.testing.assert_allclose(
          p_less_than_alpha.mean("replicates"), alpha, rtol=0.05)

  def test_cluster_bootstrap_with_equal_values_in_each_cluster(self):
    # A sanity-check for the cluster bootstrap: if each cluster consists of
    # copies of the same data point, the results should be the same as if the
    # inference had been done with 1 data point per cluster.

    effective_sample_size = 100
    repeat_factor = 2
    n_replicates = 10000

    sigma = 1.
    mean = 0.
    dist = scipy.stats.norm(loc=mean, scale=sigma)
    original_data = dist.rvs((effective_sample_size, n_replicates))
    true_dist_of_sample_mean = scipy.stats.norm(
        loc=mean, scale=sigma/np.sqrt(effective_sample_size))

    # Repeating each data point twice shouldn't change the distribution of the
    # sample mean. But if we don't take into account the correlation correctly
    # when bootstrapping, the inferred standard errors will be off.
    data = np.repeat(original_data, repeat_factor, axis=0)

    # So we tell it which data points are in the same 'cluster' and perform
    # a cluster bootstrap at the cluster level. The actual values of the cluster
    # IDs don't matter so long as they're unique per cluster, so we sample them
    # randomly to make sure implementation doesn't rely on them being sequential
    # or anything.
    cluster_ids = np.random.choice(effective_sample_size*10,
                                   size=effective_sample_size,
                                   replace=False)
    cluster_ids = np.repeat(cluster_ids, repeat_factor)
    cluster_ids = xr.DataArray(data=cluster_ids, dims=("samples",))
    data = xr.DataArray(data=data,
                        dims=("samples", "replicates"),
                        coords={"cluster": cluster_ids})

    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)
    inference = bootstrap.ClusterBootstrap(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_coord="cluster",
        n_replicates=n_replicates,
    )

    stderr_estimates = inference.standard_error_estimates()
    mean_variance_estimate = (
        stderr_estimates["mean"]["variable"]**2).mean("replicates")
    np.testing.assert_allclose(
        np.sqrt(mean_variance_estimate),
        true_dist_of_sample_mean.std(),
        rtol=0.01)

