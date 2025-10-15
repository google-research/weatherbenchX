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
from weatherbenchX.metrics import base as metrics_base
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

    # Check the point estimates have low bias relative to the true value.
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


class StationaryBootstrapTest(absltest.TestCase):

  def test_stationary_bootstrap_inference_of_mean_of_ar1_process(self):
    # For this code I wanted a test where:
    #
    # * The data has some temporal autocorrelation
    # * We know (some properties of) the true sampling distribution of our
    #   estimator and can compare our bootstrap results with these.
    #
    # Simplest case of this I could find was estimating the mean of a
    # stationary Gaussian AR(1) process, here the variance of the sample mean
    # is known exactly in closed form.
    # The sample mean is N(mu, sigma^2_marginal * correction_factor / n)
    # where correction_factor is 1 + 2phi/(1-phi) * (1 - (1 - phi^n)/(1-phi)/n))
    np.random.seed(0)
    sigma = 0.1
    phi = 0.9
    true_mean = 10.
    # We need a lot of data points to avoid finite sample bias in bootstrap
    # standard error estimates (which tends to be a downwards bias).
    n = 10000
    data = test_utils.simulate_ar1(
        mean=true_mean,
        sigma=sigma,
        phi=phi,
        steps=n,
        # These are for replicates of the entire bootstrap procedure on
        # different original datasets. For each one of these original datasets,
        # a number of bootstrap replicates will be computed from resampled
        # versions of the dataset.
        replicates=100,
    )
    data = xr.DataArray(data=data, dims=("steps", "replicates"))
    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)

    statistical_inference_method = bootstrap.StationaryBootstrap(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
        # This is referring to bootstrap-resampled replicates used internally
        # by the inference method, not the original dataset replicates above.
        n_replicates=500,
    )

    # Check the point estimates have low bias relative to the true mean.
    mean_point_estimates = statistical_inference_method.point_estimates()[
        "mean"]["variable"].mean("replicates")
    np.testing.assert_allclose(mean_point_estimates, true_mean, rtol=1e-2)

    # Now check that the (squared) standard error estimates are approximately
    # unbiased estimates of the (squared) true values under the sampling
    # distribution.
    true_stderr = test_utils.gaussian_ar1_true_stderr_of_sample_mean(
        sigma, phi, n)
    stderr_estimates = statistical_inference_method.standard_error_estimates()
    # We ideally want these standard error estimates (or rather their squares,
    # the variance estimates) to be close to unbiased estimates of the true
    # sampling variance. We can assess this by looking at the mean of the
    # estimated variance over a number of replicates of the whole bootstrap
    # procedure.
    mean_variance_estimate = (stderr_estimates["mean"]["variable"]**2).mean()
    # Even with a large-ish sample of 10000 steps the bootstrap estimate will
    # have some finite-sample bias, and some variance even when we average
    # 100 replicates of it. So the tolerance here needs to be somewhat loose.
    np.testing.assert_allclose(
        np.sqrt(mean_variance_estimate), true_stderr, rtol=0.07)

    # Now let's check the coverage of the confidence intervals.
    for alpha in [0.2, 0.1, 0.05]:
      lower, upper = statistical_inference_method.confidence_intervals(alpha)
      lower = lower["mean"]["variable"]
      upper = upper["mean"]["variable"]
      coverage_probability = (
          (lower <= true_mean) & (true_mean <= upper)).mean("replicates").data
      # Tolerance here also loose for similar reasons.
      np.testing.assert_allclose(coverage_probability, 1-alpha, rtol=0.07)

  def test_different_autocorrelation_at_different_indices(self):
    # We'll do a similar check to above, but use different time series with
    # very different amounts of autocorrelation at different indices along
    # some other dimension.
    # We want to see that the inference of standard error is still good for
    # both time series. This will require that the automatic block length
    # selection (and the bootstrap index sampling based on it) is performed
    # separately for each time series.
    np.random.seed(0)

    n = 10000

    def samples_and_true_sample_mean_stderr(mean, sigma, phi):
      samples = test_utils.simulate_ar1(
          mean=mean,
          sigma=sigma,
          phi=phi,
          steps=n,
          replicates=100,
      )
      true_stderr = test_utils.gaussian_ar1_true_stderr_of_sample_mean(
          sigma, phi, n)
      return samples, true_stderr

    high_autocorr_samples, high_autocorr_true_stderr = (
        samples_and_true_sample_mean_stderr(
            mean=-10, sigma=np.sqrt(1 - 0.9**2), phi=0.9))
    low_autocorr_samples, low_autocorr_true_stderr = (
        samples_and_true_sample_mean_stderr(
            mean=10, sigma=np.sqrt(1 - 0.2**2), phi=0.2))
    data = xr.DataArray(
        data=np.stack(
            [high_autocorr_samples, low_autocorr_samples], axis=-1),
        dims=("steps", "replicates", "autocorr"),
        coords={"autocorr": ["high", "low"]},
    )

    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)

    statistical_inference_method = bootstrap.StationaryBootstrap(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
        n_replicates=250,
    )

    stderr_estimates = statistical_inference_method.standard_error_estimates()
    root_mean_variance_estimate = np.sqrt(
        (stderr_estimates["mean"]["variable"]**2).mean("replicates"))
    np.testing.assert_allclose(
        root_mean_variance_estimate.sel(autocorr="high"),
        high_autocorr_true_stderr,
        rtol=0.07)
    np.testing.assert_allclose(
        root_mean_variance_estimate.sel(autocorr="low"),
        low_autocorr_true_stderr,
        rtol=0.07)

  def test_nonlinear_function_of_vector_mean(self):
    # Check the bootstrap performs well for a more complicated example,
    # where we test the inference of the ratio of two means, and we don't
    # rely on a closed-form formula for the correct result.
    np.random.seed(0)

    n = 10000
    true_mean_numerator = 8
    true_mean_denominator = 4
    ratio_of_true_means = true_mean_numerator / true_mean_denominator

    def metrics_and_agg_stats(dataset_replicates):
      numerators = xr.DataArray(
          data=test_utils.simulate_ar1(
              mean=true_mean_numerator,
              sigma=np.sqrt(1 - 0.1**2),
              phi=0.1,
              steps=n,
              replicates=dataset_replicates,
          ),
          dims=("steps", "replicates")
      )
      denominators = xr.DataArray(
          data=test_utils.simulate_ar1(
              mean=true_mean_denominator,
              sigma=np.sqrt(1 - 0.1**2),
              phi=0.1,
              steps=n,
              replicates=dataset_replicates,
          ),
          dims=("steps", "replicates")
      )
      return test_utils.metrics_and_agg_state_for_ratio_of_means(
          numerators, denominators)

    # We use a lot of replicates of the original dataset to estimate the true
    # std.dev of the sampling distribution of our ratio of means estimator.
    # This will be the ground truth for the test of the bootstrap estimator
    # of the std.err. below.
    metrics, agg_stats = metrics_and_agg_stats(dataset_replicates=10000)
    mean_stats = agg_stats.sum_along_dims(["steps"]).mean_statistics()
    draws_from_true_sampling_distribution = (
        metrics_base.compute_metrics_from_statistics(
            metrics, mean_stats)["ratio_of_means"]["variable"])
    estimated_true_stderr_of_ratio = draws_from_true_sampling_distribution.std(
        "replicates", ddof=1)

    # Now we compute our bootstrap estimate on each of 200 fresh replicates of
    # the original dataset.
    metrics, agg_stats = metrics_and_agg_stats(dataset_replicates=200)
    inference_method = bootstrap.StationaryBootstrap(
        metrics=metrics,
        aggregated_statistics=agg_stats,
        experimental_unit_dim="steps",
        n_replicates=300,  # Internal bootstrap replicates.
    )
    # Now check that the (squared) standard error estimates are approximately
    # unbiased estimates of the (squared) true values under the sampling
    # distribution, which we have estimated above.
    bootstrap_stderr_estimates = inference_method.standard_error_estimates()[
        "ratio_of_means"]["variable"]
    root_mean_bootstrap_variance_estimates = np.sqrt((
        bootstrap_stderr_estimates**2).mean("replicates"))
    np.testing.assert_allclose(
        root_mean_bootstrap_variance_estimates.mean(),
        estimated_true_stderr_of_ratio,
        rtol=0.05)

    # Now we check the coverage of the confidence intervals, using our draws
    # from the true sampling distribution to estimate the true mean value:
    for alpha in [0.2, 0.1, 0.05]:
      lower, upper = inference_method.confidence_intervals(alpha)
      lower = lower["ratio_of_means"]["variable"]
      upper = upper["ratio_of_means"]["variable"]
      coverage_probability = (
          (lower <= ratio_of_true_means) & (ratio_of_true_means <= upper)
      ).mean("replicates").data
      np.testing.assert_allclose(coverage_probability, 1-alpha, rtol=0.05)


if __name__ == "__main__":
  absltest.main()
