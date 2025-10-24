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
    # The sample mean is N(mu, sigma^2/n), its exp is then lognormal.
    true_sampling_dist = scipy.stats.lognorm(
        s=sigma/np.sqrt(sample_size), scale=np.exp(mean))
    # This is what our estimator is unbiased for, and what our confidence
    # intervals will be for.
    true_mean_of_sampling_dist = true_sampling_dist.mean()

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

    # Check the point estimates have low bias relative to their expected value.
    # These aren't actually computed using the bootstrap, but just to
    # sanity-check that point_estimates looks to be implemented correctly.
    mean_point_estimates = inference.point_estimates()[
        "exp_mean"]["variable"].mean("replicates")
    np.testing.assert_allclose(
        mean_point_estimates, true_mean_of_sampling_dist, rtol=0.03)

    # We want the standard error estimates (or here their squares, the
    # variance estimates) to be close to unbiased estimates of the true
    # sampling variance of the exp-of-sample-mean estimator.
    stderr_estimates = inference.standard_error_estimates()
    mean_variance_estimate = (
        stderr_estimates["exp_mean"]["variable"]**2).mean("replicates")
    np.testing.assert_allclose(
        np.sqrt(mean_variance_estimate),
        true_sampling_dist.std(), rtol=0.05)

    for alpha in [0.05, 0.1, 0.2]:
      with self.subTest(f"{alpha=}"):
        # Now let's check the coverage of the confidence intervals.
        test_utils.assert_coverage_probability_estimate_plausible(
            inference,
            # Intervals are for the expectation of the estimator under its
            # sampling distribution, so this is what we use to test coverage
            # of CIs. not the exp of the true mean since our estimator is
            # biased for this.
            true_value=true_mean_of_sampling_dist,
            metric_name="exp_mean",
            alpha=alpha,
            rtol=0.1,
            coverage_prob_significance_level=0.1,
        )
        # And that the p-values are consistent with inverting the confidence
        # intervals, i.e. the p-value is the alpha that will put the null
        # value on the boundary of the interval.
        test_utils.assert_p_value_consistent_with_confidence_interval(
            inference,
            null_value=true_mean_of_sampling_dist,
            metric_name="exp_mean",
        )

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
    sigma_marginal = 0.2
    phi = 0.9
    true_mean = 10.
    # We need a lot of data points to avoid finite sample bias in bootstrap
    # standard error estimates (which tends to be a downwards bias).
    n = 10000
    data = test_utils.simulate_ar1(
        mean=true_mean,
        sigma_marginal=sigma_marginal,
        phi=phi,
        steps=n,
        # These are for replicates of the entire bootstrap procedure on
        # different original datasets. For each one of these original datasets,
        # a number of bootstrap replicates will be computed from resampled
        # versions of the dataset.
        # We need a decent number of replicates here to get accurate enough
        # coverage probabilities for a meaningful test of CI coverage.
        replicates=500,
    )
    data = xr.DataArray(data=data, dims=("steps", "replicates"))
    metrics, aggregated_stats = test_utils.metrics_and_agg_state_for_mean(data)

    inference = bootstrap.StationaryBootstrap(
        metrics=metrics,
        aggregated_statistics=aggregated_stats,
        experimental_unit_dim="steps",
        # This is referring to bootstrap-resampled replicates used internally
        # by the inference method, not the original dataset replicates above.
        n_replicates=750,
    )

    # Check the point estimates have low bias relative to the true mean.
    mean_point_estimates = inference.point_estimates()[
        "mean"]["variable"].mean("replicates")
    np.testing.assert_allclose(mean_point_estimates, true_mean, rtol=1e-2)

    # Now check that the (squared) standard error estimates are approximately
    # unbiased estimates of the (squared) true values under the sampling
    # distribution.
    true_stderr = test_utils.gaussian_ar1_true_stderr_of_sample_mean(
        sigma_marginal, phi, n)
    stderr_estimates = inference.standard_error_estimates()
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

    # Now let's check the coverage of the confidence intervals under all the
    # different methods.
    for alpha in [0.05, 0.1, 0.2]:
      with self.subTest(f"{alpha=}"):
        test_utils.assert_coverage_probability_estimate_plausible(
            inference,
            # Our mean estimator is unbiased for the true mean so we can
            # expect the good coverage for the true mean:
            true_value=true_mean,
            alpha=alpha,
            # This means even after taking into account estimation error from
            # our 1000 replicates of the procedure, the coverage may be off
            # by 0.2 * alpha from what it should be.
            # So e.g. for 90% coverage we would tolerate results consistent
            # with anything between 88 - 92%.
            rtol=0.2,
        )

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

    def samples_and_true_sample_mean_stderr(mean, sigma_marginal, phi):
      samples = test_utils.simulate_ar1(
          mean=mean,
          sigma_marginal=sigma_marginal,
          phi=phi,
          steps=n,
          replicates=100,
      )
      true_stderr = test_utils.gaussian_ar1_true_stderr_of_sample_mean(
          sigma_marginal, phi, n)
      return samples, true_stderr

    high_autocorr_samples, high_autocorr_true_stderr = (
        samples_and_true_sample_mean_stderr(
            mean=-10, sigma_marginal=1.0, phi=0.9))
    low_autocorr_samples, low_autocorr_true_stderr = (
        samples_and_true_sample_mean_stderr(
            mean=10, sigma_marginal=1.0, phi=0.2))
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
    #
    # This is intended to be a more challenging setup: the variance of the raw
    # quantities is high relative to their means, so even after taking sample
    # means, the denominator in our ratio has enough coefficient of variation
    # for the non-linearity to have some effect and result in a sampling
    # distribution that's at least a bit skewed. There's also some temporal
    # autocorrelation thrown into the mix.
    #
    # Because the ratio function is extremely nonlinear over the range of the
    # individual per-timestep values (which include zero for the denominator),
    # the default block length selection approach (which computes the ratio on
    # a per-timestep basis) fails here, with the per-timestep division
    # destroying most of the relevant autocorrelation present in the numerator
    # and denominator. Instead we specify a mean block length to use manually.
    #
    # Much more challenging than this and the bootstrap really starts to fail.
    np.random.seed(0)
    n = 1000

    def metrics_and_agg_stats(dataset_replicates):
      numerators = xr.DataArray(
          data=test_utils.simulate_ar1(
              mean=2,
              sigma_marginal=6.6,
              phi=0.3,
              steps=n,
              replicates=dataset_replicates,
          ),
          dims=("steps", "replicates")
      )
      denominators = xr.DataArray(
          data=test_utils.simulate_ar1(
              mean=1,
              sigma_marginal=3.3,
              phi=0.3,
              steps=n,
              replicates=dataset_replicates,
          ),
          dims=("steps", "replicates")
      )
      return test_utils.metrics_and_agg_state_for_ratio_of_means(
          numerators, denominators)

    # We use a lot of replicates of the original dataset to estimate the true
    # mean and std.dev of the sampling distribution of our ratio of means
    # estimator. This will be used as ground truth in testing the bootstrap CIs
    # and the bootstrap estimator of the std.err. below.
    metrics, agg_stats = metrics_and_agg_stats(dataset_replicates=10000)
    mean_stats = agg_stats.sum_along_dims(["steps"]).mean_statistics()
    draws_from_sampling_dist = metrics_base.compute_metrics_from_statistics(
        metrics, mean_stats)["ratio_of_means"]["variable"]
    estimated_true_mean_of_sampling_dist = draws_from_sampling_dist.mean(
        "replicates")
    estimated_true_stddev_of_sampling_dist = draws_from_sampling_dist.std(
        "replicates", ddof=1)

    # Now we compute our bootstrap estimate on each of 500 fresh replicates of
    # the original dataset.
    # We need a decent number of replicates here to get accurate enough coverage
    # probabilities for a meaningful test of CI coverage.
    metrics, agg_stats = metrics_and_agg_stats(dataset_replicates=500)
    inference = bootstrap.StationaryBootstrap(
        metrics=metrics,
        aggregated_statistics=agg_stats,
        experimental_unit_dim="steps",
        n_replicates=500,  # Internal bootstrap replicates.
        # Automatic block length selection doesn't work well in this trickier
        # scenario, so we specify a sensible mean block length manually.
        mean_block_length=15,
    )
    # Now check that the (squared) standard error estimates are approximately
    # unbiased estimates of the (squared) true values under the sampling
    # distribution, which we have estimated above.
    bootstrap_stderr_estimates = inference.standard_error_estimates()[
        "ratio_of_means"]["variable"]
    root_mean_bootstrap_variance_estimates = np.sqrt((
        bootstrap_stderr_estimates**2).mean("replicates"))
    np.testing.assert_allclose(
        root_mean_bootstrap_variance_estimates.mean(),
        estimated_true_stddev_of_sampling_dist,
        rtol=0.21)

    # Now we check the coverage of the confidence intervals.
    for alpha in [0.05, 0.1, 0.2]:
      with self.subTest(f"{alpha=}"):
        test_utils.assert_coverage_probability_estimate_plausible(
            inference,
            # Our estimator is biased for the ratio of true means, the CIs
            # can be expected to cover the expectation of the estimator
            # instead (mean under its sampling distribution) for which it is
            # unbiased.
            true_value=estimated_true_mean_of_sampling_dist,
            metric_name="ratio_of_means",
            alpha=alpha,
            # Honestly the bootstrap tends to have coverage that's not
            # particularly great in more challenging cases like this, so we
            # use a fairly high tolerance. This is relative to alpha, so for
            # e.g. 90% coverage we would tolerate results consistent with
            # anything between 88 - 92%.
            rtol=0.2,
        )

if __name__ == "__main__":
  absltest.main()
