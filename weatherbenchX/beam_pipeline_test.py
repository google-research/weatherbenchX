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
from absl.testing import parameterized
from apache_beam.testing import test_pipeline
from weatherbenchX import aggregation
from weatherbenchX import beam_pipeline
from weatherbenchX import test_utils
from weatherbenchX import time_chunks
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import wrappers
import xarray as xr


class BeamPipelineTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.predictions_path = self.create_tempdir('predictions.zarr').full_path
    self.targets_path = self.create_tempdir('targets.zarr').full_path

    self.predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        lead_start='0 days',
        lead_stop='1 day',
        random=True,
        seed=0,
    )
    self.targets = test_utils.mock_target_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-05T00',
        random=True,
        seed=1,
    )

    self.predictions.to_zarr(self.predictions_path)
    self.targets.to_zarr(self.targets_path)

  @parameterized.parameters(
      {'reduce_dims': ['init_time', 'latitude', 'longitude']},
      {'reduce_dims': ['init_time']},
      {'reduce_dims': ['lead_time']},
      {'reduce_dims': ['latitude', 'longitude']},
      {'reduce_dims': []},
  )
  def test_pipeline(self, reduce_dims):
    """Test equivalence of pipeline results to directly computed results."""

    init_times = self.predictions.time.values
    lead_times = self.predictions.prediction_timedelta.values

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=1,
        lead_time_chunk_size=1,
    )
    # We're testing something non-trivial here because there are multiple chunks
    # along each of these dimensions that the beam job chunks over.
    assert len(times.init_times) > 1
    assert len(times.lead_times) > 1

    target_loader = xarray_loaders.TargetsFromXarray(
        path=self.targets_path,
    )
    prediction_loader = xarray_loaders.PredictionsFromXarray(
        path=self.predictions_path,
    )

    all_metrics = {'rmse': deterministic.RMSE(), 'mse': deterministic.MSE()}

    aggregation_method = aggregation.Aggregator(reduce_dims=reduce_dims)

    # Compute results directly
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        all_metrics,
        prediction_loader.load_chunk(init_times, lead_times),
        target_loader.load_chunk(init_times, lead_times),
    )

    aggregation_state = aggregation_method.aggregate_statistics(statistics)

    direct_results = aggregation_state.metric_values(all_metrics).compute()

    # Compute results with pipeline
    results_path = self.create_tempfile('results.nc').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=results_path,
      )
    pipeline_results = xr.open_dataset(results_path).compute()

    # There can be small differences due to numerical errors.
    xr.testing.assert_allclose(direct_results, pipeline_results, atol=1e-5)

  def test_unaggregated_pipeline(self):
    """Test equivalence of unaggregated pipeline results."""

    init_times = self.predictions.time.values
    lead_times = self.predictions.prediction_timedelta.values

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=1,
        lead_time_chunk_size=1,
    )

    target_loader = xarray_loaders.TargetsFromXarray(
        path=self.targets_path,
    )
    prediction_loader = xarray_loaders.PredictionsFromXarray(
        path=self.predictions_path,
    )

    all_metrics = {
        'rmse': deterministic.RMSE(),
        'mse': deterministic.MSE(),
        # Example metric that excludes the "lead_time" dimension.
        'bias_5_to_10_days': wrappers.WrappedMetric(
            deterministic.Bias(),
            [
                wrappers.Select(
                    which='both',
                    sel={'lead_time': slice('5D', '10D')},
                ),
                wrappers.EnsembleMean(
                    which='predictions', ensemble_dim='lead_time'
                ),
            ],
            unique_name_suffix='5_to_10_days',
        ),
    }

    # Compute results directly
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        all_metrics,
        prediction_loader.load_chunk(init_times, lead_times),
        target_loader.load_chunk(init_times, lead_times),
    )
    direct_results = xr.Dataset()
    for stat_name, var_dict in statistics.items():
      for var_name, da in var_dict.items():
        direct_results[f'{stat_name}.{var_name}'] = da
    direct_results = direct_results.transpose('init_time', 'lead_time', ...)

    # Compute results with pipeline
    results_path = self.create_tempdir('results.zarr').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_unaggregated_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          out_path=results_path,
      )
    pipeline_results = xr.open_dataset(results_path).compute()

    # There can be small differences due to numerical errors.
    xr.testing.assert_allclose(direct_results, pipeline_results, atol=1e-5)


if __name__ == '__main__':
  absltest.main()
