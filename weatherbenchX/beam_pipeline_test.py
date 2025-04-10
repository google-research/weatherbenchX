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
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import beam_pipeline
from weatherbenchX import test_utils
from weatherbenchX import time_chunks
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX.metrics import deterministic
import xarray as xr


class BeamPipelineTest(parameterized.TestCase):

  @parameterized.parameters(
      {'reduce_dims': ['init_time', 'latitude', 'longitude']},
      {'reduce_dims': ['init_time']},
      {'reduce_dims': ['latitude', 'longitude']},
      {'reduce_dims': []},
  )
  def test_pipeline(self, reduce_dims):
    """Test equivalence of pipeline results to directly computed results."""
    predictions_path = self.create_tempdir('predictions.zarr').full_path
    targets_path = self.create_tempdir('targets.zarr').full_path
    results_path = self.create_tempfile('results.nc').full_path

    predictions = (
        test_utils.mock_prediction_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-03T00',
            lead_start='0 days',
            lead_stop='1 day',
        )
        + np.random.uniform()
    )
    targets = (
        test_utils.mock_target_data(
            time_start='2020-01-01T00',
            time_stop='2020-01-05T00',
        )
        + np.random.uniform()
    )

    predictions.to_zarr(predictions_path)
    targets.to_zarr(targets_path)

    init_times = predictions.time.values
    lead_times = predictions.prediction_timedelta.values

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=1,
        lead_time_chunk_size=1,
    )

    target_loader = xarray_loaders.TargetsFromXarray(
        path=targets_path,
    )
    prediction_loader = xarray_loaders.PredictionsFromXarray(
        path=predictions_path,
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
    xr.testing.assert_allclose(direct_results, pipeline_results, rtol=1e-3)


if __name__ == '__main__':
  absltest.main()
