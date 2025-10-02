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
import pandas as pd
from weatherbenchX import aggregation
from weatherbenchX import beam_pipeline
from weatherbenchX import binning
from weatherbenchX import interpolations
from weatherbenchX import test_utils
from weatherbenchX import time_chunks
from weatherbenchX import xarray_tree
from weatherbenchX.data_loaders import sparse_parquet
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

    # Compute results directly, reading all the `times` as a single chunk:
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        all_metrics,
        prediction_loader.load_chunk(init_times, lead_times),
        target_loader.load_chunk(init_times, lead_times),
    )

    direct_aggregation_state = aggregation_method.aggregate_statistics(
        statistics
    )

    direct_metrics = direct_aggregation_state.metric_values(
        all_metrics
    ).compute()

    # Compute results with pipeline
    metrics_path = self.create_tempfile('metrics.nc').full_path
    aggregation_state_path = self.create_tempfile(
        'aggregation_state.nc'
    ).full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=metrics_path,
          aggregation_state_out_path=aggregation_state_path,
      )
    metrics_results = xr.open_dataset(metrics_path).compute()

    # There can be small differences due to numerical errors.
    xr.testing.assert_allclose(direct_metrics, metrics_results, atol=1e-5)

    aggregation_state_results = aggregation.AggregationState.from_dataset(
        xr.open_dataset(aggregation_state_path).compute()
    )
    xarray_tree.map_structure(
        lambda x, y: xr.testing.assert_allclose(x, y, atol=1e-5),
        (
            direct_aggregation_state.sum_weighted_statistics,
            direct_aggregation_state.sum_weights,
        ),
        (
            aggregation_state_results.sum_weighted_statistics,
            aggregation_state_results.sum_weights,
        ),
    )

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
                    which='predictions',
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

  def _sparse_aggregation_setup(self, init_times, lead_times):
    predictions = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-01T12',
        time_resolution='6 hours',
        lead_start='0 days',
        lead_stop='6 hours',
        lead_resolution='6 hours',
        random=False,
        seed=0,
        variables_3d=[],
    )
    predictions_path = self.create_tempdir(
        'predictions_for_sparse_tests.zarr'
    ).full_path
    predictions.to_zarr(predictions_path)

    prediction_loader = xarray_loaders.PredictionsFromXarray(
        path=predictions_path,
        interpolation=interpolations.InterpolateToReferenceCoords(
            method='nearest',
            wrap_longitude=True,
            extrapolate_out_of_bounds=True,
        ),
    )

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=1,
        lead_time_chunk_size=1,
    )
    all_metrics = {'rmse': deterministic.RMSE()}
    return prediction_loader, times, all_metrics

  def test_sparse_aggregation_different_stations_per_lead_time(self):
    """Tests sparse aggregation with overlapping but different station names per lead time.

    Without the xr.align step in ConcatPerStatisticPerVariable, this will
    cause the following error: "ValueError: Resulting object does not have
    monotonic global indexes along dimension stationName"
    """
    init_times = np.array(['2020-01-01T00'], dtype='datetime64[ns]')
    lead_times = np.arange(0, 12, 6, dtype='timedelta64[h]').astype(
        'timedelta64[ns]'
    )
    prediction_loader, times, all_metrics = self._sparse_aggregation_setup(
        init_times, lead_times
    )
    test_df = pd.DataFrame({
        'stationName': ['A', 'B', 'C', 'B'],
        'latitude': [0, 0, 0, 0],
        'longitude': [0, 0, 0, 0],
        'timeNominal': np.array(
            [
                '2020-01-01T00',
                '2020-01-01T00',
                '2020-01-01T00',
                '2020-01-01T06',
            ],
            dtype='datetime64[ns]',
        ),
        '2m_temperature': [0, 0, 0, 0],
    })

    sparse_path = self.create_tempdir('test1/year=2020/month=1').full_path
    test_df.to_parquet(f'{sparse_path}/2020-01.parquet')
    target_loader = sparse_parquet.SparseObservationsFromParquet(
        path=sparse_path.split('/year')[0],
        variables=['2m_temperature'],
        partitioned_by='month',
        time_dim='timeNominal',
        file_tolerance=np.timedelta64(0, 'h'),
        coordinate_variables=['stationName', 'latitude', 'longitude'],
    )
    bin_by = [
        binning.ByExactCoord('lead_time'),
        binning.ByExactCoord('stationName'),
    ]
    aggregation_method = aggregation.Aggregator(
        reduce_dims=['index'], bin_by=bin_by
    )
    metrics_path = self.create_tempfile('metrics.nc').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=metrics_path,
      )
    metrics_results = xr.open_dataset(metrics_path).compute()
    expected = xr.Dataset({
        'rmse.2m_temperature': xr.DataArray(
            np.array([[0, 0, 0], [np.nan, 0, np.nan]]),
            dims=['lead_time', 'stationName'],
            coords={
                'lead_time': lead_times,
                'stationName': ['A', 'B', 'C'],
            },
        )
    })
    xr.testing.assert_allclose(metrics_results, expected)

  def test_sparse_aggregation_no_stations_for_one_lead_time(self):
    """Tests sparse aggregation with one lead time having no stations.

    Using lead_time binning this will results in a zero-sized array.
    Without dropping those in ConcatPerStatisticPerVariable, this will
    cause the following error: "ValueError: Cannot handle size zero
    dimensions"
    """
    init_times = np.array(['2020-01-01T00'], dtype='datetime64[ns]')
    lead_times = np.arange(0, 12, 6, dtype='timedelta64[h]').astype(
        'timedelta64[ns]'
    )
    prediction_loader, times, all_metrics = self._sparse_aggregation_setup(
        init_times, lead_times
    )
    test_df = pd.DataFrame({
        'stationName': ['A', 'B', 'C'],
        'latitude': [0, 0, 0],
        'longitude': [0, 0, 0],
        'timeNominal': np.array(
            [
                '2020-01-01T00',
                '2020-01-01T00',
                '2020-01-01T00',
            ],
            dtype='datetime64[ns]',
        ),
        '2m_temperature': [0, 0, 0],
    })
    sparse_path = self.create_tempdir('test2/year=2020/month=1').full_path
    test_df.to_parquet(f'{sparse_path}/2020-01.parquet')
    target_loader = sparse_parquet.SparseObservationsFromParquet(
        path=sparse_path.split('/year')[0],
        variables=['2m_temperature'],
        partitioned_by='month',
        time_dim='timeNominal',
        file_tolerance=np.timedelta64(0, 'h'),
        coordinate_variables=['stationName', 'latitude', 'longitude'],
    )
    bin_by = [
        binning.ByExactCoord('lead_time'),
    ]
    aggregation_method = aggregation.Aggregator(
        reduce_dims=['index'], bin_by=bin_by
    )
    metrics_path = self.create_tempfile('metrics.nc').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=metrics_path,
      )
    metrics_results = xr.open_dataset(metrics_path).compute()
    expected = xr.Dataset({
        'rmse.2m_temperature': xr.DataArray(
            [0], dims=['lead_time'], coords={'lead_time': lead_times[:1]}
        )
    })
    xr.testing.assert_allclose(metrics_results, expected)

  def test_sparse_aggregation_no_stations_for_one_init_time(self):
    """Tests sparse aggregation with one init time having no stations.

    After the aggregation state, this will result in one array with a lead_time
    dimension of size 1 and one array with a lead_time dimension of size 0.

    Using combining sums in the aggregation (rather than simply using xarray's
    built-in sum which will use the intersection of coordinates) ensures that
    we take the sum over all stations.
    """
    init_times = np.array(
        ['2020-01-01T00', '2020-01-01T06'], dtype='datetime64[ns]'
    )
    lead_times = np.arange(0, 6, 6, dtype='timedelta64[h]').astype(
        'timedelta64[ns]'
    )
    prediction_loader, times, all_metrics = self._sparse_aggregation_setup(
        init_times, lead_times
    )
    test_df = pd.DataFrame({
        'stationName': ['A', 'B', 'C'],
        'latitude': [0, 0, 0],
        'longitude': [0, 0, 0],
        'timeNominal': np.array(
            [
                '2020-01-01T00',
                '2020-01-01T00',
                '2020-01-01T00',
            ],
            dtype='datetime64[ns]',
        ),
        '2m_temperature': [0, 0, 0],
    })
    sparse_path = self.create_tempdir('test/year=2020/month=1').full_path
    test_df.to_parquet(f'{sparse_path}/2020-01.parquet')
    target_loader = sparse_parquet.SparseObservationsFromParquet(
        path=sparse_path.split('/year')[0],
        variables=['2m_temperature'],
        partitioned_by='month',
        time_dim='timeNominal',
        file_tolerance=np.timedelta64(0, 'h'),
        coordinate_variables=['stationName', 'latitude', 'longitude'],
    )
    bin_by = [
        binning.ByExactCoord('lead_time'),
    ]
    aggregation_method = aggregation.Aggregator(
        reduce_dims=['index'], bin_by=bin_by
    )
    metrics_path = self.create_tempfile('metrics.nc').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=metrics_path,
      )
    metrics_results = xr.open_dataset(metrics_path).compute()
    self.assertLen(metrics_results.lead_time, 1)

  def test_sparse_aggregation_empty_data(self):
    """Tests sparse aggregation with all data_arrays empty.

    This should result in an empty dataset.
    In ConcatPerStatisticPerVariable this requires a special check, since
    combine_by_coords will return a Dataset, rather than a DataArray, which
    will cause the following error in AggregationState.metric_values:
    "TypeError: Cannot assign a Dataset to a single key"
    """
    init_times = np.array(['2020-01-01T00'], dtype='datetime64[ns]')
    lead_times = np.arange(0, 12, 6, dtype='timedelta64[h]').astype(
        'timedelta64[ns]'
    )
    prediction_loader, times, all_metrics = self._sparse_aggregation_setup(
        init_times, lead_times
    )
    test_df = pd.DataFrame({
        'stationName': pd.Series([], dtype=object),
        'latitude': pd.Series([], dtype=np.int64),
        'longitude': pd.Series([], dtype=np.int64),
        'timeNominal': pd.Series([], dtype='datetime64[ns]'),
        '2m_temperature': pd.Series([], dtype=np.int64),
    })
    sparse_path = self.create_tempdir('test3/year=2020/month=1').full_path
    test_df.to_parquet(f'{sparse_path}/2020-01.parquet')
    target_loader = sparse_parquet.SparseObservationsFromParquet(
        path=sparse_path.split('/year')[0],
        variables=['2m_temperature'],
        partitioned_by='month',
        time_dim='timeNominal',
        file_tolerance=np.timedelta64(0, 'h'),
        coordinate_variables=['stationName', 'latitude', 'longitude'],
    )
    bin_by = [
        binning.ByExactCoord('lead_time'),
    ]
    aggregation_method = aggregation.Aggregator(
        reduce_dims=['index'], bin_by=bin_by
    )
    metrics_path = self.create_tempfile('metrics.nc').full_path
    with test_pipeline.TestPipeline() as root:
      beam_pipeline.define_pipeline(
          root,
          times,
          prediction_loader,
          target_loader,
          all_metrics,
          aggregation_method,
          out_path=metrics_path,
      )
    metrics_results = xr.open_dataset(metrics_path).compute()
    expected = xr.Dataset({'rmse.2m_temperature': xr.DataArray()})
    xr.testing.assert_allclose(metrics_results, expected)


if __name__ == '__main__':
  absltest.main()
