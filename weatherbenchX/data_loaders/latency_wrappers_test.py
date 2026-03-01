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
"""Tests for latency wrappers."""

from absl.testing import absltest
import numpy as np
from weatherbenchX import test_utils
from weatherbenchX.data_loaders import latency_wrappers
from weatherbenchX.data_loaders import xarray_loaders


class LatencyWrappersTest(absltest.TestCase):

  def test_latency_wrapper_for_zarr(self):
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-04T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='30 hours',
        lead_resolution='6 hours',
        random=True,
    )
    prediction_path = self.create_tempdir('prediction.zarr').full_path
    prediction.to_zarr(prediction_path)

    data_loader = xarray_loaders.PredictionsFromXarray(
        path=prediction_path, variables=['2m_temperature']
    )
    init_times = np.array(
        ['2020-01-02T00', '2020-01-02T06'], dtype='datetime64[ns]'
    )
    lead_times = np.array([6, 12], dtype='timedelta64[h]')

    latency = np.timedelta64(6, 'h')

    # These are the init/lead times that should be sampled.
    available_init_times = [
        np.array(['2020-01-01T12'], dtype='datetime64[ns]'),
        np.array(['2020-01-02T00'], dtype='datetime64[ns]'),
    ]
    available_lead_times = [
        np.array([6 + 12, 12 + 12], dtype='timedelta64[h]'),
        np.array([6 + 6, 12 + 6], dtype='timedelta64[h]'),
    ]

    # Explicitly pass nominal init times to the wrapper.
    wrapped_data_loader = latency_wrappers.ConstantLatencyWrapper(
        data_loader, latency=latency, nominal_init_times=prediction.time.values
    )

    # Use the Zarr shorthand.
    wrapped_data_loader_2 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader,
        latency=latency,
    )

    wrapped_output = wrapped_data_loader.load_chunk(init_times, lead_times)
    wrapped_output_2 = wrapped_data_loader_2.load_chunk(init_times, lead_times)

    for i, (available_init_time, available_lead_time) in enumerate(
        zip(available_init_times, available_lead_times)
    ):
      correct_output = data_loader.load_chunk(
          available_init_time, available_lead_time
      )
      np.testing.assert_allclose(
          wrapped_output.isel(init_time=[i])['2m_temperature'].values,
          correct_output['2m_temperature'].values,
      )
      np.testing.assert_allclose(
          wrapped_output_2.isel(init_time=[i])['2m_temperature'].values,
          correct_output['2m_temperature'].values,
      )

  def test_multiple_latency_wrappers(self):
    prediction_0012 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-04T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='30 hours',
        lead_resolution='6 hours',
        random=True,
    )
    prediction_0618 = test_utils.mock_prediction_data(
        time_start='2020-01-01T06',
        time_stop='2020-01-04T00',
        time_resolution=np.timedelta64(12, 'h'),
        lead_start='0 hours',
        lead_stop='30 hours',
        lead_resolution='6 hours',
        random=True,
    )
    prediction_path_0012 = self.create_tempdir('prediction_0012.zarr').full_path
    prediction_path_0618 = self.create_tempdir('prediction_0618.zarr').full_path
    prediction_0012.to_zarr(prediction_path_0012)
    prediction_0618.to_zarr(prediction_path_0618)

    data_loader_0012 = xarray_loaders.PredictionsFromXarray(
        path=prediction_path_0012, variables=['2m_temperature']
    )
    data_loader_0618 = xarray_loaders.PredictionsFromXarray(
        path=prediction_path_0618, variables=['2m_temperature']
    )

    init_times = np.array(
        ['2020-01-02T00', '2020-01-02T06'], dtype='datetime64[ns]'
    )
    lead_times = np.array([6, 12], dtype='timedelta64[h]')

    latency = np.timedelta64(6, 'h')

    wrapped_data_loader_0012 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader_0012, latency=latency
    )
    wrapped_data_loader_0618 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader_0618, latency=latency
    )
    wrapped_data_loader = latency_wrappers.MultipleConstantLatencyWrapper(
        [wrapped_data_loader_0012, wrapped_data_loader_0618]
    )

    wrapped_output = wrapped_data_loader.load_chunk(init_times, lead_times)

    available_init_times = [
        np.array(['2020-01-01T18'], dtype='datetime64[ns]'),
        np.array(['2020-01-02T00'], dtype='datetime64[ns]'),
    ]
    available_lead_times = [
        np.array([6 + 6, 12 + 6], dtype='timedelta64[h]'),
        np.array([6 + 6, 12 + 6], dtype='timedelta64[h]'),
    ]
    available_data_loaders = [data_loader_0618, data_loader_0012]

    for i, (
        available_init_time,
        available_lead_time,
        available_data_loader,
    ) in enumerate(
        zip(available_init_times, available_lead_times, available_data_loaders)
    ):
      correct_output = available_data_loader.load_chunk(
          available_init_time, available_lead_time
      )
      np.testing.assert_allclose(
          wrapped_output.isel(init_time=[i])['2m_temperature'].values,
          correct_output['2m_temperature'].values,
      )

  def test_multiple_latency_wrappers_tie_breaking(self):
    # Setup two loaders with same nominal init times but different latencies.
    # We want a case where for a specific query time, they both return the SAME
    # available init time.
    prediction_1 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),  # 00, 12
        lead_start='0 hours',
        lead_stop='24 hours',
        lead_resolution='1 hours',
    ) + 1.0  # Add 1 to distinguish
    prediction_2 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(12, 'h'),  # 00, 12
        lead_start='0 hours',
        lead_stop='24 hours',
        lead_resolution='1 hours',
    ) + 2.0  # Add 2 to distinguish

    prediction_path_1 = self.create_tempdir('prediction_1.zarr').full_path
    prediction_path_2 = self.create_tempdir('prediction_2.zarr').full_path
    prediction_1.to_zarr(prediction_path_1)
    prediction_2.to_zarr(prediction_path_2)

    data_loader_1 = xarray_loaders.PredictionsFromXarray(
        path=prediction_path_1, variables=['2m_temperature']
    )
    data_loader_2 = xarray_loaders.PredictionsFromXarray(
        path=prediction_path_2, variables=['2m_temperature']
    )

    # Loader 1: Latency 6h.
    # Loader 2: Latency 12h.
    wrapper1 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader_1, latency=np.timedelta64(6, 'h')
    )
    wrapper2 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader_2, latency=np.timedelta64(12, 'h')
    )

    multi_wrapper = latency_wrappers.MultipleConstantLatencyWrapper(
        [wrapper1, wrapper2]
    )

    # Query time: 2020-01-01T13.
    # Nominal times: 00, 12.

    # Wrapper 1 (6h): Issue times T06, T18.
    # Query T13 -> Issue time T06 (nominal init T00).

    # Wrapper 2 (12h): Issue times T12, T24.
    # Query T13 -> Issue time T12 (nominal init T00).

    # Both return init T00. Wrapper 2 should be chosen (larger latency).
    init_times = np.array(['2020-01-01T13'], dtype='datetime64[ns]')
    lead_times = np.array([6], dtype='timedelta64[h]')

    wrapped_output = multi_wrapper.load_chunk(init_times, lead_times)

    # Should match prediction_2
    correct_output = data_loader_2.load_chunk(
        np.array(['2020-01-01T00'], dtype='datetime64[ns]'),
        # offset = 13 - 0 = 13h. lead = 6 + 13 = 19h.
        np.array([19], dtype='timedelta64[h]'),
    )

    np.testing.assert_allclose(
        wrapped_output.isel(init_time=[0])['2m_temperature'].values,
        correct_output['2m_temperature'].values,
    )

  def test_multiple_latency_wrappers_with_missing_init_time(self):
    prediction = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-02T00',
        time_resolution=np.timedelta64(24, 'h'),  # Only 00
        lead_start='0 hours',
        lead_stop='12 hours',
        lead_resolution='1 hours',
    )
    prediction_path = self.create_tempdir('prediction.zarr').full_path
    prediction.to_zarr(prediction_path)

    data_loader = xarray_loaders.PredictionsFromXarray(
        path=prediction_path, variables=['2m_temperature']
    )

    # Loader 1: Latency 6h. Issue time T06.
    # Loader 2: Latency 1h. Issue time T01.
    wrapper1 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader, latency=np.timedelta64(6, 'h')
    )
    wrapper2 = latency_wrappers.XarrayConstantLatencyWrapper(
        data_loader, latency=np.timedelta64(1, 'h')
    )

    multi_wrapper = latency_wrappers.MultipleConstantLatencyWrapper(
        [wrapper1, wrapper2]
    )

    # Query time: 05.
    # Wrapper 1 (6h): returns None (there is no init time before T05).
    # Wrapper 2 (1h): returns issue time T06, nominal init T00.
    # Wrapper 2 should be chosen.
    init_times = np.array(['2020-01-01T05'], dtype='datetime64[ns]')
    lead_times = np.array([1], dtype='timedelta64[h]')

    wrapped_output = multi_wrapper.load_chunk(init_times, lead_times)

    # Check that we got data (not all NaNs or error)
    # Lead loaded = 1 + 5 = 6h.
    correct_output = data_loader.load_chunk(
        np.array(['2020-01-01T00'], dtype='datetime64[ns]'),
        np.array([6], dtype='timedelta64[h]'),
    )
    np.testing.assert_allclose(
        wrapped_output.isel(init_time=[0])['2m_temperature'].values,
        correct_output['2m_temperature'].values,
    )


if __name__ == '__main__':
  absltest.main()
