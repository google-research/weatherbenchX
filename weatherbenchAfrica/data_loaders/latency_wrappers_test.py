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
from weatherbenchAfrica import test_utils
from weatherbenchAfrica.data_loaders import latency_wrappers
from weatherbenchAfrica.data_loaders import xarray_loaders


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


if __name__ == '__main__':
  absltest.main()
