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
from weatherbenchX import interpolations
from weatherbenchX import test_utils
from weatherbenchX.data_loaders import concat_loaders
from weatherbenchX.data_loaders import latency_wrappers
from weatherbenchX.data_loaders import xarray_loaders
import xarray as xr


class ConcatLoadersTest(absltest.TestCase):

  def test_concat_data_loader_matches_eager_concat(self):
    pred1 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00'
    )
    pred2 = pred1 + 1.0
    pred3 = pred1 + 2.0

    path1 = self.create_tempdir('pred1.zarr').full_path
    path2 = self.create_tempdir('pred2.zarr').full_path
    path3 = self.create_tempdir('pred3.zarr').full_path

    pred1.to_zarr(path1)
    pred2.to_zarr(path2)
    pred3.to_zarr(path3)

    variables = ['geopotential', '2m_temperature']
    loaders = [
        xarray_loaders.PredictionsFromXarray(path=p, variables=variables)
        for p in [path1, path2, path3]
    ]
    concat_loader = concat_loaders.ConcatDataLoader(
        data_loaders=loaders, concat_dim='sample'
    )

    init_times = np.arange(
        '2020-01-01T00',
        '2020-01-02T00',
        np.timedelta64(24, 'h'),
        dtype='datetime64[ns]',
    )
    lead_times = np.arange(3, dtype='timedelta64[D]').astype('timedelta64[ns]')

    chunk = concat_loader.load_chunk(init_times, lead_times)

    # Compare against eager xr.concat
    eager_ds = xr.concat(
        [
            xr.open_zarr(p)
            .sel(time=init_times, prediction_timedelta=lead_times)[variables]
            .rename({'time': 'init_time', 'prediction_timedelta': 'lead_time'})
            for p in [path1, path2, path3]
        ],
        dim='sample',
    )

    xr.testing.assert_equal(chunk, eager_ds)
    self.assertEqual(chunk.sizes['sample'], 3)

  def test_empty_loaders_raises_value_error(self):
    with self.assertRaises(ValueError):
      concat_loaders.ConcatDataLoader(data_loaders=[])

  def test_properties_and_maybe_prepare(self):
    pred = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-03T00'
    )
    path = self.create_tempdir('pred.zarr').full_path
    pred.to_zarr(path)

    variables = ['2m_temperature']
    loader = xarray_loaders.PredictionsFromXarray(
        path=path, variables=variables
    )
    concat_loader = concat_loaders.ConcatDataLoader(
        data_loaders=[loader], concat_dim='sample'
    )

    self.assertEqual(concat_loader._variables, variables)
    self.assertIsNone(concat_loader._ds)

    concat_loader.maybe_prepare_dataset()
    self.assertIsNotNone(concat_loader._ds)
    self.assertIn('2m_temperature', concat_loader._ds)

  def test_wrapped_in_latency_wrapper(self):
    pred = test_utils.mock_prediction_data(
        time_start='2020-01-01T00',
        time_stop='2020-01-03T00',
        lead_resolution='6 hours',
        lead_stop='2 days',
    )
    path = self.create_tempdir('pred.zarr').full_path
    pred.to_zarr(path)

    loader = xarray_loaders.PredictionsFromXarray(path=path)
    concat_loader = concat_loaders.ConcatDataLoader(data_loaders=[loader])

    wrapped = latency_wrappers.XarrayConstantLatencyWrapper(
        concat_loader, latency=np.timedelta64(6, 'h')
    )

    init_times = np.array(['2020-01-01T06'], dtype='datetime64[ns]')
    lead_times = np.array([np.timedelta64(6, 'h')])
    chunk = wrapped.load_chunk(init_times, lead_times)
    self.assertEqual(chunk.sizes['sample'], 1)

  def test_default_concat_kwargs_override_and_minimal(self):
    pred1 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-02T00'
    )
    pred2 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-02T00'
    )
    # Give differing non-dim coords/attrs to test compat='override'
    pred1.attrs['title'] = 'Dataset 1'
    pred2.attrs['title'] = 'Dataset 2'
    pred1 = pred1.assign_coords(extra_coord=('time', [10]))
    pred2 = pred2.assign_coords(extra_coord=('time', [20]))

    path1 = self.create_tempdir('pred1.zarr').full_path
    path2 = self.create_tempdir('pred2.zarr').full_path
    pred1.to_zarr(path1)
    pred2.to_zarr(path2)

    loader1 = xarray_loaders.PredictionsFromXarray(path=path1)
    loader2 = xarray_loaders.PredictionsFromXarray(path=path2)
    concat_loader = concat_loaders.ConcatDataLoader(
        data_loaders=[loader1, loader2], concat_dim='sample'
    )

    init_times = np.array(['2020-01-01T00'], dtype='datetime64[ns]')
    lead_times = np.array([np.timedelta64(0, 'h')])
    # Should not raise MergeError because compat='override', coords='minimal' is default
    chunk = concat_loader.load_chunk(init_times, lead_times)
    self.assertEqual(chunk.sizes['sample'], 2)

  def test_reference_forwarding_in_load_chunk(self):
    pred1 = test_utils.mock_prediction_data(
        time_start='2020-01-01T00', time_stop='2020-01-02T00'
    )
    pred2 = pred1 + 1.0
    path1 = self.create_tempdir('pred1.zarr').full_path
    path2 = self.create_tempdir('pred2.zarr').full_path
    pred1.to_zarr(path1)
    pred2.to_zarr(path2)

    interp = interpolations.InterpolateToReferenceCoords(
        dims=['latitude', 'longitude'], method='nearest'
    )
    loader1 = xarray_loaders.PredictionsFromXarray(
        path=path1, variables=['2m_temperature'], interpolation=interp
    )
    loader2 = xarray_loaders.PredictionsFromXarray(
        path=path2, variables=['2m_temperature'], interpolation=interp
    )
    concat_loader = concat_loaders.ConcatDataLoader(
        data_loaders=[loader1, loader2], concat_dim='sample'
    )

    init_times = np.array(['2020-01-01T00'], dtype='datetime64[ns]')
    lead_times = np.array([np.timedelta64(0, 'h')])
    ref_lat = pred1.latitude.values[:2]
    ref_lon = pred1.longitude.values[:3]
    reference = {
        '2m_temperature': xr.DataArray(
            np.zeros((2, 3)),
            coords={'latitude': ref_lat, 'longitude': ref_lon},
            dims=['latitude', 'longitude'],
        )
    }
    chunk = concat_loader.load_chunk(
        init_times, lead_times, reference=reference
    )
    self.assertEqual(chunk['2m_temperature'].sizes['sample'], 2)
    self.assertEqual(chunk['2m_temperature'].sizes['latitude'], 2)
    self.assertEqual(chunk['2m_temperature'].sizes['longitude'], 3)


if __name__ == '__main__':
  absltest.main()
