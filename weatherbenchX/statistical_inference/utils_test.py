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
from weatherbenchX.statistical_inference import utils
import xarray as xr


class UtilsTest(absltest.TestCase):

  def test_apply_to_slices(self):
    arg1 = xr.DataArray(np.ones((2, 3)), dims=['x', 'y'],
                        coords={'x': np.arange(2)})
    arg2 = xr.DataArray(np.ones((3,)), dims=['y'])

    def func(arg1_slice, arg2_slice):
      # We're only being called with a slice of length 1 of the x dimension
      # (specified as dim='x' below):
      self.assertEqual(arg1_slice.sizes, {'x': 1, 'y': 3})
      # arg2 doesn't have an 'x' dimension, that's fine, we don't slice it.
      self.assertEqual(arg2_slice.sizes, {'y': 3})
      return (
          # We rename some dims, to check that no assumptions are made about
          # output dims matching input dims.
          (arg1_slice + arg2_slice).rename(x='z'),
          (arg1_slice - arg2_slice).rename(y='z')
      )

    result1, result2 = utils.apply_to_slices(
        func, arg1, arg2, dim='x')
    # The result (for a broadcasting function like the above anyway) is the
    # same as applying the func to the entire array. More realistic use would
    # be cases where you need to compute the result slice-by-slice, e.g.
    # because doing it all at once uses too much memory, or because it's not
    # easily vectorized.
    xr.testing.assert_equal(result1, (arg1 + arg2).rename(x='z'))
    xr.testing.assert_equal(result2, (arg1 - arg2).rename(y='z'))


if __name__ == '__main__':
  absltest.main()
