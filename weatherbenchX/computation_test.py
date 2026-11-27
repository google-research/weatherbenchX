# Copyright 2026 Google LLC
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
"""Tests for metric and statistic computation helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchX import computation
import xarray as xr


class ComputationTest(parameterized.TestCase):

  def test_update_masks_noop_when_unmasked(self):
    predictions = {'a': xr.DataArray([1.0, 2.0], dims=('x',))}
    targets = {'a': xr.DataArray([3.0, 4.0], dims=('x',))}
    updated_preds, updated_targets = (
        computation.update_masks_for_targets_and_predictions(
            predictions, targets
        )
    )
    self.assertNotIn('mask', updated_preds['a'].coords)
    self.assertNotIn('mask', updated_targets['a'].coords)
    self.assertIs(updated_preds['a'], predictions['a'])
    self.assertIs(updated_targets['a'], targets['a'])

  @parameterized.named_parameters(
      dict(
          testcase_name='target_mask_only_preserves_unmasked_pred_nan',
          pred=xr.DataArray([np.nan, 2.0], dims=('x',)),
          target=xr.DataArray(
              [10.0, 20.0], dims=('x',), coords={'mask': ('x', [True, False])}
          ),
          allow_prediction_mask=False,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='pred_mask_only_preserves_unmasked_target_nan',
          pred=xr.DataArray(
              [1.0, 2.0], dims=('x',), coords={'mask': ('x', [True, False])}
          ),
          target=xr.DataArray([np.nan, 20.0], dims=('x',)),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='both_masked_intersects_with_and',
          pred=xr.DataArray(
              [1.0, 2.0, 3.0],
              dims=('x',),
              coords={'mask': ('x', [True, True, False])},
          ),
          target=xr.DataArray(
              [10.0, 20.0, 30.0],
              dims=('x',),
              coords={'mask': ('x', [True, False, True])},
          ),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray([True, False, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False, False], dims=('x',)),
      ),
      dict(
          testcase_name='pred_extra_dim_reuses_target_mask',
          pred=xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=('number', 'x')),
          target=xr.DataArray(
              [10.0, 20.0], dims=('x',), coords={'mask': ('x', [True, False])}
          ),
          allow_prediction_mask=False,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='pred_extra_dim_constant_both_masked',
          pred=xr.DataArray(
              [[1.0, 2.0], [3.0, 4.0]],
              dims=('number', 'x'),
              coords={
                  'mask': (('number', 'x'), [[True, False], [True, False]])
              },
          ),
          target=xr.DataArray(
              [10.0, 20.0], dims=('x',), coords={'mask': ('x', [True, True])}
          ),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='pred_extra_dim_constant_pred_only',
          pred=xr.DataArray(
              [[1.0, 2.0], [3.0, 4.0]],
              dims=('number', 'x'),
              coords={
                  'mask': (('number', 'x'), [[True, False], [True, False]])
              },
          ),
          target=xr.DataArray([10.0, 20.0], dims=('x',)),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='target_extra_unmasked_dim_allowed',
          pred=xr.DataArray([10.0, 20.0], dims=('x',)),
          target=xr.DataArray(
              [[1.0, 2.0], [3.0, 4.0]],
              dims=('time', 'x'),
              coords={'mask': ('x', [True, False])},
          ),
          allow_prediction_mask=False,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='pred_masked_target_extra_unmasked_dim_allowed',
          pred=xr.DataArray(
              [10.0, 20.0],
              dims=('x',),
              coords={'mask': ('x', [True, False])},
          ),
          target=xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=('time', 'x')),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray([True, False], dims=('x',)),
          expected_target_mask=xr.DataArray([True, False], dims=('x',)),
      ),
      dict(
          testcase_name='both_masked_different_mask_dims_broadcast',
          pred=xr.DataArray(
              [[10.0, 20.0], [30.0, 40.0]],
              dims=('time', 'x'),
              coords={'mask': ('x', [True, False])},
          ),
          target=xr.DataArray(
              [[1.0, 2.0], [3.0, 4.0]],
              dims=('time', 'x'),
              coords={'mask': (('time', 'x'), [[True, True], [False, True]])},
          ),
          allow_prediction_mask=True,
          expected_pred_mask=xr.DataArray(
              [[True, False], [False, False]], dims=('x', 'time')
          ),
          expected_target_mask=xr.DataArray(
              [[True, False], [False, False]], dims=('x', 'time')
          ),
      ),
  )
  def test_update_masks_aligns_masks(
      self,
      pred,
      target,
      allow_prediction_mask,
      expected_pred_mask,
      expected_target_mask,
  ):
    updated_preds, updated_targets = (
        computation.update_masks_for_targets_and_predictions(
            {'a': pred},
            {'a': target},
            allow_prediction_mask=allow_prediction_mask,
        )
    )
    xr.testing.assert_identical(
        updated_preds['a'].coords['mask'].variable,
        expected_pred_mask.variable,
    )
    xr.testing.assert_identical(
        updated_targets['a'].coords['mask'].variable,
        expected_target_mask.variable,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='pred_mask_only',
          target_coords={},
      ),
      dict(
          testcase_name='both_masked',
          target_coords={'mask': ('x', [True, True])},
      ),
  )
  def test_update_masks_raises_when_prediction_mask_varies_along_extra_dim(
      self, target_coords
  ):
    pred = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=('number', 'x'),
        coords={'mask': (('number', 'x'), [[True, False], [False, False]])},
        name='a',
    )
    target = xr.DataArray([10.0, 20.0], dims=('x',), coords=target_coords)
    with self.assertRaisesRegex(
        ValueError, 'constant along its extra dimensions'
    ):
      computation.update_masks_for_targets_and_predictions(
          {'a': pred}, {'a': target}, allow_prediction_mask=True
      )

  def test_update_masks_raises_when_target_mask_has_extra_dims(self):
    pred = xr.DataArray([10.0, 20.0], dims=('x',), name='a')
    target_masked = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=('time', 'x'),
        coords={'mask': (('time', 'x'), [[True, False], [False, False]])},
    )
    with self.assertRaisesRegex(ValueError, 'extra target mask dims'):
      computation.update_masks_for_targets_and_predictions(
          {'a': pred},
          {'a': target_masked},
          allow_prediction_mask=False,
      )

  def test_update_masks_raises_when_prediction_mask_disallowed(self):
    pred = xr.DataArray(
        [1.0, 2.0],
        dims=('x',),
        coords={'mask': ('x', [True, False])},
        name='a',
    )
    target = xr.DataArray([1.0, 2.0], dims=('x',))
    with self.assertRaisesRegex(ValueError, 'allow_prediction_mask=False'):
      computation.update_masks_for_targets_and_predictions(
          {'a': pred}, {'a': target}
      )


if __name__ == '__main__':
  absltest.main()
