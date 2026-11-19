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
"""Testing base class functionality and related helpers."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from weatherbenchX.metrics import base
from weatherbenchX.metrics import metrics_test_utils
import xarray as xr


class BaseTest(parameterized.TestCase):

  def test_per_variable_statistics_operating_on_subsets_of_variables(self):
    class Stat1(base.PerVariableStatistic):
      def _compute_per_variable(self, predictions, targets):
        return None if targets.name == 'a' else predictions

    class Stat2(base.PerVariableStatistic):
      def _compute_per_variable(self, predictions, targets):
        return None if targets.name == 'b' else targets

    class Metric(base.PerVariableMetric):
      statistics = {'stat1': Stat1(), 'stat2': Stat2()}

      def _values_from_mean_statistics_per_variable(self, stats):
        return stats['stat1'] + stats['stat2']

    predictions = xr.Dataset(dict(a=1, b=2, c=3))
    targets = xr.Dataset(dict(a=10, b=20, c=30))

    result = metrics_test_utils.compute_all_metrics(
        {'metric': Metric(),
         'stat1': Stat1(),
         'stat2': Stat2()},
        predictions, targets,
        reduce_dims=[],
    )

    # Stat1 defined for b and c only:
    self.assertNotIn('stat1.a', result)
    self.assertIn('stat1.b', result)
    self.assertIn('stat1.c', result)

    # Stat2 defined for a and c only:
    self.assertIn('stat2.a', result)
    self.assertNotIn('stat2.b', result)
    self.assertIn('stat2.c', result)

    # Metric based on both of them only defined for the intersection, c:
    self.assertNotIn('metric.a', result)
    self.assertNotIn('metric.b', result)
    self.assertIn('metric.c', result)

  def test_per_variable_statistic_prediction_target_different_vars(self):
    class Stat(base.PerVariableStatistic):
      def _compute_per_variable(self, predictions, targets):
        return abs(predictions - targets)

    predictions = xr.Dataset(dict(a=1, b=2))
    targets = xr.Dataset(dict(b=20, c=30))
    result = Stat().compute(predictions, targets)

    # Stat only computed for variables present in both predictions and targets:
    self.assertNotIn('a', result)
    self.assertIn('b', result)
    self.assertNotIn('c', result)

  def test_per_variable_statistic_result_not_forced_to_dataset(self):
    class Stat(base.PerVariableStatistic):
      def _compute_per_variable(self, predictions, targets):
        return abs(predictions - targets)

    # Even though predictions is a Dataset, we don't force the result to be a
    # Dataset because this can introduce problems if targets (and hence the
    # resulting statistic values) contain incompatible coordinates (e.g. mask)
    # per variable.
    predictions = xr.Dataset(dict(a=1, b=2))
    targets = dict(b=xr.DataArray(20), c=xr.DataArray(30))
    result = Stat().compute(predictions, targets)
    self.assertIsInstance(result, dict)

  def test_update_masks_noop_when_unmasked(self):
    predictions = {'a': xr.DataArray([1.0, 2.0], dims=('x',))}
    targets = {'a': xr.DataArray([3.0, 4.0], dims=('x',))}
    updated_preds, updated_targets = (
        base.update_masks_for_targets_and_predictions(predictions, targets)
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
          expected_pred_mask=xr.DataArray(
              [[True, False], [True, False]], dims=('number', 'x')
          ),
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
          expected_pred_mask=xr.DataArray(
              [[True, False], [True, False]], dims=('number', 'x')
          ),
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
        base.update_masks_for_targets_and_predictions(
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
      base.update_masks_for_targets_and_predictions(
          {'a': pred}, {'a': target}, allow_prediction_mask=True
      )

  @parameterized.named_parameters(
      dict(
          testcase_name='unmasked_prediction',
          pred=xr.DataArray([10.0, 20.0], dims=('x',), name='a'),
          allow_prediction_mask=False,
      ),
      dict(
          testcase_name='masked_prediction',
          pred=xr.DataArray(
              [[10.0, 20.0], [30.0, 40.0]],
              dims=('time', 'x'),
              coords={'mask': ('x', [True, True])},
              name='a',
          ),
          allow_prediction_mask=True,
      ),
  )
  def test_update_masks_raises_when_target_mask_has_extra_dims(
      self, pred, allow_prediction_mask
  ):
    target_masked = xr.DataArray(
        [[1.0, 2.0], [3.0, 4.0]],
        dims=('time', 'x'),
        coords={'mask': (('time', 'x'), [[True, False], [False, False]])},
    )
    with self.assertRaisesRegex(ValueError, 'extra target dims'):
      base.update_masks_for_targets_and_predictions(
          {'a': pred},
          {'a': target_masked},
          allow_prediction_mask=allow_prediction_mask,
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
      base.update_masks_for_targets_and_predictions({'a': pred}, {'a': target})


if __name__ == '__main__':
  absltest.main()
