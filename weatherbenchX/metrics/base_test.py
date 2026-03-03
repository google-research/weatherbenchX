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
from weatherbenchX.metrics import base
from weatherbenchX.metrics import metrics_test_utils
import xarray as xr


class BaseTest(absltest.TestCase):

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


if __name__ == '__main__':
  absltest.main()
