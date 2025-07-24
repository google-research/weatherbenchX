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
r"""Beam-specific utils for beam pipelines."""

import apache_beam as beam


class Sum(beam.transforms.CombineFn):
  """CombineFn which always sums one element at a time.

  This is logically equivalent to passing `sum` to e.g. `beam.CombinePerKey`,
  but avoids bringing too many elements into memory before summing.
  (Beam uses a buffer size of 10 when wrapping a callable like `sum` as a
  CombineFn).

  It also assumes '0' is OK to use as the additive identity for the sum.
  """

  def create_accumulator(self):
    return 0

  def add_input(self, accumulator, element):
    return accumulator + element

  def merge_accumulators(self, accumulators):
    return sum(accumulators)

  def extract_output(self, accumulator):
    return accumulator


class GroupAll(beam.PTransform):
  """Groups all elements into a single group."""

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        | 'AddDummyKey' >> beam.Map(lambda x: (None, x))
        | 'GroupByDummyKey' >> beam.GroupByKey()
        | 'DropDummyKey' >> beam.Values())
