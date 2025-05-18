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

import math
from typing import Sequence, Tuple, TypeVar

import apache_beam as beam
from weatherbenchAfrica import aggregation


class SumAggregationStates(beam.transforms.CombineFn):
  """An object to sum all AggregationState."""

  def create_accumulator(self) -> aggregation.AggregationState:
    return aggregation.AggregationState.zero()

  def add_input(
      self,
      accumulator: aggregation.AggregationState,
      new_element: aggregation.AggregationState,
  ) -> aggregation.AggregationState:
    return accumulator + new_element

  def merge_accumulators(
      self, accumulators: Sequence[aggregation.AggregationState]
  ) -> aggregation.AggregationState:
    return sum(accumulators, aggregation.AggregationState.zero())

  def extract_output(
      self, accumulator: aggregation.AggregationState
  ) -> aggregation.AggregationState:
    return accumulator


Element = TypeVar("Element")
ElementWithKey = Tuple[int, Element]


class CombineMultiStage(beam.PTransform):
  """Performs a Combination in multiple stages.

  It requires the input to be a `ElementWithKey`, where the first
  element is an integer key in the [0, total_num_elements) interval.

  Then it performs the aggregation such that each worker at most merges
  `max_bin_size` object, by splitting the work into multiple bins. Once
  a stage is finished, the output for each bins becomes the input for the next
  stage.

  The keys at any given stage are simply the keys of the previous stage,
  modulo the number of bins.

  The combine operation returns a single `Element` object.

  (The reason for not using Beam's built-in aggregation is that, at the point we
  tested it was only possible to split the aggregation to two stages using the 
  `fanout` parameter. However, large datasets require more than two stages.)
  """

  def __init__(
      self,
      total_num_elements: int,
      max_bin_size: int,
      combine_fn: beam.transforms.CombineFn,
  ):
    """Inits the object.

    Args:
      total_num_elements: Number of elements to aggregate, incoming with integer
        keys in the [0, total_num_elements) interval. Neither the number of
        elements nor the keys need to be exact, as this is only used to estimate
        the number of stages and number of bins per stage.
      max_bin_size: Maximum number of elements that will be aggregated in each
        bin at any given stage.
      combine_fn: `beam.transforms.CombineFn` used tocombine data.
    """
    super().__init__()

    if max_bin_size < 2:
      raise ValueError("The maximum bin size must be at least 2.")

    # We will divide the aggregation into multiple stages, such that at any
    # stage, no accumulator has to accumulate more than `max_group_size`
    # elements.
    num_current_elements = total_num_elements
    num_bins_per_stage = []
    while num_current_elements > max_bin_size:
      num_bins = math.ceil(num_current_elements / max_bin_size)
      num_bins_per_stage.append(num_bins)
      num_current_elements = num_bins
    num_bins_per_stage.append(1)
    self._num_bins_per_stage = num_bins_per_stage
    self._combine_fn = combine_fn

  def _aggregation_stage(
      self, pcoll: beam.pvalue.PCollection, num_bins: int
  ) -> beam.pvalue.PCollection:
    # Add a key according to the bin size for this stage.
    def _bin_key(inputs: ElementWithKey) -> ElementWithKey:
      input_key, element = inputs
      output_key = input_key % num_bins
      return output_key, element

    return (
        pcoll
        | f"AddKeyForBins{num_bins}" >> beam.Map(_bin_key)
        | f"SumForBins{num_bins}" >> beam.CombinePerKey(self._combine_fn)
    )

  def expand(self, pcoll: beam.pvalue.PCollection) -> beam.pvalue.PCollection:
    for num_bins in self._num_bins_per_stage:
      pcoll = self._aggregation_stage(pcoll, num_bins)

    def remove_key(inputs: ElementWithKey) -> Element:
      key, element = inputs
      assert key == 0  # All keys should be the same at this point.
      return element

    return (
        pcoll
        # Using beam.Values() seems to fail, because it does not do type
        # inference correctly, and uses the wrong encoder for the next stage.
        | "RemoveRedundantKey" >> beam.Map(remove_key)
    )
