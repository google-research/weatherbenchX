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

"""Modifier base class."""

import abc
from collections.abc import Sequence
import xarray as xr


class Modifier(abc.ABC):
  """Abstract class for modifiers (e.g. weightings, binnings)."""

  @abc.abstractmethod
  def weights(
      self,
      statistic: xr.DataArray,
  ) -> xr.DataArray:
    """Return weights for a given statistic.

    For now the implementation assumes that all information necessary to
    calculate the weights is contained in the statistic.

    Args:
      statistic: Individual DataArray with statistic values.

    Returns:
      weights: Weights whose dimensions should (with the exception of any
        added_dims) align with correspoding dimensions of the statistics.
        (Where they don't, aggregation will skip this statistic.)
    """

  @property
  def added_dims(self) -> Sequence[str]:
    """List of new dimensions added by this modifier.

    These should be dimensions present in the returned `weights` that are not
    present in the input `statistic`. Where weights contain added_dims, you
    can think of them as performing a separate weighting for each slice along
    the added_dims.

    It's used for two kinds of checks:
    * That multiple Modifiers don't try to add the same dimension.
    * That the `statistic` actually contains any non-added dimensions of the
      weights that are expected to line up with it.
    """
    return []
