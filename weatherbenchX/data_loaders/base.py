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
"""Base data loader class."""

import abc
from collections.abc import Hashable, Mapping
from typing import Collection, Callable, Optional, Union
import numpy as np
from weatherbenchX import interpolations
from weatherbenchX import xarray_tree
import xarray as xr


def add_nan_mask_to_data(
    data: Mapping[Hashable, xr.DataArray],
    variable_subset: Collection[str] | None = None,
) -> Mapping[Hashable, xr.DataArray]:
  """Adds a boolean coordinate named 'mask' to each variable with False indicating NaN values.

  When applied to targets, these masks should propagate to statistics and will
  be used by Aggregator (when masked=True) to skip evaluation units
  corresponding to NaN targets during its aggregation of statistics.

  We strongly recommend to do it this way rather than using skipna=True in the
  aggregator, because we want unexpected NaNs in the statistics (e.g. arising
  unexpected NaNs in the predictions or targets, or buggy metrics code) to
  propagate loudly and cause an error, rather than silently causing some of the
  evaluation units to be skipped, delivering biased evaluation results which
  hide a bug.

  Args:
    data: Data to add the mask to.
    variable_subset: If provided, only add masks to the variables in this list.
      Masks use memory and some compute, so we encourage limiting them to
      variables which are actually expected to contain masking information in
      the form of NaNs.

  Returns:
    The data with any masks added.
  """
  data = dict(data)
  for var in data:
    if variable_subset is None or var in variable_subset:
      data[var].coords['mask'] = ~np.isnan(data[var])
  return data


class DataLoader(abc.ABC):
  """Base class for data loaders.

  Data loaders return chunks of data compatible with the rest of the evaluation
  framework. Specifically, this should be an xr.Dataset or a dictionary of
  xr.DataArray's. It is the data loaders' job to return target and prediction
  chunks that can be broadcast against each other. If interpolation is required
  to map one dataset to another, e.g. interpolating a gridded dataset to sparse
  points, a reference dataset can be provided for this purpose.
  """

  def __init__(
      self,
      interpolation: Optional[interpolations.Interpolation] = None,
      compute: bool = True,
      add_nan_mask: bool = False,
      process_chunk_fn: Optional[Callable[[xr.Dataset], xr.Dataset]] = None,
  ):
    """Shared initialization for data loaders.

    Args:
      interpolation: (Optional) Interpolation to be applied to the data.
      compute: Load chunk into memory. Default: True.
      add_nan_mask: Adds a boolean coordinate named 'mask' to each variable
        (variables will be split into DataArrays if they aren't already), with
        False indicating NaN values. To be used for masked aggregation. Default:
        False.
      process_chunk_fn: optional function to be applied to each chunk after
        loading, interpolation and compute, but before computing a mask.
    """
    self._interpolation = interpolation
    self._compute = compute
    self._add_nan_mask = add_nan_mask
    self._process_chunk_fn = process_chunk_fn

  @abc.abstractmethod
  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    """Method to be implemented by data loaders."""

  def load_chunk(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    """Shared API for loading data chunk.

    Implements interpolation, splitting variables and loading data into memory.

    Args:
      init_times: List of init_times.
      lead_times: (Optional) List of exact lead times or lead_time interval. The
        exact behavior for each option depends on the data loader and should be
        documented there.
      reference: (Optional) A reference dataset to be used inside the data
        loader, e.g. for interpolation.

    Returns:
      data_chunk: Xarray Dataset or dictionary of DataArrays containing data for
      given times.
    """
    chunk = self._load_chunk_from_source(init_times, lead_times)

    if self._interpolation is not None:
      # TODO(srasp): Potentially implement consistency check between lead_times
      # and lead_time coordinate on reference.
      chunk = self._interpolation.interpolate(chunk, reference)

    # A temporary workaround for https://github.com/pydata/xarray/issues/10325.
    def _compute_and_keep_dtype(x: xr.DataArray) -> xr.DataArray:
      dtype = x.dtype
      return x.compute().astype(dtype)

    # Compute after interpolation avoids loading unnecessary data.
    if self._compute:
      chunk = xarray_tree.map_structure(_compute_and_keep_dtype, chunk)

    # TODO: https://github.com/google-research/weatherbenchX/issues/67 - add
    # full functionality for computing derived variables, which would complement
    # adhoc chunk processing with process_chunk_fn.
    if self._process_chunk_fn is not None:
      chunk = self._process_chunk_fn(chunk)

    if self._add_nan_mask:
      chunk = add_nan_mask_to_data(chunk)
    return chunk
