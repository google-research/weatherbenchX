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
"""Concat data loader for combining multiple data loaders."""

from collections.abc import Hashable, Mapping, Sequence
from typing import Any, Optional, Union
import numpy as np
from weatherbenchX import xarray_tree
from weatherbenchX.data_loaders import base
import xarray as xr

DEFAULT_CONCAT_KWARGS = {'compat': 'override', 'coords': 'minimal'}


class ConcatDataLoader(base.DataLoader):
  """Data loader that wraps multiple data loaders and concatenates their outputs.

  This is useful e.g. when ensemble members are stored in separate Zarr files.
  Each underlying data loader loads its own chunk lazily, and the chunks are
  then concatenated along the specified dimension (e.g. 'sample').
  """

  def __init__(
      self,
      data_loaders: Sequence[base.DataLoader],
      concat_dim: str = 'sample',
      concat_kwargs: Optional[Mapping[str, Any]] = None,
      **kwargs,
  ):
    """Initializes ConcatDataLoader.

    Args:
      data_loaders: Sequence of DataLoader instances to concatenate.
      concat_dim: Dimension along which to concatenate the loaded chunks.
      concat_kwargs: Optional additional keyword arguments to pass to
        `xr.concat`. Defaults to `{'compat': 'override', 'coords': 'minimal'}`.
      **kwargs: Additional keyword arguments passed to base.DataLoader (e.g.
        interpolation, compute, process_chunk_fn).
    """
    super().__init__(**kwargs)
    if not data_loaders:
      raise ValueError('data_loaders sequence must not be empty.')
    self._data_loaders = list(data_loaders)
    self._concat_dim = concat_dim
    merged_concat_kwargs = dict(DEFAULT_CONCAT_KWARGS)
    if concat_kwargs:
      merged_concat_kwargs.update(concat_kwargs)
    self._concat_kwargs = merged_concat_kwargs

  @property
  def data_loaders(self) -> list[base.DataLoader]:
    return self._data_loaders

  @property
  def concat_dim(self) -> str:
    return self._concat_dim

  @property
  def _ds(self) -> Optional[xr.Dataset]:
    """Returns the dataset of the first child loader if available."""
    if self._data_loaders and hasattr(self._data_loaders[0], '_ds'):
      return self._data_loaders[0]._ds  # pyrefly: ignore[bad-return]
    return None

  @property
  def _variables(self) -> Optional[Sequence[str]]:
    """Returns the variables of the first child loader if available."""
    if self._data_loaders and hasattr(self._data_loaders[0], '_variables'):
      return self._data_loaders[0]._variables  # pyrefly: ignore[bad-return]
    return None

  def maybe_prepare_dataset(self):
    """Prepares datasets on child loaders that support it."""
    for loader in self._data_loaders:
      if hasattr(loader, 'maybe_prepare_dataset'):
        loader.maybe_prepare_dataset()

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    # Call each child's full load_chunk() so children apply their own
    # interpolation, process_chunk_fn, compute, and nan_mask.
    self.maybe_prepare_dataset()
    chunks = [
        loader.load_chunk(init_times, lead_times)
        for loader in self._data_loaders
    ]
    return xarray_tree.map_structure(
        lambda *x: xr.concat(  # pyrefly: ignore[no-matching-overload]
            list(x), dim=self._concat_dim, **self._concat_kwargs
        ),
        *chunks,
    )

  def load_chunk(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    # Call each child's full load_chunk() so children apply their own
    # interpolation, process_chunk_fn, compute, and nan_mask.
    # We bypass super().load_chunk() to avoid double-applying those steps.
    self.maybe_prepare_dataset()
    chunks = [
        loader.load_chunk(init_times, lead_times, reference=reference)
        for loader in self._data_loaders
    ]
    return xarray_tree.map_structure(
        lambda *x: xr.concat(  # pyrefly: ignore[no-matching-overload]
            list(x), dim=self._concat_dim, **self._concat_kwargs
        ),
        *chunks,
    )
