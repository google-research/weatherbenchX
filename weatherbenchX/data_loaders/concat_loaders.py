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
    if not data_loaders:
      raise ValueError('data_loaders sequence must not be empty.')
    self._data_loaders = list(data_loaders)
    self._concat_dim = concat_dim
    merged_concat_kwargs = dict(DEFAULT_CONCAT_KWARGS)
    if concat_kwargs:
      merged_concat_kwargs.update(concat_kwargs)
    self._concat_kwargs = merged_concat_kwargs

    first = self._data_loaders[0]
    # pylint: disable=protected-access
    super().__init__(
        interpolation=kwargs.get('interpolation', first._interpolation),
        compute=kwargs.get('compute', first._compute),
        add_nan_mask=kwargs.get('add_nan_mask', first._add_nan_mask),
        process_chunk_fn=kwargs.get(
            'process_chunk_fn', first._process_chunk_fn
        ),
        add_values_to_coords=kwargs.get(
            'add_values_to_coords', first._add_values_to_coords
        ),
    )
    # pylint: enable=protected-access

  @property
  def data_loaders(self) -> list[base.DataLoader]:
    return self._data_loaders

  @property
  def concat_dim(self) -> str:
    return self._concat_dim

  def nominal_init_times(self, init_time_dim: str = 'init_time') -> np.ndarray:
    """Returns the nominal initialization times across child data loaders."""
    for loader in self._data_loaders:
      if not hasattr(loader, 'nominal_init_times'):
        raise ValueError(
            f'Child data loader {loader} does not implement'
            ' nominal_init_times().'
        )
    if self._concat_dim == init_time_dim:
      init_times = [
          loader.nominal_init_times(init_time_dim)  # pyrefly: ignore[missing-attribute]
          for loader in self._data_loaders
      ]
      return np.sort(np.unique(np.concatenate(init_times)))
    return self._data_loaders[0].nominal_init_times(init_time_dim)  # pyrefly: ignore[missing-attribute]

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
    self.maybe_prepare_dataset()
    if self._concat_dim == 'init_time':
      chunks = []
      for loader in self._data_loaders:
        if hasattr(loader, 'nominal_init_times'):
          loader_inits = loader.nominal_init_times('init_time')
          sub_init_times = init_times[np.isin(init_times, loader_inits)]
        else:
          sub_init_times = init_times
        if len(sub_init_times) > 0:
          chunks.append(
              loader._load_chunk_from_source(sub_init_times, lead_times)  # pylint: disable=protected-access
          )
      if not chunks:
        raise KeyError(
            f'None of the requested init_times {init_times} were found in'
            ' child data loaders.'
        )
      concatenated = xarray_tree.map_structure(
          lambda *x: xr.concat(  # pyrefly: ignore[no-matching-overload]
              list(x), dim=self._concat_dim, **self._concat_kwargs
          ).sel(init_time=init_times),
          *chunks,
      )
      return concatenated

    chunks = [
        loader._load_chunk_from_source(init_times, lead_times)  # pylint: disable=protected-access
        for loader in self._data_loaders
    ]
    return xarray_tree.map_structure(
        lambda *x: xr.concat(  # pyrefly: ignore[no-matching-overload]
            list(x), dim=self._concat_dim, **self._concat_kwargs
        ),
        *chunks,
    )
