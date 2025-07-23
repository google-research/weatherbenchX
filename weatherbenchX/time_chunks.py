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
"""Time generator that defines chunks for evaluation.

Chunks can be defined in init_time and lead_time dimensions.
"""

from collections.abc import Iterable, Iterator
import dataclasses
import itertools
from typing import Optional, Union
import numpy as np


# Tuple of (init_times, lead_times).
TimeChunk = tuple[np.ndarray, Union[np.ndarray, slice]]


@dataclasses.dataclass(frozen=True)
class TimeChunkOffsets:
  init_time: int
  lead_time: int


class TimeChunks(Iterable[TimeChunk]):
  """Iterable defining chunks in init and lead time."""

  def __init__(
      self,
      init_times: np.ndarray,
      lead_times: Union[
          np.ndarray,
          slice,
      ],
      init_time_chunk_size: Optional[int] = None,
      lead_time_chunk_size: Optional[int] = None,
  ):
    """Init.

    Args:
      init_times: Numpy array of init_times (dtype: np.datetime64).
      lead_times: To specify exact lead times, array of np.timedelta64. For a
        lead_time interval, specify slice of np.timedelta64's. End point is
        inclusive following pandas/xarray conventions. start and stop are
        mandatory for slice. step parameter is not used.
      init_time_chunk_size: Chunk size in init_time dimension. None or zero
        specifies a single chunk (default).
      lead_time_chunk_size: Chunk size in lead_time dimension. None or zero
        specifies a single chunk (default). Must be None in the case of a single
        lead_time slice.

    Iterator returns tuples of (init_times, lead_times) chunks. The chunks
    are products of the individual init_times and lead_times chunks. See example
    below.

    init_time is an array of np.datetime64's. For exact lead times, lead_time is
    an array of np.timedelta64's. For lead time intervals, lead_time is a slice
    indicating start and stop as np.timedelta64's.

    Example 1: Exact lead times
        >>> from weatherbenchX import time_chunks
        >>> init_times = np.arange(
        >>>     '2020-01-01T00',
        >>>     '2020-01-02T00',
        >>>     np.timedelta64(6, 'h'),
        >>>     dtype="datetime64"
        >>>     )
        >>> lead_times = np.arange(0, 18, 6, dtype='timedelta64[h]')
        >>> times = time_chunks.TimeChunks(
        >>>     init_times,
        >>>     lead_times,
        >>>     init_time_chunk_size=2,
        >>>     lead_time_chunk_size=2
        >>>     )
        >>> list(times)
        [(array(['2020-01-01T00', '2020-01-01T06'], dtype='datetime64[h]'),
        array([0, 6], dtype='timedelta64[h]')),
        (array(['2020-01-01T00', '2020-01-01T06'], dtype='datetime64[h]'),
        array([12], dtype='timedelta64[h]')),
        (array(['2020-01-01T12', '2020-01-01T18'], dtype='datetime64[h]'),
        array([0, 6], dtype='timedelta64[h]')),
        (array(['2020-01-01T12', '2020-01-01T18'], dtype='datetime64[h]'),
        array([12], dtype='timedelta64[h]'))]

    Example 2: Lead time interval
        >>> lead_times = slice(np.timedelta64(0), np.timedelta64(6, 'h'))
        >>> times = time_chunks.TimeChunks(
        >>>     init_times,
        >>>     lead_times,
        >>>     init_time_chunk_size=2,
        >>>     lead_time_chunk_size=None   # Must be None for slice
        >>>     )
        >>> list(times)
        [(array(['2020-01-01T00', '2020-01-01T06'], dtype='datetime64[h]'),
          slice(numpy.timedelta64(0), numpy.timedelta64(6,'h'), None)),
        (array(['2020-01-01T12', '2020-01-01T18'], dtype='datetime64[h]'),
          slice(numpy.timedelta64(0), numpy.timedelta64(6,'h'), None))]
    """

    # -1 is used in xarray_beam, but results in a silent failure here.
    if init_time_chunk_size is not None and init_time_chunk_size < 0:
      raise ValueError(
          f'{init_time_chunk_size=} but should be non-negative or None'
      )
    if lead_time_chunk_size is not None and lead_time_chunk_size < 0:
      raise ValueError(
          f'{lead_time_chunk_size=} but should be non-negative or None'
      )

    init_times = init_times.astype('datetime64[ns]')

    # If chunk size is None, return all elements in a single chunk.
    if not init_time_chunk_size:
      init_time_chunk_size = len(init_times)
    # Split init_times into chunks
    self._init_time_chunks = [
        init_times[i : i + init_time_chunk_size]
        for i in range(0, len(init_times), init_time_chunk_size)
    ]

    if isinstance(lead_times, slice):
      # Enforce slice start and stop to be specified and step be None.
      if lead_times.start is None or lead_times.stop is None:
        raise ValueError('Slice start and stop must be specified.')
      if lead_times.step is not None:
        raise ValueError('Slice step must be None.')

      if lead_time_chunk_size:
        raise ValueError('Chunking in lead time not compatible for slice.')
      self._lead_time_chunks = [lead_times]
    elif isinstance(lead_times, np.ndarray):
      lead_times = lead_times.astype('timedelta64[ns]')
      if not lead_time_chunk_size:
        lead_time_chunk_size = len(lead_times)
      # Split lead_times into chunks
      self._lead_time_chunks = [
          lead_times[i : i + lead_time_chunk_size]
          for i in range(0, len(lead_times), lead_time_chunk_size)
      ]
    else:
      raise ValueError('Lead times must be either np.ndarray or slice.')

    self._init_times = init_times
    self._lead_times = lead_times
    self._init_time_chunk_size = init_time_chunk_size
    self._lead_time_chunk_size = lead_time_chunk_size
    self._num_init_chunks = len(self._init_time_chunks)
    self._num_lead_chunks = len(self._lead_time_chunks)

  @property
  def init_times(self) -> np.ndarray:
    return self._init_times

  @property
  def lead_times(self) -> Union[np.ndarray, slice]:
    return self._lead_times

  @property
  def init_time_chunk_size(self) -> int:
    return self._init_time_chunk_size

  @property
  def lead_time_chunk_size(self) -> int:
    return self._lead_time_chunk_size  # pytype: disable=bad-return-type

  def __iter__(self) -> Iterator[TimeChunk]:
    return itertools.product(self._init_time_chunks, self._lead_time_chunks)

  def __len__(self) -> int:
    return self._num_init_chunks * self._num_lead_chunks

  def __getitem__(self, index: int) -> TimeChunk:
    if index < 0 or index >= len(self):
      raise IndexError(f'TimeChunks index out of range: {index}')
    init_chunk = self._init_time_chunks[index // self._num_lead_chunks]
    lead_chunk = self._lead_time_chunks[index % self._num_lead_chunks]
    return init_chunk, lead_chunk

  def iter_with_chunk_offsets(self) -> Iterator[
      tuple[TimeChunkOffsets, TimeChunk]]:
    """Yields time chunks with keys describing the offsets of the chunk.

    Yields:
      (offsets, (init_chunk, lead_chunk)) where offsets refers to offsets of
      the chunk within the full arrays of init_time and lead_time values.
    """
    for index, (init_chunk, lead_chunk) in enumerate(self):
      init_index = self._init_time_chunk_size * (index // self._num_lead_chunks)
      lead_index = self._lead_time_chunk_size * (index % self._num_lead_chunks)
      offsets = TimeChunkOffsets(init_time=init_index, lead_time=lead_index)
      yield offsets, (init_chunk, lead_chunk)
