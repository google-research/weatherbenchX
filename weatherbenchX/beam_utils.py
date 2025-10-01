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

from collections.abc import Iterable
import contextlib
import os
import uuid

import apache_beam as beam
import fsspec
from weatherbenchX import aggregation
import xarray as xr


_Accumulator = xr.DataArray | None


class CombiningSum(beam.transforms.CombineFn):
  """CombineFn for DataArrays, wrapping aggregation.combining_sum."""

  def create_accumulator(self) -> _Accumulator:
    return None

  def add_input(
      self, accumulator: _Accumulator, element: xr.DataArray
  ) -> _Accumulator:
    if accumulator is None:
      return element
    else:
      return aggregation.combining_sum([accumulator, element])

  def merge_accumulators(
      self, accumulators: Iterable[_Accumulator]) -> _Accumulator:
    accumulators = [a for a in accumulators if a is not None]
    return aggregation.combining_sum(accumulators) if accumulators else None

  def extract_output(self, accumulator: _Accumulator) -> _Accumulator:
    return accumulator


class GroupAll(beam.PTransform):
  """Groups all elements into a single group."""

  def expand(self, pcoll: beam.PCollection) -> beam.PCollection:
    return (
        pcoll
        | 'AddDummyKey' >> beam.Map(lambda x: (None, x))
        | 'GroupByDummyKey' >> beam.GroupByKey()
        | 'DropDummyKey' >> beam.Values())


def atomic_write(
    file_path: str,
    data: bytes,
    auto_mkdir: bool = True,
) -> None:
  """Writes bytes to an fsspec path, atomically for supporting filesystems.

  This is important to avoid write races when multiple beam workers attempt to
  write to the same file, which can happen e.g. due to a beam runner scheduling
  redundant backup attempts for slow workers at the final stage.

  This assumes that the fsspec.mv move operation is atomic for the filesystem
  in use, which is not necessarily the case for all filesystems, but is about
  the best we can do using a general API like fsspec.

  Args:
    file_path: The path to write to.
    data: The data to write.
    auto_mkdir: Whether to create directories if they don't exist.
  """
  filesystem, file_path = fsspec.core.url_to_fs(file_path)

  dir_path, name = os.path.split(file_path)

  if auto_mkdir:
    filesystem.makedirs(dir_path, exist_ok=True)
  tmp_name = f'tmp.{uuid.uuid1()}.{name}'
  tmp_file_path = os.path.join(dir_path, tmp_name)

  try:
    with filesystem.open(tmp_file_path, mode='wb') as f:
      f.write(data)
  except BaseException:
    with contextlib.suppress(FileNotFoundError):
      filesystem.rm(tmp_file_path)
    raise
  else:
    filesystem.mv(tmp_file_path, file_path, overwrite=True)
