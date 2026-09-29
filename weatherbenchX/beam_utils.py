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
r"""Beam-specific utils for beam pipelines."""

from collections.abc import Iterable, Iterator
import contextlib
import errno
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
    return aggregation.combining_sum(accumulators) if accumulators else None  # pyrefly: ignore[bad-argument-type]

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


def _is_already_exists_err(e: BaseException) -> bool:
  """Returns True if the exception indicates the destination already exists."""
  if not isinstance(e, Exception):
    # For example, KeyboardInterrupt or SystemExit.
    return False
  if isinstance(e, FileExistsError):
    return True
  if isinstance(e, OSError) and e.errno in (errno.EEXIST, errno.ENOTEMPTY):
    return True
  err_str = str(e).lower()
  return 'already exists' in err_str or 'already_exists' in err_str


@contextlib.contextmanager
def atomic_write_dir(
    dir_path: str,
    auto_mkdir: bool = True,
) -> Iterator[str]:
  """Yield a temporary directory to caller and atomically move it to dir_path.

  Avoids write races in distributed pipelines when redundant backup workers
  attempt to write to the same output directory via the following mechanism:
  1. Yields the temporary directory path/URL to the caller (e.g. ds.to_zarr),
  cleaning up the temporary directory if the caller fails to write anything to
  it.
  2. If the caller succeeds to write to the temporary directory, tries to
  atomically move it to the final target directory. If this fails, re-raises
  the exception, except if the target directory already exists, in which case
  it assumes another worker has already completed the job and cleans up the
  temporary directory.

  Args:
    dir_path: The final target directory path.
    auto_mkdir: Whether to create parent directories if they don't exist.

  Yields:
    A temporary directory path/URL to write into.
  """
  filesystem, target_fs_path = fsspec.core.url_to_fs(dir_path)

  # Maintain protocol prefix on the yielded temporary URL.
  parent_url, name = os.path.split(dir_path.rstrip('/'))
  tmp_name = f'tmp.{uuid.uuid4().hex}.{name}'
  tmp_url = os.path.join(parent_url, tmp_name)
  _, tmp_fs_path = fsspec.core.url_to_fs(tmp_url)

  parent_fs_dir, _ = os.path.split(target_fs_path.rstrip('/'))
  if auto_mkdir and parent_fs_dir:
    filesystem.makedirs(parent_fs_dir, exist_ok=True)
  if auto_mkdir:
    filesystem.makedirs(tmp_fs_path, exist_ok=True)

  try:
    yield tmp_url
  # If any exception occurred in the caller (e.g. ds.to_zarr) in the
  # attempt to write to the temporary directory, attempt to clean up the
  # temporary directory.
  except BaseException:
    # Don't crash the cleanup if the temporary directory doesn't exist because
    # the caller failed to write anything to it, so that we re-raise the
    # original exception.
    with contextlib.suppress(OSError):
      filesystem.rm(tmp_fs_path, recursive=True)
    raise
  # If no exception occurred in the caller, we attempt to move the temporary
  # directory to the final target directory.
  else:
    try:
      filesystem.mv(tmp_fs_path, target_fs_path, recursive=True)
    except BaseException as e:
      # Always clean up our temporary directory regardless of why mv failed.
      with contextlib.suppress(OSError):
        filesystem.rm(tmp_fs_path, recursive=True)
      # Only swallow the error if it is explicitly an "already exists" error
      # AND the target directory exists. Otherwise (e.g. running out of disk
      # space mid-move, permissions error, or KeyboardInterrupt), re-raise.
      target_already_exists = _is_already_exists_err(e) and filesystem.exists(
          target_fs_path
      )
      if not target_already_exists:
        raise
