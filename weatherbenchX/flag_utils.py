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
# ==============================================================================
"""WeatherBenchX utilities for working with command line flags."""
import re
from typing import Any

from absl import flags


def _chunks_string_is_valid(chunks_string: str) -> bool:
  return re.fullmatch(r'(\w+=-?\d+(,\w+=-?\d+)*)?', chunks_string) is not None


def _parse_chunks(chunks_string: str) -> dict[str, int]:
  """Parse a chunks string into a dict."""
  chunks = {}
  if chunks_string:
    for entry in chunks_string.split(','):
      key, value = entry.split('=')
      chunks[key] = int(value)
  return chunks


class _ChunksParser(flags.ArgumentParser):
  """Parser for Xarray-Beam chunks flags."""

  syntactic_help: str = (
      'comma separate list of dim=size pairs, e.g., "time=10,longitude=100"'
  )

  def parse(self, argument: str) -> dict[str, int]:
    if not _chunks_string_is_valid(argument):
      raise ValueError(f'invalid chunks string: {argument}')
    return _parse_chunks(argument)

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'dict[str, int]'


class _DimValuePairSerializer(flags.ArgumentSerializer):
  """Serializer for dim=value pairs."""

  def serialize(self, value: dict[str, int]) -> str:
    return ','.join(f'{k}={v}' for k, v in value.items())


def DEFINE_chunks(  # pylint: disable=invalid-name
    name: str,
    default: str,
    help: str,  # pylint: disable=redefined-builtin
    **kwargs: Any,
):
  """Define a flag for defining Xarray-Beam chunks."""
  parser = _ChunksParser()
  serializer = _DimValuePairSerializer()
  return flags.DEFINE(
      parser, name, default, help, serializer=serializer, **kwargs
  )


# Key/value pairs of the form dimension=integer have the same requirements as
# chunks.
DEFINE_dim_integer_pairs = DEFINE_chunks


class _StringKeyValueParser(flags.ArgumentParser):
  """Parser for key=value pairs, both interpreted as strings."""

  syntactic_help: str = (
      'comma separate list of key=svalue pairs, both interpreted as strings.'
  )

  def parse(self, argument: str) -> dict[str, str]:
    pairs = {}
    if not argument:
      return pairs
    for entry in argument.split(','):
      key, value = entry.split('=')
      pairs[key] = value
    return pairs

  def flag_type(self) -> str:
    """Returns a string representing the type of the flag."""
    return 'dict[str, str]'


def DEFINE_string_key_value_pairs(  # pylint: disable=invalid-name
    name: str,
    default: str,
    help: str,  # pylint: disable=redefined-builtin
    **kwargs: Any,
):
  """Define a flag for parsing key=value pairs, both interpreted as strings."""
  parser = _StringKeyValueParser()
  serializer = _DimValuePairSerializer()
  return flags.DEFINE(
      parser, name, default, help, serializer=serializer, **kwargs
  )
