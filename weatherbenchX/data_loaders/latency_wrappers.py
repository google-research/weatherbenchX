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
"""Latency wrappers."""

from typing import Hashable, Mapping, Optional, Union
import numpy as np
from weatherbenchX import xarray_tree
from weatherbenchX.data_loaders import base
from weatherbenchX.data_loaders import xarray_loaders
import xarray as xr
from absl import logging


class ConstantLatencyWrapper(base.DataLoader):
  """Wraps a data loader to adjust init/lead_times based on a constant latency.

  Terminology used here:

  - Nominal init time: Initialization time of raw file(s), or in other words,
    what the underlying model considers t=0.
  - Nominal lead time: Lead time of raw file(s).
  - Latency: Delay from the nominal init time to the time when the forecast
    would be available in an operational setting.
  - Issue time: Init time at which the forecast is actually available = nominal
    init time + latency.
  - Queried init/lead time: Actual time/lead time a forecast is requested for
    in an operational setting.

  This works by picking the most recently available nominal init time (i.e. init
  time on file) for the requested init_time given a constant latency.

  Lead and init times are adjusted to reflect the latency and
  _load_chunk_from_source is called with these adjusted values for the given
  data loader. The returned values are then assigned the requested init/lead
  times.

  Because this has to be done for each init time separately, the results are
  concatenated. For non-sparse data (i.e. where init_time is data dimension),
  the concatenation is done along the init_time dimension. For sparse data,
  where init_time is simply a coordinate, the concatenation is done along the
  index dimension. This has to be passed explicitly as an argument.

  Examples:
    1. For a latency of 6h with nominal init_times at 00/12UTC, querying an init
    time of 12UTC and a lead time of 6h, will internally load the 00UTC init
    time and a lead_time of 12h. The returned data will still have an init_time
    of 12UTC and a lead_time of 6h.

    2. For a forecast initialized at 00/12UTC with a 5 hour latency (meaning
    issue times of 05/17UTC), querying an init time of 16UTC and a lead time of
    1h will internally load the 00UTC nominal init time and a lead time of 16h.
  """

  def __init__(
      self,
      data_loader: base.DataLoader,
      latency: np.timedelta64,
      nominal_init_times: np.ndarray,
      concat_dim: str = 'init_time',
  ):
    """Init.

    Args:
      data_loader: The data loader to wrap.
      latency: Constant latency as np.timedelta64 object.
      nominal_init_times: A numpy array containing the nominal init times of the
        predictions for the entire dataset of predictions as numpy datetime64.
        Example array: np.array(['2020-01-01T00', '2020-01-01T12', ...])
      concat_dim: The dimension to concatenate along. Default: 'init_time'. Set
        to 'index' for sparse data.
    """
    self.data_loader = data_loader
    self.latency = latency
    self.nominal_init_times = nominal_init_times
    self._concat_dim = concat_dim
    super().__init__(
        interpolation=data_loader._interpolation,
        compute=data_loader._compute,
        add_nan_mask=data_loader._add_nan_mask,
    )

  def get_available_init_time(self, init_time: np.datetime64) -> np.datetime64:
    """Return most recent available nominal init time for requested init time."""
    issue_time = self.nominal_init_times + self.latency
    diff = (issue_time - init_time).astype(int)
    # Find index of issue time that is closest to requested init_time.
    # on the left, i.e. with issue_time > nominal init_time.
    available_idx = np.nanargmax(np.where(diff <= 0, diff, np.nan))
    available_init_time = self.nominal_init_times[available_idx]
    return available_init_time

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    """Modified loading method that adjusts init/lead times.

    Args:
      init_times: Queried init times.
      lead_times: Queried lead times.

    Returns:
      chunk: Chunk loaded from adjusted nominal init/lead times but with the
        requested init/lead times assigned as coordinates.
    """
    if lead_times is None:
      raise ValueError('Latency adjustement is only valid with lead times.')

    chunk = []
    # Has to be done for each init time separately because the offset can differ
    # depending on the init time.
    for init_time in init_times:
      available_init_time = self.get_available_init_time(init_time)
      # lead_time_offset is positive for a positive latency.
      lead_time_offset = init_time - available_init_time

      adjusted_lead_times = lead_times + lead_time_offset
      logging.info(
          'LatencyWrapper: loading chunk for init time %s, using available init'
          ' time %s, adjusted lead times %s',
          init_time,
          available_init_time,
          adjusted_lead_times.astype('timedelta64[m]'),
      )
      raw_chunk = self.data_loader._load_chunk_from_source(  # pystyle: disable=protected-access
          np.array([available_init_time]), adjusted_lead_times
      )

      # Adjust nominal init and lead times to the query values.
      # Doing this by adding/subtracting the offset also works for sparse data
      # where init/lead_time is a coordinate but not a data dimension.
      def adjust_init_and_lead_times(x):
        x = x.assign_coords(init_time=x.init_time + lead_time_offset)  # pylint: disable=cell-var-from-loop
        x = x.assign_coords(lead_time=x.lead_time - lead_time_offset)  # pylint: disable=cell-var-from-loop
        return x

      # TODO(srasp): Potentially assert that raw_chunk coordinates match
      # adjusted times before and query times after. However, since for sparse
      # data init/lead times are coordinates with dimension index (with
      # possibly missing init/lead times), this isn't trivial.
      raw_chunk = xarray_tree.map_structure(  # pytype: disable=wrong-arg-types
          adjust_init_and_lead_times,
          raw_chunk,
      )
      chunk.append(raw_chunk)

    # Concatenate init_times. For sparse data, the index dimension should be
    # used.
    chunk = xarray_tree.map_structure(
        lambda *x: xr.concat([*x], dim=self._concat_dim), *chunk
    )
    return chunk


class XarrayConstantLatencyWrapper(ConstantLatencyWrapper):
  """Shortcut for wrapping a xarray_loaders.XarrayDataLoader data loader in a latency wrapper.

  This simply uses the init_time coordinate on the Zarr file to determine the
  nominal init times.
  """

  def __init__(
      self,
      data_loader: xarray_loaders.XarrayDataLoader,
      latency: np.timedelta64,
      init_time_dim: str = 'init_time',
      concat_dim: str = 'init_time',
  ):
    super().__init__(
        data_loader,
        latency,
        data_loader._ds[init_time_dim].values,
        concat_dim=concat_dim,
    )


class MultipleConstantLatencyWrapper(base.DataLoader):
  """Extension to multiple data loaders with different nominal init times.

  This is to serve the case where e.g. 00/12UTC and 06/18UTC forecasts are
  stored in different e.g. Zarr files. This data loader then uses the most
  recent available init time across all data loaders.

  It works internally by wrapping load_chunk, determining the wrapped data
  loader to call and then concatenating the results across init_time. If there
  is a tie (i.e. multiple underling data loaders that have the same available
  init time), ties will be broken by picking the data loader with the largest
  latency, with the assumption that a larger latency implies a larger lookahead.

  As for the regular LatencyWrapper, the concatenation is done along the
  init_time dimension for non-sparse data and along the index dimension for
  sparse data. However, this has to be explicitly passed as an argument.

  One difference to the regular LatencyWrapper is that the concatenation is done
  using data from .load_chunk() instead of ._load_chunk_from_source() for single
  latency wrappers. This means that here the data is already interpolated.
  """

  def __init__(
      self,
      data_loaders: list[ConstantLatencyWrapper],
      concat_dim: str = 'init_time',
  ):
    super().__init__()
    self._data_loaders = data_loaders
    self._concat_dim = concat_dim

  def _load_chunk_from_source(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    raise NotImplementedError(
        'This should only be called for the individual data loaders.'
    )

  def _get_data_loader(self, init_time):
    lead_time_offsets_and_latencies = []
    for data_loader in self._data_loaders:
      available_init_time = data_loader.get_available_init_time(init_time)
      lead_time_offset = init_time - available_init_time
      # Break ties by picking the data loader with largest latency -- note that
      # we make latency negative here because we want the smallest
      # lead_time_offset, but the largest data loader latency.
      lead_time_offsets_and_latencies.append(
          (lead_time_offset, -data_loader.latency)
      )
    lead_time_offsets_and_latencies = np.array(
        lead_time_offsets_and_latencies,
        dtype=[
            ('lead_time_offset', 'timedelta64[s]'),
            ('neg_latency', 'timedelta64[s]'),
        ],
    )
    idx = np.argsort(
        lead_time_offsets_and_latencies,
        order=('lead_time_offset', 'neg_latency'),
    )
    most_recent_data_loader = self._data_loaders[idx[0]]
    logging.info(
        'Init time: %s, data loader latency: %s',
        init_time,
        most_recent_data_loader.latency,
    )
    return most_recent_data_loader

  def load_chunk(
      self,
      init_times: np.ndarray,
      lead_times: Optional[Union[np.ndarray, slice]] = None,
      reference: Optional[Mapping[Hashable, xr.DataArray]] = None,
  ) -> Mapping[Hashable, xr.DataArray]:
    chunk = []
    # TODO(srasp): Interpolation gets tricky here because the reference data
    # may contain data for multiple init times. For dense data this is fine,
    # because you are only interpolating spatially, but for sparse data this
    # can cause issues. Need to add some checks here and document this better.
    for init_time in init_times:
      most_recent_data_loader = self._get_data_loader(init_time)
      chunk.append(
          most_recent_data_loader.load_chunk([init_time], lead_times, reference)
      )

    # Concatenate init_times. For sparse data, the index dimension should be
    # used.
    chunk = xarray_tree.map_structure(
        lambda *x: xr.concat([*x], dim=self._concat_dim), *chunk
    )
    return chunk
