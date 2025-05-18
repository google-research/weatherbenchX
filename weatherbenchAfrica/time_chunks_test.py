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

from absl.testing import absltest
import numpy as np
from weatherbenchAfrica import time_chunks


class TimeChunksTest(absltest.TestCase):

  def test_expected_output_length(self):
    """Tests that the output length is correct."""

    # 4 init_times
    init_times = np.arange(
        '2020-01-01T00',
        '2020-01-02T00',
        np.timedelta64(6, 'h'),
        dtype='datetime64[ns]',
    )

    # Case #1: 3 specific lead_times
    lead_times = np.arange(0, 18, 6, dtype='timedelta64[h]')

    # Case #1.1: Single chunks
    times = time_chunks.TimeChunks(init_times=init_times, lead_times=lead_times)
    self.assertLen(list(times), 1)

    # Case #1.2: Chunks in both dimensions
    times = time_chunks.TimeChunks(
        init_times=init_times,
        lead_times=lead_times,
        init_time_chunk_size=2,
        lead_time_chunk_size=2,
    )
    self.assertLen(list(times), 4)

    # Case #2: Lead time interval
    lead_times = slice(np.timedelta64(0, 'h'), np.timedelta64(6, 'h'))

    times = time_chunks.TimeChunks(
        init_times=init_times,
        lead_times=lead_times,
        init_time_chunk_size=2,
    )
    self.assertLen(list(times), 2)


if __name__ == '__main__':
  absltest.main()
