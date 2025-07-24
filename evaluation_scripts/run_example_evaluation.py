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
r"""Example WB-X evaluation script.

Example usage:

export BUCKET=my-bucket
export PROJECT=my-project
export REGION=us-central1

python run_example_evaluation.py \
  --prediction_path=gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr \
  --target_path=gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr \
  --time_start=2020-01-01 \
  --time_stop=2020-01-02 \
  --output_path=./results.nc \
  --runner=DirectRunner

or to run on DataFlow:
  --output_path=gs://$BUCKET/results.nc \
  --runner=DataflowRunner \
  -- \
  --project=$PROJECT \
  --region=$REGION \
  --temp_location=gs://$BUCKET/tmp/ \
  --setup_file=../setup.py \
  --job_name=wbx-evaluation
"""

from collections.abc import Sequence

from absl import app
from absl import flags
import apache_beam as beam
import numpy as np
from weatherbenchX import aggregation
from weatherbenchX import beam_pipeline
from weatherbenchX import binning
from weatherbenchX import time_chunks
from weatherbenchX import weighting
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import deterministic


_DEFAULT_VARIABLES = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'specific_humidity',
    '2m_temperature',
    'mean_sea_level_pressure',
]

_DEFAULT_LEVELS = ['500', '700', '850']

PREDICTION_PATH = flags.DEFINE_string(
    'prediction_path',
    None,
    help='Path to forecasts to evaluate in Zarr format',
)
TARGET_PATH = flags.DEFINE_string(
    'target_path',
    None,
    help='Path to ground-truth to evaluate in Zarr format',
)
TIME_START = flags.DEFINE_string(
    'time_start',
    '2020-01-01',
    help='ISO 8601 timestamp (inclusive) at which to start evaluation',
)
TIME_STOP = flags.DEFINE_string(
    'time_stop',
    '2020-12-31',
    help='ISO 8601 timestamp (exclusive) at which to stop evaluation',
)
TIME_FREQUENCY = flags.DEFINE_integer(
    'time_frequency', 12, help='Init frequency.'
)
TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'time_chunk_size', None, help='Time chunk size.'
)
LEAD_TIME_START = flags.DEFINE_integer(
    'lead_time_start', 0, help='Lead time start in hours.'
)
LEAD_TIME_STOP = flags.DEFINE_integer(
    'lead_time_stop', 24 * 10, help='Lead time end in hours(exclusive).'
)
LEAD_TIME_FREQUENCY = flags.DEFINE_integer(
    'lead_time_frequency', 6, help='Lead time frequency in hours.'
)
LEAD_TIME_CHUNK_SIZE = flags.DEFINE_integer(
    'lead_time_chunk_size', None, help='Lead time chunk size.'
)
LEVELS = flags.DEFINE_list(
    'levels',
    None,
    help='Comma delimited list of pressure levels to select for evaluation',
)
VARIABLES = flags.DEFINE_list(
    'variables',
    _DEFAULT_VARIABLES,
    help='Comma delimited list of variables to select from weather.',
)
REDUCE_DIMS = flags.DEFINE_list(
    'reduce_dims',
    ['init_time', 'latitude', 'longitude'],
    help='Comma delimited list of dimensions to reduce over.',
)
OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    None,
    help='File to save evaluation results in netCDF format',
)
RUNNER = flags.DEFINE_string('runner', None, 'beam.runners.Runner')


def main(argv: Sequence[str]) -> None:
  init_times = np.arange(
      TIME_START.value,
      TIME_STOP.value,
      np.timedelta64(TIME_FREQUENCY.value, 'h'),
      dtype='datetime64[ns]',
  )
  lead_times = np.arange(
      LEAD_TIME_START.value,
      LEAD_TIME_STOP.value,
      LEAD_TIME_FREQUENCY.value,
      dtype='timedelta64[h]',
  ).astype('timedelta64[ns]')

  times = time_chunks.TimeChunks(
      init_times,
      lead_times,
      init_time_chunk_size=TIME_CHUNK_SIZE.value,
      lead_time_chunk_size=LEAD_TIME_CHUNK_SIZE.value,
  )

  if LEVELS.value is not None:
    sel_kwargs = {'level': [int(level) for level in LEVELS.value]}
  else:
    sel_kwargs = {}

  target_loader = xarray_loaders.TargetsFromXarray(
      path=TARGET_PATH.value,
      variables=VARIABLES.value,
      sel_kwargs=sel_kwargs,
  )

  prediction_loader = xarray_loaders.PredictionsFromXarray(
      path=PREDICTION_PATH.value,
      variables=VARIABLES.value,
      sel_kwargs=sel_kwargs,
  )

  all_metrics = {'rmse': deterministic.RMSE(), 'mse': deterministic.MSE()}
  weigh_by = [weighting.GridAreaWeighting()]

  regions = {
      'global': ((-90, 90), (0, 360)),
      'northern-hemisphere': ((20, 90), (0, 360)),
  }
  bin_by = [binning.Regions(regions)]

  aggregation_method = aggregation.Aggregator(
      reduce_dims=REDUCE_DIMS.value,
      weigh_by=weigh_by,
      bin_by=bin_by,
  )

  with beam.Pipeline(runner=RUNNER.value, argv=argv) as root:
    beam_pipeline.define_pipeline(
        root,
        times,
        prediction_loader,
        target_loader,
        all_metrics,
        aggregation_method,
        out_path=OUTPUT_PATH.value,
    )


if __name__ == '__main__':
  app.run(main)
