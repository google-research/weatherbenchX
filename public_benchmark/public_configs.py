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
"""Public configs."""

import copy

upper_level_variables = [
    'geopotential',
    'temperature',
    'u_component_of_wind',
    'v_component_of_wind',
    'wind_speed',
    'specific_humidity',
]

surface_variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '10m_wind_speed',
]

standard_variables = upper_level_variables + surface_variables

precipitation_variables = [
    'total_precipitation_6hr',
    'total_precipitation_24hr',
]

fuxi_variables = [
    v
    for v in standard_variables + precipitation_variables
    if v != 'specific_humidity'
]

gc_kwargs = {
    'rename_dimensions': {
        'time': 'init_time',
        'prediction_timedelta': 'lead_time',
        'lat': 'latitude',
        'lon': 'longitude',
    }
}
deterministic_prediction_configs = {
    # HRES
    **dict.fromkeys(  # Use this syntax if several years map to the same path.
        ['hres_64x32_2018', 'hres_64x32_2020', 'hres_64x32_2022'],
        {
            'path': 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        ['hres_240x121_2018', 'hres_240x121_2020', 'hres_240x121_2022'],
        {
            'path': 'gs://weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        ['hres_1440x721_2018', 'hres_1440x721_2020', 'hres_1440x721_2022'],
        {
            'path': (
                'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
            ),
            'variables': standard_variables + precipitation_variables,
        },
    ),
    # ENS Mean
    **dict.fromkeys(
        ['ens_mean_64x32_2018', 'ens_mean_64x32_2020', 'ens_mean_64x32_2022'],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative_mean.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        [
            'ens_mean_240x121_2018',
            'ens_mean_240x121_2020',
            'ens_mean_240x121_2022',
        ],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        [
            'ens_mean_1440x721_2018',
            'ens_mean_1440x721_2020',
            'ens_mean_1440x721_2022',
        ],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    # ERA5 Forecast
    'era5_forecast_64x32_2018': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2018-64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
    'era5_forecast_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2020-64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
    'era5_forecast_240x121_2018': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2018-240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables,
    },
    'era5_forecast_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2020-240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables,
    },
    'era5_forecast_1440x721_2018': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2018-1440x721.zarr',
        'variables': standard_variables,
    },
    'era5_forecast_1440x721_2020': {
        'path': 'gs://weatherbench2/datasets/era5-forecasts/2020-1440x721.zarr',
        'variables': standard_variables,
    },
    # Keisler
    'keisler_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/keisler/2020-64x32_equiangular_conservative.zarr',
        'variables': upper_level_variables,
        'data_loader_kwargs': {'add_nan_mask': True},
    },
    'keisler_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/keisler/2020-240x121_equiangular_with_poles_conservative.zarr',
        'variables': upper_level_variables,
        'data_loader_kwargs': {'add_nan_mask': True},
    },
    # Pangu
    **dict.fromkeys(
        ['pangu_64x32_2018', 'pangu_64x32_2020', 'pangu_64x32_2022'],
        {
            'path': 'gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr',
            'variables': standard_variables,
        },
    ),
    **dict.fromkeys(
        [
            'pangu_240x121_2018',
            'pangu_240x121_2020',
            'pangu_240x121_2022',
        ],
        {
            'path': 'gs://weatherbench2/datasets/pangu/2018-2022_0012_240x121_equiangular_with_poles_conservative.zarr',
            'variables': standard_variables,
        },
    ),
    **dict.fromkeys(
        [
            'pangu_1440x721_2018',
            'pangu_1440x721_2020',
            'pangu_1440x721_2022',
        ],
        {
            'path': (
                'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
            ),
            'variables': standard_variables,
        },
    ),
    # Pangu (HRES_init)
    'pangu_hres_init_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/pangu_hres_init/2020_0012_64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
    'pangu_hres_init_64x32_2022': {
        'path': 'gs://weatherbench2/datasets/pangu_hres_init/2022_0012_64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
    'pangu_hres_init_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/pangu_hres_init/2020_0012_240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables,
    },
    'pangu_hres_init_240x121_2022': {
        'path': 'gs://weatherbench2/datasets/pangu_hres_init/2022_0012_240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables,
    },
    'pangu_hres_init_1440x721_2020': {
        'path': (
            'gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr'
        ),
        'variables': standard_variables,
    },
    'pangu_hres_init_1440x721_2022': {
        'path': (
            'gs://weatherbench2/datasets/pangu_hres_init/2022_0012_0p25.zarr'
        ),
        'variables': standard_variables,
    },
    # NeuralGCM 0.7
    'neuralgcm_hres_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_deterministic/2020-64x32_equiangular_conservative.zarr',
        'variables': upper_level_variables,
    },
    'neuralgcm_hres_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr',
        'variables': upper_level_variables,
    },
    # NeuralGCM ENS mean
    'neuralgcm_ens_mean_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative_mean.zarr',
        'variables': upper_level_variables,
    },
    'neuralgcm_ens_mean_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_ens/2020-240x121_equiangular_with_poles_conservative_mean.zarr',
        'variables': upper_level_variables,
    },
    # FuXi
    'fuxi_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/fuxi/2020-64x32_equiangular_conservative.zarr',
        'variables': fuxi_variables,
        'levels': [500, 850],
        # 'data_loader_kwargs': fuxi_kwargs,
    },
    'fuxi_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/fuxi/2020-240x121_equiangular_with_poles_conservative.zarr',
        'variables': fuxi_variables,
        'levels': [500, 850],
        # 'data_loader_kwargs': fuxi_kwargs,
    },
    'fuxi_1440x721_2020': {
        'path': 'gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr',
        'variables': fuxi_variables,
        'levels': [500, 850],
        # 'data_loader_kwargs': fuxi_kwargs,
    },
}
# For ensembles, add single member config

add_single_member_config = ['ens', 'neuralgcm_ens']


def select_first_member(ds):
  if 'number' in ds.dims:
    return ds.isel(number=0)
  elif 'sample' in ds.dims:
    return ds.isel(sample=0)
  elif 'member' in ds.dims:
    return ds.isel(member=0)
  elif 'realization' in ds.dims:
    return ds.isel(realization=0)
  else:
    raise ValueError('Dataset does not have a member dimension.')


single_member_configs = {}
for model, config in deterministic_prediction_configs.items():
  if any(model.startswith(m) for m in add_single_member_config):
    single_member_config = copy.deepcopy(config)
    single_member_config['path'] = single_member_config['path'].replace(  # pytype: disable=attribute-error
        '_mean.zarr', '.zarr'
    )
    if 'data_loader_kwargs' in single_member_config:
      assert (
          'preprocessing_fn' not in single_member_config['data_loader_kwargs']
      )
      single_member_config['data_loader_kwargs'][
          'preprocessing_fn'
      ] = select_first_member
    else:
      single_member_config['data_loader_kwargs'] = {
          'preprocessing_fn': select_first_member
      }
    single_member_configs[model.replace('_mean', '_single_member')] = (
        single_member_config
    )
deterministic_prediction_configs.update(single_member_configs)

probabilistic_prediction_configs = {
    # ENS
    **dict.fromkeys(
        ['ens_64x32_2018', 'ens_64x32_2020', 'ens_64x32_2022'],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        [
            'ens_240x121_2018',
            'ens_240x121_2020',
            'ens_240x121_2022',
        ],
        {
            'path': 'gs://weatherbench2/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        [
            'ens_1440x721_2018',
            'ens_1440x721_2020',
            'ens_1440x721_2022',
        ],
        {
            'path': (
                'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721.zarr'
            ),
            'variables': standard_variables + precipitation_variables,
        },
    ),
    # NeuralGCM ENS
    'neuralgcm_ens_64x32_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_ens/2020-64x32_equiangular_conservative.zarr',
        'variables': upper_level_variables,
        'data_loader_kwargs': {
            'rename_dimensions': {
                'realization': 'number',
                'time': 'init_time',
                'prediction_timedelta': 'lead_time',
            }
        },
    },
    'neuralgcm_ens_240x121_2020': {
        'path': 'gs://weatherbench2/datasets/neuralgcm_ens/2020-240x121_equiangular_with_poles_conservative.zarr',
        'variables': upper_level_variables,
        'data_loader_kwargs': {
            'rename_dimensions': {
                'realization': 'number',
                'time': 'init_time',
                'prediction_timedelta': 'lead_time',
            }
        },
    },
}

target_configs = {
    # ERA5
    'era5_64x32': {
        'path': 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr',
        'variables': standard_variables + precipitation_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    'era5_240x121': {
        'path': 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables + precipitation_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    'era5_1440x721': {
        'path': 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr',
        'variables': standard_variables + precipitation_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    # HRES T0
    'hres_t0_64x32': {
        'path': 'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr',
        'variables': standard_variables,
    },
    'hres_t0_240x121': {
        'path': 'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables,
    },
    'hres_t0_1440x721': {
        'path': (
            'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr'
        ),
        'variables': standard_variables,
    },
}

climatology_configs = {
    'era5_64x32_2018': {
        'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr',
        'variables': standard_variables + precipitation_variables,
    },
    'era5_240x121_2016': {
        'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables + precipitation_variables,
    },
    'era5_240x121_2018': {
        'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr',
        'variables': standard_variables + precipitation_variables,
    },
    'era5_1440x721_2018': {
        'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2017_6h_1440x721.zarr',
        'variables': standard_variables + precipitation_variables,
        'data_loader_kwargs': {
            'preprocessing_fn': lambda ds: ds.sortby('latitude')
        },
    },
    **dict.fromkeys(
        ['era5_64x32_2020', 'era5_64x32_2022'],
        {
            'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        ['era5_240x121_2020', 'era5_240x121_2022'],
        {
            'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr',
            'variables': standard_variables + precipitation_variables,
        },
    ),
    **dict.fromkeys(
        ['era5_1440x721_2020', 'era5_1440x721_2022'],
        {
            'path': 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr',
            'variables': standard_variables + precipitation_variables,
            'data_loader_kwargs': {
                'preprocessing_fn': lambda ds: ds.sortby('latitude')
            },
        },
    ),
}
