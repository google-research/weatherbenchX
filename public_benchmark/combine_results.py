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
r"""Combine benchmark results into a single file.

This can take a few minutes to run locally.

python combine_results.py \
    --input_dir=gs://weatherbench2/benchmark_results \
    --output_dir=./ \
    --mode=deterministic
"""

from os import path

from absl import app
from absl import flags
import fsspec
import numpy as np
import xarray as xr

MODE = flags.DEFINE_enum(
    "mode",
    None,
    ["deterministic", "probabilistic"],
    "Type of results to combine.",
    required=True,
)
INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Directory to read results from.",
)
OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Directory to write results to.",
)

# These names should match those in the evaluation config.
DETERMINISTIC_MODELS = {
    "hres_vs_hres_t0": "IFS HRES vs Analysis",
    "hres_vs_era5": "IFS HRES vs ERA5",
    "ens_mean_vs_hres_t0": "IFS ENS (mean) vs Analysis",
    "ens_single_member_vs_hres_t0": "IFS ENS (1st member) vs Analysis",
    "ens_mean_vs_era5": "IFS ENS (mean) vs ERA5",
    "ens_single_member_vs_era5": "IFS ENS (1st member) vs ERA5",
    "era5_forecast_vs_era5": "ERA5-Forecasts vs ERA5",
    "climatology_vs_era5": "Climatology vs ERA5",
    "persistence_vs_era5": "Persistence vs ERA5",
    "keisler_vs_era5": "Keisler (2022) vs ERA5",
    "pangu_vs_era5": "Pangu-Weather vs ERA5",
    "pangu_hres_init_vs_era5": "Pangu-Weather (oper.) vs ERA5",
    "pangu_hres_init_vs_hres_t0": "Pangu-Weather (oper.) vs Analysis",
    "graphcast_vs_era5": "GraphCast vs ERA5",
    "graphcast_hres_init_vs_era5": "GraphCast (oper.) vs ERA5",
    "graphcast_hres_init_vs_hres_t0": "GraphCast (oper.) vs Analysis",
    "gencast_mean_vs_era5": "GenCast (mean) vs ERA5",
    "gencast_single_member_vs_era5": "GenCast (1st member) vs ERA5",
    "gencast_operational_100m_uv_mean_vs_era5": (
        "GenCast (oper.) (mean) vs ERA5"
    ),
    "gencast_operational_100m_uv_single_member_vs_era5": (
        "GenCast (oper.) (1st member) vs ERA5"
    ),
    "gencast_operational_100m_uv_mean_vs_hres_t0": (
        "GenCast (oper.) (mean) vs Analysis"
    ),
    "gencast_operational_100m_uv_single_member_vs_hres_t0": (
        "GenCast (oper.) (1st member) vs Analysis"
    ),
    "neuralgcm_hres_vs_era5": "NeuralGCM 0.7 vs ERA5",
    "neuralgcm_ens_mean_vs_era5": "NeuralGCM ENS (mean) vs ERA5",
    "neuralgcm_ens_single_member_vs_era5": "NeuralGCM ENS (1st member) vs ERA5",
    "fuxi_vs_era5": "FuXi vs ERA5",
    "stormer_ens_mean_vs_era5": "Stormer ENS (mean) vs ERA5",
    "arches_weather_mx4_vs_era5": "ArchesWeather-Mx4 vs ERA5",
    "arches_weather_gen_mean_vs_era5": "ArchesWeatherGen (mean) vs ERA5",
    "swin_vs_era5": "Swin vs ERA5",
    "excarta_vs_era5": "Excarta (HEAL-ViT) vs ERA5",
    "aurora_vs_hres_t0": "Aurora (oper.) vs Analysis",
    "baguan_vs_era5": "Baguan vs ERA5",
    "weathermesh4_vs_era5": "WeatherMesh4 vs ERA5",
}
PROBABILISTIC_MODELS = {
    "ens_vs_hres_t0": "IFS ENS vs Analysis",
    "ens_vs_era5": "IFS ENS vs ERA5",
    "neuralgcm_ens_vs_era5": "NeuralGCM ENS vs ERA5",
    "probabilistic_climatology_vs_era5": "Probabilistic Climatology vs ERA5",
    "gencast_vs_era5": "GenCast vs ERA5",
    "gencast_operational_100m_uv_vs_era5": "GenCast (oper.) vs ERA5",
    "gencast_operational_100m_uv_vs_hres_t0": "GenCast (oper.) vs Analysis",
    "arches_weather_gen_vs_era5": "ArchesWeatherGen vs ERA5",
}
REGION_NAMES = {
    "global": "Global",
    "tropics": "Tropics",
    # 'Extra-tropics': 'Extra-Tropics',
    "northern-hemisphere": "Northern Hemisphere",
    "southern-hemisphere": "Southern Hemisphere",
    "europe": "Europe",
    "north-america": "North America",
    "north-atlantic": "North Atlantic",
    "north-pacific": "North Pacific",
    "east-asia": "East Asia",
    "ausnz": "Australia/New Zealand",
    "arctic": "Arctic",
    "antarctic": "Antarctic",
    "northern-africa": "Northern Africa",
    "southern-africa": "Southern Africa",
    "south-america": "South America",
    "west-asia": "West Asia",
    "south-east-asia": "South-East Asia",
}
VARIABLE_NAMES = {
    "geopotential": "Geopotential",
    "temperature": "Temperature",
    "specific_humidity": "Specific Humidity",
    "u_component_of_wind": "U Component of Wind",
    "v_component_of_wind": "V Component of Wind",
    "10m_u_component_of_wind": "10m U Component of Wind",
    "10m_v_component_of_wind": "10m V Component of Wind",
    "mean_sea_level_pressure": "Sea Level Pressure",
    "2m_temperature": "2m Temperature",
    "total_precipitation_6hr": "6h Precipitation",
    "total_precipitation_24hr": "24h Precipitation",
    "wind_speed": "Wind Speed",
    "10m_wind_speed": "10m Wind Speed",
    "wind_vector": "Wind Vector",
    "10m_wind_vector": "10m Wind Vector",
}
DETERMINISTIC_METRIC_NAMES = {
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias",
    "seeps": "SEEPS",
    "acc": "ACC",
    "mse": "MSE",
    "prediction_activity": "Forecast Activity",
}
PROBABILISTIC_METRIC_NAMES = {
    "crps": "CRPS",
    "spread_skill": "Spread/Skill",
    "unbiased_spread_skill": "Unbiased Spread/Skill",
    "unbiased_mean_rmse": "Unbiased Mean RMSE",
    "mean_rmse": "Mean RMSE",
}

UNITS = {
    "Geopotential": "m<sup>2</sup>/s<sup>2</sup>",
    "Temperature": "K",
    "Specific Humidity": "g/kg",
    "U Component of Wind": "m/s",
    "V Component of Wind": "m/s",
    "10m U Component of Wind": "m/s",
    "10m V Component of Wind": "m/s",
    "2m Temperature": "K",
    "Sea Level Pressure": "Pa",
    "6h Precipitation": "mm",
    "24h Precipitation": "mm",
    "Wind Speed": "m/s",
    "10m Wind Speed": "m/s",
    "Wind Vector": "m/s",
    "10m Wind Vector": "m/s",
    "500hPa Geopotential": "m<sup>2</sup>/s<sup>2</sup>",
    "700hPa Specific Humidity": "g/kg",
    "850hPa Temperature": "K",
    "850hPa Wind Speed": "m/s",
}

YEARS = [2018, 2020, 2022]
RESOLUTIONS = ["64x32", "240x121", "1440x721"]


def open_nc(filename: str, **kwargs) -> xr.Dataset:
  """Open NetCDF file from filesystem."""
  with fsspec.open(filename, "rb") as f:
    ds = xr.open_dataset(f, decode_timedelta=True, **kwargs).compute()
  return ds


def _rename_region(region):
  if region.endswith("_land"):
    return REGION_NAMES[region.split("_land")[0]] + " (Land)"
  else:
    return REGION_NAMES[region]


def process_results(model, year, resolution):
  """Process a single results file."""

  fn = path.join(INPUT_DIR.value, f"{model}_{resolution}_{year}.nc")
  try:
    ds = open_nc(fn)
  except Exception:  # pylint: disable=broad-exception-caught
    print(fn, "does not exist.")
    return

  # GraphCast models retain a random lead_time_secs coordinate that we drop.
  if hasattr(ds, "lead_time_secs"):
    ds = ds.drop_vars("lead_time_secs")

  if "vector_rmse.wind" in ds:
    ds = ds.rename({"vector_rmse.wind": "rmse.wind_vector"})
  if "vector_rmse.10m_wind" in ds:
    ds = ds.rename({"vector_rmse.10m_wind": "rmse.10m_wind_vector"})
  metric_variables = list(ds.data_vars)
  variables = np.unique([v.split(".")[1] for v in metric_variables])
  cat_variables = {}
  for v in variables:
    metrics_for_variable = [
        mv.split(".")[0] for mv in metric_variables if mv.endswith("." + v)
    ]
    cat_variable = xr.concat(
        [ds[f"{mv}.{v}"] for mv in metrics_for_variable],
        dim=xr.DataArray(metrics_for_variable, dims=["metric"]),
    )
    cat_variables[v] = cat_variable
  ds = xr.Dataset(cat_variables)
  ds = ds.rename({k: v for k, v in VARIABLE_NAMES.items() if k in ds})
  if MODE.value == "deterministic":
    metric_names = DETERMINISTIC_METRIC_NAMES
  else:
    metric_names = PROBABILISTIC_METRIC_NAMES
  ds = ds.assign_coords(metric=[metric_names[m] for m in ds.metric.values])
  ds = ds.assign_coords(region=[_rename_region(r) for r in ds.region.values])
  return ds


def main(_):
  if MODE.value == "deterministic":
    models = DETERMINISTIC_MODELS
  else:
    models = PROBABILISTIC_MODELS

  results = xr.Dataset()
  for model in models:
    for year in YEARS:
      for resolution in RESOLUTIONS:
        print(model, year, resolution)
        ds = process_results(model, year, resolution)
        if ds is not None:
          ds = ds.expand_dims(
              model=[models[model]],
              resolution=[resolution],
              year=[str(year)],
          )
          # TODO(srasp): Accumulate datasets in a list and merge using
          # xr.combine_by_coords(). More efficient.
          results = xr.merge([results, ds])
  results.coords["lead_time_h"] = results.lead_time.values.astype(
      "timedelta64[h]"
  ).astype("int")
  print(results)

  results = results.chunk({
      "model": -1,
      "resolution": 1,
      "year": 1,
      "metric": -1,
      "region": 1,
      "level": -1,
      "lead_time": -1,
  })
  if MODE.value == "deterministic":
    results_fn = path.join(OUTPUT_DIR.value, "deterministic.zarr")
  else:
    results_fn = path.join(OUTPUT_DIR.value, "probabilistic.zarr")
  print(results_fn)
  results.to_zarr(results_fn)


if __name__ == "__main__":
  app.run(main)
