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
import seaborn as sns

c = sns.color_palette("tab10", 10)
colors = list(c.as_hex())
blue, orange, green, red, purple, brown, pink, grey, yellow, lightblue = colors

c = sns.color_palette("husl", 12)
husl = list(c.as_hex())

SURFACE_VARIABLES = [
    "Sea Level Pressure",
    "2m Temperature",
    "6h Precipitation",
    "24h Precipitation",
    "10m Wind Speed",
    "10m Wind Vector",
]

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


DEFAULT_MODELS = [
    "IFS HRES vs Analysis",
    "IFS ENS (mean) vs Analysis",
    "GraphCast vs ERA5",
    "NeuralGCM 0.7 vs ERA5",
]
DEFAULT_MODELS_PRECIP = [
    "IFS HRES vs ERA5",
    "IFS ENS (mean) vs ERA5",
    "GraphCast vs ERA5",
]
DEFAULT_MODELS_PROB = [
    "IFS ENS vs Analysis",
    "GenCast vs ERA5",
    "NeuralGCM ENS vs ERA5",
]

# Pobabilistic + deterministic
COLORS = {
    "Climatology vs ERA5": "darkgrey",
    "Probabilistic Climatology vs ERA5": "darkgrey",
    "Persistence vs ERA5": "lightgray",
    "IFS HRES vs Analysis": blue,
    "IFS HRES vs ERA5": blue,
    "IFS ENS (mean) vs Analysis": green,
    "IFS ENS (mean) vs ERA5": green,
    "IFS ENS (1st member) vs Analysis": green,
    "IFS ENS (1st member) vs ERA5": green,
    "IFS ENS vs Analysis": green,
    "IFS ENS vs ERA5": green,
    "GraphCast vs ERA5": red,
    "GraphCast (oper.) vs ERA5": pink,
    "GraphCast (oper.) vs Analysis": pink,
    "Pangu-Weather vs ERA5": purple,
    "Pangu-Weather (oper.) vs ERA5": "purple",
    "Pangu-Weather (oper.) vs Analysis": "purple",
    "Keisler (2022) vs ERA5": husl[6],
    "ERA5-Forecasts vs ERA5": yellow,
    "NeuralGCM 0.7 vs ERA5": "orange",
    "NeuralGCM ENS vs ERA5": "orange",
    "NeuralGCM ENS (mean) vs ERA5": orange,
    "NeuralGCM ENS (1st member) vs ERA5": orange,
    # "sphericalcnn_vs_era": lightblue,
    "FuXi vs ERA5": brown,
    "GenCast vs ERA5": "firebrick",
    "GenCast (mean) vs ERA5": "firebrick",
    "GenCast (1st member) vs ERA5": "firebrick",
    "GenCast (oper.) vs ERA5": "maroon",
    "GenCast (oper.) (mean) vs ERA5": "maroon",
    "GenCast (oper.) (1st member) vs ERA5": "maroon",
    "GenCast (oper.) vs Analysis": "maroon",
    "GenCast (oper.) (mean) vs Analysis": "maroon",
    "GenCast (oper.) (1st member) vs Analysis": "maroon",
    # "GenCast 100m U/V (oper.) vs ERA5": "maroon",
    # "GenCast 100m U/V (oper.) (mean) vs ERA5": "maroon",
    # "GenCast 100m U/V (oper.) (1st member) vs ERA5": "maroon",
    # "GenCast 100m U/V (oper.) vs Analysis": "maroon",
    # "GenCast 100m U/V (oper.) (mean) vs Analysis": "maroon",
    # "GenCast 100m U/V (oper.) (1st member) vs Analysis": "maroon",
    "Aurora (oper.) vs Analysis": husl[8],
    "Stormer ENS (mean) vs ERA5": husl[2],
    "ArchesWeather-Mx4 vs ERA5": husl[3],
    "ArchesWeatherGen vs ERA5": husl[3],
    "ArchesWeatherGen (mean) vs ERA5": husl[3],
    "Swin vs ERA5": husl[4],
    "Excarta (HEAL-ViT) vs ERA5": husl[5],
    "Baguan vs ERA5": husl[6],
    "WeatherMesh4 vs ERA5": husl[7],
}

SYMBOLS = {
    "Climatology vs ERA5": "circle",
    "Probabilistic Climatology vs ERA5": "circle",
    "Persistence vs ERA5": "circle",
    "IFS HRES vs Analysis": "circle",
    "IFS HRES vs ERA5": "triangle-up",
    "IFS ENS (mean) vs Analysis": "circle",
    "IFS ENS (mean) vs ERA5": "triangle-up",
    "IFS ENS (1st member) vs Analysis": "triangle-down",
    "IFS ENS (1st member) vs ERA5": "square",
    "IFS ENS vs Analysis": "circle",
    "IFS ENS vs ERA5": "triangle-up",
    "GraphCast vs ERA5": "circle",
    "GraphCast (oper.) vs ERA5": "triangle-up",
    "GraphCast (oper.) vs Analysis": "circle",
    "Pangu-Weather vs ERA5": "circle",
    "Pangu-Weather (oper.) vs ERA5": "triangle-up",
    "Pangu-Weather (oper.) vs Analysis": "circle",
    "Keisler (2022) vs ERA5": "circle",
    "ERA5-Forecasts vs ERA5": "circle",
    "NeuralGCM 0.7 vs ERA5": "circle",
    "neuralgcm_hres_vs_era": "circle",
    "NeuralGCM ENS vs ERA5": "square",
    "NeuralGCM ENS (mean) vs ERA5": "square",
    "NeuralGCM ENS (1st member) vs ERA5": "triangle-down",
    # "sphericalcnn_vs_era": "circle",
    "FuXi vs ERA5": "circle",
    "GenCast vs ERA5": "circle",
    "GenCast (mean) vs ERA5": "circle",
    "GenCast (1st member) vs ERA5": "triangle-down",
    "GenCast (oper.) vs ERA5": "triangle-up",
    "GenCast (oper.) (mean) vs ERA5": "triangle-up",
    "GenCast (oper.) (1st member) vs ERA5": "square",
    "GenCast (oper.) vs Analysis": "circle",
    "GenCast (oper.) (mean) vs Analysis": "circle",
    "GenCast (oper.) (1st member) vs Analysis": "triangle-down",
    # "GenCast 100m U/V (oper.) vs ERA5": "triangle-up",
    # "GenCast 100m U/V (oper.) (mean) vs ERA5": "triangle-up",
    # "GenCast 100m U/V (oper.) (1st member) vs ERA5": "square",
    # "GenCast 100m U/V (oper.) vs Analysis": "circle",
    # "GenCast 100m U/V (oper.) (mean) vs Analysis": "circle",
    # "GenCast 100m U/V (oper.) (1st member) vs Analysis": "triangle-down",
    "Aurora (oper.) vs Analysis": "triangle-down",
    "Stormer ENS (mean) vs ERA5": "circle",
    "ArchesWeather-Mx4 vs ERA5": "circle",
    "ArchesWeatherGen (mean) vs ERA5": "square",
    "ArchesWeatherGen vs ERA5": "circle",
    "Swin vs ERA5": "circle",
    "Excarta (HEAL-ViT) vs ERA5": "circle",
    "Baguan vs ERA5": "circle",
    "WeatherMesh4 vs ERA5": "circle",
}
