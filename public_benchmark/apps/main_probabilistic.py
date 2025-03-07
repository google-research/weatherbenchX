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
from dash_app import make_app
import xarray as xr
import sys

ds = xr.open_zarr('gs://wb2-app-data/v4/probabilistic.zarr')
app = make_app(ds, 'probabilistic')
server = app.server

if __name__ == '__main__':
    if sys.argv[1] == 'local':
        app.run_server(debug=True, use_reloader=True)
    else:
        app.run_server(host='0.0.0.0', port=8080, debug=False, use_reloader=False)