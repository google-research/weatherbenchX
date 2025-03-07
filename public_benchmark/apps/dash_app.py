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
import xarray as xr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import pandas as pd
import dash

# from dash import Dash, Input, Output, callback
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.express as px

import dash_bootstrap_components as dbc

# Import the necessaries libraries
import plotly.offline as pyo
import plotly.graph_objs as go

import numpy as np

from config import *

RELATIVE_NO = "Absolute scores"
RELATIVE_YES = "Compare to model"


def make_app(ds, det_or_prob):
    # ds = ds.assign_coords(region=[region_names[r] for r in ds.region.values])
    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.SANDSTONE,
            "https://codepen.io/chriddyp/pen/bWLwgP.css",
        ],
        # title="WeatherBench 2 - Deterministic Scores",
    )
    # dropdown_style = {"width": f"{100/1}%", "display": "block", "fontSize": 13}
    dropdown_style = {"fontSize": 13, "width": f"{100}%"}
    w = 3
    menu = [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Variable"),
                                dcc.Dropdown(
                                    list(ds),
                                    "Temperature",
                                    id="variable",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Metric"),
                                dcc.Dropdown(
                                    list(ds.metric.values),
                                    "RMSE"
                                    if det_or_prob == "deterministic"
                                    else "CRPS",
                                    id="metric",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Level"),
                                dcc.Dropdown(
                                    list(ds.level.values),
                                    850,
                                    id="level",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Region"),
                                dcc.Dropdown(
                                    list(ds.region.values),
                                    "Global",
                                    id="region",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Year"),
                                dcc.Dropdown(
                                    list(ds.year.values), 
                                    "2020", 
                                    id="year",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Resolution"),
                                dcc.Dropdown(
                                    list(ds.resolution.values),
                                    "240x121",
                                    id="resolution",
                                    clearable=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Mode"),
                                dcc.RadioItems(
                                    [RELATIVE_NO, RELATIVE_YES],
                                    RELATIVE_NO,
                                    id="relative",
                                    inline=False,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Model to compare to"),
                                dcc.Dropdown(
                                        list(ds.model.values),
                                    'IFS HRES vs Analysis'
                                    if det_or_prob == "deterministic"
                                    else 'IFS ENS vs Analysis',
                                    id="relative_to",
                                    clearable=False,
                                    disabled=True,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                dbc.Label("Appearance"),
                                dbc.Checklist(
                                    options=[{"label": "Toggle markers", "value": 1}],
                                    value=[1],
                                    id="markers",
                                    switch=True,
                                ),
                            ],
                            style=dropdown_style,
                            className="mb-4",
                        )
                    ],
                    width=w,
                ),
            ]
        )
    ]

    alert_tbd = dbc.Alert(
        "We are still computing these results. Stay tuned!",
        id="alert-tbd",
        dismissable=True,
        fade=True,
        is_open=False,
        duration=4000,
        color="secondary",
    )
    alert_seeps = dbc.Alert(
        'SEEPS score only applies to precipitation and "vs ERA5".',
        id="alert-seeps",
        dismissable=True,
        fade=True,
        is_open=False,
        duration=4000,
        color="secondary",
    )
    alert_relative = dbc.Alert(
        "The model chosen to compare to has no values for the current selection. Please chose a different model or different parameters.",
        id="alert-relative",
        dismissable=True,
        fade=True,
        is_open=False,
        # duration=4000,
        color="secondary",
    )
    app.layout = html.Div(
        menu
        + [alert_tbd, alert_seeps, alert_relative]
        + [dcc.Graph(id="graph", style={"width": "95vw", "height": "45vw"})],
        style={"width": "95vw", "height": "45vw"},
    )

    @dash.callback(Output("level", "disabled"), Input("variable", "value"))
    def disable_level(variable):
        if variable in SURFACE_VARIABLES:
            return True
        else:
            return False

    @dash.callback(Output("relative_to", "disabled"), Input("relative", "value"))
    def disable_relative_to(relative):
        if relative == RELATIVE_YES:
            return False
        else:
            return True

    @dash.callback(
        [
            Output("graph", "figure"),
            Output("alert-relative", "is_open"),
            Output("alert-seeps", "is_open"),
            Output("alert-tbd", "is_open"),
        ],
        [
            Input("variable", "value"),
            Input("metric", "value"),
            Input("level", "value"),
            Input("region", "value"),
            Input("year", "value"),
            Input("resolution", "value"),
            Input("relative", "value"),
            Input("relative_to", "value"),
            Input("markers", "value"),
        ],
        State("graph", "figure")
    )
    def update_graph(
        variable, metric, level, region, year, resolution, relative, relative_to, markers, previous_fig=None
    ):
        relative_alert = False
        seeps_alert = False
        tbd_alert = False
        # year = "2020"
        fig = go.Figure()

        if metric == "SEEPS":
            if variable not in ["24h Precipitation", "6h Precipitation"]:
                seeps_alert = True
            if relative == RELATIVE_YES:
                if "vs Analysis" in relative_to:
                    seeps_alert = True
        if seeps_alert:
            return fig, relative_alert, seeps_alert, tbd_alert

        # if resolution == "1440x721" and metric == "ACC":
        #     tbd_alert = True
        # elif resolution == "1440x721" and metric == "SEEPS":
        #     tbd_alert = True
        # elif metric == "SEEPS" and variable == "6h Precipitation":
        #     tbd_alert = True
        # if tbd_alert:
        #     return fig, relative_alert, seeps_alert, tbd_alert

        selection = ds.sel(
            region=region,
            metric=metric,
            level=level,
            year=year,
            resolution=resolution,
            drop=True,
        )[variable].compute()

        if relative == RELATIVE_YES:
            baseline = selection.sel(model=relative_to)
            if np.isnan(baseline).mean() == 1:
                relative_alert = True
            if metric == "ACC":
                selection = (selection - baseline) / (1 - baseline) * 100
            else:
                selection = (selection - baseline) / baseline * -100
            # No relative at t=0 to avoid large values
            selection = selection.where(selection.lead_time > np.timedelta64(0, "h"))
        if metric == "Spread/Skill Ratio":
            selection = selection.where(selection.lead_time > np.timedelta64(0, "h"))

        for model in selection.model.values:
            # # if "climatology" in model:
            # #     if not metric == "acc":
            # #         fig.add_hline(
            # #             y=selection.sel(model=model).mean("lead_time").values,
            # #             name=model,
            # #             line_color=colors[model],
            # #         )
            # if True:
            visible = "legendonly"
            if "Precipitation" in variable:
                if model in DEFAULT_MODELS_PRECIP:
                    visible = None
            else:
                if det_or_prob == 'deterministic':
                    if model in DEFAULT_MODELS:
                        visible = None
                else:
                    if model in DEFAULT_MODELS_PROB:
                        visible = None

            if np.isfinite(selection.sel(model=model)).mean() > 0:
                fig.add_trace(
                    go.Scatter(
                        x=ds.lead_time_h,
                        y=selection.sel(model=model),
                        mode="lines+markers" if len(markers) == 1 else "lines",
                        name=model,
                        line={
                            "color": COLORS[model],
                            # "dash": dashes[model] if len(markers) == 1 else "solid",
                        },
                        marker={
                            "symbol": SYMBOLS[model]
                        },
                        visible=visible,
                    )
                )
                fig.update_traces(connectgaps=True)   # Required to plot despite of NaNs

        # Retain visibility selection
        # https://community.plotly.com/t/preserve-trace-visibility-in-callback/5118
        if previous_fig:
            visible_state = {}
            for i in previous_fig['data']:
                visible = i['visible'] if 'visible' in i.keys() else True
                visible_state[i['name']] = visible
            for j in fig['data']:
                if j['name'] in visible_state.keys():
                    j['visible'] = visible_state[j['name']]
                else:
                    j['visible'] = True if j['name'] in DEFAULT_MODELS else 'legendonly'

        if relative == RELATIVE_YES:
            ylabel = f"% improvement relative to {relative_to}"
        elif metric == "ACC":
            ylabel = "ACC"
        elif metric == "rms_bias":
            ylabel = rf"RMS Bias [{UNITS[variable]}]"
        else:
            ylabel = rf"{metric} [{UNITS[variable]}]"

        fig.update_layout(
            xaxis=dict(tickmode="linear", tick0=0, dtick=24),
            # title="Deterministic Scores",
            xaxis_title="Lead time [hours]",
            yaxis_title=ylabel,
            legend_title="Models",
            margin_b=0,
            margin_t=40,
            # uirevision=relative
        )
        fig.update_xaxes(
            range=[0, 10 * 24 if det_or_prob == "deterministic" else 15 * 24]
        )

        return fig, relative_alert, seeps_alert, tbd_alert

    return app


# import logging


# def make_temporal_app():

#     logging.info('[INFO] Start')
#     # ds = xr.concat([
#     #     xr.open_zarr('gs://wb2-app-data/v3/temporal_2018.zarr'),
#     #     xr.open_zarr('gs://wb2-app-data/v3/temporal_2020.zarr')], dim='year')
#     ds = xr.open_zarr('gs://wb2-app-data/v3/temporal_2020.zarr')
#     ds2 = xr.open_zarr('gs://wb2-app-data/v3/temporal_2018.zarr')
#     ds = ds.assign_coords(region=[region_names[r] for r in ds.region.values])
#     ds2 = ds2.assign_coords(region=[region_names[r] for r in ds2.region.values])

#     logging.info("[INFO] In app")
#     app = dash.Dash(
#         __name__,
#         external_stylesheets=[
#             dbc.themes.SANDSTONE,
#             "https://codepen.io/chriddyp/pen/bWLwgP.css",
#         ],
#         # title="WeatherBench 2 - Deterministic Scores",
#     )

#     dropdown_style = {"fontSize": 13, "width": f"{100}%"}
#     w = 3
#     menu = [
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Variable"),
#                                 dcc.Dropdown(
#                                     list(ds),
#                                     "Geopotential",
#                                     id="variable",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Metric"),
#                                 dcc.Dropdown(
#                                     [m for m in list(ds.metric.values) if m != 'RMSB'],
#                                     "RMSE",
#                                     id="metric",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Region"),
#                                 dcc.Dropdown(
#                                     list(ds.region.values),
#                                     "Global",
#                                     id="region",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Level"),
#                                 dcc.Dropdown(
#                                     list(ds.level.values),
#                                     500,
#                                     id="level",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#             ],
#             justify="center",
#         ),
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Resolution"),
#                                 dcc.Dropdown(
#                                     list(ds.resolution.values),
#                                     "240x121",
#                                     id="resolution",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Lead time [h]"),
#                                 dcc.Dropdown(
#                                     list(ds.lead_time_h.values),
#                                     3 * 24,
#                                     id="lead_time_h",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Relative"),
#                                 dcc.RadioItems(
#                                     [RELATIVE_NO, RELATIVE_YES],
#                                     RELATIVE_NO,
#                                     id="relative",
#                                     inline=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Model to compare to"),
#                                 dcc.Dropdown(
#                                     list(ds.model.values),
#                                     'IFS HRES vs Analysis',
#                                     id="relative_to",
#                                     clearable=False,
#                                     disabled=True,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Year"),
#                                 dcc.Dropdown(
#                                     list(ds.year.values),
#                                     # ['2018', '2020', '2022'], 
#                                     "2020", 
#                                     id="year",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Appearance"),
#                                 dbc.Checklist(
#                                     options=[{"label": "Toggle markers", "value": 1}],
#                                     value=[],
#                                     id="markers",
#                                     switch=True,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#             ]
#         ),
#     ]

#     alert_tbd = dbc.Alert(
#         "We are still computing these results. Stay tuned!",
#         id="alert-tbd",
#         dismissable=True,
#         fade=True,
#         is_open=False,
#         duration=4000,
#         color="secondary",
#     )
#     alert_seeps = dbc.Alert(
#         'SEEPS score only applies to precipitation and "vs ERA5".',
#         id="alert-seeps",
#         dismissable=True,
#         fade=True,
#         is_open=False,
#         duration=4000,
#         color="secondary",
#     )
#     alert_relative = dbc.Alert(
#         "The model chosen to compare to has no values for the current selection. Please chose a different model or different parameters.",
#         id="alert-relative",
#         dismissable=True,
#         fade=True,
#         is_open=False,
#         # duration=4000,
#         color="secondary",
#     )

#     app.layout = html.Div(
#         menu
#         + [alert_tbd, alert_seeps, alert_relative]
#         + [dcc.Graph(id="graph", style={"width": "95vw", "height": "45vw"})],
#         style={"width": "95vw", "height": "45vw"},
#     )

#     @dash.callback(Output("level", "disabled"), Input("variable", "value"))
#     def disable_level(variable):
#         if variable in SURFACE_VARIABLES:
#             return True
#         else:
#             return False

#     @dash.callback(Output("relative_to", "disabled"), Input("relative", "value"))
#     def disable_relative_to(relative):
#         if relative == RELATIVE_YES:
#             return False
#         else:
#             return True

#     @dash.callback(
#         [
#             Output("graph", "figure"),
#             Output("alert-relative", "is_open"),
#             Output("alert-seeps", "is_open"),
#             Output("alert-tbd", "is_open"),
#         ],
#         [
#             Input("variable", "value"),
#             Input("metric", "value"),
#             Input("level", "value"),
#             Input("region", "value"),
#             Input("year", "value"),
#             Input("resolution", "value"),
#             Input("lead_time_h", "value"),
#             Input("relative", "value"),
#             Input("relative_to", "value"),
#             Input("markers", "value"),
#         ],
#         State("graph", "figure")
#     )
#     def update_graph(
#         variable,
#         metric,
#         level,
#         region,
#         year,
#         resolution,
#         lead_time_h,
#         relative,
#         relative_to,
#         markers,
#         previous_fig=None
#     ):
#         relative_alert = False
#         seeps_alert = False
#         tbd_alert = False
#         logging.info("[INFO] In update")
#         fig = go.Figure()

#         if metric == "SEEPS":
#             if variable not in ["24h Precipitation", "6h Precipitation"]:
#                 seeps_alert = True
#             if relative == RELATIVE_YES:
#                 if "vs Analysis" in relative_to:
#                     seeps_alert = True
#         if seeps_alert:
#             return fig, relative_alert, seeps_alert, tbd_alert

#         # if resolution == "1440x721" and metric == "ACC":
#         #     tbd_alert = True
#         # elif resolution == "1440x721" and metric == "SEEPS":
#         #     tbd_alert = True
#         # elif metric == "SEEPS" and variable == "6h Precipitation":
#         #     tbd_alert = True
#         # if tbd_alert:
#         #     return fig, relative_alert, seeps_alert, tbd_alert
#         if year == '2020':
#             tmp = ds
#         elif year == '2018':
#             tmp = ds2
#         selection = (
#             tmp.sel(
#                 region=region,
#                 metric=metric,
#                 level=level,
#                 resolution=resolution,
#                 lead_time_h=lead_time_h,
#                 year=int(year),
#                 drop=True,
#             )[variable]
#             .compute()
#         )
#         # To ensure correct xlim
#         selection = selection.dropna('init_time', how='all')
#         logging.info("[INFO] After loading data")

#         if relative == RELATIVE_YES:
#             baseline = selection.sel(model=relative_to)
#             if np.isnan(baseline).mean() == 1:
#                 relative_alert = True
#             if metric == "acc":
#                 selection = (selection - baseline) / (1 - baseline) * 100
#             else:
#                 selection = (selection - baseline) / baseline * -100

#         for model in selection.model.values:
#             visible = "legendonly"
#             if "Precipitation" in variable:
#                 if model in DEFAULT_MODELS_PRECIP:
#                     visible = None
#             else:
#                 if model in DEFAULT_MODELS:
#                     visible = None
#             # print(selection)
#             if np.isfinite(selection.sel(model=model)).mean() > 0:
#                 # print(selection.sel(model=model))
#                 fig.add_trace(
#                     go.Scatter(
#                         x=selection.init_time,
#                         y=selection.sel(model=model),
#                         mode="lines+markers" if len(markers) == 1 else "lines",
#                         name=model,
#                         line={
#                             "color": COLORS[model],
#                             # "dash": dashes[model] if len(markers) == 1 else "solid",
#                         },
#                         marker={
#                             "symbol": SYMBOLS[model]
#                         },
#                         visible=visible,
#                     )
#                 )
#                 fig.update_traces(connectgaps=True)   # Required to plot despite of NaNs

#         # Retain visibility selection
#         # https://community.plotly.com/t/preserve-trace-visibility-in-callback/5118
#         if previous_fig:
#             visible_state = {}
#             for i in previous_fig['data']:
#                 visible = i['visible'] if 'visible' in i.keys() else True
#                 visible_state[i['name']] = visible
#             for j in fig['data']:
#                 if j['name'] in visible_state.keys():
#                     j['visible'] = visible_state[j['name']]
#                 else:
#                     j['visible'] = True if j['name'] in DEFAULT_MODELS else 'legendonly'

#         if relative == "Yes":
#             ylabel = f"% improvement relative to {relative_to}"
#         elif metric == "ACC":
#             ylabel = "ACC"
#         elif metric == "rms_bias":
#             ylabel = rf"RMS Bias [{UNITS[variable]}]"
#         else:
#             ylabel = rf"{metric} [{UNITS[variable]}]"

#         fig.update_layout(
#             # title="Deterministic Scores",
#             xaxis_title="Initialization time",
#             yaxis_title=ylabel,
#             legend_title="Models",
#             margin_b=0,
#             margin_t=40,
#         )

#         return fig, relative_alert, seeps_alert, tbd_alert

#     return app


# def make_spectra_app(ds):
#     app = dash.Dash(
#         __name__,
#         external_stylesheets=[
#             dbc.themes.SANDSTONE,
#             "https://codepen.io/chriddyp/pen/bWLwgP.css",
#         ],
#         # title="WeatherBench 2 - Deterministic Scores",
#     )
#     # dropdown_style = {"width": f"{100/6}%", "display": "inline-block", "fontSize": 13}
#     dropdown_style = {"fontSize": 13, "width": f"{100}%"}
#     w = 3

#     menu = [
#         dbc.Row(
#             [
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Variable"),
#                                 dcc.Dropdown(
#                                     list(ds),
#                                     "Geopotential",
#                                     id="variable",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         ),
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Level"),
#                                 dcc.Dropdown(
#                                     list(ds.level.values),
#                                     500,
#                                     id="level",
#                                     clearable=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         ),
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("X-Axis"),
#                                 dcc.RadioItems(
#                                     ['Wavelength', 'Wavenumber'],
#                                     'Wavelength',
#                                     id="wavenumber",
#                                     inline=False,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#                 dbc.Col(
#                     [
#                         html.Div(
#                             [
#                                 dbc.Label("Appearance"),
#                                 dbc.Checklist(
#                                     options=[{"label": "Toggle line styles", "value": 1}],
#                                     value=[],
#                                     id="markers",
#                                     switch=True,
#                                 ),
#                             ],
#                             style=dropdown_style,
#                             className="mb-4",
#                         )
#                     ],
#                     width=w,
#                 ),
#             ]
#         )
#     ]

#     app.layout = html.Div(
#         menu
#         + [dcc.Graph(id="graph")]
#         + [
#             html.Div(
#                 [
#                     "Lead time [h]",
#                     dcc.Slider(
#                         0,
#                         ds.lead_time_h.values.max(),
#                         12,
#                         value=3 * 24,
#                         id="lead_time_h",
#                         updatemode="drag",
#                         marks={
#                             t: {"label": str(t) if t % 24 == 0 else ""}
#                             for t in range(0, 360 + 1, 12)
#                         },
#                     ),
#                 ],
#                 style={"width": f"{100}%", "display": "inline-block", "fontSize": 13},
#             ),
#         ],
#         style={"width": "95vw"},
#     )

#     @dash.callback(Output("level", "disabled"), Input("variable", "value"))
#     def disable_level(variable):
#         if variable in surface_variables:
#             return True
#         else:
#             return False

#     @dash.callback(
#         Output("graph", "figure"),
#         [
#             Input("variable", "value"),
#             Input("level", "value"),
#             Input("lead_time_h", "value"),
#             Input("wavenumber", "value"),
#             Input("markers", "value"),
#         ],
#         State("graph", "figure")
#     )
#     def update_graph(variable, level, lead_time_h, wavenumber, markers, previous_fig=None):
#         fig = go.Figure()
#         selection = ds.sel(
#             level=level,
#             drop=True,
#         )[variable].compute()
#         mn = selection.min()
#         mx = selection.max()

#         for model in selection.model.values:
#             # visible = "legendonly"
#             # if reverse_labels[model] in default_models:
#             #     visible = None
#             da = selection.sel(model=model, lead_time_h=int(lead_time_h)).dropna(
#                 "zonal_wavenumber"
#             )
#             if np.isfinite(da).mean() > 0:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=da.zonal_wavenumber if wavenumber == 'Wavenumber' else da.wavelength,
#                         y=da,
#                         # mode="lines+markers" if len(markers) == 1 else "lines",
#                         mode="lines",
#                         name=spectra_labels[model],
#                         line={
#                             "color": spectra_colors[model],
#                             "dash": spectra_dashes[model]
#                             if len(markers) == 1
#                             else "solid",
#                         },
#                         # marker={
#                         #     "symbol": spectra_symbols[model]
#                         # },
#                         # visible=visible,
#                     )
#                 )

#         # Retain visibility selection
#         # https://community.plotly.com/t/preserve-trace-visibility-in-callback/5118
#         if previous_fig:
#             visible_state = {}
#             for i in previous_fig['data']:
#                 visible = i['visible'] if 'visible' in i.keys() else True
#                 visible_state[i['name']] = visible
#             for j in fig['data']:
#                 if j['name'] in visible_state.keys():
#                     j['visible'] = visible_state[j['name']]
#                 else:
#                     j['visible'] = True if reverse_spectra_labels[j['name']] in default_models else 'legendonly'

#         fig.update_layout(
#             # xaxis=dict(tickmode="linear", tick0=0, dtick=24),
#             # title="Deterministic Scores",
#             xaxis_title="Wavenumber" if wavenumber == 'Wavenumber' else "Wave length [km]",
#             yaxis_title="Power",
#             legend_title="Models",
#             margin_b=0,
#             margin_t=40,
#             uirevision=f'{variable}{level}'

#         )
#         fig.update_xaxes(type="log")
#         fig.update_yaxes(type="log")
#         fig.update_yaxes(range=[np.log10(mn), np.log10(mx)])

#         return fig

#     return app
