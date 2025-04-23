# This script contain the front end of the side bar aswell as the back end of the class sources manager
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from Tools.SourcesManager import SourcesManager
from Tools.SpatialObject import SpatialObject
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import Tools.graphing.mpl_template
import numpy as np

pio.templates.default = "ECS"

fs = 44100
grid_resolution = 100
n_fft = 128
data_m = np.zeros((grid_resolution**2, n_fft))


app_ui = ui.page_fluid(
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("x_length", "Width", value=10, min=1, max=100),
            ui.input_numeric("y_length", "Length", value=10, min=1, max=100),
            ui.input_numeric("target_freq", "Frequency", value=1000, min=20, max=20000),
            ui.card(
                ui.card_header("Source settings"),
                ui.input_numeric("radius", "Radius (m)", 0.1, min=0.01, max=1),
                ui.input_numeric(
                    "orientation", "Orientation (deg)", 0.0, min=0.0, max=360
                ),
                ui.input_select(
                    "sel_dir",
                    "Directivity",
                    ["monopole", "cardioid", "bessel"],
                    selected="monopole",
                ),
                ui.layout_column_wrap(
                    ui.input_slider("posX", "x (m)", value=0, min=-10 / 2, max=10 / 2),
                    ui.input_slider("posY", "y (m)", value=0, min=-10 / 2, max=10 / 2),
                    width=1 / 2,
                ),
            ),
            ui.card(
                ui.input_action_button("create_instance", "New instance"),
                ui.card_header("Data Frame as table"),
                ui.output_table("instance_table"),
            ),
        ),
        # Main Card
        ui.card(ui.card_header("display field"), output_widget("plot_field")),
    ),
    class_="p-3",
)


def server(input, output, session):

    Src_inst = SourcesManager()

    New_inst = reactive.Value(0)

    update_data = reactive.value(None)

    @reactive.calc()
    def define_grid():
        x = np.linspace(-input.x_length() / 2, input.x_length() / 2, grid_resolution)
        y = np.linspace(-input.y_length() / 2, input.y_length() / 2, grid_resolution)
        X, Y = np.meshgrid(x, y)
        return X, Y, x, y

    @reactive.effect()
    @reactive.event(input.create_instance)
    def _():
        X, Y, _, _ = define_grid()
        Src_inst.create_instance(
            f"{input.sel_dir()[0].upper()}_{New_inst.get()}",
            SpatialObject,
            fs=fs,
            dist_v=0,
            norm_s="cartesian",
            data_m=data_m,
            radius=input.radius(),
            position_v=[input.posX(), input.posY(), 0],
            orientation_v=[input.orientation(), 0],
            azim_v=X.flatten(),
            elev_v=Y.flatten(),
            directivity=input.sel_dir(),
            src_resp=1.0,
        )

        New_inst.set(New_inst() + 1)

    @reactive.calc()
    @reactive.event(input.orientation)
    def reoriente():
        print("reoriente")
        instances = Src_inst.get_instances(Src_inst.list_instances())
        Src_datas = [
            hp.update_orientation(new_orientation_v=input.orientation(), target="dataF")
            for hp in instances
        ]
        data = sum(Src_datas)
        update_data.set("orientation")
        return data

    @reactive.calc()
    def get_source_info():
        New_inst()
        x = [
            Src_inst.get_instance(_id).position_v[0]
            for _id in Src_inst.list_instances()
        ]
        y = [
            Src_inst.get_instance(_id).position_v[1]
            for _id in Src_inst.list_instances()
        ]
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        Infos = [
            f"ID: {SrcArgs[ii]["ID"]} <br>Dir: {SrcArgs[ii]["directivity"]} <br>G: {SrcArgs[ii]["gain"]} <br>Deg: {SrcArgs[ii]["orientation"]}"
            for ii in range(len(SrcArgs))
        ]
        return x, y, Infos

    @reactive.calc()
    def compute_field():
        New_inst()
        instances = Src_inst.get_instances(Src_inst.list_instances())
        Src_datas = [
            hp.resp_for_f(freq=input.target_freq(), reshape=True) for hp in instances
        ]
        data = sum(Src_datas)
        update_data.set("create_instance")
        return data

    @render_widget
    @reactive.event(input.create_instance, input.orientation, ignore_init=True)
    def plot_field():

        if update_data == "orientation":
            data = reoriente()

        elif update_data == "create_instance":
            data = compute_field()

        _, _, x, y = define_grid()
        # Once the plotting will be clean -> function to do contour plot inside graphing.py

        fig = go.Figure(
            data=go.Contour(
                z=SpatialObject._Lp(data),
                x=x,
                y=y,
                colorbar=dict(title=dict(text=r"Pressure level (dB re 2e-5 Pa)")),
            )
        )
        xS, yS, Infos = get_source_info()

        fig.add_trace(
            go.Scatter(x=xS, y=yS, hovertext=Infos, hoverinfo="text", mode="markers")
        )  # Should change using reactive value instead
        # Params to twick : labelfont(color), coloring("heatmap","lines","fill"), .. colorscale
        widget = go.FigureWidget(fig.data, fig.layout)
        return widget

    @render.table
    def instance_table():
        New_inst()
        if len(Src_inst.list_instances()) > 0:
            df = pd.DataFrame(
                [
                    {
                        "ID": _id,
                        "X": Src_inst.get_instance(_id).position_v[0],
                        "Y": Src_inst.get_instance(_id).position_v[1],
                    }
                    for _id in Src_inst.list_instances()
                ]
            )

        else:
            df = pd.DataFrame(columns=["ID", "X", "Y"])
        return df


app = App(app_ui, server)


# plotly graph
# @render_widget()
# def plot_field():
#     New_inst()
#     if len(Src_inst.list_instances()) == 0:
#         X, Y = define_grid()

#         fig = go.Figure(
#             data=go.Contour(z=np.zeros_like(X), x=np.unique(X), y=np.unique(Y))
#         )
#         return fig
#     data = compute_field()
#     X, Y = define_grid()
#     fig = go.Figure(data=go.Contour(z=SpatialObject._Lp(data), x=X, y=Y))
#     return fig
# output_widget("plot_field"),
