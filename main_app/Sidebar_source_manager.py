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
from pathlib import Path

pio.templates.default = "ECS"

fs = 44100
grid_resolution = 100
n_fft = 128
data_m = np.zeros((grid_resolution**2, n_fft))


app_ui = ui.page_fluid(
    ui.include_css(path=Path(__file__).parent / "www" / "custom.css"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Field settings",
                    ui.input_numeric("x_length", "Width", value=10, min=1, max=100),
                    ui.input_numeric("y_length", "Length", value=10, min=1, max=100),
                    ui.input_numeric(
                        "target_freq", "Frequency", value=1000, min=20, max=20000
                    ),
                ),
                ui.accordion_panel(
                    "Source settings",
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
                        ui.input_slider(
                            "posX", "x (m)", value=0, min=-10 / 2, max=10 / 2
                        ),
                        ui.input_slider(
                            "posY", "y (m)", value=0, min=-10 / 2, max=10 / 2
                        ),
                        width=1 / 2,
                    ),
                ),
                multiple=False,
                open=False,
            ),
            ui.card(
                ui.input_action_button("create_instance", "New Source"),
                ui.popover(
                    ui.input_action_button("del_instance", "Delete Source"),
                    ui.p(
                        "Select the source you whant to delete (helper: click on the ID of the source you want to delete)"
                    ),
                    ui.input_text(
                        id="DelID",
                        label="Source to delete",
                        placeholder="ID of source",
                    ),
                    ui.input_action_button("confirm_delete", "Confirm"),
                ),
            ),
            ui.card(
                ui.card_header(
                    "Sources Attribute",
                ),
                ui.output_data_frame("instance_table"),
                full_screen=True,
            ),
            width=250,
        ),
        # Main Card
        ui.layout_column_wrap(
            ui.card(
                ui.card_header(
                    "display field",
                    ui.input_dark_mode(mode="light"),
                    class_="d-flex justify-content-between align-items-center",
                ),
                output_widget("plot_field"),
                style="width: 600px; height: 600px;",
                class_="resizable-card",
                full_screen=True,
            ),
            ui.card(
                ui.p("her"),
                class_="resizable-card",
            ),
            width=1 / 2,
        ),
    ),
    class_="p-3",
)


def server(input, output, session):

    Src_inst = SourcesManager()

    New_inst = reactive.Value(0)
    attr = Src_inst.attributes.columns.to_list()
    Attribute_dict = {}
    for ii in range(len(attr)):
        Attribute_dict[attr[ii]] = reactive.Value(0)

    attributes_changed = reactive.Value(0)
    rerun = reactive.Value(0)

    Del_sources = reactive.Value(0)

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
            deg=True,
        )
        # Reseting source setting
        ui.update_numeric("radius", label="Radius (m)", value=0.1, min=0.01, max=1)
        ui.update_numeric(
            "orientation", label="Orientation (deg)", value=0.0, min=0.0, max=360
        )
        ui.update_slider("posX", label="x (m)", value=0, min=-10 / 2, max=10 / 2),
        ui.update_slider("posY", label="y (m)", value=0, min=-10 / 2, max=10 / 2),

        New_inst.set(New_inst() + 1)
        attributes_changed.set(attributes_changed() + 1)

    @reactive.effect()
    @reactive.event(Attribute_dict["orientation"], ignore_init=True)
    def reoriente():

        instances = Src_inst.get_instances(Src_inst.list_instances())
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        for ii, hp in enumerate(instances):
            hp.orientation_v[0] = SrcArgs[ii]["orientation"]
            hp.set_directivity(SrcArgs[ii]["directivity"])
        rerun.set(rerun() + 1)

    @reactive.effect()
    @reactive.event(Attribute_dict["x"], Attribute_dict["y"], ignore_init=True)
    def update_pos():

        print("position")
        instances = Src_inst.get_instances(Src_inst.list_instances())
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        for ii, hp in enumerate(instances):
            hp.position_v[:2] = [SrcArgs[ii]["x"], SrcArgs[ii]["y"]]
            print("position of the source", hp.position_v[:2])
            hp.set_directivity(SrcArgs[ii]["directivity"])
        rerun.set(rerun() + 1)

    @reactive.calc()
    def get_source_info():
        New_inst()
        rerun()
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
        print("field computed")
        New_inst()
        Del_sources()
        rerun()

        instances = Src_inst.get_instances(Src_inst.list_instances())
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        for ii in range(len(instances)):
            print(SrcArgs[ii]["orientation"])
        Src_datas = [
            hp.resp_for_f(
                freq=input.target_freq(),
                reshape=True,
                storedArgs=Src_inst.Stored.iloc[ii],
            )
            for ii, hp in enumerate(instances)
        ]
        data = sum(Src_datas)
        return data

    @reactive.effect()
    @reactive.event(input.confirm_delete, ignore_init=True)
    def del_source():
        _id = input.DelID()
        Src_inst.remove_instance(_id)
        ui.update_text(
            id="DelID", label="Source to delete", placeholder="ID of source", value=""
        )
        attributes_changed.set(attributes_changed() + 1)
        Del_sources.set(Del_sources() + 1)

    @render_widget
    @reactive.event(
        input.create_instance,
        input.confirm_delete,
        Attribute_dict["orientation"],
        Attribute_dict["x"],
        Attribute_dict["y"],
        ignore_init=True,
    )
    def plot_field():
        if New_inst.get() == 0:
            return
        data = compute_field()

        _, _, x, y = define_grid()
        # Once the plotting will be clean -> function to do contour plot inside graphing.py
        if isinstance(data, (list, np.ndarray)):
            fig = go.Figure()
            fig.add_trace(
                go.Contour(
                    z=SpatialObject._Lp(data),
                    x=x,
                    y=y,
                    colorbar=dict(title=dict(text=r"Pressure level (dB re 2e-5 Pa)")),
                )
            )
            xS, yS, Infos = get_source_info()

            fig.add_trace(
                go.Scatter(
                    x=xS, y=yS, hovertext=Infos, hoverinfo="text", mode="markers"
                )
            )
            fig.update_layout(
                xaxis=dict(
                    range=[x.min(), x.max()],
                    # scaleanchor="x",
                    # scaleratio=1,
                    # constrain="domain",
                ),
                yaxis=dict(
                    range=[y.min(), y.max()],
                    # scaleanchor="x",
                    # scaleratio=1,
                    # constrain="domain",
                ),
                margin=dict(l=30, b=30),
            )
            # Should change using reactive value instead
            # Params to twick : labelfont(color), coloring("heatmap","lines","fill"), .. colorscale
            widget = go.FigureWidget(fig.data, fig.layout)
            return widget
        else:
            return

    @render.data_frame
    def instance_table():
        attributes_changed()
        return render.DataTable(
            Src_inst.attributes,
            editable=True,
            selection_mode="row",
            height="200px",
        )

    @instance_table.set_patch_fn
    def _(*, patch: render.CellPatch):
        update_data_with_patch(patch)
        return patch["value"]

    def update_data_with_patch(patch):
        col_idx = patch["column_index"]
        col_name = Src_inst.attributes.columns[col_idx]
        if col_idx == 0:
            fn = str
        elif col_name in ["ID"]:
            # ui.update_popover("DelSourcePopover", show=True)
            fn = str

        elif col_name in ["directivity"]:
            fn = str
        elif col_name in ["fmin", "fmax"]:
            fn = int
        else:
            fn = float
        Src_inst.Stored = Src_inst.attributes.copy()
        Src_inst.attributes.iat[patch["row_index"], patch["column_index"]] = fn(
            patch["value"]
        )

        Attribute_dict[col_name].set(Src_inst.attributes[col_name])
        attributes_changed.set(attributes_changed() + 1)


app = App(app_ui, server)


# New_inst()
# if len(Src_inst.list_instances()) > 0:
#     df = pd.DataFrame(
#         [
#             {
#                 "ID": _id,
#                 "X": Src_inst.get_instance(_id).position_v[0],
#                 "Y": Src_inst.get_instance(_id).position_v[1],
#             }
#             for _id in Src_inst.list_instances()
#         ]
#     )

# else:
#     df = pd.DataFrame(columns=["ID", "X", "Y"])
# return df


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


# Reactive value that reset the calculation
# Srcs_state = reactive.value({"create_instance": 0, "orientation": 0})


# Might need to be a function appart -> get_the_data_from_right_func
# state_create_instance = input.create_instance()
# state_orientation = input.orientation()
# stored = Srcs_state.get()
# if state_orientation != stored["orientation"]:
#     print("plot field: orientation", state_orientation)
#     if stored["create_instance"] > 0:

#     else:
#         return

# Srcs_state.set(
#     {"create_instance": state_create_instance, "orientation": state_orientation}
# )
