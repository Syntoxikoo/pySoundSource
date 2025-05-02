# This script contain the front end of the side bar aswell as the back end of the class sources manager
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
from Tools.SourcesManager import SourcesManager
from Tools.SpatialObject import SpatialObject
import plotly.graph_objects as go
import plotly.io as pio
import Tools.graphing.mpl_template
import numpy as np
from pathlib import Path
from Tools.icon_script import *
from scipy import signal

pio.templates.default = "ECS"

fs = 16000


app_ui = ui.page_fluid(
    ui.include_css(path=Path(__file__).parent / "www" / "custom.css"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Field settings",
                    ui.input_numeric(
                        "fs",
                        "Sampling Frequency (Hz)",
                        value=16000,
                        min=8000,
                        max=48000,
                        step=8000,
                    ),
                    ui.layout_column_wrap(
                        ui.input_numeric("x_length", "Width", value=10, min=1, max=100),
                        ui.input_numeric(
                            "y_length", "Length", value=10, min=1, max=100
                        ),
                        width=1 / 2,
                    ),
                    ui.input_numeric(
                        "target_freq",
                        label="Frequency (Hz)",
                        value=1000,
                        min=20,
                        max=20000,
                        step=20,
                    ),
                    ui.input_select(
                        "nfft",
                        label="Nfft",
                        choices={
                            "64": 64,
                            "128": 128,
                            "256": 256,
                            "512": 512,
                            "1024": 1024,
                            "2048": 2048,
                        },
                        selected="128",
                    ),
                    ui.input_slider(
                        "grid_res",
                        label="grid precision",
                        min=10,
                        value=100,
                        max=200,
                        step=10,
                        ticks=True,
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
                    ui.input_slider(
                        "set_delay",
                        label="Delay (ms)",
                        min=0,
                        value=0,
                        max=20,
                        step=0.1,
                    ),
                    ui.input_slider(
                        "set_f_range",
                        label="range",
                        min=20,
                        value=[20, 20000],
                        max=20000,
                        step=1,
                    ),
                ),
                multiple=False,
                open=False,
            ),
            ui.card(
                ui.input_action_button(
                    "create_instance",
                    "New Source",
                    icon=instanceIcon,
                    class_="d-flex justify-content-between align-items-center",
                ),
                ui.popover(
                    ui.input_action_button(
                        "del_instance",
                        "Delete Source",
                        icon=DelInstanceIcon,
                        class_="d-flex justify-content-between align-items-center",
                    ),
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
        ui.layout_columns(
            ui.card(
                ui.card_header(
                    "display field",
                    ui.input_dark_mode(id="darkmode", mode="light"),
                    class_="d-flex justify-content-between align-items-center",
                ),
                output_widget("plot_field"),
                class_="displayfield-card",
                full_screen=True,
                max_height=600,
            ),
            ui.card(
                ui.value_box(
                    "Happy Birthday",
                    f"{20} years",
                    "Ruben <3",
                    showcase=birthdayIcon,
                    theme="bg-gradient-blue-purple",
                    showcase_layout="top right",
                ),
                # class_="displayfield-card",
                ui.output_ui("pressure_axis_box"),
                fill=False,
            ),
            col_widths=[7, 5],
            fixed_width=True,
        ),
    ),
    class_="p-3",
)


def server(input, output, session):

    Src_inst = SourcesManager()

    # -- REACTIVITY --
    New_inst = reactive.Value(0)
    attr = Src_inst.attributes.columns.to_list()
    Attribute_dict = {}
    for ii in range(len(attr)):
        Attribute_dict[attr[ii]] = reactive.Value(0)

    attributes_changed = reactive.Value(0)
    rerun = reactive.Value(0)
    target_pos = reactive.Value([3, 0])
    Del_sources = reactive.Value(0)
    p_axis = reactive.Value(0)

    # ----------------

    @reactive.calc()
    def define_grid():
        x = np.linspace(-input.x_length() / 2, input.x_length() / 2, input.grid_res())
        y = np.linspace(-input.y_length() / 2, input.y_length() / 2, input.grid_res())
        X, Y = np.meshgrid(x, y)

        return X, Y, x, y

    @reactive.effect()
    @reactive.event(input.create_instance)
    def instanciate():

        X, Y, _, _ = define_grid()
        Src_inst.create_instance(
            f"{input.sel_dir()[0].upper()}{New_inst.get()}",
            SpatialObject,
            fs=input.fs(),
            dist_v=0,
            norm_s="cartesian",
            data_m=np.zeros((input.grid_res() ** 2, int(input.nfft()))),
            radius=input.radius(),
            position_v=[input.posX(), input.posY(), 0],
            orientation_v=[input.orientation(), 0],
            azim_v=X.flatten(),
            elev_v=Y.flatten(),
            directivity=input.sel_dir(),
            src_resp=1.0,
            deg=True,
            n_fft=int(input.nfft()),
            delay=input.set_delay() * 1e-3,
        )

        # Reseting source setting
        ui.update_numeric("radius", label="Radius (m)", value=0.1, min=0.01, max=1)
        ui.update_numeric(
            "orientation", label="Orientation (deg)", value=0.0, min=0.0, max=360
        )
        ui.update_slider("posX", label="x (m)", value=0, min=-10 / 2, max=10 / 2)
        ui.update_slider("posY", label="y (m)", value=0, min=-10 / 2, max=10 / 2)

        # -- REACTIVITY --
        New_inst.set(New_inst() + 1)
        attributes_changed.set(attributes_changed() + 1)
        # ----------------

    @reactive.effect()
    @reactive.event(input.confirm_delete, ignore_init=True)
    def del_source():
        _id = input.DelID()
        Src_inst.remove_instance(_id)
        ui.update_text(
            id="DelID", label="Source to delete", placeholder="ID of source", value=""
        )
        # -- REACTIVITY --
        attributes_changed.set(attributes_changed() + 1)
        Del_sources.set(Del_sources() + 1)
        # ----------------

    @reactive.calc()
    def get_source_info():
        # ---- TRIGER ----
        New_inst()
        rerun()
        Del_sources()
        # ----------------
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

        # -- REACTIVITY --
        rerun.set(rerun() + 1)
        # ----------------

    @reactive.effect()
    @reactive.event(Attribute_dict["x"], Attribute_dict["y"], ignore_init=True)
    def update_pos():

        instances = Src_inst.get_instances(Src_inst.list_instances())
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        for ii, hp in enumerate(instances):
            hp.position_v[:2] = [SrcArgs[ii]["x"], SrcArgs[ii]["y"]]
            hp.set_directivity(SrcArgs[ii]["directivity"])

        # -- REACTIVITY --
        rerun.set(rerun() + 1)
        # ----------------

    @reactive.effect()
    @reactive.event(Attribute_dict["delay"], ignore_init=True)
    def update_delay():

        instances = Src_inst.get_instances(Src_inst.list_instances())
        SrcArgs = [
            Src_inst.get_attributes_dict(_id) for _id in Src_inst.list_instances()
        ]
        for ii, hp in enumerate(instances):
            hp.delay = SrcArgs[ii]["delay"]

        # -- REACTIVITY --
        rerun.set(rerun() + 1)
        # ----------------

    @reactive.effect()
    @reactive.event(
        input.fs,
        input.grid_res,
        input.nfft,
        input.x_length,
        input.y_length,
        ignore_init=True,
    )
    def update_fieldParams():

        instances = Src_inst.get_instances(Src_inst.list_instances())
        X, Y, _, _ = define_grid()
        print(input.y_length.get())

        for ii, hp in enumerate(instances):
            hp.fs = input.fs()
            hp.n_fft = int(input.nfft())
            hp.azim_v = X.flatten()
            hp.elev_v = Y.flatten()

        # -- REACTIVITY --
        rerun.set(rerun() + 1)
        # ----------------

    @reactive.calc()
    def compute_field():
        # ---- TRIGER ----
        New_inst()
        Del_sources()
        rerun()
        # ----------------

        instances = Src_inst.get_instances(Src_inst.list_instances())

        Src_datas = [
            hp.resp_for_f(
                freq=input.target_freq(),
                reshape=True,
                storedArgs=Src_inst.Stored.iloc[ii],
                FieldSettings=Src_inst.FieldSettings,
            )
            for ii, hp in enumerate(instances)
        ]
        data = sum(Src_datas)

        return data

    @reactive.calc()
    @reactive.event(input.set_f_range, ignore_init=True)
    def compute_filter():
        pass

    @reactive.calc()
    def compute_pos():
        # ---- TRIGER ----
        New_inst()
        Del_sources()
        rerun()
        # ----------------

        instances = Src_inst.get_instances(Src_inst.list_instances())
        Src_datas = [
            hp.resp_for_M(
                target_pos=target_pos.get(),
                storedArgs=Src_inst.Stored.iloc[ii],
                FieldSettings=Src_inst.FieldSettings,
            )
            for ii, hp in enumerate(instances)
        ]
        data = sum(Src_datas)
        p_axis.set(np.mean(SpatialObject._Lp(data)))

        return data

    @output
    @render.ui
    @reactive.event(p_axis)
    def pressure_axis_box():
        return ui.value_box(
            "Pressure in axis at 3m",
            f"{p_axis():.1f} dB SPL",
            output_widget("plot_OV_axis"),
            id="pAxisBox",
            showcase=sound_pressureIcon,
            theme="bg-orange",
            showcase_layout="top right",
        )

    @render_widget
    @reactive.event(
        input.create_instance,
        input.confirm_delete,
        input.target_freq,
        input.darkmode,
        input.nfft,
        input.grid_res,
        input.fs,
        input.x_length,
        input.y_length,
        Attribute_dict["orientation"],
        Attribute_dict["x"],
        Attribute_dict["y"],
        Attribute_dict["delay"],
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
                    zmax=100,
                    zmin=50,
                )
            )

            xS, yS, Infos = (
                get_source_info()
            )  # Should change using reactive value instead ?

            fig.add_trace(
                go.Scatter(
                    x=xS,
                    y=yS,
                    hovertext=Infos,
                    hoverinfo="text",
                    mode="markers",
                    showlegend=False,
                )
            )
            micInfo = " <b>Mic in axis</b>"

            fig.add_trace(
                go.Scatter(
                    x=np.array(target_pos.get()[0]),
                    y=np.array(target_pos.get()[1]),
                    hovertext=micInfo,
                    hoverinfo="text",
                    mode="markers",
                    marker_symbol="circle-x",
                    marker_line_color="#F45100",
                    marker_color=None,
                    marker_line_width=2,
                    showlegend=False,
                )
            )
            fig.update_layout(
                xaxis=dict(
                    range=[x.min(), x.max()],
                ),
                yaxis=dict(
                    range=[y.min(), y.max()],
                ),
                margin=dict(l=30, b=30),
                height=580,
                width=580,
            )

            if input.darkmode() == "dark":
                fig.update_layout(
                    paper_bgcolor="#1D1F21",
                    plot_bgcolor="#1D1F21",
                )
            # Params to twick : labelfont(color), coloring("heatmap","lines","fill"), .. colorscale
            widget2 = go.FigureWidget(fig.data, fig.layout)
            return widget2
        else:
            return

    @render_widget
    @reactive.event(
        input.create_instance,
        input.confirm_delete,
        input.target_freq,
        input.darkmode,
        input.nfft,
        input.grid_res,
        input.fs,
        input.x_length,
        input.y_length,
        Attribute_dict["orientation"],
        Attribute_dict["x"],
        Attribute_dict["y"],
        Attribute_dict["delay"],
        ignore_init=True,
    )
    def plot_OV_axis():
        if New_inst.get() == 0:
            return

        data = compute_pos()
        # return

        if isinstance(data, (list, np.ndarray)):
            instances = Src_inst.get_instances(Src_inst.list_instances())
            xf = instances[0].xaxis_v
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=xf,
                    y=SpatialObject._Lp(data),
                    mode="lines",
                    line=dict(color="white"),
                    hovertemplate="%{y:.1f} dB" + "<extra></extra>",
                )
            )
            fig.update_layout(
                hovermode="x",
                height=150,
                width=200,
                margin=dict(l=30, b=30),
                xaxis=dict(
                    type="log",
                    tickfont=dict(color="black"),
                    showgrid=False,
                    zeroline=False,
                    minor=dict(showgrid=False),
                    showline=True,
                    mirror=False,
                ),
                yaxis=dict(
                    tickformat=".0f",
                    tickfont=dict(color="black"),
                    range=([p_axis.get() - 40, p_axis.get() + 10]),
                    showgrid=False,
                    zeroline=False,
                    minor=dict(showgrid=False),
                    showline=True,
                    mirror=False,
                ),
                paper_bgcolor="#F45100",
                plot_bgcolor="#F45100",
            )
            widget = go.FigureWidget(fig.data, fig.layout)
            return widget
        else:
            return

    @render.data_frame
    def instance_table():
        # ---- TRIGER ----
        attributes_changed()
        # ----------------
        return render.DataTable(
            Src_inst.attributes,
            editable=True,
            selection_mode="row",
            height="130px",
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
