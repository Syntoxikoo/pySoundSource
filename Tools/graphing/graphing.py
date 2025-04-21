import datetime
import math
import os
import plotly.io as pio
from ICEInstrumentation.tools.mpl_template import *
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from ..tools import logging_functions

pio.templates.default = "matplotlib"


def visualize(t, input_signal, output_signals):
    n_signals = np.shape(output_signals)[1]
    if n_signals == 1:
        plt.plot(t, input_signal)
        plt.plot(t, output_signals)
    else:
        if n_signals == 2:
            shape = [2, 1]
        elif n_signals == 3:
            shape = [3, 1]
        elif n_signals == 4:
            shape = [2, 2]
        elif n_signals <= 9:
            shape = [3, int(n_signals / 3)]
        else:
            shape = [4, 4]
        fig, axs = plt.subplots(shape[0], shape[1], sharex=True, sharey=True)
        for itt, ax_row in enumerate(axs):
            if shape[1] == 1:
                signal = output_signals[:, itt]
                ax_row.plot(t, input_signal)
                ax_row.plot(t, signal)
                channel = itt
                zone = math.ceil(channel / 2)
                lr = "l" if channel % 2 == 1 else "r"
                ax_row.set_title(f"{zone}{lr}")
            else:
                for i, (ax, signal) in enumerate(
                    zip(
                        ax_row,
                        output_signals.T[itt * shape[0] : itt * shape[0] + shape[1]],
                    )
                ):
                    ax.plot(t, input_signal)
                    ax.plot(t, signal)
                    channel = itt * shape[0] + i + 1
                    zone = math.ceil(channel / 2)
                    lr = "l" if channel % 2 == 1 else "r"
                    ax.set_title(f"{zone}{lr}")
    plt.show()


def save_image(
    x, ys, name, path=".", show=False, fig_size_x=6.4, fig_size_y=4.8, **kwargs
):
    """
    TODO : documentation"""
    logging_level = logging_functions._disable_logger()

    if "method" in kwargs and kwargs["method"] == "plotly":

        ppi = 96  # pixel per inch
        fig_plotly = go.Figure(
            layout=go.Layout(width=fig_size_x * ppi, height=fig_size_y * ppi)
        )
        _config_plotly(fig_plotly, x, ys, **kwargs)
        fig = None
    elif "method" in kwargs and kwargs["method"] == "mpl":
        fig = plt.figure(figsize=(fig_size_x, fig_size_y))
        _configure_image(plt.gca(), x, ys, **kwargs)
        fig_plotly = None

    else:
        ppi = 96  # pixel per inch
        fig_plotly = go.Figure(
            layout=go.Layout(width=fig_size_x * ppi, height=fig_size_y * ppi)
        )
        fig = plt.figure(figsize=(fig_size_x, fig_size_y))
        _configure_image(plt.gca(), x, ys, **kwargs)
        _config_plotly(fig_plotly, x, ys, **kwargs)

    if not os.path.exists(path):
        os.makedirs(path)

    if name.endswith(".html"):
        if fig_plotly is not None:

            dir = Path(__file__).parent
            path_js = Path.joinpath(dir, "plotly_3_0_1.min.js")

            fig_plotly.write_html(
                os.path.join(path, name),
                include_plotlyjs=str(path_js),
            )
        if fig is not None:
            mpld3.save_html(fig, os.path.join(path, name))
    else:
        if fig is None:
            raise TypeError("plotly figure cannot be saved as png")
        plt.savefig(os.path.join(path, name))
    if show:
        if fig_plotly is not None:
            fig_plotly.show()
        if fig is not None:
            plt.show()
    plt.close("all")
    logging_functions._enable_logger(logging_level)


def save_image_stacked(
    graph_sets: list[tuple],
    name: str,
    path=".",
    title=None,
    show=False,
    fig_size_x=6.4,
    fig_size_y=4.8,
    method: str = "both",
):
    try:
        logging_level = logging_functions._disable_logger()
        if not os.path.exists(path):
            os.makedirs(path)

        if method == "plotly":
            fig = None
            fig_plotly = make_subplots(
                rows=len(graph_sets),
                cols=1,
                subplot_titles=[kwargs["title"] for _, _, kwargs in graph_sets],
            )
            fig_plotly.update_layout(
                margin=dict(l=20, r=20, t=40, b=40),
            )
            ii = 1
            for x, ys, kwargs in graph_sets:
                kwargs["row"] = ii

                return_values = _config_plotly(fig_plotly, x, ys, **kwargs)
                ii += 1
        elif method == "mpl":
            fig_plotly = None
            fig, axs = plt.subplots(len(graph_sets), figsize=(fig_size_x, fig_size_y))
            if title is not None:
                plt.suptitle(title, wrap=True)
            if not isinstance(
                axs, np.ndarray
            ):  # single graph set passed - do a regular plot
                axs = np.array([axs])

            for ax, (x, ys, kwargs) in zip(axs, graph_sets):
                return_values = _configure_image(ax, x, ys, **kwargs)

            fig.tight_layout()
        else:
            fig, axs = plt.subplots(len(graph_sets), figsize=(fig_size_x, fig_size_y))
            fig_plotly = make_subplots(
                rows=len(graph_sets),
                cols=1,
                subplot_titles=[kwargs["title"] for _, _, kwargs in graph_sets],
            )

            fig_plotly.update_layout(
                margin=dict(l=20, r=20, t=40, b=40),
            )
            if title is not None:
                plt.suptitle(title, wrap=True)
            if not isinstance(
                axs, np.ndarray
            ):  # single graph set passed - do a regular plot
                axs = np.array([axs])
            ii = 1
            for ax, (x, ys, kwargs) in zip(axs, graph_sets):
                kwargs["row"] = ii
                return_values = _configure_image(ax, x, ys, **kwargs)
                return_values = _config_plotly(fig_plotly, x, ys, **kwargs)
                ii += 1
            fig.tight_layout()

        if name.endswith(".html"):
            if fig_plotly is not None:
                dir = Path(__file__).parent
                path_js = Path.joinpath(dir, "plotly_3_0_1.min.js")
                fig_plotly.write_html(
                    os.path.join(path, name),
                    include_plotlyjs=str(path_js),
                )
            if fig is not None:
                mpld3.save_html(fig, os.path.join(path, name))
        else:
            if fig is None:
                raise TypeError("plotly figure cannot be saved as png")
            plt.savefig(os.path.join(path, name))
        if show:
            if fig_plotly is not None:
                fig_plotly.show()
            if fig is not None:
                plt.show()
        plt.close("all")
        logging_functions._enable_logger(logging_level)
        return return_values
    except Exception as e:
        logging_functions._enable_logger(logging_level)
        raise e


def _config_plotly(fig: go.Figure, x, ys, **kwargs):
    """
    Configure a Plotly figure from desired design.

    Parameters:
    - fig: plotly.graph_objects.Figure - The figure to configure
    - x: array-like - x-axis data
    - ys: list of array-like - List of y-axis data series
    - **kwargs: Various options

    Returns:
    - Trace added, or figure if no traces were added
    """

    if isinstance(ys, np.ndarray):
        if ys.ndim < 2:
            ys = [ys]
        else:
            ys = ys.T
    elif not isinstance(ys, (list, tuple)):
        ys = [ys]

    if "grid" in kwargs and kwargs["grid"]:
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
    else:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    if "minor" in kwargs and kwargs["minor"]:
        fig.update_xaxes(
            minor=dict(showgrid=True, ticklen=2),
        )
    else:
        fig.update_xaxes(
            minor=dict(showgrid=False, ticklen=0),
        )
    colors = kwargs.get("colors", [None] * len(ys))
    markers = kwargs.get("markers", [None] * len(ys))
    linestyles = kwargs.get("linestyles", [None] * len(ys))
    linewidths = kwargs.get("linewidth", [None] * len(ys))

    if "mask" in kwargs:
        masks = kwargs["mask"]
        if not isinstance(masks, list):
            masks = [masks] * len(ys)
        for itt, mask in enumerate(masks):
            if mask is None:
                masks[itt] = np.array([True] * len(x))
    else:
        masks = [np.array([True] * len(x))] * len(ys)

    marker_mapping = {
        ".": "circle",
        "o": "circle",
        "v": "triangle-down",
        "^": "triangle-up",
        "<": "triangle-left",
        ">": "triangle-right",
        "s": "square",
        "p": "pentagon",
        "*": "star",
        "h": "hexagon",
        "+": "cross",
        "x": "x",
        "D": "diamond",
        "d": "diamond",
    }

    line_type_mapping = {
        "-": "solid",
        "--": "dash",
        ":": "dot",
        "-.": "dashdot",
        "None": None,
    }

    traces = []
    for y, color, marker, linestyle, linewidth, mask in zip(
        ys, colors, markers, linestyles, linewidths, masks
    ):

        x_data = np.array(x)[mask]
        y_data = np.array(y).flatten()[mask]

        line_dict = {}
        if color:
            line_dict["color"] = color
        if linewidth is not None:
            line_dict["width"] = linewidth
        if linestyle is not None:
            plotly_dash = line_type_mapping.get(linestyle)
            if plotly_dash is not None:
                line_dict["dash"] = plotly_dash

        marker_dict = {}
        if marker:
            marker_dict["symbol"] = marker_mapping.get(marker, "circle")
            if color:
                marker_dict["color"] = color

        # To trace either the marker, the line or both
        mode_parts = []
        if linestyle not in (None, "None", ""):
            mode_parts.append("lines")
        if marker is not None:
            mode_parts.append("markers")
        mode_str = "+".join(mode_parts) if mode_parts else "lines"

        trace = go.Scatter(
            x=x_data,
            y=y_data,
            line=line_dict,
            marker=marker_dict,
            mode=mode_str,
            showlegend=True if "legend" in kwargs else False,
        )
        if "row" in kwargs:
            fig.add_trace(trace, row=kwargs["row"], col=1)
        else:
            fig.add_trace(trace)
        traces.append(trace)

    # trace some value point : point should be defined as a tuple with coordinate in list
    if "point" in kwargs:
        point_kwargs = kwargs.get("point_kwargs", {})

        for i, pointxy in enumerate(kwargs["point"]):
            fig.add_trace(
                go.Scatter(
                    x=[pointxy[0]],
                    y=[pointxy[1]],
                    mode="markers",
                    marker=dict(
                        color=point_kwargs.get("color", "red"),
                        size=point_kwargs.get("s", 8),
                        symbol=marker_mapping[point_kwargs.get("marker", "x")],
                    ),
                    showlegend=False,
                )
            )
    if "annotation" in kwargs:
        annotation_kwargs = kwargs.get("annotation_kwargs", {})
        if isinstance(kwargs["annotation"], tuple):
            for i in range(len(kwargs["annotation"])):
                try:
                    xoffset = annotation_kwargs.get("xoffset", [20])[i]
                    yoffset = annotation_kwargs.get("yoffset", [40])[i]
                except (IndexError, TypeError):
                    xoffset = 20
                    yoffset = 40
                fig.add_annotation(
                    x=kwargs["point"][i][0],  # might be changed for more flexibility
                    y=kwargs["point"][i][1],
                    text=kwargs["annotation"][i],
                    showarrow=annotation_kwargs.get("showarrow", True),
                    ax=xoffset,
                    ay=yoffset,
                )
        else:
            fig.add_annotation(
                x=kwargs["point"][0][0],
                y=kwargs["point"][0][1],
                text=kwargs["annotation"],
                showarrow=annotation_kwargs.get("showarrow", True),
                ax=annotation_kwargs.get("xoffset", 20),
                ay=annotation_kwargs.get("yoffset", 40),
            )

    if "semilogx" in kwargs and kwargs["semilogx"]:
        fig.update_xaxes(type="log")

    if "log" in kwargs:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")

    layout_updates = {}
    if "title" in kwargs and "row" not in kwargs:
        layout_updates["title"] = kwargs["title"]

    if "xlimit" in kwargs or "xlim" in kwargs:
        rangeax = kwargs.get("xlimit", kwargs.get("xlim"))
        if "semilogx" in kwargs and kwargs["semilogx"]:
            if rangeax[0] == 0:
                rangeax[0] += min(abs(x_data[1] - x_data[0]), 1)
            rangeax = np.log10(rangeax)
        elif "log" in kwargs:
            if rangeax[0] == 0:
                rangeax[0] += min(abs(x_data[1] - x_data[0]), 1)
            rangeax = np.log10(rangeax)

        fig.update_xaxes(range=rangeax)
    if "ylim" in kwargs or "ylimit" in kwargs:
        rangeax = kwargs.get("ylimit", kwargs.get("ylim"))
        if "log" in kwargs:
            if rangeax[0] == 0:
                rangeax[0] += min(abs(x_data[1] - x_data[0]), 1)
            rangeax = np.log10(rangeax)
        fig.update_yaxes(range=rangeax)
    if "row" in kwargs:
        if "ylabel" in kwargs:
            fig.update_yaxes(title_text=kwargs["ylabel"], row=kwargs["row"], col=1)
        if "xlabel" in kwargs:
            fig.update_xaxes(
                title_text=kwargs["xlabel"], row=kwargs["row"], col=1, title_standoff=0
            )
    else:
        if "ylabel" in kwargs:
            layout_updates["yaxis_title"] = kwargs["ylabel"]
        if "xlabel" in kwargs:
            layout_updates["xaxis_title"] = kwargs["xlabel"]

    if "vline" in kwargs:
        if isinstance(kwargs["vline"], list):
            for line, color in kwargs["vline"]:
                fig.add_shape(
                    type="line",
                    x0=line,
                    x1=line,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color=color, dash="dot"),
                )
        elif isinstance(kwargs["vline"], tuple):
            fig.add_shape(
                type="line",
                x0=kwargs["vline"][0],
                x1=kwargs["vline"][0],
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color=kwargs["vline"][1], dash="dot"),
            )
        else:
            fig.add_shape(
                type="line",
                x0=kwargs["vline"],
                x1=kwargs["vline"],
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", dash="dot"),
            )

    if "hline" in kwargs:
        if isinstance(kwargs["hline"], list):
            for line, hline_kwargs in kwargs["hline"]:
                fig.add_shape(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=line,
                    y1=line,
                    xref="paper",
                    line=dict(
                        color=hline_kwargs.get("color", "blue"),
                        dash="dot" if hline_kwargs.get("linestyle") == ":" else "solid",
                        width=hline_kwargs.get("linewidth", 1),
                    ),
                )
        elif isinstance(kwargs["hline"], tuple):
            hline_value, hline_color = kwargs["hline"][0], kwargs["hline"][1]
            hline_style = "solid"
            if len(kwargs["hline"]) > 2:
                if kwargs["hline"][2] == ":":
                    hline_style = "dot"
                elif kwargs["hline"][2] == "--":
                    hline_style = "dash"
            hline_width = kwargs["hline"][3] if len(kwargs["hline"]) > 3 else 1

            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=hline_value,
                y1=hline_value,
                xref="paper",
                line=dict(color=hline_color, dash=hline_style, width=hline_width),
            )
        else:
            fig.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=kwargs["hline"],
                y1=kwargs["hline"],
                xref="paper",
                line=dict(color="red", dash="dot"),
            )

    if "axvspan" in kwargs:
        axvspans = (
            kwargs["axvspan"]
            if isinstance(kwargs["axvspan"], list)
            else [kwargs["axvspan"]]
        )
        axvspans_kwargs = kwargs.get("axvspan_kwargs", [{}] * len(axvspans))
        if not isinstance(axvspans_kwargs, list):
            axvspans_kwargs = [axvspans_kwargs]

        for axvspan, axvspan_kwargs in zip(axvspans, axvspans_kwargs):
            vspan_min, vspan_max = axvspan
            fig.add_shape(
                type="rect",
                x0=vspan_min,
                x1=vspan_max,
                y0=0,
                y1=1,
                xref="paper",
                yref="paper",
                fillcolor=axvspan_kwargs.get("color", "rgba(0,0,255,0.1)"),
                opacity=axvspan_kwargs.get("alpha", 0.2),
                line=dict(width=0),
            )

    if "axhspan" in kwargs and kwargs["axhspan"] is not None:
        axhspans = (
            kwargs["axhspan"]
            if isinstance(kwargs["axhspan"], list)
            else [kwargs["axhspan"]]
        )
        axhspans_kwargs = kwargs.get("axhspan_kwargs", [{}] * len(axhspans))
        if not isinstance(axhspans_kwargs, list):
            axhspans_kwargs = [axhspans_kwargs]

        for axhspan, axhspan_kwargs in zip(axhspans, axhspans_kwargs):
            hspan_min, hspan_max = axhspan
            fig.add_shape(
                type="rect",
                x0=0,
                x1=1,
                y0=hspan_min,
                y1=hspan_max,
                xref="paper",
                yref="paper",
                fillcolor=axhspan_kwargs.get("color", "rgba(0,0,255,0.1)"),
                opacity=axhspan_kwargs.get("alpha", 0.2),
                line=dict(width=0),
            )

    if "legend" in kwargs:
        legend_kwargs = kwargs.get("legend_kwargs", {})

        for i, name in enumerate(kwargs["legend"]):
            if i < len(fig.data):
                fig.data[i].name = name

        if "loc" in legend_kwargs:
            loc = legend_kwargs["loc"]
            legend_pos = {}
            if loc == "upper right":
                legend_pos = dict(x=1, y=1, xanchor="right", yanchor="top")
            elif loc == "upper left":
                legend_pos = dict(x=0, y=1, xanchor="left", yanchor="top")
            elif loc == "lower right":
                legend_pos = dict(x=1, y=0, xanchor="right", yanchor="bottom")
            elif loc == "lower left":
                legend_pos = dict(x=0, y=0, xanchor="left", yanchor="bottom")
            elif loc == "out":
                legend_pos = dict(
                    x=1.01, y=1, xanchor="left", yanchor="top", orientation="v"
                )
        else:
            legend_pos = dict(
                x=1.01, y=1, xanchor="left", yanchor="top", orientation="v"
            )

        layout_updates["legend"] = legend_pos

    if "margins" in kwargs:
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50, pad=int(kwargs["margins"] * 100))
        )
    if "row" in kwargs:
        fig.update_layout(xaxis=dict(automargin=True), yaxis=dict(automargin=True))
    fig.update_layout(**layout_updates)

    return traces[0] if traces else fig


def _configure_image(ax: plt.Axes, x, ys, **kwargs):
    if "semilogx" in kwargs and kwargs["semilogx"]:
        plot_fun = ax.semilogx
    else:
        plot_fun = ax.plot

    if "grid" in kwargs.keys():
        if kwargs["grid"]:
            ax.grid(visible=kwargs["grid"], color="grey", linestyle="-", which="both")

    if "colors" in kwargs:
        colors = kwargs["colors"]
    else:
        colors = [None] * len(ys)

    if "markers" in kwargs:
        markers = kwargs["markers"]
    else:
        markers = [None] * len(ys)

    if "linestyles" in kwargs:
        linestyles = kwargs["linestyles"]
    else:
        linestyles = [None] * len(ys)

    if "linewidth" in kwargs:
        linewidths = kwargs["linewidth"]
    else:
        linewidths = [None] * len(ys)

    if "mask" in kwargs:
        masks = kwargs["mask"]
        if not isinstance(masks, list):
            masks = [masks] * len(ys)
        for itt, mask in enumerate(masks):
            if mask is None:
                masks[itt] = np.array([True] * len(x))
    else:
        masks = [np.array([True] * len(x))] * len(ys)

    for y, color, marker, linestyle, linewidth, mask in zip(
        ys, colors, markers, linestyles, linewidths, masks
    ):
        return_values = plot_fun(
            np.array(x)[mask],
            np.array(y)[mask],
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
        )

    if "point" in kwargs.keys():
        point_kwargs = kwargs.get("point_kwargs", {})
        ax.scatter(x=kwargs["point"][0], y=kwargs["point"][1], **point_kwargs)

    if "xticks" in kwargs.keys():
        plt.xticks(kwargs["xticks"]["ticks"], kwargs["xticks"]["labels"])
    if "yticks" in kwargs.keys():
        plt.yticks(kwargs["yticks"]["ticks"], kwargs["yticks"]["labels"])

    if "title" in kwargs.keys():
        ax.set_title(kwargs["title"])

    if "log" in kwargs.keys():
        ax.set_xscale("log", base=kwargs["log"][0])
        ax.set_yscale("log", base=kwargs["log"][1])
    if "xlim" in kwargs.keys():
        ax.set_xlim(left=kwargs["xlim"][0], right=kwargs["xlim"][1])
    if "xlimit" in kwargs.keys():
        ax.set_xlim(left=kwargs["xlimit"][0], right=kwargs["xlimit"][1])
    if "ylim" in kwargs.keys():
        ax.set_ylim(bottom=kwargs["ylim"][0], top=kwargs["ylim"][1])
    if "ylimit" in kwargs.keys():
        ax.set_ylim(bottom=kwargs["ylimit"][0], top=kwargs["ylimit"][1])
    if "ylabel" in kwargs.keys():
        ax.set_ylabel(kwargs["ylabel"])
    if "xlabel" in kwargs.keys():
        ax.set_xlabel(kwargs["xlabel"])
    if "vline" in kwargs.keys():
        if isinstance(kwargs["vline"], list):
            for line, color in kwargs["vline"]:
                ax.axvline(line, color=color, linestyle=":")
        elif isinstance(kwargs["vline"], tuple):
            ax.axvline(kwargs["vline"][0], color=kwargs["vline"][1], linestyle=":")
        else:
            ax.axvline(kwargs["vline"], color="red", linestyle=":")
    if "hline" in kwargs.keys():
        if isinstance(kwargs["hline"], list):
            for line, _kwargs in kwargs["hline"]:
                ax.axhline(line, **_kwargs)
        elif isinstance(kwargs["hline"], tuple):
            hline_value = kwargs["hline"][0]
            hline_color = kwargs["hline"][1]
            hline_linestyle = None
            hline_linewidth = None
            if len(kwargs["hline"]) > 2:
                hline_linestyle = kwargs["hline"][2]
            if len(kwargs["hline"]) > 3:
                hline_linewidth = kwargs["hline"][3]
            ax.axhline(
                hline_value,
                color=hline_color,
                linestyle=hline_linestyle,
                linewidth=hline_linewidth,
            )
        else:
            ax.axhline(kwargs["hline"], color="red", linestyle=":")
    if "axvspan" in kwargs.keys():
        axvspans = kwargs["axvspan"]
        if not isinstance(axvspans, list):
            axvspans = [axvspans]
        axvspans_kwargs = kwargs.get("axvspan_kwargs", [{}] * len(axvspans))
        if not isinstance(axvspans_kwargs, list):
            axvspans_kwargs = [axvspans_kwargs]
        for axvspan, axvspan_kwargs in zip(axvspans, axvspans_kwargs):
            vspan_min, vspan_max = axvspan
            ax.axvspan(vspan_min, vspan_max, **axvspan_kwargs)
    if "axhspan" in kwargs.keys():
        axhspans = kwargs["axhspan"]
        if not isinstance(axhspans, list):
            axhspans = [axhspans]
        axhspans_kwargs = kwargs.get("axhspan_kwargs", [{}] * len(axhspans))
        if not isinstance(axhspans_kwargs, list):
            axhspans_kwargs = [axhspans_kwargs]
        for axhspan, axhspan_kwargs in zip(axhspans, axhspans_kwargs):
            hspan_min, hspan_max = axhspan
            ax.axhspan(hspan_min, hspan_max, **axhspan_kwargs)
    # if "legend" in kwargs.keys():
    # legend_kwargs = kwargs.get("legend_kwargs", {})
    # ax.legend(kwargs["legend"], **legend_kwargs)
    if "margins" in kwargs.keys():
        ax.margins(x=kwargs["margins"])

    return return_values


def save_timeline(filename, length, points, title, path=".", show=False):
    """
    Save a timeline from a set of points.
    :param filename: Name of the file.
    :param length: Length of the graph in x-values.
    :param points: List of tuples, time and a label.
    :param title: Title of the graph.
    :param show: Whether or not to show the graph.
    """
    points.sort(key=lambda x: x[0])
    level = logging_functions._disable_logger()
    fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)
    ax.set_ylim(-2, 1.75)
    # ax.set_xlim(0, length)
    ax.axhline(0, xmin=0.05, xmax=0.95, c="deeppink", zorder=1)
    x_points = [point[0] for point in points]

    # create the points
    ax.scatter(x_points, np.zeros(len(x_points)), s=120, c="palevioletred", zorder=2)
    ax.scatter(x_points, np.zeros(len(x_points)), s=30, c="darkmagenta", zorder=3)

    # create the labels
    label_offsets = np.zeros(len(x_points))
    label_offsets[::2] = 0.35
    label_offsets[1::2] = -0.7
    for offset, (t, label) in zip(label_offsets, points):
        ax.text(
            t,
            offset,
            label,
            ha="center",
            fontfamily="serif",
            fontweight="bold",
            color="royalblue",
            fontsize=12,
        )

    # create the stems
    stems = np.zeros(len(x_points))
    stems[::2] = 0.3
    stems[1::2] = -0.3
    markerline, stemline, baseline = ax.stem(x_points, stems)
    plt.setp(markerline, marker=",", color="darkmagenta")
    plt.setp(stemline, color="darkmagenta")

    # hide lines around chart
    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # hide tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        title, fontweight="bold", fontfamily="serif", fontsize=16, color="royalblue"
    )

    if show:
        plt.show()
    plt.savefig(os.path.join(path, filename))
    plt.close()
    logging_functions._enable_logger(level)


def save_status_bars(filename, dates, colors=None, width=0.3, show=False, bins=10):
    fig = plt.figure(figsize=(10, 2), dpi=70)
    ax = fig.add_subplot(111)
    dates = sorted(dates)
    latest_date = max(dates)
    latest_date = datetime.datetime(
        latest_date.year, latest_date.month, latest_date.day
    )
    date_bins = [0] * bins
    x_days = [latest_date - datetime.timedelta(days=i) for i in reversed(range(bins))]
    for color in set(colors):
        date_bins = [0] * bins
        for date, _color in zip(dates, colors):
            if _color != color:
                continue
            reduced_date = datetime.datetime(date.year, date.month, date.day)
            delta_days = (latest_date - reduced_date).days
            if delta_days > bins:
                continue
            date_bins[bins - 1 - delta_days] += 1

        ax.bar(x_days, date_bins, width=width, align="center", label=color, color=color)
    plt.tight_layout()

    # hide lines around chart
    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_visible(False)

    # hide tick labels
    y_ticks = [min(date_bins), max(date_bins)]
    x_labels = [None] * bins
    x_labels[0] = x_days[0].strftime("%d/%m/%Y")
    x_labels[-1] = x_days[-1].strftime("%d/%m/%Y")
    ax.set_xticks(x_days)
    ax.set_xticklabels(x_labels, fontsize=25)
    ax.tick_params(axis="both", length=0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=25)

    ticks = ax.xaxis.get_majorticklabels()
    ticks[0].set_horizontalalignment("left")
    ticks[-1].set_horizontalalignment("right")

    # make background/foreground transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    if show:
        plt.show()

    plt.savefig(filename, transparent=True)
    plt.close()
