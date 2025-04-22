import plotly.graph_objects as go
import plotly.io as pio

# Template that mimic matplotlib default settings
colors_scheme = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


font_scheme = {
    "title": {"family": "Arial", "size": 22, "color": "black", "weight": "bold"},
    "legend": {"family": "Arial", "size": 16, "color": "#333333"},
    "axes": {
        "title_font": {"family": "Courier New", "size": 20, "color": "#444444"},
        "tick_font": {"family": "Courier New", "size": 16, "color": "#666666"},
    },
    "annotations": {"family": "Arial", "size": 16, "color": "#222222"},
}
ppi = 96

ECS_template = go.layout.Template(
    layout=go.Layout(
        colorway=colors_scheme,
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(font=font_scheme["title"]),
        width=7.7 * ppi,
        height=460,
        annotationdefaults=dict(
            font=font_scheme["annotations"],
            showarrow=True,
            arrowhead=5,
            arrowsize=0.5,
        ),
        hoverlabel=dict(
            font=font_scheme["axes"]["tick_font"] | {"color": "white"},
            bgcolor="#555555",
        ),
        xaxis=dict(
            title=dict(font=font_scheme["axes"]["title_font"]),
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            minor=dict(ticks="inside", ticklen=2, showgrid=True),
            showline=True,
            linecolor="black",
            linewidth=1.5,
            ticks="inside",
            tickcolor="black",
            tickwidth=1.5,
            ticklen=5,
            mirror=True,
            showticklabels=True,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            tickfont=font_scheme["axes"]["tick_font"],
        ),
        yaxis=dict(
            title=dict(font=font_scheme["axes"]["title_font"]),
            showgrid=True,
            gridcolor="#E0E0E0",
            gridwidth=1,
            showline=True,
            linecolor="black",
            linewidth=1.5,
            ticks="inside",
            tickcolor="black",
            tickwidth=1.5,
            ticklen=5,
            mirror=True,
            showticklabels=True,
            zeroline=True,
            zerolinecolor="gray",
            zerolinewidth=1,
            tickfont=font_scheme["axes"]["tick_font"],
        ),
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bordercolor="black",
            borderwidth=1,
            bgcolor="#FFFFFF",
            font=font_scheme["legend"],
        ),
    ),
    data=dict(
        contour=[
            go.Contour(
                colorscale="Electric",
                hovertemplate="SPL: %{z:.2f} dB<extra></extra>",
                contours=dict(
                    showlabels=True,
                    labelfont=font_scheme["annotations"] | {"color": "white"},
                    coloring="heatmap",  # or "lines" or "fill"
                ),
                colorbar=dict(
                    title=dict(
                        text="",  # Default empty title
                        font=font_scheme["axes"]["title_font"],
                        side="right",
                    ),
                    thickness=20,
                    len=0.9,
                    x=1.02,
                    xanchor="left",
                    y=0.5,
                    yanchor="middle",
                    outlinewidth=1,
                    outlinecolor="black",
                    tickfont=font_scheme["axes"]["tick_font"],
                    ticklabelposition="outside",
                ),
                ncontours=10,
                showscale=True,
                line_smoothing=0.85,
                line=dict(
                    width=0.5,
                    color="black",
                ),
            ),
        ],
        scatter=[
            go.Scatter(marker=dict(symbol="circle", color="black", size=10)),
        ],
    ),
)

pio.templates["ECS"] = ECS_template
pio.templates.default = "ECS"
