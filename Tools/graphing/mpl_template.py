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

matplotlib_template = go.layout.Template(
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
)

pio.templates["matplotlib"] = matplotlib_template
pio.templates.default = "matplotlib"
