import numpy as np
import plotly.graph_objects as go
from Tools.SpatialObject import SpatialObject
from rich import print

grid_res = 10
x = np.linspace(-10 / 2, 10 / 2, grid_res)
y = np.linspace(-10 / 2, 10 / 2, grid_res)
X, Y = np.meshgrid(x, y)
nfft = 64
# print(X)
hps = [
    SpatialObject(
        src_domain="time",
        norm_s="cartesian",
        azim_v=X.flatten(),
        elev_v=Y.flatten(),
        dist_v=0,
        data_m=np.zeros((grid_res**2, nfft + 1)),
        n_fft=nfft,
        deg=True,
        position_v=[X.flatten()[ii], Y.flatten()[ii], 0],
    )
    for ii in range(len(X.flatten()))
]

target = [2.5, 0.5]
datas = np.zeros((nfft + 1, len(hps)), dtype=complex)
for ii, hp in enumerate(hps):
    hp.set_directivity("monopole")
    datas[:, ii] = hp.resp_for_M(
        target_pos=target,
    )
datas[:, 57] = np.mean(datas[:, :57:])
data = SpatialObject._Lp(datas[4, :].reshape(len(x), len(y)))
fig = go.Figure(
    data=[
        go.Surface(
            z=data,
            x=X,
            y=Y,
            colorscale="Viridis",
            opacity=0.8,
        )
    ]
)

fig.update_layout(
    title="Meshgrid Visualization (3D)",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
)
fig.show()
