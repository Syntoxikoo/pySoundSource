import numpy as np
from Tools.SpatialObject import SpatialObject as Sobj
from Tools.plot_tools import mask_src2D
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import Tools.graphing.mpl_template

fs = 16000
n_fft = 128

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Xv = X.flatten()
Yv = Y.flatten()
Z = 0


src_pos = [[0.0, 0.0, 0], [0.0, 0.0, 0], [0, 0, 0]]
radius = 0.05 * 2.54 / 2

hps = [
    Sobj(
        fs=fs,
        src_domain="time",
        azim_v=Xv,
        elev_v=Yv,
        dist_v=Z,
        radius=radius,
        position_v=src_pos[m],
        norm_s="cartesian",
        data_m=np.zeros((len(Xv), n_fft)),
    )
    for m in range(len(src_pos))
]
directivity_s = ["monopole", "cardioid", "bessel"]
data = 0
fig, ax = plt.subplots(ncols=3, figsize=(11, 3))

for ihp, hp in enumerate(hps):
    hp.set_directivity(directivity_s[ihp])
    hp_grid = hp.get_grid("cartesian")

    data = hp.resp_for_f(hp_grid=hp_grid, freq=8000, reshape=True)
    # data = mask_src2D(data, position_v=src_pos[ihp], x=x, y=y)
    # mask = hp.enclosure(db=True)

    ax[ihp].contourf(
        X,
        Y,
        Sobj._Lp(data),
        # - mask[:, np.argmin(np.abs(hp.xaxis_v - 3000))].reshape(len(x), len(y)),
        # 10,
        # vmin=np.max(Sobj._Lp(data)),
        vmax=np.max(Sobj._Lp(data)),
        cmap="viridis",
    )
    ax[ihp].set_xticks([])
    ax[ihp].set_yticks([])
    ax[ihp].set_title(directivity_s[ihp])
    m = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
    plt.colorbar(m, ax=ax[ihp])

fig = go.Figure(data=go.Contour(z=Sobj._Lp(data), x=x, y=y))
fig.show()


# ax.scatter(hp_grid.coords_m[:,0], hp_grid.coords_m[:,1], c=20*np.log10(abs(p)/2e-5), cmap='viridis')


# ax.set_xlabel("x")
# ax.set_ylabel("y")


# ax.set_title("Sound Pressure Level (dB SPL)")

plt.show()
