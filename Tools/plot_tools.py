import numpy as np


def mask_src2D(data, position_v: list, x: np.ndarray, y: np.ndarray):
    """
    to mask source near-singularity on a 2D grid
    make definition
    """
    assert data.shape[0] == len(x) and data.shape[1] == len(
        y
    ), "data shape should follow the x-y grid axis"
    mask = np.ones((len(x), len(y)))
    if np.array(position_v).ndim < 2:
        position_v = [position_v]
    for ii in range(len(position_v)):

        ind_xS = np.argmin(abs(x - position_v[ii][0]))
        ind_yS = np.argmin(abs(y - position_v[ii][1]))

    y_min = max(0, ind_yS - 1)
    y_max = min(len(y), ind_yS + 2)
    x_min = max(0, ind_xS - 1)
    x_max = min(len(x), ind_xS + 2)
    mask[y_min:y_max, x_min:x_max] = np.mean(
        np.abs(data[y_min:y_max, x_min:x_max]) / np.abs(data[ind_xS, ind_yS])
    )
    return data * mask
