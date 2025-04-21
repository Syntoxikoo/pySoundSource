# Version : 0.1 - February 2025 : making use of it for beamformer
# Version : 0.2 - April 2025 - modified for phase simulator
import numpy as np
import copy
from scipy.fft import rfft, irfft, rfftfreq
import scipy.signal as ssi


def freq2time(fft_inp, norm_b=True):
    norm = "ortho" if norm_b else None

    nb_frq = fft_inp.shape[0]
    if nb_frq % 2 == 0:
        n_time = 2 * nb_frq - 1
    else:
        n_time = 2 * nb_frq - 2

    return irfft(fft_inp, n=n_time, axis=0, norm=norm)


def time2freq(frm_inp, nb_fft=None, norm_b=True, fs=None):
    if nb_fft is None:
        nb_fft = frm_inp.shape[0]
    norm = "ortho" if norm_b else None

    # Compute the FFT using rfft : returns only the first half.
    Y = rfft(frm_inp, n=nb_fft, axis=0, norm=norm)
    if fs is not None:
        xf = rfftfreq(n=nb_fft, d=1 / fs)
        return Y, xf
    else:
        return Y


def mag2db(mag):
    db = 20 * np.log10(np.abs(mag) + np.finfo(float).eps)
    return db


def db2mag(db):
    return 10 ** (db / 20.0)


def record_signal(hps, sig_m, listening_pos_v):
    tmp_sig_m = copy.deepcopy(sig_m)
    assert len(hps) == sig_m.shape[0], "Must have consistent number of loudspeakers."
    for i_hp, hp in enumerate(hps):
        hp.freq2time()
        hp_grid = hp.get_grid()
        _, id_dir, _ = hp_grid.find_closest_point(
            point_v=listening_pos_v, norm_s="spherical_1"
        )
        tmp_sig_m[i_hp, :] = ssi.fftconvolve(
            sig_m[i_hp, :], hps[i_hp].data_m[id_dir, :], mode="same"
        )
    output_v = np.sum(tmp_sig_m, axis=0)
    return output_v


def rms(x, axis=-1):
    return np.sqrt(np.mean(np.abs(x) ** 2, axis=axis))


def linalg_concatreal2complex(r_m):
    """
    ## Typical Use Cases
    1.	Complex Data Storage:
        2.	Memory Optimization: Storing complex data as two concatenated real
        arrays reduces memory overhead (avoys complex dtype storage).
        3.	GPU/Accelerator Compatibility: Some hardware accelerators require
        real-valued tensors, necessitating splitting/reconstruction."""
    if len(r_m.shape) == 1:
        c_m = r_m[: r_m.shape[0] // 2] + 1j * r_m[r_m.shape[0] // 2 :]
    elif len(r_m.shape) == 2:
        c_m = (
            r_m[: r_m.shape[0] // 2 :, : r_m.shape[1] // 2]
            + 1j * r_m[r_m.shape[0] // 2 :, : r_m.shape[1] // 2]
        )
    else:
        print("linalg_complex2concatreal: wrong shape.")
        c_m = None
    return c_m
