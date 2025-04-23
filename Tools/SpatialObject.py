import numpy as np
import scipy.signal as ssi
import scipy.special as ssp
from scipy.interpolate import interp1d
import sys
from Tools import sig_proc_tools as spt
from Tools import space_tools as st
from time import time
from scipy import ndimage


class SpatialObject:
    def __init__(
        self,
        fs: int = 16000,
        src_domain: str = "time",
        dist_v=None,
        azim_v=None,
        elev_v=None,
        data_m=None,
        src_resp=1.0,
        position_v=(0, 0, 0),
        orientation_v=(0, 0),
        radius=0.01,
        **kwargs,
    ):
        """
        This class is a spatial object that represents a source, as an object, the source has it's own properties
        and grids coordinates, meaning the impulse response of one particular source can be compute, each source can
        have different radius, T&S, directivity pattern, etc.

        ### INPUTS
        - **fs**: sampling frequency (default: 16000 Hz)
        - **src_domain**: source domain ('time', 'freq') (default: 'time')
        - **dist_v**: sph : distance source ->mic [(n_azim, n_elev).flatten()], cart : Z coordinate of the space
        - **azim_v**: sph : azimuth of each mic [(n_azim, n_elev).flatten()], cart : X coordinate of the space
        - **elev_v**: sph : elevation of each mic [(n_azim, n_elev).flatten()], cart : Y coordinate of the space
        - **data_m**: impulse response or frequency response (nb_dir x nb_tap) or (nb_dir x nb_freq)
        - **src_resp**: source weighting (default: 1.0)
        - **position_v**: position of the spk [n_hp,(x,y,z).T]
        - **orientation_v**: orientation of the spk [azim,elev]
        - **radius**: radius of the spk

        - kwargs:
            - **c**: speed of sound (default: 343 m/s)
            - **n_fft**: number of samples for the fft (default: 1024) - scalar
            - **DIM**: dimension of the array (default: 3) - 2 or 3
            - **deg**: boolean to specify if the angles are in degrees (default: False)
            - **freq**: frequency of interest (default: None) - scalar
            - **norm_s**: normalization system (default: 'spherical_1') - 'cartesian'

        ### TODO
        - src_resp : should accept the transfer function of the loudspeaker from thieles and small
        - refine function :
            - get_hplane_data
            - plot_hplane_imshow

        Version : 0.1 - February 2025 : making use of it for beamformer
        Version : 0.2 - April 2025 - modified for phase simulator
        """
        # Kwargs
        params = {
            "c": 343,
            "rho": 1.21,
            "n_fft": 1024,
            "DIM": 3,
            "deg": False,
            "freq": None,
            "norm_s": "spherical_1",
        }
        params.update(kwargs)

        self.norm_s = params["norm_s"]
        self.c = params["c"]
        self.rho = params["rho"]
        self.n_fft = params["n_fft"]
        self.DIM = params["DIM"]
        self.deg = params["deg"]

        if dist_v is None:
            dist_v = np.array([0.0])
        if elev_v is None:
            elev_v = np.array([0.0])
        if azim_v is None:
            azim_v = np.array([0.0])
        if data_m is None:
            data_m = np.concatenate(([[1.0]], np.zeros((1, 127))), axis=1)

        # Check input parameters
        assert src_domain in ["time", "freq"], "src_domain must be time or freq"
        if self.norm_s == "spherical_1":
            assert (
                azim_v.shape == elev_v.shape == dist_v.shape
            ), "azim_v, elev_v and dist_v have to have the same shape"
            self.DIM = 3
        elif self.norm_s == "cartesian":
            if isinstance(dist_v, (float, int)):
                dist_v = np.array([dist_v])
            if dist_v.shape[0] != azim_v.shape[0]:
                assert (
                    azim_v.shape == elev_v.shape
                ), "X and Y have to have the same shape"
                self.DIM = 2
            else:
                assert (
                    azim_v.shape == elev_v.shape == dist_v.shape
                ), "azim_v, elev_v and dist_v have to have the same shape"
                self.DIM = 3
        assert (
            azim_v.shape[0] == data_m.shape[0]
        ), "azim_v and data_m have to have the same number of lign"

        # CONSTRUCTOR
        self.src_domain = src_domain
        self.fs = fs
        self.azim_v = azim_v
        self.elev_v = elev_v
        self.dist_v = dist_v
        self.data_m = data_m
        self.xaxis_v = np.linspace(
            0, self.data_m.shape[1], self.data_m.shape[1]
        )  # time axis or freq axis
        self.src_resp = src_resp
        if self.DIM == 2:
            self.position_v = position_v[0:2]
            self.orientation_v = orientation_v[0:1]
        else:
            self.position_v = position_v
            self.orientation_v = orientation_v
        self.radius = radius
        self.directed = False
        self.resp_computed = False
        self.Q = None
        self.respF = None

    @staticmethod
    def _Lp(data):
        return 20 * np.log10(np.abs(data) / 2e-5 + 1e-9)

    @staticmethod
    def _pXY(data, x, y, Mx: float = 0.0, My: float = 0.0):
        """
        method to get the pressure at a specific point in the grid
        data has to have the shape (len(x),len(y))
        """
        idx_x = np.argmin(np.abs(x - Mx))
        idx_y = np.argmin(np.abs(y - My))
        return data[idx_x, idx_y]

    def freq_range(self, fmin, fmax):
        pass

    def _Sresp(self, freq=None):
        """
        TODO : still not sure if should put the freq or the frequency bin
        """
        if self.Q is not None:
            if self.src_domain == "time":
                print("SpatObj: _Sresp: Convert time to freq")
                self.time2freq()
            if freq is not None:
                Sresp = (
                    1j
                    * self.rho
                    * (2 * np.pi * freq)
                    * self.Q[np.argmin(np.abs(self.xaxis_v - freq))]
                )
            else:
                Sresp = 1j * self.rho * (2 * np.pi * self.xaxis_v) * self.Q

            return Sresp
        else:
            return 1.0

    def update_xaxis(self):
        """
        Update the time axis or freq axis according to the data type
        Useful for plot data
        """
        num_sample_n = self.data_m.shape[1]

        if self.src_domain == "time":
            self.xaxis_v = np.linspace(
                0.0, float(num_sample_n / self.fs), num=num_sample_n
            )  # Tmax = N/fs
        elif self.src_domain == "freq":
            self.xaxis_v = np.linspace(
                0.0, float(self.fs / 2.0), num=num_sample_n
            )  # Fmax = fs/2
        else:
            print("Unvalid type", file=sys.stderr)
        return

    def time2freq(self, num_freq_n=None):
        """
        This function converts the data from time domain to frequency domain using the FFT algorithm
        developed in the sig_proc_tools.py file (time2freq function)

        ### INPUTS
        - num_freq_n: number of frequency bins

        """
        # ENSURE TO BE ODD
        if num_freq_n is not None:
            if np.mod(num_freq_n, 2) == 0:
                num_freq_n += 1
        # CONVERT
        if self.src_domain == "time":
            if num_freq_n is None:
                num_freq_n = self.data_m.shape[1] // 2
                num_freq_n += 1 - np.mod(num_freq_n, 2)
            self.src_domain = "freq"
            self.n_fft = (num_freq_n - 1) * 2
            self.data_m = spt.time2freq(self.data_m.T, nb_fft=self.n_fft).T
        elif self.src_domain == "freq":
            if self.data_m.shape[1] is not None or self.data_m.shape[1] != num_freq_n:
                self.freq2time()
                self.time2freq(num_freq_n)
        else:
            print("Unvalid type", file=sys.stderr)

        self.update_xaxis()

        return

    def freq2time(self, num_sample_n=None):
        """
        Conversion of self.data_m from frequency domain to time domain using the IFFT algorithm
        developed in the sig_proc_tools.py file (freq2time function)

        ### INPUTS
        - num_sample_n: number of time samples
        """
        if self.src_domain == "freq":
            self.src_domain = "time"
            self.data_m = spt.freq2time(self.data_m.T).T
            if np.sum(np.imag(self.data_m)) > 0.0:
                print(
                    "SpatObject::freq2time: IMAGINARY PART IS NOT EQUAL TO ZERO: %f"
                    % np.sum(np.imag(self.data_m)),
                    file=sys.stderr,
                )
        if self.src_domain == "time":
            pass
        else:
            print("SpatObject::freq2time: Unvalid type", file=sys.stderr)

        self.update_xaxis()

        return

    def get_grid(self, new_norm_s="spherical_1", norm_s=None):
        """
        This function returns a Grid object corresponding to the simulated space
        the simulated space must be defined by azim_v, elev_v and dist_v

        ### FUTURE IMPROVEMENTS
        - add a way to define a cartesian grid
        - add a way to define a spherical grid with a different norm

        ### INPUTS
        - norm_s: coordinates norm 'cartesian', 'spherical_1' (depends on the input data)
        - new_norm_s: new coordinates norm 'cartesian', 'spherical_1' (format of the output grid)

        ### OUTPUTS
        - Grid object corresponding simulated space

        """
        if norm_s is not None:
            self.norm_s = norm_s
        if self.norm_s == "spherical_1":
            tmp_m = np.zeros((self.dist_v.shape[0], 3), dtype=float)
            tmp_m[:, 0] = self.dist_v
            tmp_m[:, 1] = self.azim_v
            tmp_m[:, 2] = self.elev_v
        elif self.norm_s == "cartesian":
            if self.dist_v.shape[0] != self.azim_v.shape[0]:
                tmp_m = np.zeros((self.azim_v.shape[0], 2), dtype=float)
                tmp_m[:, 0] = self.azim_v
                tmp_m[:, 1] = self.elev_v
                self.DIM = 2
            else:
                tmp_m = np.zeros((self.dist_v.shape[0], 3), dtype=float)
                tmp_m[:, 0] = self.dist_v
                tmp_m[:, 1] = self.azim_v
                tmp_m[:, 2] = self.elev_v
        else:
            print("get_grid: Unknown coordinates norm %s" % norm_s)
        grid = st.Grid(norm_s=self.norm_s, coords_m=tmp_m, DIM=self.DIM, deg=self.deg)

        if any(pos != 0 for pos in self.position_v) and not self.directed:
            # Offset the grid to account for the position of the source
            if self.norm_s == "cartesian":
                grid.coords_m = grid.coords_m - np.array(self.position_v)
            else:
                grid.convert_coordinates(new_norm_s="cartesian")
                grid.coords_m = grid.coords_m - np.array(self.position_v)

        grid.convert_coordinates(new_norm_s=new_norm_s)

        return grid

    def set_directivity(self, method):
        """
        This function sets the directivity of the loudspeaker
        it chooses the directivity method to apply

        Args:
            method: directivity method ('cardioid', 'bessel', 'monopole')

        """

        if method == "cardioid":
            self._set_directivity_cardioid()
        elif method == "bessel":
            self._set_directivity_bessel()
        elif method == "monopole":
            if self.src_domain == "time":
                self.time2freq()
                self.src_domain = "freq"
            self.pattern = np.ones_like(self.data_m)
            self.directed = True
        else:
            print("set_directivity: Unknown method %s" % method)
        return

    def _set_directivity_cardioid(self):
        """
        Set weight for a cardioid directivity pattern
        TODO :
        - Add a way to just multiply the datas by the cardioid pattern
        - check if the normalization distance do not conflict
        """
        assert not self.directed, "Directivity already set"
        if self.DIM == 2:
            xy_trg_v = st.pol2cart(1.0, self.orientation_v[0], deg=self.deg)
        else:
            xy_trg_v = st.sph2cart(
                1.0, self.orientation_v[0], self.orientation_v[1], deg=self.deg
            )
        hp_grid = self.get_grid("spherical_1")
        hp_grid.coords_m[:, 0] = 1.0  # force to 1 for scalar product (distance)
        hp_grid.convert_coordinates("cartesian")
        if self.src_domain == "time":
            self.time2freq()
        pattern_t = (1.0 + hp_grid.coords_m @ xy_trg_v) / 2
        self.pattern = np.tile(pattern_t[:, np.newaxis], (1, self.data_m.shape[1]))
        self.directed = True
        return

    def _set_directivity_bessel(self):
        """
        ### TODO
        - Done : refine the problem with the position of the loudspeaker and the radiation pattern
        - - Add a way to just multiply the datas by the bessel pattern
        """
        # GET TARGET DIRECTION
        assert not self.directed, "Directivity already set"
        if self.DIM == 2:
            xy_trg_v = st.pol2cart(1.0, self.orientation_v[0], deg=self.deg)
            hp_grid = self.get_grid("spherical_1")
            angle = hp_grid.coords_m[:, 1] - np.deg2rad(self.orientation_v[0])
            if self.src_domain == "time":
                self.time2freq()
            k_v = 2 * np.pi * (np.abs(self.xaxis_v) + 1e-9) / self.c
            tmp_m = (
                k_v[None, :].repeat(hp_grid.nb_dir, axis=0)
                * self.radius
                * np.sin(angle[:, None].repeat(self.xaxis_v.shape[0], axis=1))
            )
            self.pattern = (2 * ssp.j1(tmp_m + 1e-9) / (tmp_m + 1e-9)) * 2
            self.directed = True
            return
        else:
            xy_trg_v = st.sph2cart(
                1.0, self.orientation_v[0], self.orientation_v[1], deg=self.deg
            )
            hp_grid = self.get_grid("spherical_1")
            hp_grid.coords_m[:, 0] = 1.0  # force to 1 for scalar product
            hp_grid.convert_coordinates("cartesian")
            angles_v = np.array(
                [
                    np.arccos(np.dot(hp_grid.coords_m[ii, :], np.array(xy_trg_v)))
                    for ii in range(hp_grid.nb_dir)
                ]
            )  # compute the angle between the target direction and the grid points
            if self.src_domain == "time":
                self.time2freq()
            k_v = 2 * np.pi * (np.abs(self.xaxis_v) + 1e-9) / self.c
            tmp_m = (
                k_v[None, :].repeat(hp_grid.nb_dir, axis=0)
                * self.radius
                * np.sin(angles_v[:, None].repeat(self.xaxis_v.shape[0], axis=1))
            )
            self.pattern = (2 * ssp.j1(tmp_m + 1e-6) / (tmp_m + 1e-6)) * 2
            self.directed = True
        return

    def resample(self, fs_new=96000):
        self.freq2time()
        nb_smp = int(self.data_m.shape[1] * fs_new / self.fs)
        nb_smp += np.mod(nb_smp, 2)  # Ensure nb samp being even
        self.data_m = ssi.resample_poly(self.data_m, fs_new, self.fs, axis=1)
        self.fs = fs_new
        self.update_xaxis()
        return

    def compute_response(self, hp_grid=None, freq=None, sig=None, **kwargs):
        """
        This function returns the pressure of the loudspeaker

        ### INPUTS
        - hp_grid: Grid object
        - freq: if not none give the data for only one given frequency

        ### OUTPUTS
        - pressure: pressure of the loudspeaker

        ### Future improvements
        - add a way to compute a real pressure by shaping the data with LEM, (computing the velocity of the speaker)

        """
        kwargs.setdefault("noise", False)
        kwargs.setdefault("fast", False)
        noise = kwargs["noise"]
        fast = kwargs["fast"]
        if hp_grid is None:
            hp_grid = self.get_grid("cartesian")

        if self.resp_computed:
            if freq is not None:
                idx_freq = np.argmin(np.abs(self.xaxis_v - freq))
                if "reshape" in kwargs and kwargs["reshape"]:

                    return self.data_m[:, idx_freq].reshape(
                        hp_grid.length_x, hp_grid.length_y
                    )
                else:
                    return self.data_m[:, idx_freq]
            else:
                return self.data_m

        if self.src_domain == "time":
            print("Compute response: Convert time to freq")
            self.time2freq()

        rel_pos = hp_grid.coords_m - np.asarray(
            self.position_v
        )  # Compute the relative position between the grid points and the loudspeaker

        dist = np.linalg.norm(
            rel_pos, axis=1
        )  # Calculating the distance between the grid points and the loudspeaker
        delay = dist / self.c
        # delay += 0.02  # Adding a delay to recentre the filter
        nb_dir, nb_freq = self.data_m.shape
        freqs_v = self.xaxis_v
        omega_v = 2 * np.pi * freqs_v
        self.data_m = self._Sresp() * self.data_m.astype(complex)
        if fast:
            k = omega_v / self.c
            return k, dist, self.data_m
        phase = -1j * omega_v * delay[:, None]
        self.data_m *= 1j * self.rho * np.exp(phase)
        self.data_m /= dist[:, None] * 4 * np.pi
        self.resp_computed = True

        # Maybe there is no point to compute the response to a signal in all dir, especially in free field
        if isinstance(sig, (np.ndarray, list)):
            if noise:
                # mean the noise
                y = np.zeros((self.n_fft / 2 + 1), dtype=complex)
                for ii in range(int(len(sig) / (self.n_fft))):
                    sig_t = sig[ii * self.n_fft : (ii + 1) * self.n_fft]
                    y += np.fft.rfft(sig_t, self.n_fft) / self.fs
            else:
                y = np.fft.rfft(sig, self.n_fft) / self.fs

            y_m = self.data_m * y[:, None].repeat(nb_dir, axis=1)
            return y_m

        if freq is not None:
            idx_freq = np.argmin(np.abs(freqs_v - freq))
            if "reshape" in kwargs and kwargs["reshape"]:

                return self.data_m[:, idx_freq].reshape(
                    hp_grid.length_x, hp_grid.length_y
                )
            else:
                return self.data_m[:, idx_freq]
        else:
            return

    def resp_for_f(self, hp_grid=None, freq: float = 1000, reshape=True):
        """
        Compute the response of the source on the grid for a specific frequency bin.

        TODO:
            add a way to compute a real pressure by shaping the data with LEM, (computing the velocity of the speaker)
        Args:
            hp_grid: Grid object
                instance of grid containing the field points.
            freq: float
                if not none give the data for only one given frequency

        Returns:
            pressure: pressure of the loudspeaker for 1 frequency bin

        Exemple:

        """
        start = time()
        if hp_grid is None:
            hp_grid = self.get_grid("cartesian")

        if self.respF and self.respF == freq:
            return self.dataF

        if self.src_domain == "time":
            print("Compute response: Convert time to freq")
            self.time2freq()

        rel_pos = hp_grid.coords_m - np.asarray(self.position_v)
        dist = np.linalg.norm(rel_pos, axis=1)
        delay = dist / self.c

        omega = 2 * np.pi * freq
        self.dataF = self._Sresp(freq) * self.pattern[
            :, np.argmin(np.abs(self.xaxis_v - freq))
        ].astype(complex)
        print("shape:", self.dataF.shape)
        phase = -1j * omega * delay
        self.dataF *= 1j * self.rho * np.exp(phase)
        self.dataF /= dist * 4 * np.pi
        self.respF = freq

        if reshape:
            self.dataF = self.dataF.reshape(hp_grid.length_x, hp_grid.length_y)
        print("computing response for f took :", time() - start)
        return self.dataF

    def resp_for_M(self):
        """
        Compute the response of the source on its frequency range for a specific point on the grid.
        """
        pass

    def enclosure(self, TL=None, dB=True, **kwargs):
        """
        simulate the attenuation of the rear of an enclosure
        use absorption or admitance ?
        Parameter:
            TL - np.ndarray : transmission loss per oct band [freq,TL(dBSPL)]
        """
        if TL is None:
            TL = np.array(
                [
                    [63, 9.57],
                    [125, 15.52],
                    [250, 21.54],
                    [500, 27.56],
                    [1000, 33.58],
                    [2000, 39.60],
                    [4000, 45.62],
                    [8000, 51.65],
                ]
            )

            # TL[:, 1] = 10 ** (TL[:, 1] / 20) * 2e-5  # spl to mag

        hp_grid = self.get_grid("cartesian")

        if self.DIM == 2:
            xy_trg_v = st.pol2cart(1.0, self.orientation_v[0], deg=self.deg)

        angles = hp_grid.coords_m @ xy_trg_v

        angle_TL = np.array([0 if angle >= 0 else 1 for angle in angles])

        if self.src_domain == "time":
            self.time2freq()
        attenuation = interp1d(TL[:, 0], TL[:, 1], fill_value="extrapolate")
        freq_TL = attenuation(self.xaxis_v)

        mask = angle_TL[:, None] * freq_TL[None, :]

        if self.DIM == 3:
            raise NotImplementedError("enclosure is not implemented for 3D now")

        if not dB:
            non_zero_mask = mask != 0

            mask_mag = np.zeros_like(mask)
            mask_mag[non_zero_mask] = 10 ** (mask[non_zero_mask] / 20) * 2e-5

            mask = mask_mag
        return mask

    def update_orientation(self, new_orientation_v, deg=None, target="pattern"):
        """
        Update the orientation of the cardioid directivity pattern using rotation matrices
        without recomputing the entire pattern.

        Does this function make any sense (computationnally), for the moment scipy approach for 2D but
        full python approach can be more efficient

        Args:
            new_orientation_v : array-like
                New orientation vector. For 2D, it's [azimuth], for 3D it's [azimuth, elevation]
            deg : bool, optional
                Whether the angles are in degrees. If None, uses the class's deg attribute.
            target : str
                data which would be updated after rotation ("pattern", "data_M", "dataF", "all" )
        """
        if deg is not None:
            self.deg = deg

        if not self.directed:
            raise ValueError(
                "Directivity pattern not set yet. Call _set_directivity_cardioid first."
            )

        if self.DIM == 2:

            rotation_angle = new_orientation_v - self.orientation_v[0]

            cos_the = np.cos(rotation_angle)
            sin_the = np.sin(rotation_angle)
            R = np.array([[cos_the, -sin_the], [sin_the, cos_the]])

            hp_grid = self.get_grid("cartesian")
            if target == "pattern":

                self.pattern = self.pattern * R
            elif target == "dataF":
                self.dataF = self.dataF * R
                return self.dataF
            # R_coords = np.zeros_like(hp_grid.coords_m)
            # R_coords[:, :2] = hp_grid.coords_m[:, :2] @ R.T

        else:
            return
