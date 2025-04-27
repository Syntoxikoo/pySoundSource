# Version : 0.1 - February 2025 - making use of it for the beamformer
# Version : 0.2 - April 2025 - modified for phase simulator

import numpy as np
import copy
import sys


def sph2cart(rad_v, azim_v, elev_v, deg=True):
    """
    Convert spherical coordinates to cartesian
    ### INPUT:
    - **rad_v**: radius -- scalar or vector <float>
    - **azim_v**: azimuth -- scalar or vector <float>
    - **elev_v**: elevation -- scalar or vector <float>
    - **deg**: <bool> if True, the input angle is in degree, otherwise in radians

    ### OUTPUT:
    - **x_v**: x coordinate -- scalar or vector <float>
    - **y_v**: y coordinate -- scalar or vector <float>
    - **z_v**: z coordinate -- scalar or vector <float>
    """
    # CHECK
    if isinstance(rad_v, float):
        assert isinstance(azim_v, float) and isinstance(
            elev_v, float
        ), "First input argument is float, then others have to be also float"
    elif isinstance(rad_v, np.ndarray):
        assert (
            rad_v.shape == azim_v.shape == elev_v.shape
        ), "Shape of the input arguments have to be the same"
        assert azim_v.dtype == float, "Input argument has to be float"
        assert elev_v.dtype == float, "Input argument has to be float"
    # PROCESS
    if deg:
        azim_v = np.deg2rad(azim_v)
        elev_v = np.deg2rad(elev_v)
    x_v = rad_v * np.cos(elev_v) * np.cos(azim_v)
    y_v = rad_v * np.cos(elev_v) * np.sin(azim_v)
    z_v = rad_v * np.sin(elev_v)
    return x_v, y_v, z_v


def cart2sph(x_v, y_v, z_v, deg=True):
    """
    Convert cartesian coordinates to spherical
    ### INPUT:
    - **x_v**: x coordinate -- scalar or vector <float>
    - **y_v**: y coordinate -- scalar or vector <float>
    - **z_v**: z coordinate -- scalar or vector <float>
    - **deg**: <bool> if True, the input angle is in degree, otherwise in radians

    ### OUTPUT:
    - **rad_v**: radius -- scalar or vector <float>
    - **azim_v**: azimuth -- scalar or vector <float>
    - **elev_v**: elevation -- scalar or vector <float>
    """
    # CHECK
    assert (
        x_v.shape == y_v.shape == z_v.shape
    ), "Shape of the input arguments have to be the same"
    if len(x_v.shape) == 0:
        x_v = x_v[None]
        y_v = y_v[None]
        z_v = z_v[None]
    # PROCESS
    mat_m = np.concatenate((x_v[:, None], y_v[:, None], z_v[:, None]), axis=1)
    rad_v = np.linalg.norm(mat_m, axis=1)

    subrad_v = np.linalg.norm(
        np.concatenate((x_v[:, None], y_v[:, None]), axis=1), axis=1
    )
    if deg:
        azim_v = np.rad2deg(np.arctan2(y_v, x_v))
        elev_v = np.rad2deg(np.arctan2(z_v, subrad_v))
    else:
        azim_v = np.arctan2(y_v, x_v)
        elev_v = np.arctan2(z_v, subrad_v)
    return rad_v, azim_v, elev_v


def cart2pol(x, y, deg=True):
    """
    Convert cartesian coordinates to polar
    ### INPUT:
    - **x_v**: x coordinate -- scalar or vector <float>
    - **y_v**: y coordinate -- scalar or vector <float>
    - **deg**: <bool> if True, the input angle is in degree, otherwise in radians
    ### OUTPUT:
    - **rho**: radius -- scalar or vector <float>
    - **phi**: azimuth -- scalar or vector <float>
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    if deg:
        phi = np.rad2deg(phi)
    return rho, phi


def pol2cart(rho, phi, deg=True):
    """
    Convert polar coordinates to cartesian
    ### INPUT:
    - **rho**: radius -- scalar or vector <float>
    - **phi**: azimuth -- scalar or vector <float>
    - **deg**: <bool> if True, the input angle is in degree, otherwise in radians

    ### OUTPUT:
    - **x_v**: x coordinate -- scalar or vector <float>
    - **y_v**: y coordinate -- scalar or vector <float>
    """
    if deg:
        phi = np.deg2rad(phi)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


class Grid:
    def __init__(
        self, norm_s="spherical_1", coords_m=np.zeros((1, 3)), DIM=3, deg=True
    ):
        """
        This class is used to create a grid object for each sources and for the beamformer,
        it essentially contains the coordinates of the grid points and the norm of the coordinates,
        it also deals with all the geometrical transformations.
        ### INPUT
        - **norm_s**: <str> coordinates norm ('cartesian', 'spherical_1', 'lateral-polar')
        - **coords_m**: <Nx3> [x, y, z] [radius, azimuth, elevation] [radius, lateral, polar]
        - **DIM**: <int> dimension of the simulated space (2 or 3)

        ### METHODS
        - **convert_coordinates**: convert the coordinates of the grid to the new norm
        - **find_closest_point**: find the closest point on the grid compare to the given point (steering vector point)
        - **resolve_space**: calculate the spacing between sources and the target microphone
        - **dist_src_mic**: calculate the distance between each sources and the microphones
        - **get_spherical_weighting_harder**: compute the spherical weighting for each direction of the grid
        ### PROPERTIES
        - **nb_dir**: number of directions
        - **n_angles**: number of angles across the azimuth and elevation

        ### TODO
        - Implementing 2D grid, done - need to check
        - find_closest_point : nb_out_point_n is not supported, make it


        """
        # CHECK
        assert isinstance(norm_s, str), "First input argument must be a string"
        assert isinstance(
            coords_m, np.ndarray
        ), "Second input argument has to be numpy array"
        assert isinstance(DIM, int), "Third input argument has to be integer"

        assert len(coords_m.shape) == 2, "Second input argument has to be 2D"
        if DIM == 3:
            assert (
                coords_m.shape[1] == 3
            ), "Second input argument has to have 3 columns for 3D grid"
        elif DIM == 2:
            assert (
                coords_m.shape[1] == 2
            ), "Second input argument has to have 2 columns for 2D grid"

        self.norm_s = norm_s
        self.coords_m = coords_m.astype(float)
        self.DIM = DIM
        self.position_v = None
        self.deg = deg

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------ PROPERTIES ------------------------------------------------------------------------------------
    @property
    def nb_dir(self):
        return self.coords_m.shape[0]

    @property
    def n_angles(self):
        if self.DIM > 2:
            return (
                len(np.unique(self.coords_m[:, 1])),
                len(np.unique(self.coords_m[:, 2])),
            )
        else:
            return len(np.unique(self.coords_m[:, 1]))

    @property
    def length_x(self):
        return len(np.unique(self.coords_m[:, 0]))

    @property
    def length_y(self):
        return len(np.unique(self.coords_m[:, 1]))

    @property
    def get_norm_s(self):
        """Returns the current coordinate system identifier"""
        return self.norm_s

    def find_pos(self, point=None):
        """
        Find the closest grid point to the given (x,y) coordinates

        """

        grid_t = copy.deepcopy(self)

        if self.norm_s != "cartesian":
            grid_copy.convert_coordinates(new_norm_s="cartesian")

        if point == None:
            point = [0, 0]

        distances = np.sqrt(
            (grid_t.coords_m[:, 0] - point[0]) ** 2
            + (grid_t.coords_m[:, 1] - point[1]) ** 2
        )
        idx = np.argmin(distances)

        # Return closest point, its index, and the distance
        return self.coords_m[idx], idx, distances[idx]

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------ METHODS ---------------------------------------------------------------------------------------
    def convert_coordinates(self, new_norm_s):
        """
        Convert the coordinates of the grid to the new norm
        ### INPUT
            - **new_norm_s**: <str> coordinates norm ('cartesian', 'spherical_1', 'lateral-polar')

        TODO:
        - try to use lateral polar for steering vector
        """
        # CHECK
        assert isinstance(
            new_norm_s, str
        ), "Grid::convert_coordinates: First input argument must be a string"
        # IF THE SAME COORD. SYSTEM THEN DO NOTHING
        if self.norm_s == new_norm_s:
            return

        # PROCESS
        if new_norm_s == "spherical_1":
            if self.norm_s == "cartesian":
                if self.DIM == 2:
                    self.coords_m[:, 0], self.coords_m[:, 1] = cart2pol(
                        self.coords_m[:, 0], self.coords_m[:, 1], deg=self.deg
                    )
                else:
                    self.coords_m[:, 0], self.coords_m[:, 1], self.coords_m[:, 2] = (
                        cart2sph(
                            self.coords_m[:, 0],
                            self.coords_m[:, 1],
                            self.coords_m[:, 2],
                            deg=self.deg,
                        )
                    )
            elif self.norm_s == "lateral-polar":
                self.convert_coordinates("cartesian")
                self.convert_coordinates("spherical_1")

        elif new_norm_s == "cartesian":
            if self.norm_s == "spherical_1":
                if self.DIM == 2:
                    self.coords_m[:, 0], self.coords_m[:, 1] = pol2cart(
                        self.coords_m[:, 0], self.coords_m[:, 1], deg=self.deg
                    )
                else:
                    self.coords_m[:, 0], self.coords_m[:, 1], self.coords_m[:, 2] = (
                        sph2cart(
                            self.coords_m[:, 0],
                            self.coords_m[:, 1],
                            self.coords_m[:, 2],
                            deg=self.deg,
                        )
                    )
            elif self.norm_s == "lateral-polar":
                grid_inp = copy.deepcopy(self)
                # Get Y
                self.coords_m[:, 1] = np.sin(np.deg2rad(grid_inp.coords_m[:, 1]))
                # Get X Z
                self.coords_m[:, 0], self.coords_m[:, 2] = pol2cart(
                    grid_inp.coords_m[:, 2], grid_inp.coords_m[:, 0], deg=self.deg
                )
                self.coords_m[:, 0] = self.coords_m[:, 0] * np.cos(
                    np.deg2rad(grid_inp.coords_m[:, 1])
                )
                self.coords_m[:, 2] = self.coords_m[:, 2] * np.cos(
                    np.deg2rad(grid_inp.coords_m[:, 1])
                )

        elif new_norm_s == "lateral-polar":
            grid_cart = copy.deepcopy(self)
            grid_sphr = copy.deepcopy(self)
            grid_cart.convert_coordinates(new_norm_s="cartesian")
            grid_sphr.convert_coordinates(new_norm_s="spherical_1")
            # Radii
            self.coords_m[:, 0] = grid_sphr.coords_m[:, 0]
            # Lateral angle
            self.coords_m[:, 1] = np.rad2deg(np.arcsin(grid_cart.coords_m[:, 1]))
            # Polar angle
            self.coords_m[:, 2], _ = cart2pol(
                grid_cart.coords_m[:, 0], grid_cart.coords_m[:, 2], deg=self.deg
            )
            # Polar angle[-90;-179] change to[270; 181]
            self.coords_m[self.coords_m[:, 2] < -90, 2] = (
                360 + self.coords_m[self.coords_m[:, 2] < -90, 2]
            )
        else:
            print(
                "Grid::convert_coordinates: Unknown coordinates norm %s" % new_norm_s,
                file=sys.stderr,
            )
        self.norm_s = new_norm_s
        return

    def find_closest_point(self, point_v, norm_s, nb_out_point_n=1):
        """
        Find the closest point on the grid compare to the given point (steering vector point)
        ### INPUT
            - point_v: <1x3> (could be : [x, y, z] for cart, [radius, azimuth, elevation] for sph, [radius, lateral, polar] for latpol)
            - norm_s: <str> coordinates norm ('cartesian', 'spherical_1', 'lateral-polar')
            - nb_out_point_n: <int> number of closest points (default: 1)
        ### OUTPUT
            - closest_point_v: <1x3> (could be : [x, y, z] for cart, [radius, azimuth, elevation] for sph, [radius, lateral, polar] for latpol)
            - idx: <int> index of the closest point
            - dist: <float> distance to the closest point
        """
        # todo: nb_out_point_n is not supported, make it
        # CHECK
        assert isinstance(norm_s, str), "First input argument must be a string"
        assert isinstance(nb_out_point_n, int), "First input argument must be a string"
        grid_xyz = copy.deepcopy(self)
        tmppoint_v = np.copy(point_v)

        # Convert to cart to find the euclidean distance
        grid_xyz.convert_coordinates(new_norm_s="cartesian")
        if norm_s == "spherical_1":
            if self.DIM == 2:
                tmppoint_v[0], tmppoint_v[1] = pol2cart(
                    tmppoint_v[0], tmppoint_v[1], deg=self.deg
                )
            else:
                tmppoint_v[0], tmppoint_v[1], tmppoint_v[2] = sph2cart(
                    tmppoint_v[0], tmppoint_v[1], tmppoint_v[2], deg=self.deg
                )
        elif norm_s == "cartesian":
            pass
        else:
            print(
                "Simspace::Grid::find_closest_point: Unknown coordinates norm",
                file=sys.stderr,
            )

        dist_v = np.linalg.norm(grid_xyz.coords_m - tmppoint_v, axis=1)
        idx = np.argmin(dist_v)
        closest_point_v = grid_xyz.coords_m[idx, :]
        return closest_point_v, idx, dist_v[idx]

    def resolve_space(self, spk_array=None, steer_v=None, arr_center=[0, 0, 0]):
        """
        Resolve the space of the speakers and calculate the spacing between sources and the target microphone
        ### Inputs:
        - spk_array: array of sources [n_hp,(x,y,z).T]
        - steer_v: steering vector [x,y,z]
        - arr_center: center of the array [x,y,z] default [0,0,0]
        ### Outputs:
        - dist_target: distance between the source and the microphones [n_hp]
        """
        self.steer_v = steer_v
        if isinstance(self.position_v, type(None)):
            self.position_v = spk_array
            self.n_hp = len(self.position_v)

        target_mic, _, _ = self.find_closest_point(self.steer_v, "spherical_1")

        # calculate the distance between the target and the microphones
        dist_target = np.zeros(self.n_hp)
        for ii in range(self.n_hp):
            if self.DIM == 2:
                dist_target[ii] = np.linalg.norm(
                    target_mic - (self.position_v[ii, :2] - arr_center[0:2])
                )
            elif self.DIM == 3:
                dist_target[ii] = np.linalg.norm(
                    target_mic - (self.position_v[ii, :] - arr_center)
                )

        return dist_target

    def dist_src_mic(self, spk_array, arr_center=[0, 0, 0]):
        """
        calculate the distance between the source and the microphones
        - add coordinate system handler
        TODO:
        - Implement arr_center
        """
        self.position_v = spk_array
        self.n_hp = len(self.position_v)
        mic_array = copy.deepcopy(self)
        mic_array.convert_coordinates(new_norm_s="cartesian")
        pos_mic_array = mic_array.coords_m
        if self.DIM == 2:
            self.dist_src = np.zeros((self.n_hp, self.nb_dir))
            for ii in range(self.n_hp):
                self.dist_src[ii, :] = np.sqrt(
                    (pos_mic_array[:, 0] - self.position_v[ii, 0]) ** 2
                    + (pos_mic_array[:, 1] - self.position_v[ii, 1]) ** 2
                )
        else:
            self.dist_src = np.zeros((self.n_hp, self.nb_dir))
            for ii in range(self.n_hp):
                self.dist_src[ii, :] = np.linalg.norm(
                    pos_mic_array - self.position_v[ii, :], axis=1
                )

        return self.dist_src

    def get_spherical_weighting_harder(self, nb_dz=10000):
        """
        Compute the spherical weighting for each direction of the grid, preserving the surface area of each direction
        even if the grid is not uniform. It computes a weighting for each direction in a spherical grid so
        that each weight is proportional to the actual surface area on a sphere associated with that grid cell.

        ### INPUT
            - nb_dz: <int> number of points to compute the elevation
        """

        # PREPARE
        # Essentially retrieve the original azim and elev vector before mesh
        uni_az_v = np.unique(self.coords_m[:, 1])
        uni_el_v = np.unique(self.coords_m[:, 2])

        # AVOID MODULO ISSUE
        uni_az_bound_v = np.concatenate(
            (
                uni_az_v[len(uni_az_v) - 1][np.newaxis, np.newaxis] - 360,
                uni_az_v[:, np.newaxis],
                uni_az_v[0][np.newaxis, np.newaxis] + 360,
            ),
            axis=0,
        )
        uni_el_bound_v = np.concatenate(
            (
                uni_el_v[len(uni_el_v) - 1][np.newaxis, np.newaxis] - 360,
                uni_el_v[:, np.newaxis],
                uni_el_v[0][np.newaxis, np.newaxis] + 360,
            ),
            axis=0,
        )
        nb_az = len(uni_az_bound_v)
        nb_el = len(uni_el_bound_v)

        # ELEVATION(EQ 10.3)
        dz = 2 / nb_dz
        zz_v = np.linspace(-1, 1, nb_dz)
        Sel_v = np.zeros((nb_el,))
        for id_el in range(1, nb_el - 1):
            u_f = np.sin(
                np.deg2rad(uni_el_bound_v[id_el] + uni_el_bound_v[id_el + 1]) / 2
            )
            l_f = np.sin(
                np.deg2rad(uni_el_bound_v[id_el] + uni_el_bound_v[id_el - 1]) / 2
            )
            mask_b = np.logical_and(zz_v > l_f, zz_v < u_f)
            segment_v = zz_v[mask_b]

            Sel_v[id_el] = np.sum(
                2
                * np.pi
                * np.sqrt(1 - segment_v**2)
                * np.sqrt(1 + (-segment_v / np.sqrt(1 - segment_v**2)) ** 2)
                * dz
            )

        # COMPUTE SURFACE AREA(EQ 10.4)
        Sea_m = np.zeros((nb_az, nb_el))
        for id_el in range(nb_el):
            for id_az in range(1, nb_az - 1):
                Sea_m[id_az, id_el] = (
                    np.deg2rad(uni_az_bound_v[id_az + 1] - uni_az_bound_v[id_az - 1])
                    / (4 * np.pi)
                    * Sel_v[id_el]
                )
        Sea_m = Sea_m[1 : Sea_m.shape[0] - 1, 1 : Sea_m.shape[0] - 1]

        # VECTORIZE IT TO CORRESPOND TO THE GRID STRUCT
        Sea_v = np.zeros((self.nb_dir,))
        for dd in range(self.nb_dir):
            id_az = np.where(uni_az_v == self.coords_m[dd, 1])
            id_el = np.where(uni_el_v == self.coords_m[dd, 2])
            Sea_v[dd] = Sea_m[id_az, id_el]

        weight_v = Sea_v / (4 * np.pi)
        weight_v = weight_v / np.sum(weight_v)
        return weight_v

    # def get_grid_hplane(self):
    #     Grid_hp = copy.deepcopy(self)
    #     mask_hp = self.coords_m[:, 2] == 0
    #     Grid_hp.coords_m = self.coords_m[mask_hp, :]
    #     Grid_hp.coords_m[Grid_hp.coords_m[:, 1] > 180, 1] = Grid_hp.coords_m[
    #                                                             Grid_hp.coords_m[:, 1] > 180, 1] - 360
    #     idx_order = np.argsort(Grid_hp.coords_m[:, 1])
    #     Grid_hp.coords_m = Grid_hp.coords_m[idx_order, :]
    #     return Grid_hp, mask_hp, idx_order
