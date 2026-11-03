#%% Import
from typing import Optional
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
import numpy as np
import ppigrf

from .utils import calc_chaos_internal_field_spherical

# Constants
d2r = np.pi / 180


def _prediction_std(G, Cmpost):
    """Return sqrt(diag(G @ Cmpost @ G.T)) without forming the full product."""

    G_Cmpost = G.dot(Cmpost)
    variance = np.einsum('ij,ij->i', G_Cmpost, G)
    return np.sqrt(np.maximum(variance, 0))


def _transform_horizontal_G(Ge, Gn, first_e, first_n, second_e, second_n):
    """Transform horizontal prediction matrices with a pointwise linear map."""

    G_first = first_e[:, None]*Ge + first_n[:, None]*Gn
    G_second = second_e[:, None]*Ge + second_n[:, None]*Gn
    return G_first, G_second

#%% Model class
class Evaluator(object):
    def __init__(self, model):

        self.set_model(model)

        self.reset_main_field()

        self.reset_ev()

#%% Set model

    def set_model(self, model):

        from .model import Model
        if not isinstance(model, Model):
            raise ValueError('model has to be of class ezie.Model.')

        self.model = model

    @property
    def grid(self):
        return self.model.grid

    @property
    def data(self):
        return self.model.data

#%% Main field

    def reset_main_field(self):
        self._Be0 = None
        self._Bn0 = None
        self._Bu0 = None
        self._B0 = None
        self._b0 = None

    def calc_main_field(self,
                        lat: Optional[np.ndarray] = None,
                        lon: Optional[np.ndarray] = None,
                        r: Optional[float] = None,
                        ret=False):

        if lat is None:
            lat = self.grid.lat.flatten()
        if lon is None:
            lon = self.grid.lon.flatten()
        if r is None:
            r = (6371.2+80)*1e3

        if self.model.B0_model == 'igrf':
            Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(lon, lat, (r - self.model.map_params['RE'])*1e-3, self.data.date))
        elif self.model.B0_model == 'chaos':

            if not isinstance(r, np.ndarray):
                r = np.ones_like(lat)*r

            # Use the complete internal field: time-dependent core plus static
            # lithosphere. External GSM/SM sources are not included.
            B_radius, B_theta, B_phi = calc_chaos_internal_field_spherical(
                self.data.date, r*1e-3, 90-lat, lon, include_static=True)
            Be0, Bn0, Bu0 = B_phi, -B_theta, B_radius

        # ppigrf and CHAOS return nT; EZIE model fields use T internally.
        Be0, Bn0, Bu0 = Be0*1e-9, Bn0*1e-9, Bu0*1e-9

        B0_vector = np.vstack((Be0, Bn0, Bu0))
        B0 = np.linalg.norm(B0_vector, axis = 0)
        b0 = B0_vector / B0

        if ret:
            return B0, Be0, Bn0, Bu0, b0

        self._Be0 = Be0
        self._Bn0 = Bn0
        self._Bu0 = Bu0
        self._B0 = B0
        self._b0 = b0

    @property
    def Be0(self):
        if self._Be0 is None:
            self.calc_main_field()
        return self._Be0

    @property
    def Bn0(self):
        if self._Bn0 is None:
            self.calc_main_field()
        return self._Bn0

    @property
    def Bu0(self):
        if self._Bu0 is None:
            self.calc_main_field()
        return self._Bu0

    @property
    def B0(self):
        if self._B0 is None:
            self.calc_main_field()
        return self._B0

    @property
    def b0(self):
        if self._b0 is None:
            self.calc_main_field()
        return self._b0

#%% Evaludate
    def reset_ev(self):
        self._Be = None
        self._Bn = None
        self._Bu = None

        self._Be_u = None
        self._Bn_u = None
        self._Bu_u = None

        self._Je = None
        self._Jn = None
        self._Je_u = None
        self._Jn_u = None

        self._Jxi = None
        self._Jeta = None
        self._Jxi_u = None
        self._Jeta_u = None

    @property
    def Be(self):
        if self._Be is None:
            self.get_B_predictions()
        return self._Be

    @property
    def Bn(self):
        if self._Bn is None:
            self.get_B_predictions()
        return self._Bn

    @property
    def Bu(self):
        if self._Bu is None:
            self.get_B_predictions()
        return self._Bu

    @property
    def Be_u(self):
        if self._Be_u is None:
            self.get_B_u_predictions()
        return self._Be_u

    @property
    def Bn_u(self):
        if self._Bn_u is None:
            self.get_B_u_predictions()
        return self._Bn_u

    @property
    def Bu_u(self):
        if self._Bu_u is None:
            self.get_B_u_predictions()
        return self._Bu_u

    def get_B_predictions(self,
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None,
                          r: Optional[float] = None,
                          ret=False):
        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()
        if r is None:
            r = (6371.2+80)*1e3

        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r,
                                           self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                           singularity_limit=self.model.slim)

        Be, Bn, Bu = Ge.dot(self.model.m), Gn.dot(self.model.m), Gu.dot(self.model.m)

        if ret:
            return Be, Bn, Bu

        self._Be, self._Bn, self._Bu = Be, Bn, Bu

    def get_B_u_predictions(self,
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None,
                          r: Optional[float] = None,
                          ret=False):
        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()
        if r is None:
            r = (6371.2+80)*1e3

        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r,
                                           self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                           singularity_limit=self.model.slim)

        Be_u = _prediction_std(Ge, self.model.Cmpost)
        Bn_u = _prediction_std(Gn, self.model.Cmpost)
        Bu_u = _prediction_std(Gu, self.model.Cmpost)

        if ret:
            return Be_u, Bn_u, Bu_u

        self._Be_u, self._Bn_u, self._Bu_u = Be_u, Bn_u, Bu_u

    def get_B_qd_u_predictions(self,
                               lat: Optional[np.ndarray] = None,
                               lon: Optional[np.ndarray] = None,
                               r: Optional[float] = None,
                               h: Optional[float] = None):
        """Return magnetic standard deviations in the local QD frame."""

        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()
        if r is None:
            r = (6371.2+80)*1e3
        if h is None:
            h = (r - self.model.map_params['RE'])*1e-3

        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r,
                                           self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                           singularity_limit=self.model.slim)

        ones = np.ones_like(lat, dtype=float)
        zeros = np.zeros_like(lat, dtype=float)
        qde_from_e, qdn_from_e = self.vec_geo2qd(lat, lon, h, ones, zeros)
        qde_from_n, qdn_from_n = self.vec_geo2qd(lat, lon, h, zeros, ones)
        Gqde, Gqdn = _transform_horizontal_G(Ge, Gn,
                                             qde_from_e, qde_from_n,
                                             qdn_from_e, qdn_from_n)

        Bqde_u = _prediction_std(Gqde, self.model.Cmpost)
        Bqdn_u = _prediction_std(Gqdn, self.model.Cmpost)
        Bu_u = _prediction_std(Gu, self.model.Cmpost)
        return Bqde_u, Bqdn_u, Bu_u

    def get_B_magnitude_u_predictions(self, lat, lon, r, dB=None):
        """Return first-order uncertainty of ``|B0 + dB| - |B0|``."""

        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r,
                                           self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                           singularity_limit=self.model.slim)

        if dB is None:
            dBe = Ge.dot(self.model.m)
            dBn = Gn.dot(self.model.m)
            dBu = Gu.dot(self.model.m)
        else:
            dBe, dBn, dBu = dB

        _, Be0, Bn0, Bu0, _ = self.calc_main_field(lat, lon, r, ret=True)
        B_total = np.vstack((Be0 + dBe, Bn0 + dBn, Bu0 + dBu))
        b_total = B_total / np.linalg.norm(B_total, axis=0)

        G_magnitude = (b_total[0, :, None]*Ge +
                       b_total[1, :, None]*Gn +
                       b_total[2, :, None]*Gu)
        return _prediction_std(G_magnitude, self.model.Cmpost)

    @property
    def Je(self):
        if self._Je is None:
            self.get_J_predictions()
        return self._Je

    @property
    def Jn(self):
        if self._Jn is None:
            self.get_J_predictions()
        return self._Jn

    @property
    def Je_u(self):
        if self._Je_u is None:
            self.get_J_u_predictions()
        return self._Je_u

    @property
    def Jn_u(self):
        if self._Jn_u is None:
            self.get_J_u_predictions()
        return self._Jn_u

    def get_J_predictions(self,
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None):
        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()

        Ge, Gn = get_SECS_J_G_matrices(lat, lon,
                                       self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                       singularity_limit=self.model.slim)
        self._Je = Ge.dot(self.model.m)
        self._Jn = Gn.dot(self.model.m)

    def get_J_u_predictions(self,
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None):
        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()

        Ge, Gn = get_SECS_J_G_matrices(lat, lon,
                                       self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                       singularity_limit=self.model.slim)

        self._Je_u = _prediction_std(Ge, self.model.Cmpost)
        self._Jn_u = _prediction_std(Gn, self.model.Cmpost)

    def get_J_qd_u_predictions(self,
                               lat: Optional[np.ndarray] = None,
                               lon: Optional[np.ndarray] = None,
                               h: Optional[float] = None):
        """Return current-density standard deviations in the local QD frame."""

        if lat is None:
            lat = self.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.grid.lon_mesh.flatten()
        if h is None:
            h = self.model.map_params['HI']*1e-3

        Ge, Gn = get_SECS_J_G_matrices(lat, lon,
                                       self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                       singularity_limit=self.model.slim)

        ones = np.ones_like(lat, dtype=float)
        zeros = np.zeros_like(lat, dtype=float)
        qde_from_e, qdn_from_e = self.vec_geo2qd(lat, lon, h, ones, zeros)
        qde_from_n, qdn_from_n = self.vec_geo2qd(lat, lon, h, zeros, ones)
        Gqde, Gqdn = _transform_horizontal_G(Ge, Gn,
                                             qde_from_e, qde_from_n,
                                             qdn_from_e, qdn_from_n)

        Jqde_u = _prediction_std(Gqde, self.model.Cmpost)
        Jqdn_u = _prediction_std(Gqdn, self.model.Cmpost)
        return Jqde_u, Jqdn_u

    @property
    def Jxi(self):
        if self._Jxi is None:
            self.get_J_cs()
        return self._Jxi

    @property
    def Jeta(self):
        if self._Jeta is None:
            self.get_J_cs()
        return self._Jeta

    @property
    def Jxi_u(self):
        if self._Jxi_u is None:
            self.get_J_u_cs()
        return self._Jxi_u

    @property
    def Jeta_u(self):
        if self._Jeta_u is None:
            self.get_J_u_cs()
        return self._Jeta_u

    def get_J_cs(self):
        Jxi, Jeta = self.grid.projection.vector_cube_projection(self.Je, self.Jn,
                                                                self.grid.lon_mesh.flatten(),
                                                                self.grid.lat_mesh.flatten(),
                                                                return_xi_eta=False)
        self._Jxi = Jxi
        self._Jeta = Jeta

    def get_J_u_cs(self):
        lat = self.grid.lat_mesh.flatten()
        lon = self.grid.lon_mesh.flatten()
        Ge, Gn = get_SECS_J_G_matrices(lat, lon,
                                       self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.grid.R,
                                       singularity_limit=self.model.slim)

        ones = np.ones_like(lat, dtype=float)
        zeros = np.zeros_like(lat, dtype=float)
        xi_from_e, eta_from_e = self.grid.projection.vector_cube_projection(
            ones, zeros, lon, lat, return_xi_eta=False)
        xi_from_n, eta_from_n = self.grid.projection.vector_cube_projection(
            zeros, ones, lon, lat, return_xi_eta=False)
        Gxi, Geta = _transform_horizontal_G(Ge, Gn,
                                            xi_from_e, xi_from_n,
                                            eta_from_e, eta_from_n)

        self._Jxi_u = _prediction_std(Gxi, self.model.Cmpost)
        self._Jeta_u = _prediction_std(Geta, self.model.Cmpost)

#%% Magnetic frame

    def geo2qd(self, lat, lon, h, xe, xn):
        mlat, mlon, mlt = self.coords_geo2qd(lat, lon, h)
        mxe, mxn = self.vec_geo2qd(lat, lon, h, xe, xn)
        return mlat, mlon, mlt, mxe, mxn

    def coords_geo2qd(self, lat, lon, h):
        mlat, mlon = self.model.apx.geo2qd(lat, lon, h)
        mlt = self.model.apx.mlon2mlt(mlon, self.data.date)
        return mlat, mlon, mlt

    def vec_geo2qd(self, lat, lon, h, xe, xn):
        f1, f2 = self.model.apx.basevectors_qd(lat.flatten(), lon.flatten(), h, coords='geo')
        F = np.zeros(f1.shape[1])
        for j in range(0, len(F)):
            F[j] = np.cross(f1[:, j], f2[:, j])
        mxe = (f1[0] * xe.flatten() + f1[1] * xn.flatten())/F
        mxn = (f2[0] * xe.flatten() + f2[1] * xn.flatten())/F
        return mxe.reshape(xe.shape), mxn.reshape(xe.shape)
