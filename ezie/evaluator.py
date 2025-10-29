#%% Import
from typing import Optional
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
import numpy as np
import ppigrf
import chaosmagpy as cp

# Constants
d2r = np.pi / 180

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
            lat = self.model.grid.lat.flatten()
        if lon is None:
            lon = self.model.grid.lon.flatten()
        if r is None:
            r = (6371.2+80)*1e3
        
        if self.model.B0_model == 'igrf':
            Be0, Bn0, Bu0 = map(np.ravel, ppigrf.igrf(lon, lat, (r - self.model.map_params['RE'])*1e-3, self.model.data.date))
        elif self.model.B0_model == 'chaos':
            
            if not isinstance(r, np.ndarray):
                r = np.ones_like(lat)*r
            
            time = cp.data_utils.mjd2000(self.model.data.date.year, 
                                         self.model.data.date.month, 
                                         self.model.data.date.day)  # modified Julian date
            
            chaos_model = cp.load_CHAOS_matfile('/home/bing/Dropbox/work/code/repos/EZIE/data/CHAOS-8.4.mat')
            B_radius, B_theta, B_phi = chaos_model.synth_values_tdep(time, r*1e-3, 90-lat, lon)
            Be0, Bn0, Bu0 = B_phi, -B_theta, B_radius
        
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
            self.get_B_predictions()
        return self._Be_u

    @property
    def Bn_u(self):
        if self._Bn_u is None:
            self.get_B_predictions()
        return self._Bn_u
    
    @property
    def Bu_u(self):
        if self._Bu_u is None:
            self.get_B_predictions()
        return self._Bu_u

    def get_B_predictions(self,
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None,
                          r: Optional[float] = None,
                          ret=False):
        if lat is None:
            lat = self.model.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.model.grid.lon_mesh.flatten()
        if r is None:
            r = (6371.2+80)*1e3
        
        Ge, Gn, Gu = get_SECS_B_G_matrices(lat, lon, r, 
                                           self.model.grid.lat.flatten(), self.model.grid.lon.flatten(), RI=self.model.grid.R,
                                           singularity_limit=self.model.slim)
        
        Be, Bn, Bu = Ge.dot(self.model.m), Gn.dot(self.model.m), Gu.dot(self.model.m)
        Be_u = np.sqrt(np.diag(Ge.dot(self.model.Cmpost).dot(Ge.T)))
        Bn_u = np.sqrt(np.diag(Gn.dot(self.model.Cmpost).dot(Gn.T)))
        Bu_u = np.sqrt(np.diag(Gu.dot(self.model.Cmpost).dot(Gu.T)))
        
        if ret:
            return Be, Bn, Bu, Be_u, Bn_u, Bu_u
        
        self._Be, self._Bn, self._Bu = Be, Bn, Bu
        self._Be_u, self._Bn_u, self._Bu_u = Be_u, Bn_u, Bu_u
        
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
            self.get_J_predictions()
        return self._Je_u
    
    @property
    def Jn_u(self):
        if self._Jn_u is None:
            self.get_J_predictions()
        return self._Jn_u
    
    def get_J_predictions(self, 
                          lat: Optional[np.ndarray] = None,
                          lon: Optional[np.ndarray] = None):
        if lat is None:
            lat = self.model.grid.lat_mesh.flatten()
        if lon is None:
            lon = self.model.grid.lon_mesh.flatten()

        Ge, Gn = get_SECS_J_G_matrices(lat, lon, 
                                       self.model.grid.lat.flatten(), self.model.grid.lon.flatten(), RI=self.model.grid.R,
                                       singularity_limit=self.model.slim)
        self._Je = Ge.dot(self.model.m)
        self._Jn = Gn.dot(self.model.m)
        
        self._Je_u = np.sqrt(np.diag(Ge.dot(self.model.Cmpost).dot(Ge.T)))
        self._Jn_u = np.sqrt(np.diag(Gn.dot(self.model.Cmpost).dot(Gn.T)))

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
            self.get_J_cs()
        return self._Jxi_u

    @property
    def Jeta_u(self):
        if self._Jeta_u is None:
            self.get_J_cs()
        return self._Jeta_u

    def get_J_cs(self):
        Jxi, Jeta = self.model.grid.projection.vector_cube_projection(self.Je, self.Jn, 
                                                                self.model.grid.lon_mesh.flatten(), 
                                                                self.model.grid.lat_mesh.flatten(),
                                                                return_xi_eta=False)
        self._Jxi = Jxi
        self._Jeta = Jeta
        
        Jxi_u, Jeta_u = self.model.grid.projection.vector_cube_projection(self.Je_u, self.Jn_u, 
                                                                self.model.grid.lon_mesh.flatten(), 
                                                                self.model.grid.lat_mesh.flatten(),
                                                                return_xi_eta=False)
        self._Jxi_u = Jxi_u
        self._Jeta_u = Jeta_u
    
#%% Magnetic frame
    
    def geo2qd(self, lat, lon, h, xe, xn):
        mlat, mlon, mlt = self.coords_geo2qd(lat, lon, h)
        mxe, mxn = self.vec_geo2qd(lat, lon, h, xe, xn)
        return mlat, mlon, mlt, mxe, mxn

    def coords_geo2qd(self, lat, lon, h):
        mlat, mlon = self.model.apx.geo2qd(lat, lon, h)
        mlt = self.model.apx.mlon2mlt(mlon, self.model.data.date)
        return mlat, mlon, mlt
    
    def vec_geo2qd(self, lat, lon, h, xe, xn):
        f1, f2 = self.model.apx.basevectors_qd(lat.flatten(), lon.flatten(), h, coords='geo')
        F = np.zeros(f1.shape[1])
        for j in range(0, len(F)):
            F[j] = np.cross(f1[:, j], f2[:, j])
        mxe = (f1[0] * xe.flatten() + f1[1] * xn.flatten())/F
        mxn = (f2[0] * xe.flatten() + f2[1] * xn.flatten())/F
        return mxe.reshape(xe.shape), mxn.reshape(xe.shape)
