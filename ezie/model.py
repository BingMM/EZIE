#%% Import
from typing import Union, Optional
from .data import Data
from .regularization_optimizer import RegularizationOptimizer
from .evaluator import Evaluator
from secsy import spherical
from secsy import get_SECS_B_G_matrices, get_SECS_J_G_matrices
from secsy import CSgrid, CSprojection
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from apexpy import Apex
import ppigrf

# Constants
d2r = np.pi / 180

#%% Model class
class Model(object):
    def __init__(self,
                 map_params: Optional[dict] = None,
                 center_mem: Optional[int] = 0,
                 l1: Optional[Union[float, int]] = None,
                 l2: Optional[Union[float, int]] = None,
                 B0_model: Optional[str] = 'chaos'):
        
        self.B0_model = B0_model
        
        self.reset_model(map_params, center_mem, l1, l2)

#%% Reset everything
    
    def reset_model(self,
                    map_params:Optional[dict] = None,
                    center_mem: Optional[int] = 0,
                    l1: Optional[Union[float, int]] = None,
                    l2: Optional[Union[float, int]] = None):
        
        self.clear_data()
        self.reset_grid(map_params, center_mem)
        self.reset_all_regularization(l1, l2)
        self.reset_design()
        self.reset_solution()
        self.reset_ev()

#%% Data
    
    def add_data(self, data: Data):
        self.clear_data()
        self.data = data
        
    def clear_data(self):
        self.data = None
        self._d = None
        self._Q = None
        self._Qinv = None

    @property
    def d(self):
        if self._d is None:
            self.generate_d()
        return self._d

    def generate_d(self):
        
        self._d = []
        if 's' in self.data.comp:
            B0 = self.ev.calc_main_field(self.data.lat, self.data.lon, self.data.r, ret=True)[0]
            dB_parallel = ((self.data.B + B0)**2 - B0**2) / (2 * B0)
            self._d.append(dB_parallel)
            del dB_parallel
        if 'e' in self.data.comp:
            self._d.append(self.data.Be)
        if 'n' in self.data.comp:
            self._d.append(self.data.Bn)
        if 'u' in self.data.comp:
            self._d.append(self.data.Bu)
        
        self._d = np.hstack(self._d)

    @property
    def Q(self):
        if self._Q is None:
            self.generate_Q()
        return self._Q

    def generate_Q(self):
        if self.data.comp == 's' or len(self.data.comp) == 1:
            self.generate_Q_diag()
        else:
            self.generate_Q_dense()
        
    def generate_Q_dense(self):
        self._Q = np.vstack([np.hstack([np.diag(self.data.cov_ee), np.diag(self.data.cov_en), np.diag(self.data.cov_eu)]), 
                             np.hstack([np.diag(self.data.cov_en), np.diag(self.data.cov_nn), np.diag(self.data.cov_nu)]), 
                             np.hstack([np.diag(self.data.cov_eu), np.diag(self.data.cov_nu), np.diag(self.data.cov_uu)])])
        
    def generate_Q_diag(self):
        
        self._Q = []
        if 's' in self.data.comp:
            self._Q.append(self.data.cov_mag)
        if 'e' in self.data.comp:
            self._Q.append(self.data.cov_ee)
        if 'n' in self.data.comp:
            self._Q.append(self.data.cov_nn)
        if 'u' in self.data.comp:
            self._Q.append(self.data.cov_uu)
        
        self._Q = np.diag(np.hstack(self._Q))
    
    @property
    def Qinv(self):
        if self._Qinv is None:
            self._Qinv = np.linalg.inv(self.Q)
        return self._Qinv
    
#%% Grid
    
    def reset_grid(self,
                   map_params:Optional[dict] = None,
                   center_mem: Optional[int] = 0):

        self.map_params = {'LRES': 40e3,
                           'WRES': 20e3,
                           'W': 2000e3,
                           'L': 1500e3,#700e3,
                           'RE': 6371.2e3,
                           'HI': 110e3}
        
        if not map_params is None:
            self.map_params.update(map_params)
        
        self.RI = self.map_params['RE'] + self.map_params['HI']
        self.cm = center_mem
        self._grid = None
        self.slim = np.min([self.map_params['WRES'], self.map_params['LRES']])/2

    @property
    def grid(self):
        if self._grid is None and self.data == None:
            raise ValueError('Add data before trying to access grid object.')
        elif self._grid is None:
            self._grid = self.create_grid(self.data.mems[self.cm].lat,
                                          self.data.mems[self.cm].lon)
        return self._grid

    def create_grid(self, lat, lon):
        ### Timespan and satellite velocity
        # calculate SC velocity
        te, tn = spherical.tangent_vector(lat[:-1], lon[:-1],
                                          lat[1:], lon[1:])

        ve = np.hstack((te, np.nan))
        vn = np.hstack((tn, np.nan))

        # get index of central point of analysis interval:
        tm = te.size//2

        # spacecraft velocity at central time:
        v = np.array((ve[tm], vn[tm]))

        # spacecraft lat and lon at central time:
        sc_lat0 = lat[tm]
        sc_lon0 = lon[tm]

        # limits of analysis interval:
        t0 = 0
        t1 = te.size - 1

        # get unit vectors pointing at satellite (Cartesian vectors)
        rs = []
        for t in [t0, tm, t1]:
            rs.append(np.array([np.cos(lat[t] * d2r) * np.cos(lon[t] * d2r),
                                np.cos(lat[t] * d2r) * np.sin(lon[t] * d2r),
                                np.sin(lat[t] * d2r)]))

        ### Define map paramters
        # dimensions of analysis region/d (in km)
        W = self.map_params['W'] + self.RI * np.arccos(np.sum(rs[0]*rs[-1]))

        position = (sc_lon0, sc_lat0)
        orientation = (v[1], -v[0]) # align coordinate system such that xi axis points right wrt to satellite velocity vector, and eta along velocity
        projection = CSprojection(position, orientation)
        return CSgrid(projection, self.map_params['L'], W, self.map_params['LRES'], self.map_params['WRES'], R = self.RI)

#%% Regularization
    
    def reset_regularization(self,
                             l1: Optional[Union[float, int]] = None,
                             l2: Optional[Union[float, int]] = None):
        self._l1 = l1
        self._l2 = l2
        self._reg = None
        #self._regOpt = None # Found the problem!!!!!!

    def reset_all_regularization(self,
                             l1: Optional[Union[float, int]] = None,
                             l2: Optional[Union[float, int]] = None):
        self._regOpt = None
        self._apx = None
        self._LTL = None
        self._ltl_mag = None
        self.reset_regularization(l1, l2)

    @property
    def l1(self):
        if self._l1 is None:
            self._l1 = self.regOpt.l1_opt
        return self._l1

    @property
    def l2(self):
        if self._l2 is None:
            self._l2 = self.regOpt.l2_opt
        return self._l2
    
    @property
    def regOpt(self):
        if self._regOpt is None:
            self._regOpt = RegularizationOptimizer(self)
        return self._regOpt            

    @property
    def LTL(self):
        if self._LTL is None:
            self._LTL = self.get_LL()
        return self._LTL
    
    @property
    def ltl_mag(self):
        if self._ltl_mag is None:
           self._ltl_mag = np.median(np.diag(self.LTL)) 
        return self._ltl_mag

    @property
    def reg(self):
        if self._reg is None:
            self._reg = 10**self.l1 * self.gtg_mag * np.eye(self.LTL.shape[0]) + 10**self.l2 * self.gtg_mag / self.ltl_mag * self.LTL
        return self._reg

    @property
    def apx(self):
        if self._apx is None:
            self._apx = Apex(self.data.date.year, refh=self.map_params['HI']*1e-3)
        return self._apx

    def get_LL(self):#.grid, apx, hI=110):
        # set up matrix that produces gradients in the magnetic eastward direction, and use to construct regularization matrix LL:
        Le, Ln = self.grid.get_Le_Ln()
        f1, f2 = self.apx.basevectors_qd(self.grid.lat.flatten(), self.grid.lon.flatten(), self.map_params['HI']*1e-3, coords='geo')
        f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
        L = Le * f1[0].reshape((-1, 1)) + Ln * f1[1].reshape((-1, 1))
        LL = L.T.dot(L)
        return LL

#%% Design

    def reset_design(self):
        self._G = None
        self._GTG = None
        self._gtg_mag = None
        self._GTd = None

    def generate_G(self):
        Ge, Gn, Gu = get_SECS_B_G_matrices(self.data.lat, self.data.lon, self.data.r, 
                              self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.RI,
                              singularity_limit=self.slim)
        
        self._G = []
        if 's' in self.data.comp:
            #G_scalar = Ge * self.ev.b0[0].reshape((-1, 1)) + Gn * self.ev.b0[1].reshape((-1, 1)) + Gu * self.ev.b0[2].reshape((-1, 1))
            G_scalar = Ge * self.ev.b0[0] + Gn * self.ev.b0[1] + Gu * self.ev.b0[2]
            self._G.append(G_scalar)
            del G_scalar
        if 'e' in self.data.comp:
            self._G.append(Ge)
            del Ge
        if 'n' in self.data.comp:
            self._G.append(Gn)
            del Gn
        if 'u' in self.data.comp:
            self._G.append(Gu)
            del Gu

        self._G = np.vstack(self._G)

    @property
    def G(self):
        if self._G is None:
            self.generate_G()
        return self._G

    @property
    def GTG(self):
        if self._GTG is None:
            self._GTG = self.G.T@self.Qinv@self.G
        return self._GTG
    
    @property
    def gtg_mag(self):
        if self._gtg_mag is None:
            self._gtg_mag = np.median(np.diag(self.GTG))
        return self._gtg_mag
    
    @property
    def GTd(self):
        if self._GTd is None:
            self._GTd = self.G.T@self.Qinv@self.d
        return self._GTd

#%% Evaluator

    def reset_ev(self):
        self._ev = None

    @property
    def ev(self):
        if self._ev is None:
            self._ev = Evaluator(self)
        return self._ev

#%% Solve inverse problem

    def reset_solution(self):
        self._c_factor = None
        self._Cmpost = None
        self._Rm = None
        self._m = None

    @property
    def c_factor(self):
        if self._c_factor is None:
            self._c_factor = cho_factor(self.GTG + self.reg, overwrite_a=False, check_finite=False)
        return self._c_factor

    @property
    def Cmpost(self):
        if self._Cmpost is None:
            inv_L = solve_triangular(self.c_factor[0], np.eye(self.c_factor[0].shape[0]), lower=self.c_factor[1])
            self._Cmpost = inv_L.T @ inv_L
        return self._Cmpost
    
    @property
    def Rm(self):
        if self._Rm is None:
            self._Rm = self.Cmpost.dot(self.GTG)
        return self._Rm
    
    @property
    def m(self):
        if self._m is None:
            self._m = cho_solve(self.c_factor, self.GTd, check_finite=False)
        return self._m
    
