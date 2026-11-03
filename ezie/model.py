#%% Import
from typing import Union, Optional
from .data import Data
from .regularization_optimizer import RegularizationOptimizer
from .evaluator import Evaluator
from .resolution import Resolution
from secsy import spherical
from secsy import get_SECS_B_G_matrices
from secsy import CSgrid, CSprojection
import numpy as np
from apexpy import Apex
from nandir import QuadraticRegularization, Solver
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import block_diag, csr_matrix, diags

# Constants
d2r = np.pi / 180


def _normalize_fit_mode(fit_mode):
    if not isinstance(fit_mode, str):
        raise TypeError("fit_mode must be 'absolute' or 'gradient'")

    fit_mode = fit_mode.lower()
    if fit_mode not in ('absolute', 'gradient'):
        raise ValueError("fit_mode must be 'absolute' or 'gradient'")

    return fit_mode

#%% Model class
class Model(object):
    def __init__(self,
                 map_params: Optional[dict] = None,
                 center_mem: Optional[int] = 0,
                 l1: Optional[Union[float, int]] = None,
                 l2: Optional[Union[float, int]] = None,
                 B0_model: Optional[str] = 'chaos',
                 fit_mode: Optional[str] = 'absolute'):
        
        self.B0_model = B0_model
        self.fit_mode = _normalize_fit_mode(fit_mode)
        
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
        self.reset_resolution()

#%% Data
    
    def add_data(self, data: Data):
        if self.fit_mode == 'gradient' and getattr(data, 'comp', None) != 's':
            raise ValueError("fit_mode='gradient' currently requires scalar-only data")

        self.clear_data()
        self.data = data

        # Every downstream quantity depends directly or indirectly on the
        # observations. Preserve the model configuration, but rebuild all
        # data-dependent objects lazily for the new event.
        self._grid = None
        self.reset_all_regularization(self._configured_l1, self._configured_l2)
        self.reset_design()
        self.reset_solution()
        self.reset_ev()
        self.reset_resolution()
        
    def clear_data(self):
        self.data = None
        self._d_absolute = None
        self._d = None
        self._D = None
        self._Q_absolute = None
        self._Q = None
        self._Q_factor = None
        self._Qinv = None
        self._track_offsets = None

    @property
    def d_absolute(self):
        """Observations before the optional fit-space transformation."""

        if self._d_absolute is None:
            values = []
            if 's' in self.data.comp:
                values.append(self.data.B)
            if 'e' in self.data.comp:
                values.append(self.data.Be)
            if 'n' in self.data.comp:
                values.append(self.data.Bn)
            if 'u' in self.data.comp:
                values.append(self.data.Bu)
            self._d_absolute = np.hstack(values)

        return self._d_absolute

    @property
    def d(self):
        if self._d is None:
            self.generate_d()
        return self._d

    def generate_d(self):
        if self.fit_mode == 'gradient':
            self._d = self.D @ self.d_absolute
            return

        values = []
        if 's' in self.data.comp:
            B0 = self.ev.calc_main_field(self.data.lat, self.data.lon, self.data.r, ret=True)[0]
            dB_parallel = ((self.data.B + B0)**2 - B0**2) / (2 * B0)
            values.append(dB_parallel)
            del dB_parallel
        if 'e' in self.data.comp:
            values.append(self.data.Be)
        if 'n' in self.data.comp:
            values.append(self.data.Bn)
        if 'u' in self.data.comp:
            values.append(self.data.Bu)

        self._d = np.hstack(values)

    @property
    def D(self):
        """Blockwise first-difference operator for scalar MEM tracks."""

        if self.fit_mode != 'gradient':
            return None
        if self._D is None:
            blocks = []
            for mem in self.data.mems:
                if mem.nd < 2:
                    raise ValueError('Each MEM needs at least two samples for gradient fitting')
                ones = np.ones(mem.nd-1)
                blocks.append(diags((-ones, ones), (0, 1), shape=(mem.nd-1, mem.nd)))
            self._D = block_diag(blocks, format='csr')
        return self._D

    @property
    def Q_absolute(self):
        """Observation covariance before the optional fit transformation."""

        if self._Q_absolute is None:
            self.generate_Q_absolute()
        return self._Q_absolute

    @property
    def Q(self):
        if self._Q is None:
            if self.fit_mode == 'gradient':
                Q_sparse = csr_matrix(self.Q_absolute)
                self._Q = (self.D @ Q_sparse @ self.D.T).toarray()
            else:
                self._Q = self.Q_absolute
        return self._Q

    def generate_Q_absolute(self):
        if self.data.comp == 's' or len(self.data.comp) == 1:
            self.generate_Q_absolute_diag()
        else:
            self.generate_Q_absolute_dense()
        
    def generate_Q_absolute_dense(self):
        covariance = {('e', 'e'): 'cov_ee',
                      ('e', 'n'): 'cov_en',
                      ('e', 'u'): 'cov_eu',
                      ('n', 'e'): 'cov_en',
                      ('n', 'n'): 'cov_nn',
                      ('n', 'u'): 'cov_nu',
                      ('u', 'e'): 'cov_eu',
                      ('u', 'n'): 'cov_nu',
                      ('u', 'u'): 'cov_uu'}

        blocks = [[np.diag(getattr(self.data, covariance[(row, column)])) for column in self.data.comp]
                  for row in self.data.comp]
        self._Q_absolute = np.block(blocks)
        
    def generate_Q_absolute_diag(self):
        values = []
        if 's' in self.data.comp:
            values.append(self.data.cov_mag)
        if 'e' in self.data.comp:
            values.append(self.data.cov_ee)
        if 'n' in self.data.comp:
            values.append(self.data.cov_nn)
        if 'u' in self.data.comp:
            values.append(self.data.cov_uu)

        self._Q_absolute = np.diag(np.hstack(values))

    @property
    def Q_factor(self):
        if self._Q_factor is None:
            self._Q_factor = cho_factor(self.Q, overwrite_a=False, check_finite=False)
        return self._Q_factor

    def solve_Q(self, rhs):
        """Apply the inverse fit covariance without explicitly forming it."""

        return cho_solve(self.Q_factor, rhs, check_finite=False)

    def weighted_residual_norm(self, residual):
        return residual.T @ self.solve_Q(residual)
    
    @property
    def Qinv(self):
        if self._Qinv is None:
            self._Qinv = self.solve_Q(np.eye(self.Q.shape[0]))
        return self._Qinv
    
#%% Grid
    
    def reset_grid(self,
                   map_params:Optional[dict] = None,
                   center_mem: Optional[int] = 0):

        self.map_params = {'LRES': 20e3,
                           'WRES': 40e3,
                           'W': 2400e3,
                           'L': 2000e3,#700e3,
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
        # Level-2 geolocation is stored as float32. Promote it before the
        # near-pole projection arithmetic, where float32 cancellation can
        # violate secsy's rotation consistency checks.
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

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

        # Align eta with the satellite velocity and xi toward its right.
        position = (float(sc_lon0), float(sc_lat0))
        orientation = (float(v[1]), float(-v[0]))
        projection = CSprojection(position, orientation)
        return CSgrid(projection, self.map_params['L'], W, self.map_params['LRES'], self.map_params['WRES'], R = self.RI)

#%% Regularization
    
    def reset_regularization(self,
                             l1: Optional[Union[float, int]] = None,
                             l2: Optional[Union[float, int]] = None):
        self._l1 = l1
        self._l2 = l2
        self._reg = None
        self._reg_sparse = None
        self._reg_terms = None
        #self._regOpt = None # Found the problem!!!!!!

    def reset_all_regularization(self,
                             l1: Optional[Union[float, int]] = None,
                             l2: Optional[Union[float, int]] = None):
        self._configured_l1 = l1
        self._configured_l2 = l2
        self._regOpt = None
        self._apx = None
        self._L = None
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
    def L(self):
        if self._L is None:
            self._L = self.get_L()
        return self._L
    
    @property
    def ltl_mag(self):
        if self._ltl_mag is None:
            self._ltl_mag = np.median(self.LTL.diagonal())
        return self._ltl_mag

    @property
    def reg(self):
        if self._reg is None:
            self._reg = self.reg_sparse.toarray()
        return self._reg

    @property
    def reg_sparse(self):
        """Return the complete regularization matrix in sparse form."""

        if self._reg_sparse is None:
            ridge = 10**self.l1 * self.gtg_mag
            smoothness = 10**self.l2 * self.gtg_mag / self.ltl_mag
            self._reg_sparse = self.LTL.copy()
            self._reg_sparse *= smoothness
            self._reg_sparse.setdiag(
                self._reg_sparse.diagonal() + ridge)
            self._reg_sparse.eliminate_zeros()
        return self._reg_sparse

    @property
    def reg_terms(self):
        """Return sparse regularization terms for the nandir solver."""

        if self._reg_terms is None:
            ridge = 10**self.l1 * self.gtg_mag
            smoothness = 10**self.l2 * self.gtg_mag / self.ltl_mag
            identity = diags(np.ones(self.LTL.shape[0]), format='csr')
            self._reg_terms = (
                QuadraticRegularization(identity, lreg=ridge),
                QuadraticRegularization(self.LTL, lreg=smoothness))
        return self._reg_terms

    @property
    def apx(self):
        if self._apx is None:
            self._apx = Apex(self.data.date.year, refh=self.map_params['HI']*1e-3)
        return self._apx

    def get_L(self):
        """Return the sparse magnetic-eastward derivative operator."""

        Le, Ln = self.grid.get_Le_Ln(return_sparse=True)
        f1, f2 = self.apx.basevectors_qd(self.grid.lat.flatten(), self.grid.lon.flatten(), self.map_params['HI']*1e-3, coords='geo')
        f1 = f1/np.linalg.norm(f1, axis = 0) # normalize
        L = (Le.multiply(f1[0].reshape((-1, 1))) +
             Ln.multiply(f1[1].reshape((-1, 1))))
        return L.tocsr()

    def get_LL(self):
        """Return sparse ``L.T @ L`` for magnetic-east smoothing."""

        return (self.L.T @ self.L).tocsr()

#%% Design

    def reset_design(self):
        self._G_absolute = None
        self._G = None
        self._GTG = None
        self._gtg_mag = None
        self._GTd = None

    @property
    def G_absolute(self):
        """SECS observation matrix before the optional fit transformation."""

        if self._G_absolute is None:
            self.generate_G_absolute()
        return self._G_absolute

    def generate_G_absolute(self):
        Ge, Gn, Gu = get_SECS_B_G_matrices(self.data.lat, self.data.lon, self.data.r, 
                              self.grid.lat.flatten(), self.grid.lon.flatten(), RI=self.RI,
                              singularity_limit=self.slim)
        
        values = []
        if 's' in self.data.comp:
            if self.fit_mode == 'gradient' and all(
                    value is not None for mem in self.data.mems
                    for value in (mem.B0e, mem.B0n, mem.B0u, mem.B0)):
                b0 = np.vstack((self.data.B0e, self.data.B0n, self.data.B0u)) / self.data.B0
            else:
                b0 = self.ev.calc_main_field(
                    self.data.lat, self.data.lon, self.data.r, ret=True)[4]
            G_scalar = (Ge * b0[0].reshape((-1, 1)) +
                        Gn * b0[1].reshape((-1, 1)) +
                        Gu * b0[2].reshape((-1, 1)))
            values.append(G_scalar)
            del G_scalar
        if 'e' in self.data.comp:
            values.append(Ge)
            del Ge
        if 'n' in self.data.comp:
            values.append(Gn)
            del Gn
        if 'u' in self.data.comp:
            values.append(Gu)
            del Gu

        self._G_absolute = np.vstack(values)

    def generate_G(self):
        if self.fit_mode == 'gradient':
            self._G = self.D @ self.G_absolute
        else:
            self._G = self.G_absolute

    @property
    def G(self):
        if self._G is None:
            self.generate_G()
        return self._G

    @property
    def GTG(self):
        if self._GTG is None:
            self._GTG = self.G.T @ self.solve_Q(self.G)
        return self._GTG
    
    @property
    def gtg_mag(self):
        if self._gtg_mag is None:
            self._gtg_mag = np.median(np.diag(self.GTG))
        return self._gtg_mag
    
    @property
    def GTd(self):
        if self._GTd is None:
            self._GTd = self.G.T @ self.solve_Q(self.d)
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
        self._solver = None
        self._track_offsets = None

    @property
    def solver(self):
        if self._solver is None:
            self._solver = Solver(
                GTG=self.GTG, GTd=self.GTd, reg=self.reg_terms)
        return self._solver

    @property
    def c_factor(self):
        return self.solver.c_factor

    @property
    def Cmpost(self):
        return self.solver.Cmpost
    
    @property
    def Rm(self):
        return self.solver.Rm
    
    @property
    def m(self):
        if self.solver.m is None:
            self.solver.solve()
        return self.solver.m

    @property
    def track_offsets(self):
        """Return scalar MEM offsets conditional on the gradient-fit solution."""

        if self.fit_mode != 'gradient':
            raise ValueError("track_offsets are available only for fit_mode='gradient'")

        if self._track_offsets is None:
            offsets = []
            start = 0
            prediction = self.G_absolute @ self.m
            for mem in self.data.mems:
                stop = start + mem.nd
                residual = self.d_absolute[start:stop] - prediction[start:stop]
                variance = mem.cov_mag
                valid = np.isfinite(residual) & np.isfinite(variance) & (variance > 0)
                if not np.any(valid):
                    offsets.append(np.nan)
                else:
                    weights = 1 / variance[valid]
                    offsets.append(np.sum(weights*residual[valid]) / np.sum(weights))
                start = stop
            self._track_offsets = np.asarray(offsets)

        return self._track_offsets

#%% Resolution

    def reset_resolution(self):
        self._resolution = None

    @property
    def resolution(self):
        if self._resolution is None:
            self._resolution = Resolution(self)
        return self._resolution
