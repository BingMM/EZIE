#%% Import
from typing import Optional
import numpy as np
from scipy.linalg import cho_solve
from scipy.interpolate import UnivariateSpline
from kneed import KneeLocator
from itertools import product
from tqdm import tqdm

#%% Constants
d2r = np.pi / 180

#%% Functions

def calc_gini(x):
    N = x.size
    x = np.sort(np.abs(x))
    norm_l1 = np.sum(np.abs(x))
    k = np.arange(1, N + 1)
    gini = np.sum((x / norm_l1) * ((N - k + 0.5) / N))
    gini = 1 - 2 * gini
    return gini

def calc_gini_matrix(x):
    N = x.shape[0]
    k = np.tile(np.arange(1, N+1), (N, 1))    
    x = np.sort(abs(x), axis=1)
    norm_l1 = np.tile(np.sum(abs(x), axis=1).reshape(-1, 1), (1, N))    
    gini = (x / norm_l1) * ((N - k + 0.5)/N)
    gini = 1 - 2*np.sum(gini, axis=1)    
    return gini

#%% Model class

class RegularizationOptimizer(object):
    def __init__(self, 
                 model,
                 l1_lower: Optional[int] = -3, 
                 l1_upper: Optional[int] =  3, 
                 l1_steps: Optional[int] = 10,
                 l2_lower: Optional[int] = -2, 
                 l2_upper: Optional[int] =  6, 
                 l2_steps: Optional[int] = 10,
                 iterations: Optional[int] = 2,
                 fit_steps: Optional[int] = 50,
                 m_id: Optional[int] = None):
        """Optimizer for determining optimal L1/L2 regularization parameters."""
        
        # --- Core parameters
        self.l1_lower = l1_lower
        self.l1_upper = l1_upper
        self.l1_steps = l1_steps
        self.l2_lower = l2_lower
        self.l2_upper = l2_upper
        self.l2_steps = l2_steps
        self.iterations = iterations
        self.fit_steps = fit_steps
        self._m_id = m_id

        # --- Everything else
        self.set_model(model)

    @property
    def grid(self):
        return self.model.grid
    
    @property
    def data(self):
        return self.model.data

#%% Reset

    def set_model(self, model):
        
        from .model import Model
        if not isinstance(model, Model):
            raise ValueError('model has to be of class ezie.Model.')
        
        self.model = model
        # Initialize parameter grids and all internal variables
        self._init_parameter_grids()
        self._reset_internal_state()
        self._reset_results()
        
    def _init_parameter_grids(self):
        """Initialize the L1/L2 parameter grids used for searching."""
        self.l1s = np.linspace(self.l1_lower, self.l1_upper, self.l1_steps)
        self.l2s = np.linspace(self.l2_lower, self.l2_upper, self.l2_steps)
        self.l1_fit = np.linspace(self.l1_lower, self.l1_upper, self.fit_steps)
        self._l2_fit = None

    def _reset_internal_state(self):
        """Reset caches or temporary internal variables."""
        self._gini_cube = None
        self._gini_map = None
        self._l2_ridge = None
        self._opt_id = None
        self._offset = None
        self._l1_opt = None
        self._l2_opt = None
        self.ridge_exp = None
        self._m_id = None

    def _reset_results(self):
        """Reset stored results from L-curve and ridge searches."""
        self._rnorm = None
        self._mnorm = None

#%%

    @property
    def m_id(self):
        if self._m_id is None:
            self._m_id = self.automate_m_id()
        return self._m_id

    def automate_m_id(self):
        xis = [np.nanmedian(self.grid.projection.geo2cube(MEM.lon, MEM.lat)[0]) for MEM in self.grid.mems]
        xi = self.grid.xi[0, :]
        xi_min, xi_max = np.min(xis), np.max(xis)
        left_id, right_id = np.argmin(abs(xi-xi_min)), np.argmin(abs(xi-xi_max))
        xi_crop = xi[left_id:right_id+1]
        dists = np.vstack([xi_crop - xi for xi in xis])
        xi_id = np.argmax(np.min(abs(dists), axis=0))
        return self.grid.shape[0]//2*self.grid.shape[1] + (left_id + xi_id)

#%% L-curve
   
    @property
    def l1_opt(self):
        if self._l1_opt is None:
            self.get_opt_l()
        return self._l1_opt
    
    @property
    def l2_opt(self):
        if self._l2_opt is None:
            self.get_opt_l()
        return self._l2_opt

    def get_opt_l(self):    
        self._l1_opt = self.l1_fit[self.opt_id]
        self._l2_opt = self.l2_fit[self.opt_id]

    @property
    def opt_id(self):
        if self._opt_id is None:
            #self._opt_id, self._offset = self.robust_Kneedle()
            self._opt_id, self._offset = self.max_curvature()
        return self._opt_id

    @property
    def offset(self):
        if self._offset is None:
            #self._opt_id, self._offset = self.robust_Kneedle()
            self._opt_id, self._offset = self.max_curvature()
        return self._offset
    
    @property
    def rnorm(self):
        if self._rnorm is None:
            self.get_L_curve()
        return self._rnorm
    
    @property
    def mnorm(self):
        if self._mnorm is None:
            self.get_L_curve()
        return self._mnorm
    
    def get_L_curve(self):
        
        l1_store = self.model._l1
        l2_store = self.model._l2
        
        self._rnorm = np.ones(self.fit_steps)
        self._mnorm = np.ones(self.fit_steps)
        loop = tqdm(enumerate(zip(self.l1_fit, self.l2_fit)),
                    total=self.fit_steps, desc='Generating L-curve')
        for i, (l1 ,l2) in loop:
            self.model.reset_regularization(l1, l2)
            self.model.reset_solution()
            res = self.model.G.dot(self.model.m) - self.model.d
            self._rnorm[i] = np.log10(res.T.dot(self.model.Qinv).dot(res))
            self._mnorm[i] = np.log10(self.model.m.T.dot(self.model.m) + self.model.m.T.dot(self.model.LTL).dot(self.model.m))

        self.model.reset_regularization(l1_store, l2_store)

    def max_curvature(self):
        # Create the spline
        spl = UnivariateSpline(self.rnorm, self.mnorm, k=3, s=0)
        rnorm_fit = np.linspace(self.rnorm.min(), self.rnorm.max(), 1000)
        mnorm_fit = spl(rnorm_fit)
        
        self.rnorm_fit, self.mnorm_fit = rnorm_fit, mnorm_fit
        
        # Get first and second derivatives
        spl_prime = spl.derivative(1)  # First derivative
        spl_double_prime = spl.derivative(2)  # Second derivative
        
        # Evaluate derivatives at the fitted points
        y_prime = spl_prime(rnorm_fit)
        y_double_prime = spl_double_prime(rnorm_fit)
        
        # Calculate curvature: κ = |y''| / (1 + y'^2)^(3/2)
        curvature = np.abs(y_double_prime) / (1 + y_prime**2)**(3/2)
        
        # Find convex region (where second derivative is positive)
        convex_mask = y_double_prime > 0
        stable_spline_mask = rnorm_fit >= self.rnorm[4]
        covex_mask = convex_mask & stable_spline_mask
        
        if not np.any(convex_mask):
            # No convex region found
            return None, None
        
        # Get curvature only in convex region
        convex_curvature = curvature[convex_mask]
        convex_rnorm = rnorm_fit[convex_mask]
        
        # Find maximum curvature in convex region
        max_curv_idx = np.argmax(convex_curvature)
        max_curvature_value = convex_curvature[max_curv_idx]
        max_curvature_point = convex_rnorm[max_curv_idx]
        
        # Get the corresponding y-value
        #max_curvature_y = spl(max_curvature_point)
    
        return np.argmin(abs(self.rnorm - max_curvature_point)), 3
        #return max_curvature_point, max_curvature_value
    

    def robust_Kneedle(self):
        start = self.find_convex_start_finite_diff()
        knee = KneeLocator(self.rnorm[start:], self.mnorm[start:], curve='convex', direction='decreasing')
        return np.argmin(abs(self.rnorm - knee.knee)), start

    def find_convex_start_finite_diff(self):
        """
        Find convex start using finite differences.
        """
        # Compute first differences (approximate first derivative)
        x, y = self.rnorm, self.mnorm
        dx = np.diff(x)
        dy = np.diff(y)
        first_deriv = dy / dx
    
        # Compute second differences (approximate second derivative)
        d2y = np.diff(dy)
        dx_mid = (dx[:-1] + dx[1:]) / 2
        second_deriv = d2y / (dx_mid * dx[:-1])
    
        # Check conditions (note: arrays are shorter due to diff)
        is_decreasing = first_deriv[:-1] < 0
        is_convex = second_deriv > 0
    
        valid_points = np.where(is_decreasing & is_convex)[0]
    
        if len(valid_points) > 0:
            return valid_points[0] + 1  # +1 to account for diff reducing array size
        else:
            return 0

#%% Lambda relation
    
    @property
    def l2_fit(self):
        if self._l2_fit is None:
            spl = UnivariateSpline(self.l1s, self.l2_ridge, k=3)
            self._l2_fit = spl(self.l1_fit)
        return self._l2_fit
    
    @property
    def l2_ridge(self):
        if self._l2_ridge is None:
            self.explore_ridge()
        return self._l2_ridge
        
    def explore_ridge(self):
        self.ridge_exp = np.zeros((self.l1_steps, self.iterations, 3)) # axis=2: max, lower, upper
        loop = tqdm(enumerate(self.l1s), total=self.l1_steps,
                    desc='Optimizing l2 for each l1.')
        for i, l1 in loop:
            l2l = self.l2_lower
            l2u = self.l2_upper
            for j in range(self.iterations):
                l2s = np.linspace(l2l, l2u, self.l2_steps)
                dl2s = np.diff(l2s)[0]
                
                gini = np.zeros(self.l2_steps)                
                for k, l2 in enumerate(l2s):
                    gini[k] = self.compute_gini_for_pair(l1, l2, self.m_id)
                
                l2_max = l2s[np.argmax(gini)]
                l2l = l2_max - dl2s
                l2u = l2_max + dl2s                
                self.ridge_exp[i, j, :] = [l2_max, l2l, l2u]
        self._l2_ridge = self.ridge_exp[:, -1, 0]
            
    def compute_gini_for_pair(self, l1, l2, idx=None):
        """Compute the Gini index for a single (l1, l2) pair and a single row idx."""
        
        l1_store = self.model._l1
        l2_store = self.model._l2
        
        self.model.reset_regularization(l1, l2)
        self.model.reset_solution()
        
        if idx is None:                
            R = cho_solve(self.model.c_factor, self.model.GTG, check_finite=False)
        else:
            e_idx = np.zeros(self.grid.size)
            e_idx[idx] = 1.0
            R = cho_solve(self.model.c_factor, e_idx, check_finite=False) @ self.model.GTG
        
        self.model.reset_regularization(l1_store, l2_store)
        self.model.reset_solution()
        
        # Gini calculation
        if idx is None:
            gini = calc_gini_matrix(R)
        else:
            gini = calc_gini(R)
        return gini

#%% Grid search approach for validation

    @property
    def gini_cube(self):
        if self._gini_cube is None:
            self.get_gini_cube()
        return self._gini_cube
    
    def get_gini_cube(self):        
        self._gini_cube = np.zeros((self.l1_steps, self.l2_steps, self.grid.size))
        loop = tqdm(product(enumerate(self.l1s), enumerate(self.l2s)), 
                    total=self.l1_steps*self.l2_steps,
                    desc='Generating gini cube')
        for (i, l1_), (j, l2_) in loop:
            self._gini_cube[i, j, :] = self.compute_gini_for_pair(l1_, l2_)
    
    @property
    def gini_map(self):
        if self._gini_map is None:
            self.get_gini_map()
        return self._gini_map
    
    def get_gini_map(self):
        self._gini_map = np.zeros((self.l1_steps, self.l2_steps))
        loop = tqdm(product(enumerate(self.l1s), enumerate(self.l2s)), 
                    total=self.l1_steps*self.l2_steps,
                    desc='Generating gini map')
        for (i, l1_), (j, l2_) in loop:
            self._gini_map[i, j] = self.compute_gini_for_pair(l1_, l2_, self.m_id)