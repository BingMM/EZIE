#%% Import
from typing import Optional
import numpy as np
from secsy import CSgrid


#%% Resolution class

class Resolution(object):
    def __init__(self, 
                 model: Optional = None,
                 R: Optional[np.ndarray] = None, 
                 grid: Optional[CSgrid] = None):
        
        if model is None and (R is None or grid is None):
            raise ValueError('Either model has to be specified or R and grid.')
        
        if model is None:
            self.R = R
            self.grid = grid
        else:
            self.R = model.Rm
            self.grid = model.grid
        
        if self.R.shape[0] != self.R.shape[1]:
            raise ValueError('R has to be square.')
        
        if self.R.shape[0] != self.n:
            raise ValueError('R and grid does not have the same dimensions.')
        
        self.reset()

    @property
    def shape(self):
        return self.grid.shape
    
    @property
    def n(self):
        return self.grid.size

#%% Reset

    def reset(self):
        
        self._res_xi = None
        self._res_eta = None
        self._res = None
        self._flag_xi = None
        self._flag_eta = None
        self._flag = None

#%% Calculate resolution

    @property
    def res(self):
        if self._res is None:
            self._res = (self.res_xi + self.res_eta)/2
        return self._res
    
    @property
    def res_xi(self):
        if self._res_xi is None:
            self.calc_resolution()
        return self._res_xi.reshape(self.shape)
    
    @property
    def res_eta(self):
        if self._res_eta is None:
            self.calc_resolution()
        return self._res_eta.reshape(self.shape)
    
    @property
    def flag(self):
        if self._flag is None:
            self._flag = self.flag_xi & self.flag_eta
        return self._flag
    
    @property
    def flag_xi(self):
        if self._flag_xi is None:
            self.calc_resolution()
        return self._flag_xi.reshape(self.shape)
    
    @property
    def flag_eta(self):
        if self._flag_eta is None:
            self.calc_resolution()
        return self._flag_eta.reshape(self.shape)
    
    def calc_resolution(self):
        
        self._res_xi = -np.ones(self.n)
        self._res_eta = -np.ones(self.n)
        self._flag_xi = np.zeros(self.n)
        self._flag_eta = np.zeros(self.n)
        
        for i in range(self.n):
            PSF = abs(self.R[:, i].reshape(self.shape))
            
            PSF_xi = np.sum(PSF, axis=0)
            left, right, flag = calc_PSF_fwhm(PSF_xi)
            if flag:                
                self._res_xi[i] = (right-left)*self.grid.Lres
            self._flag_xi[i] = flag
            
            PSF_eta = np.sum(PSF, axis=1)
            left, right, flag = calc_PSF_fwhm(PSF_eta)
            if flag:
                self._res_eta[i] = (right-left)*self.grid.Wres
            self._flag_eta[i] = flag

#%%

def calc_PSF_fwhm(PSF):
    id_max = np.argmax(PSF)
    M = PSF[id_max]
    HM = M/2
    
    if (id_max == 0) or (id_max == (PSF.size-1)):
        return None, None, False

    left_segment    = np.flip(PSF[:id_max+1])
    right_segment   = PSF[id_max:]

    left_id = find_crossing(left_segment, HM)
    right_id = find_crossing(right_segment, HM)

    if left_id is None or right_id is None:
        return None, None, False

    left_id = id_max - left_id
    right_id = id_max + right_id
    
    return left_id, right_id, True

def find_crossing(y, hm):
    if np.all(y > hm):
        return None
    
    x_hm = np.argmax(y <= hm)
    
    if np.any(y[x_hm:] > hm):
        return None
    
    x1 = x_hm - 1
    y1 = y[x1]
    if x_hm == (y.size-1):
        x2 = x_hm
    else:
        x2 = x_hm+1
    y2 = y[x2]
    
    a = (y1-y2)/(x1-x2)
    b = y1 - a*x1    
    id_hm = (hm - b) / a    
    
    return id_hm
    