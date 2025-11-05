from typing import Union, Optional
from datetime import datetime
import numpy as np
from .mem import MEM

class Data(object):
    def __init__(self, 
                 date: Union[datetime, list[datetime]], 
                 lat: list, 
                 lon: list, 
                 r: Union[list, float, int], 
                 Be: Optional[list] = None,
                 Bn: Optional[list] = None,
                 Bu: Optional[list] = None,
                 B: Optional[list] = None,
                 cov_ee: Optional[list] = None,
                 cov_nn: Optional[list] = None,
                 cov_uu: Optional[list] = None,
                 cov_en: Optional[list] = None,
                 cov_eu: Optional[list] = None,
                 cov_nu: Optional[list] = None,
                 cov_mag: Optional[list] = None):
        
        if isinstance(date, list):
            self.dates = date
            self.date = self.dates[len(self.dates)//2]
        else:
            self.date = date
        self.nmem = len(lat)
        
        if not isinstance(r, list):
            r = [self.nmem]*r
        
        if np.any([(lst != None) and (len(lst) != self.nmem)  for lst in [lat, lon, r, Be, Bn, Bu, B, cov_ee, cov_nn, cov_uu, cov_en, cov_eu, cov_nu, cov_mag]]):
            raise ValueError('lat, lon, (r), B_i, and cov_i have to be the same length.')
        
        def _safe_iter(x, n):
            return x if x is not None else [None] * n

        self.mems = [MEM(lat_, lon_, r_, Be_, Bn_, Bu_, B_, cov_ee_, cov_nn_, cov_uu_, cov_en_, cov_eu_, cov_nu_, cov_mag_) for (lat_, lon_, r_, Be_, Bn_, Bu_, B_, cov_ee_, cov_nn_, cov_uu_, cov_en_, cov_eu_, cov_nu_, cov_mag_) in zip(lat,lon,r,_safe_iter(Be, self.nmem),_safe_iter(Bn, self.nmem),_safe_iter(Bu, self.nmem),_safe_iter(B, self.nmem),_safe_iter(cov_ee, self.nmem),_safe_iter(cov_nn, self.nmem),_safe_iter(cov_uu, self.nmem),_safe_iter(cov_en, self.nmem),_safe_iter(cov_eu, self.nmem),_safe_iter(cov_nu, self.nmem),_safe_iter(cov_mag, self.nmem),)]
        #self.mems = [MEM(lat_, lon_, r_, B_, B_u_, comp) for (lat_, lon_, r_, B_, B_u_) in zip(lat, lon, r, B, B_u)]
        self.comp = self.mems[0].comp
        
        self._lat = None
        self._lon = None
        self._r = None
        
        self._Be = None
        self._Bn = None
        self._Bu = None
        self._B = None
        
        self._cov_ee = None
        self._cov_nn = None
        self._cov_uu = None
        self._cov_en = None
        self._cov_eu = None
        self._cov_nu = None
        self._cov_mag = None
    
    @property
    def lat(self):
        if self._lat is None:
            self._lat = np.concatenate([mem.lat for mem in self.mems])
        return self._lat
    
    @property
    def lon(self):
        if self._lon is None:
            self._lon = np.concatenate([mem.lon for mem in self.mems])
        return self._lon
    
    @property
    def r(self):
        if self._r is None:
            self._r = np.concatenate([mem.r for mem in self.mems])
        return self._r
    
    @property
    def B(self):
        if self._B is None:
            self._B = np.concatenate([mem.B for mem in self.mems])
        return self._B
    
    @property
    def Be(self):
        if self._Be is None:
            self._Be = np.concatenate([mem.Be for mem in self.mems])
        return self._Be
    
    @property
    def Bn(self):
        if self._Bn is None:
            self._Bn = np.concatenate([mem.Bn for mem in self.mems])
        return self._Bn
    
    @property
    def Bu(self):
        if self._Bu is None:
            self._Bu = np.concatenate([mem.Bu for mem in self.mems])
        return self._Bu
    
    @property
    def cov_ee(self):
        if self._cov_ee is None:
            self._cov_ee = np.concatenate([mem.cov_ee for mem in self.mems])
        return self._cov_ee
    
    @property
    def cov_nn(self):
        if self._cov_nn is None:
            self._cov_nn = np.concatenate([mem.cov_nn for mem in self.mems])
        return self._cov_nn
    
    @property
    def cov_uu(self):
        if self._cov_uu is None:
            self._cov_uu = np.concatenate([mem.cov_uu for mem in self.mems])
        return self._cov_uu
    
    @property
    def cov_en(self):
        if self._cov_en is None:
            self._cov_en = np.concatenate([mem.cov_en for mem in self.mems])
        return self._cov_en
    
    @property
    def cov_eu(self):
        if self._cov_eu is None:
            self._cov_eu = np.concatenate([mem.cov_eu for mem in self.mems])
        return self._cov_eu
    
    @property
    def cov_nu(self):
        if self._cov_nu is None:
            self._cov_nu = np.concatenate([mem.cov_nu for mem in self.mems])
        return self._cov_nu
    
    @property
    def cov_mag(self):
        if self._cov_mag is None:
            self._cov_mag = np.concatenate([mem.cov_mag for mem in self.mems])
        return self._cov_mag
    
    