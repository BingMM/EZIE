from typing import Union, Optional
import numpy as np

class MEM(object):
    def __init__(self,
                 lat: np.ndarray, 
                 lon: np.ndarray, 
                 r: Union[np.ndarray, float, int], 
                 Be: Optional[np.ndarray] = None,
                 Bn: Optional[np.ndarray] = None,
                 Bu: Optional[np.ndarray] = None,
                 B: Optional[np.ndarray] = None,
                 cov_ee: Optional[np.ndarray] = None,
                 cov_nn: Optional[np.ndarray] = None,
                 cov_uu: Optional[np.ndarray] = None,
                 cov_en: Optional[np.ndarray] = None,
                 cov_eu: Optional[np.ndarray] = None,
                 cov_nu: Optional[np.ndarray] = None,
                 cov_mag: Optional[np.ndarray] = None):
        
        # Check coordinates
        if len(lat.shape) != 1:
            raise ValueError('lat has to be 1D')
        self.lat = lat
        self.nd = self.lat.size
        
        if len(lon.shape) != 1:
            raise ValueError('lon has to be 1D')
        self.lon = lon
        
        if self.lon.size != self.nd:
            raise ValueError('lat and lon has to have the same length')
        
        if isinstance(r, np.ndarray) and (len(r.shape) != 1 or self.nd != r.size):
            raise ValueError('If r is array it has to be 1D with size as lat and lon')
        
        if isinstance(r, float) or isinstance(r, int):
            self.r = np.ones(self.nd)*r
        else:
            self.r = r
        
        for var in [Be, Bn, Bu, B, cov_ee, cov_nn, cov_uu, cov_en, cov_eu, cov_nu, cov_mag]:
            if var is None:
                continue
            if len(var) != self.nd:
                raise ValueError('Magnetic field and associated covariances have to have the same size as lat.')
        
        self.Be, self.Bn, self.Bu, self.B = Be, Bn, Bu, B
        self.cov_ee, self.cov_nn, self.cov_uu, self.cov_mag = cov_ee, cov_nn, cov_uu, cov_mag
        self.cov_en, self.cov_eu, self.cov_nu = cov_en, cov_eu, cov_nu
        
        self.comp = 'senu'
        if self.B is None:
            self.comp = self.comp.replace('s', '')
        if self.Be is None:
            self.comp = self.comp.replace('e', '')
        if self.Bn is None:
            self.comp = self.comp.replace('n', '')
        if self.Bu is None:
            self.comp = self.comp.replace('u', '')
        