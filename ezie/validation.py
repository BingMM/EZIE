#%% Import

import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from .supermag_api import SuperMAGGetInventory, SuperMAGGetData, sm_grabme
from tqdm import tqdm

#%% Model class

class Validation(object):
    def __init__(self, 
                 model,
                 userid: str):
        
        self.userid = userid
        self.set_model(model)
        self.reset_sm_st_list()
        self.reset_sm_avail()
        self.reset_sm_grid()
        self.reset_sm_data()
        self.reset_predictions()

#%% set model

    def set_model(self, model):
        
        from .model import Model
        if not isinstance(model, Model):
            raise ValueError('model has to be of class ezie.Model.')
        
        self.dates = model.data.dates
        self.start = list(self.dates[0].timetuple()[:6]) # alt: start='2019-11-15T10:40'
        self.duration = int((self.dates[-1]-self.dates[0]).total_seconds())
        self.grid = model.grid
        self.ev = model.ev

#%% Fetch list of all possible stations

    def reset_sm_st_list(self):
        self._IAGA_all = None
        self._lat_all = None
        self._lon_all = None

    @property
    def IAGA_all(self):
        if self._IAGA_all is None:
            self.fetch_sm_all()
        return self._IAGA_all

    @property
    def lat_all(self):
        if self._lat_all is None:
            self.fetch_sm_all()
        return self._lat_all
    
    @property
    def lon_all(self):
        if self._lon_all is None:
            self.fetch_sm_all()
        return self._lon_all

    def fetch_sm_all(self):
        
        # Get path to this file (validation.py)
        here = Path(__file__).resolve()

        # Go up two directories: ezie/validation.py → EZIE/
        repo_root = here.parents[1]

        # Construct full path to your data file
        sm_station_list = os.path.join(repo_root, 'data', '20251104-09-24-supermag-stations.csv')

        # Read file        
        st_data = pd.read_csv(sm_station_list, usecols=range(3))
        self._IAGA_all = st_data['IAGA'].to_numpy()
        self._lat_all  = st_data['GEOLAT'].to_numpy()
        self._lon_all  = st_data['GEOLON'].to_numpy()

#%% Fetch all available stations

    def reset_sm_avail(self):
        self._IAGA_avail = None
        self._lat_avail = None
        self._lon_avail = None

    @property
    def IAGA_avail(self):
        if self._IAGA_avail is None:
            self.fetch_sm_avail()
        return self._IAGA_avail

    @property
    def lat_avail(self):
        if self._lat_avail is None:
            self.fetch_sm_avail()
        return self._lat_avail
    
    @property
    def lon_avail(self):
        if self._lon_avail is None:
            self.fetch_sm_avail()
        return self._lon_avail

    def fetch_sm_avail(self):
                
        # Fetch names of all stations with available data
        #(status, stations) = SuperMAGGetInventory(self.userid, self.start, self.duration)
        (status, stations) = SuperMAGGetInventory(self.userid, self.start, 3600) # doesn't work with anything less than 3600...
        if status == 0:
            raise ValueError('Fetching SuperMAG inventory list failed.')
            
        self._IAGA_avail = np.array(stations)
        
        f = np.isin(self.IAGA_all, self._IAGA_avail)
        
        self._IAGA_avail = self.IAGA_all[f]
        self._lat_avail  = self.lat_all[f]
        self._lon_avail  = self.lon_all[f]

#%% Find all stations inside the grid

    def reset_sm_grid(self):
        self._IAGA_grid = None
        self._lat_grid = None
        self._lon_grid = None

    @property
    def IAGA_grid(self):
        if self._IAGA_grid is None:
            self.fetch_sm_grid()
        return self._IAGA_grid

    @property
    def lat_grid(self):
        if self._lat_grid is None:
            self.fetch_sm_grid()
        return self._lat_grid
    
    @property
    def lon_grid(self):
        if self._lon_grid is None:
            self.fetch_sm_grid()
        return self._lon_grid

    def fetch_sm_grid(self):
        f = self.grid.ingrid(self.lon_avail, self.lat_avail)
        self._IAGA_grid = self.IAGA_avail[f]
        self._lat_grid  = self.lat_avail[f]
        self._lon_grid  = self.lon_avail[f]
        
#%%% Fetch data from supermag

    def reset_sm_data(self):
        self._sm = None

    @property
    def IAGA(self):
        return self.IAGA_grid
    
    @property
    def lat(self):
        return self.lat_grid

    @property
    def lon(self):
        return self.lon_grid

    @property
    def sm(self):
        if self._sm is None:
            self.fetch_data()
        return self._sm
    
    def fetch_data(self):
        t0 = datetime(1970, 1, 1)
        
        t, iaga, Be, Bn, Bu = [], [], [], [], []
        loop = tqdm(self.IAGA, total=len(self.IAGA), desc='Downloading data from SuperMAG')
        for IAGA in loop:
            attempt = 1
            while attempt < 4:
                (status, data) = SuperMAGGetData(self.userid, self.start, self.duration, 'all', IAGA)
                if data.empty:
                    attempt += 1
                else:
                    break
            if attempt >= 4:
                raise ValueError(f'Supermag API call for {IAGA} failed.')
            t.append(np.array([t0 + timedelta(seconds=s) for s in data['tval']]))
            iaga.append(data['iaga'].to_numpy())
            Be.append(np.array(sm_grabme(data, 'E', 'geo'))*1e-9)
            Bn.append(np.array(sm_grabme(data, 'N', 'geo'))*1e-9)
            Bu.append(-1*np.array(sm_grabme(data, 'Z', 'geo'))*1e-9)
        
        self._sm = pd.DataFrame({'IAGA': np.hstack(iaga),
                                 't': np.hstack(t),
                                 'Be': np.hstack(Be),
                                 'Bn': np.hstack(Bn),
                                 'Bu': np.hstack(Bu)
                                 })
        
        self._sm = self._sm.set_index(['IAGA', 't'])

#%% Evaluate model at stations

    def reset_predictions(self):
        self._Be_ev = None
        self._Bn_ev = None
        self._Bu_ev = None
        self._Be_std_ev = None
        self._Bn_std_ev = None
        self._Bu_std_ev = None
        
    @property
    def Be_ev(self):
        if self._Be_ev is None:
            self.ev_sm()
        return self._Be_ev
    
    @property
    def Bn_ev(self):
        if self._Bn_ev is None:
            self.ev_sm()
        return self._Bn_ev
    
    @property
    def Bu_ev(self):
        if self._Bu_ev is None:
            self.ev_sm()
        return self._Bu_ev
    
    @property
    def Be_std_ev(self):
        if self._Be_std_ev is None:
            self.ev_sm()
        return self._Be_std_ev
    
    @property
    def Bn_std_ev(self):
        if self._Bn_std_ev is None:
            self.ev_sm()
        return self._Bn_std_ev
    
    @property
    def Bu_std_ev(self):
        if self._Bu_std_ev is None:
            self.ev_sm()
        return self._Bu_std_ev

    def ev_sm(self):
        B_ = self.ev.get_B_predictions(self.lat, self.lon, 6371.2e3, ret=True)
        self._Be_ev, self._Bn_ev, self._Bu_ev, self._Be_std_ev, self._Bn_std_ev, self._Bu_std_ev = B_
        del B_

#%% Get data for a single station

    def get_IAGA(self, IAGA):
        f = np.isin(self.IAGA, IAGA)
        
        ev_B = np.array([self.Be_ev[f], self.Bn_ev[f], self.Bu_ev[f]]).flatten()
        ev_B_std = np.array([self.Be_std_ev[f], self.Bn_std_ev[f], self.Bu_std_ev[f]]).flatten()
        sm_B = list(self.sm.loc[IAGA, ['Be', 'Bn', 'Bu']].to_numpy().T)
        
        return sm_B, ev_B, ev_B_std
    






















