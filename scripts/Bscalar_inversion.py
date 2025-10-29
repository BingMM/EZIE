#%% Import

import os
import ezie
import numpy as np
from netCDF4 import Dataset
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import polplot as pp

#%% Set base

base = '/home/bing/Dropbox/work/code/repos/EZIE'

#%% Import data

filename = os.path.join(base, 'data', 'ezie_l1_20250505_023909_sva_v002_r001.nc4')

file = Dataset(filename)

n_obs = file.dimensions['n_obs'].size
n_vec = file.dimensions['n_b_vector'].size

alt = file['altitude'][:].filled(np.nan) # meters

time = file['time_utc'][:].filled(np.nan) * 3600
lat = np.zeros((n_obs, 4))
lon = np.zeros((n_obs, 4))
dB = np.zeros((n_obs, 4, n_vec))

for i in range(4):
    lat[:, i] = file[f'obs_lat{i+1}'][:].filled(np.nan)
    lon[:, i] = file[f'obs_lon{i+1}'][:].filled(np.nan)
    dB[:, i, :] = file[f'B_ret{i+1}'][:].filled(np.nan) - file[f'B_prior{i+1}'][:].filled(np.nan)

file.close()

#%%
fig = plt.figure()
ax = plt.gca()
pax = pp.Polarplot(ax, sector='night')