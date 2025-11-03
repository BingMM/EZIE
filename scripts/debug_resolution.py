#%% Import

import os
import ezie
import numpy as np
from netCDF4 import Dataset
from datetime import datetime
import matplotlib.pyplot as plt
from secsy import get_SECS_B_G_matrices

#%% Set base

base = '/home/bing/Dropbox/work/code/repos/EZIE/Rafael_l2_Oct_2025/'

#%% Import data

#post_fix = '20250504_054739'
post_fix = '20250504_072309'

date = datetime.strptime(post_fix, '%Y%m%d_%H%M%S')

filename = os.path.join(base, 'data', f'{post_fix}.nc4')

file = Dataset(filename)

n_obs = file.dimensions['n_obs'].size
n_b = file.dimensions['n_beam'].size

time = file['times'][:].filled(np.nan)
lat = file['lat'][:].filled(np.nan)
lon = file['lon'][:].filled(np.nan)
r = np.ones_like(lat)*(6371.2 + 80)*1e3
dB = -file['Bd_no_bias'][:].filled(np.nan)*1e-9
cov = np.ones_like(dB)*(450*1e-9)**2

file.close()

#%% Make model

data = ezie.Data(date=date, 
                 lat=list(lat.T), lon=list(lon.T), r=list(r.T), 
                 Bu=list(dB.T), cov_uu=list(cov.T))

model = ezie.Model(l1=0.24, l2=1.88)
#model = ezie.Model()
model.add_data(data)
plotter = ezie.Plotter(model)

#%% Plot resolution - ezie

fig = plt.figure()
res = model.resolution.res
plt.contourf(model.grid.xi, model.grid.eta, res*1e-3, cmap='Reds', levels=np.linspace(0, 2000, 40))
plotter.plot_tracks(plt.gca())
plt.colorbar()

#%% Plot resolution - old

import copy

def get_resolution(R, grid):
    xi_FWHM  = np.zeros(grid.size)
    eta_FWHM = np.zeros(grid.size)
    xi_flag  = np.zeros(grid.size).astype(int)
    eta_flag = np.zeros(grid.size).astype(int)
    for i in range(R.shape[0]):
        
        PSF = abs(R[:, i].reshape(grid.shape))
        #PSF = R[:, i].reshape(grid.shape)
        
        PSF_xi = np.sum(PSF, axis=0)
        i_left, i_right, flag = left_right(PSF_xi)
        xi_FWHM[i] = (i_right-i_left)*grid.Lres
        xi_flag[i] = flag
        
        PSF_eta = np.sum(PSF, axis=1)
        i_left, i_right, flag = left_right(PSF_eta)
        eta_FWHM[i] = (i_right-i_left)*grid.Wres
        eta_flag[i] = flag
        
    xi_FWHM = xi_FWHM.reshape(grid.shape)
    eta_FWHM = eta_FWHM.reshape(grid.shape)
    xi_flag = xi_flag.reshape(grid.shape)
    eta_flag = eta_flag.reshape(grid.shape)
        
    return xi_FWHM, eta_FWHM, xi_flag, eta_flag

def left_right(PSF_i, fraq=0.5, inside=False, x='', x_min='', x_max=''):
    
    if inside:
        PSF_ii = copy.deepcopy(PSF_i)
        valid = False
        while not valid:
            i_max = np.argmax(PSF_ii)
            
            x_i = x[i_max]
            if (x_i >= x_min) and (x_i <= x_max):
                valid = True
            else:
                PSF_ii[i_max] = np.min(PSF_i)            
                        
    else:
        i_max = np.argmax(PSF_i)    
    
    PSF_max = PSF_i[i_max]
        
    j = 0
    i_left = 0
    left_edge = True
    while (i_max - j) >= 0:
        if PSF_i[i_max - j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max - j + 1] - PSF_i[i_max - j]
            dx = (fraq*PSF_max - PSF_i[i_max - j]) / dPSF
            i_left = i_max - j + dx
            
            left_edge = False
            
            break
        else:
            j += 1

    j = 0
    i_right = len(PSF_i) - 1
    right_edge = True
    while (i_max + j) < len(PSF_i):
        if PSF_i[i_max + j] < fraq*PSF_max:
            
            dPSF = PSF_i[i_max + j] - PSF_i[i_max + j - 1]
            dx = (fraq*PSF_max - PSF_i[i_max + j - 1]) / dPSF
            i_right = i_max + j - 1 + dx 
            
            right_edge = False
            
            break
        else:
            j += 1
    
    flag = True
    if left_edge and right_edge:
        print('I think something is wrong')
        flag = False
    elif left_edge:
        i_left = i_max - (i_right - i_max)
        flag = False
    elif right_edge:
        i_right = i_max + (i_max - i_left)
        flag = False
    
    return i_left, i_right, flag

xi_FWHM, eta_FWHM, xi_flag, eta_flag = get_resolution(model.Rm, model.grid)
res = (xi_FWHM + eta_FWHM)/2
#flag = (xi_flag == 0) | (eta_flag == 0)
#res[flag] = np.nan

fig = plt.figure()
plt.contourf(model.grid.xi, model.grid.eta, res*1e-3, cmap='Reds', levels=np.linspace(0, 2000, 40))
plotter.plot_tracks(plt.gca())
plt.colorbar()

#%%
row = model.grid.shape[0]//2
col = int(model.grid.shape[1]*3/5)+5
idx = row*model.grid.shape[1] + col #5102
PSF = abs(model.Rm[:, idx].reshape(model.grid.shape))

fig = plt.figure()
ax = plt.gca()
plt.contourf(model.grid.xi, model.grid.eta, PSF/np.max(abs(PSF)), cmap='bwr', levels=41)
plt.plot(model.grid.xi.flatten()[idx], model.grid.eta.flatten()[idx], '*')
plotter.plot_tracks(ax)
plt.colorbar()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(np.arange(model.grid.shape[1]), np.sum(PSF, axis=0))
axs[0].vlines(np.arange(model.grid.shape[1])[col], axs[0].get_ylim()[0], axs[0].get_ylim()[1])
axs[1].plot(np.arange(model.grid.shape[0]), np.sum(PSF, axis=1))
axs[1].vlines(np.arange(model.grid.shape[0])[row], axs[1].get_ylim()[0], axs[1].get_ylim()[1])







