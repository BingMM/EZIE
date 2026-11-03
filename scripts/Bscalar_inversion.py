#%% Import

import os
import ezie
import matplotlib.pyplot as plt
from ezie.utils import generate_ezie_data
import numpy as np
import polplot as pp

#%% Set base

base = '/media/bing/LaCie/EZIE_L2/a/20250515'

#%% Compare scalar inversions

file = os.path.join(base, 'ezie_l2_20250515_022454_sva_v003_r000.nc4')

# No gradients
ezie_data = generate_ezie_data(file, comp='F', background='chaos')
#model = ezie.Model(l1=0.75, l2=2.5)
#model = ezie.Model(l1=2.4, l2=3.9)
model = ezie.Model(l1=1.65, l2=3.9)
model.add_data(ezie_data)

plotter = ezie.Plotter(model, userid='mih008')
fig, axs = plotter.plot_solution_cs()

fig, axs1 = plt.subplots(2,2, figsize=(10,10))
fig, axs2 = plt.subplots(2,2, figsize=(10,10))
for i, (ax1, ax2) in enumerate(zip(axs1.flatten(), axs2.flatten())):
    ax1.plot(ezie_data.mems[i].B)
    Be, Bn, Bu = model.ev.get_B_predictions(lat=model.data.mems[i].lat, lon=model.data.mems[i].lon, r=model.data.mems[i].r, ret=True)
    Be = Be + model.data.mems[i].B0e
    Bn = Bn + model.data.mems[i].B0n
    Bu = Bu + model.data.mems[i].B0u
    B = np.sqrt(Be**2 + Bn**2 + Bu**2)# + model.track_offsets[i]
    dB = B - model.data.mems[i].B0
    ax1.plot(dB)
    
    ax2.plot(np.diff(ezie_data.mems[i].B))
    ax2.plot(np.diff(dB))

# gradients
ezie_data = generate_ezie_data(file, comp='F', background='chaos', scalar_source='retrieved')
#model = ezie.Model(l1=0.75, l2=2.5, fit_mode='gradient')
model = ezie.Model(l1=2.4, l2=3.9, fit_mode='gradient')
model.add_data(ezie_data)

plotter = ezie.Plotter(model, userid='mih008')
fig, axs = plotter.plot_solution_cs()

fig, axs1 = plt.subplots(2,2, figsize=(10,10))
fig, axs2 = plt.subplots(2,2, figsize=(10,10))
for i, (ax1, ax2) in enumerate(zip(axs1.flatten(), axs2.flatten())):
    ax1.plot(ezie_data.mems[i].B)
    Be, Bn, Bu = model.ev.get_B_predictions(lat=model.data.mems[i].lat, lon=model.data.mems[i].lon, r=model.data.mems[i].r, ret=True)
    Be = Be + model.data.mems[i].B0e
    Bn = Bn + model.data.mems[i].B0n
    Bu = Bu + model.data.mems[i].B0u
    B = np.sqrt(Be**2 + Bn**2 + Bu**2) + model.track_offsets[i]
    dB = B - model.data.mems[i].B0
    ax1.plot(dB)
    
    ax2.plot(np.diff(ezie_data.mems[i].B))
    ax2.plot(np.diff(dB))


#%% Load data

#file = os.path.join(base, 'ezie_l2_20250515_004708_sva_v003_r000.nc4')
file = os.path.join(base, 'ezie_l2_20250515_022454_sva_v003_r000.nc4')
#ezie_data = generate_ezie_data(file, comp='ENU', background='chaos')
ezie_data = generate_ezie_data(file, comp='F', background='chaos')
#ezie_data = generate_ezie_data(file, comp='ENU', background='chaos', offset_corrected=True)

#%%

model = ezie.Model(l1=0.75, l2=2.5)
#model = ezie.Model(l1=0.75, l2=2.5, fit_mode='gradient')
#model = ezie.Model()
model.add_data(ezie_data)

#%%

fig = plt.figure()
pax = pp.Polarplot(plt.gca())
pax.plot(model.grid.lat_mesh[:, 0], (model.grid.lon_mesh[:, 0]/15)%24, color='k')
pax.plot(model.grid.lat_mesh[:, -1], (model.grid.lon_mesh[:, -1]/15)%24, color='k')
pax.plot(model.grid.lat_mesh[0, :], (model.grid.lon_mesh[0, :]/15)%24, color='k')
pax.plot(model.grid.lat_mesh[-1, :], (model.grid.lon_mesh[-1, :]/15)%24, color='k')

for mem in ezie_data.mems:
    pax.plot(mem.lat, (mem.lon/15)%24)


#%%

plt.figure()
plt.plot(model.m)

#%%

plotter = ezie.Plotter(model, userid='mih008')
fig, axs = plotter.plot_solution_cs()

#%%

fig, axs = plt.subplots(2,2, figsize=(10,10))
for i, ax in enumerate(axs.flatten()):
    ax.plot(ezie_data.mems[i].B)
    Be, Bn, Bu = model.ev.get_B_predictions(lat=model.data.mems[i].lat, lon=model.data.mems[i].lon, r=model.data.mems[i].r, ret=True)
    Be = Be + model.data.mems[i].B0e
    Bn = Bn + model.data.mems[i].B0n
    Bu = Bu + model.data.mems[i].B0u
    B = np.sqrt(Be**2 + Bn**2 + Bu**2)# + model.track_offsets[i]
    dB = B - model.data.mems[i].B0
    ax.plot(dB)



#%%











plotter = ezie.Plotter(model, userid='mih008')
fig, axs = plotter.plot_regOpt()

#%%

model2 = ezie.Model(l1=0.75, l2=2.5)
model2.add_data(ezie_data)
regOpt = ezie.RegularizationOptimizer(model2, 
                                      l1_lower=-3, l1_upper=-1, l1_steps=10, 
                                      l2_lower=-1.5, l2_upper=1.5, l2_steps=11)

y, x = np.meshgrid(regOpt.l2s, regOpt.l1s)
gini = np.copy(regOpt.gini_map)
gini /= np.max(gini, axis=1, keepdims=True)

#%%

plt.figure()
ax = plt.gca()
ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=100, cmap='magma')
#ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=np.linspace(0.99, 1, 100), cmap='magma', extend='both')
ax.plot(model.regOpt.l1_fit, model.regOpt.l2_fit)
ax.plot(regOpt.l1_fit, regOpt.l2_fit)
ax.set_xlim(-3, -1)
ax.set_ylim(-1.5, 1.5)
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 0, 0])
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 1, 0])

#%%

model2 = ezie.Model(l1=0.75, l2=2.5)
model2.add_data(ezie_data)
regOpt = ezie.RegularizationOptimizer(model2, iterations=3)

y, x = np.meshgrid(regOpt.l2s, regOpt.l1s)
gini = np.copy(regOpt.gini_map)
gini /= np.max(gini, axis=1, keepdims=True)

#%%

plt.figure()
ax = plt.gca()
ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=100, cmap='magma')
ax.plot(regOpt.l1_fit, regOpt.l2_fit)

#%%
plt.figure()
ax = plt.gca()
ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=100, cmap='magma')
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 0, 0])
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 1, 0])
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 2, 0])
ax.plot(regOpt.l1_fit, regOpt.l2_fit)



#%%%

model2 = ezie.Model(l1=0.75, l2=2.5)
model2.add_data(ezie_data)
regOpt = ezie.RegularizationOptimizer(model2, l1_steps=30, l2_steps=21, iterations=3)

y, x = np.meshgrid(regOpt.l2s, regOpt.l1s)
gini = np.copy(regOpt.gini_map)
gini /= np.max(gini, axis=1, keepdims=True)

plt.figure()
ax = plt.gca()
ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=100, cmap='magma')
ax.plot(regOpt.l1_fit, regOpt.l2_fit)

plt.figure()
ax = plt.gca()
ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), levels=100, cmap='magma')
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 0, 0])
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 1, 0])
ax.plot(regOpt.l1s, regOpt.ridge_exp[:, 2, 0])
ax.plot(regOpt.l1_fit, regOpt.l2_fit)
ax.plot(model.regOpt.l1_fit, model.regOpt.l2_fit)

plt.figure()
ax = plt.gca()
ax.plot(regOpt.l1_fit, regOpt.l2_fit)
ax.plot(model.regOpt.l1_fit, model.regOpt.l2_fit)

#%%

plotter = ezie.Plotter(model, userid='mih008')

figs, axs = plotter.plot_validation()
path = os.path.join(base, 'figures', 'supermag_comparison', 'orbit_' + str(orbit+1).zfill(2))
for fig in figs:
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, fig._suptitle.get_text() + '.png'), bbox_inches='tight', dpi=300)
plt.close('all')

fig, ax = plotter.plot_validation_map()
if not fig is None:
    plt.savefig(os.path.join(base, 'figures', 'supermag_comparison_map', 'orbit_' + str(orbit+1).zfill(2) + '.png'), bbox_inches='tight', dpi=300)
    plt.close('all')

fig, axs = plotter.plot_data_model_comparison()
plt.savefig(os.path.join(base, 'figures', 'data_model_comp_scalar', 'orbit_' + str(orbit+1).zfill(2) + '.png'), bbox_inches='tight', dpi=300)
plt.close('all')

#fig, axs = plotter.plot_solution()
#plt.savefig(os.path.join(base, 'figures', 'test', 'orbit_' + str(orbit+1).zfill(2) + '.png'), bbox_inches='tight', dpi=300)
#plt.close('all')

fig, axs = plotter.plot_solution_cs()
plt.savefig(os.path.join(base, 'figures', 'orbit_reconstruction_scalar', 'orbit_' + str(orbit+1).zfill(2) + '.png'), bbox_inches='tight', dpi=300)
plt.close('all')

fig, axs = plotter.plot_regOpt()
plt.savefig(os.path.join(base, 'figures', 'regularization_diagnostics_scalar', 'orbit_' + str(orbit+1).zfill(2) + '.png'), bbox_inches='tight', dpi=300)
plt.close('all')

l1s[orbit], l2s[orbit] = model.l1, model.l2
l1s_fit.append(model.regOpt.l1_fit)
l2s_fit.append(model.regOpt.l2_fit)
























