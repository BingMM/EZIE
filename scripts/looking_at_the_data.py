#%% Import

import os
import ezie
import matplotlib.pyplot as plt
from ezie.utils import generate_ezie_data
import numpy as np
import polplot as pp

#%% Set base

base = '/media/bing/LaCie/EZIE_L2/a/20250515'

#%%

file = os.path.join(base, 'ezie_l2_20250515_022454_sva_v003_r000.nc4')

#%%

data = ezie.utils.load_file(file)

#%%

fig, axs = plt.subplots(2, 4, figsize=(15, 10))

for i in range(4):
    mem = i+1
    axs[0, i].plot(data[f'Be{mem}'])
    axs[0, i].plot(data[f'Bn{mem}'])
    axs[0, i].plot(data[f'Bu{mem}'])
    axs[1, i].plot(data[f'B{mem}'])
    axs[1, i].plot(np.sqrt(data[f'Be{mem}']**2 + data[f'Bn{mem}']**2 + data[f'Bu{mem}']**2))

#%%

fig, axs = plt.subplots(3, 4, figsize=(15, 10))

for i in range(4):
    mem = i+1
    axs[0, i].plot(data[f'dBe{mem}'])
    axs[0, i].plot(data[f'dBn{mem}'])
    axs[0, i].plot(data[f'dBu{mem}'])
    axs[1, i].plot(data[f'dB{mem}'])
    axs[1, i].plot(np.sqrt(data[f'dBe{mem}']**2 + data[f'dBn{mem}']**2 + data[f'dBu{mem}']**2))
    Be = data[f'dBe{mem}'] + data[f'Be{mem}']
    Bn = data[f'dBn{mem}'] + data[f'Bn{mem}']
    Bu = data[f'dBu{mem}'] + data[f'Bu{mem}']
    B = np.sqrt(Be**2 + Bn**2 + Bu**2)
    dB = B - data[f'B{mem}']
    axs[2, i].plot(data[f'dB{mem}'])
    axs[2, i].plot(dB)

#%%

fig, axs = plt.subplots(4, 4, figsize=(15, 15))

for i in range(4):
    mem = i+1
    axs[0, i].plot(data[f'dBe{mem}'])
    axs[0, i].plot(data[f'dBn{mem}'])
    axs[0, i].plot(data[f'dBu{mem}'])
    
    axs[1, i].plot(data[f'dBe_c{mem}'])
    axs[1, i].plot(data[f'dBn_c{mem}'])
    axs[1, i].plot(data[f'dBu_c{mem}'])
    
    Be = data[f'dBe{mem}'] + data[f'Be{mem}']
    Bn = data[f'dBn{mem}'] + data[f'Bn{mem}']
    Bu = data[f'dBu{mem}'] + data[f'Bu{mem}']
    B = np.sqrt(Be**2 + Bn**2 + Bu**2)
    dB = B - data[f'B{mem}']
    axs[2, i].plot(data[f'dB{mem}'])
    axs[2, i].plot(dB)
    
    Be = data[f'dBe_c{mem}'] + data[f'Be{mem}']
    Bn = data[f'dBn_c{mem}'] + data[f'Bn{mem}']
    Bu = data[f'dBu_c{mem}'] + data[f'Bu{mem}']
    B = np.sqrt(Be**2 + Bn**2 + Bu**2)
    dB = B - data[f'B{mem}']
    axs[3, i].plot(data[f'dB{mem}'])
    axs[3, i].plot(dB)







#%% Load data

#file = os.path.join(base, 'ezie_l2_20250515_004708_sva_v003_r000.nc4')
file = os.path.join(base, 'ezie_l2_20250515_022454_sva_v003_r000.nc4')
#ezie_data = generate_ezie_data(file, comp='ENU', background='chaos')
ezie_data = generate_ezie_data(file, comp='ENU')

#%%

model = ezie.Model(l1=0.75, l2=2.5)
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

fig, axs = plotter.plot_regOpt()


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
























