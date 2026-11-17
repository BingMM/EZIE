#%% Import

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ezie.sml import nearest_sml_indices, read_sml_cache
from ezie.utils import load_file

from secsy import CSgrid, CSprojection

import polplot as pp
import matplotlib.pyplot as plt

#%% Configuration

input_base = Path('/media/bing/LaCie/EZIE_L2')
output_file = input_base.parent / 'EZIE_lithosphere_maps_l2' / 'scalar_50km.nc'
sats = ['a', 'b', 'c']
max_files = None  # Set to e.g. 11 for a short interactive run.
sml_cache_file = input_base.parent / 'EZIE_lithosphere_maps' / 'supermag_sml.nc'
sml_limit_nt = 80.0
sml_derivative_limit_nt_per_minute = 10.0
sml_match_tolerance_seconds = 60.0

#%%

files = sorted(input_base.rglob('*.nc4'))
if max_files is not None:
    files = files[:max_files]
if not files:
    raise FileNotFoundError(f'No Level-2 files found under {input_base}')

df = []

for file in tqdm(files):
    data = load_file(file, background='igrf')

    df.append(pd.DataFrame({'sat': data['sat'][-1],
                            'time': data['time'],
                            'lat1': data['lat1'],
                            'lat2': data['lat2'],
                            'lat3': data['lat3'],
                            'lat4': data['lat4'],
                            'lon1': data['lon1'],
                            'lon2': data['lon2'],
                            'lon3': data['lon3'],
                            'lon4': data['lon4'],
                            'dB1': data['dB1'],
                            'dB2': data['dB2'],
                            'dB3': data['dB3'],
                            'dB4': data['dB4'],
                            'dB1_var': data['mag_var1'],
                            'dB2_var': data['mag_var2'],
                            'dB3_var': data['mag_var3'],
                            'dB4_var': data['mag_var4']}))

df = pd.concat(df, ignore_index=True)

#%% Match the existing minute-resolution SML cache to each L2 observation.
# dB and dB_var retain the loader's T and T^2 units; SML is in nT.

sml_data = read_sml_cache(sml_cache_file)
if sml_data is None:
    raise FileNotFoundError(f'No complete SML cache found at {sml_cache_file}')
sml_time, sml, sml_derivative = sml_data

# Normalize to UTC nanoseconds for the shared nearest-minute matcher.
observation_time = pd.to_datetime(df['time'], utc=True)
observation_time_ns = observation_time.to_numpy(dtype='datetime64[ns]').astype(np.int64)
nearest, matched = nearest_sml_indices(observation_time_ns, sml_time, tolerance_seconds=sml_match_tolerance_seconds)
matched &= observation_time.notna().to_numpy()

df['SML'] = np.where(matched, sml[nearest], np.nan)
df['dSML_dt'] = np.where(matched, sml_derivative[nearest], np.nan)
df['sml_matched'] = matched
df['quiet_sml'] = (matched & (df['SML'].abs() < sml_limit_nt) & (df['dSML_dt'].abs() < sml_derivative_limit_nt_per_minute))

print(f'SML cache: {pd.to_datetime(sml_time[0], unit="s", utc=True)} to {pd.to_datetime(sml_time[-1], unit="s", utc=True)}')
print(f'SML matched {matched.sum():,}/{len(df):,} L2 rows; {df["quiet_sml"].sum():,} pass the quiet-time limits.')

# Keep all observations in df; use this view for a quiet-time map.
df_quiet = df.loc[df['quiet_sml']]

#%% Generate grid

projection = CSprojection([270.0, 85.0], [0.0, 1.0])
grid = CSgrid(projection, L=50000e3, W=50000e3, Lres=50e3, Wres=50e3, R=(6371.2+80)*1e3)

#%% Do statistics

# Pool the four MEMs, keeping each scalar observation at its own position.
latitude = df_quiet[[f'lat{mem}' for mem in range(1, 5)]].to_numpy().ravel()
longitude = df_quiet[[f'lon{mem}' for mem in range(1, 5)]].to_numpy().ravel()
scalar_values = df_quiet[[f'dB{mem}' for mem in range(1, 5)]].to_numpy().ravel()

finite = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(scalar_values)
latitude = latitude[finite]
longitude = longitude[finite]
scalar_values = scalar_values[finite]

bin_i, bin_j = grid.bin_index(longitude, latitude)
in_grid = (bin_i >= 0) & (bin_j >= 0)
cell_index = np.ravel_multi_index((bin_i[in_grid], bin_j[in_grid]), grid.shape)

cell_statistics = pd.Series(scalar_values[in_grid]).groupby(cell_index).agg(['median', 'count'])

# Empty cells have NaN medians and zero counts. Medians retain units of T.
medians = np.full(grid.shape, np.nan)
counts = np.zeros(grid.shape, dtype=np.int64)
medians.ravel()[cell_statistics.index.to_numpy()] = cell_statistics['median'].to_numpy()
counts.ravel()[cell_statistics.index.to_numpy()] = cell_statistics['count'].to_numpy()

#%% Plot

fig = plt.figure(figsize=(15,15))
ax = plt.gca()
pax = pp.Polarplot(ax, minlat=40)
pax.coastlines()

pax.plot(grid.lat[:, 0], (grid.lon[:, 0]/15)%24, color='k')
pax.plot(grid.lat[:, -1], (grid.lon[:, -1]/15)%24, color='k')
pax.plot(grid.lat[0, :], (grid.lon[0, :]/15)%24, color='k')
pax.plot(grid.lat[-1, :], (grid.lon[-1, :]/15)%24, color='k')

tmp = medians + 0
tmp[counts < 10] = np.nan

pax.plotimg(grid.lat, (grid.lon/15)%24, tmp, cmap='bwr', crange=(-300e-9, 300e-9))
