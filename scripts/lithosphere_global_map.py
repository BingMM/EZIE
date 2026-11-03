#%% Import

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from polplot import Polarplot
from scipy.ndimage import gaussian_filter
from secsy import CSgrid, CSprojection
from tqdm import tqdm

from ezie.supermag_api import SuperMAGGetIndices


#%% Configuration

input_base = Path('/media/bing/LaCie/EZIE_lithosphere_l3')
output_file = input_base.parent / 'EZIE_lithosphere_maps' / 'scalar_50km.nc'

scalar_variable = 'predicted_dB_exact'
satellite_ids = ('a', 'b', 'c')
mem_count = 4
minimum_latitude = 50.0
resolution_m = 50_000.0
footprint_size_m = 50_000.0
minimum_coverage_count = 10
earth_radius_m = 6_371_200.0

supermag_userid = 'mih008'
sml_cache_file = output_file.parent / 'supermag_sml.nc'
sml_limit_nt = 80.0
sml_derivative_limit_nt_per_minute = 10.0
sml_match_tolerance_seconds = 60.0

map_schema_version = '1.3'


#%% Global northern cubed-sphere grid

def create_global_grid(minimum_latitude=50.0, resolution_m=50_000.0,
                       radius_m=6_371_200.0):
    """Create a north-polar cubed-sphere grid covering the requested cap."""

    if not 45 < minimum_latitude < 90:
        raise ValueError('minimum_latitude must be between 45 and 90 degrees')
    if resolution_m <= 0:
        raise ValueError('resolution_m must be positive')

    projection = CSprojection(
        np.array([0.0, 90.0]), np.array([0.0, 1.0]))

    half_width = np.deg2rad(90.0-minimum_latitude)
    target_step = np.arctan(resolution_m/radius_m)
    cell_count = int(np.ceil(2*half_width/target_step))
    edges = np.linspace(-half_width, half_width, cell_count+1)

    # L and W are retained as grid metadata. Explicit xi/eta edges define the
    # actual grid and give symmetric coverage around the geographic pole.
    dimension = radius_m*np.tan(2*half_width)
    grid = CSgrid(
        projection, dimension, dimension, resolution_m, resolution_m,
        edges=(edges, edges), R=radius_m)
    return grid


#%% Checkpoint loading and binning

def discover_checkpoints(directory):
    """Return a stable snapshot of the available checkpoint paths."""

    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f'Checkpoint directory does not exist: {directory}')
    return sorted(directory.rglob('*.nc'))


def checkpoint_time_range(paths, show_progress=True):
    """Return the finite observation-time range of complete checkpoints."""

    minimum = None
    maximum = None
    missing_time = np.iinfo(np.int64).min
    complete_count = 0
    for path in tqdm(
            paths, desc='Scanning checkpoint times', disable=not show_progress):
        with Dataset(path) as dataset:
            if getattr(dataset, 'processing_complete', 0) != 1:
                continue
            if 'time' not in dataset.variables:
                raise ValueError(f'{path} is missing required variable: time')
            time = np.asarray(
                np.ma.filled(dataset.variables['time'][:], missing_time),
                dtype=np.int64)
            valid = time != missing_time
            if not np.any(valid):
                continue
            file_minimum = np.min(time[valid])
            file_maximum = np.max(time[valid])
            minimum = file_minimum if minimum is None else min(minimum, file_minimum)
            maximum = file_maximum if maximum is None else max(maximum, file_maximum)
            complete_count += 1

    if minimum is None:
        raise ValueError('No complete checkpoints contain observation times')
    return minimum, maximum, complete_count


#%% SuperMAG SML cache and quiet-time selection

def calculate_sml_derivative(time_seconds, sml_nt):
    """Calculate dSML/dt in nT/min without bridging gaps in the index."""

    time_seconds = np.asarray(time_seconds, dtype=float)
    sml_nt = np.asarray(sml_nt, dtype=float)
    if time_seconds.ndim != 1 or sml_nt.shape != time_seconds.shape:
        raise ValueError('SML time and values must be equal-length 1D arrays')
    if time_seconds.size < 2 or np.any(np.diff(time_seconds) <= 0):
        raise ValueError('SML time must contain at least two increasing samples')

    derivative = np.gradient(sml_nt, time_seconds/60)
    steps = np.diff(time_seconds)
    regular = np.zeros(time_seconds.size, dtype=bool)
    regular[0] = steps[0] <= 90
    regular[-1] = steps[-1] <= 90
    if regular.size > 2:
        regular[1:-1] = (steps[:-1] <= 90) & (steps[1:] <= 90)
    derivative[~regular] = np.nan
    return derivative


def write_sml_cache(path, time_seconds, sml_nt):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    derivative = calculate_sml_derivative(time_seconds, sml_nt)

    try:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            dataset.processing_complete = 0
            dataset.source = 'SuperMAG SML index service'
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            dataset.createDimension('time', len(time_seconds))
            time = dataset.createVariable('time', 'i8', ('time',))
            time[:] = np.asarray(time_seconds, dtype=np.int64)
            time.units = 'seconds since 1970-01-01 00:00:00 UTC'
            sml = dataset.createVariable('SML', 'f8', ('time',))
            sml[:] = sml_nt
            sml.units = 'nT'
            d_sml = dataset.createVariable('dSML_dt', 'f8', ('time',))
            d_sml[:] = derivative
            d_sml.units = 'nT/min'
            dataset.processing_complete = 1
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_sml_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            return None
        required = {'time', 'SML', 'dSML_dt'}
        if not required.issubset(dataset.variables):
            return None
        return (
            np.asarray(dataset.variables['time'][:], dtype=np.int64),
            np.asarray(dataset.variables['SML'][:], dtype=float),
            np.asarray(dataset.variables['dSML_dt'][:], dtype=float))


def download_sml(start_seconds, stop_seconds, userid=supermag_userid,
                 chunk_days=7, maximum_attempts=3,
                 retry_delay_seconds=5.0):
    """Download minute SML in bounded chunks and return sorted unique samples."""

    if maximum_attempts < 1:
        raise ValueError('maximum_attempts must be at least one')
    if retry_delay_seconds < 0:
        raise ValueError('retry_delay_seconds must be non-negative')
    start_seconds = int(np.floor(start_seconds/60)*60)
    stop_seconds = int(np.ceil(stop_seconds/60)*60)
    chunk_seconds = int(chunk_days*24*60*60)
    time_chunks = []
    sml_chunks = []

    starts = range(start_seconds, stop_seconds, chunk_seconds)
    loop = tqdm(starts, desc='Downloading SuperMAG SML')
    for chunk_start in loop:
        chunk_stop = min(chunk_start+chunk_seconds, stop_seconds)
        start = datetime.fromtimestamp(chunk_start, tz=timezone.utc)
        failure = None
        for attempt in range(1, maximum_attempts+1):
            try:
                status, data = SuperMAGGetIndices(
                    userid, start, chunk_stop-chunk_start, 'sml')
                valid = (
                    status == 1 and not data.empty and
                    {'tval', 'SML'}.issubset(data.columns))
                if valid:
                    break
                failure = f'status {status} or incomplete response'
            except Exception as error:
                failure = f'{type(error).__name__}: {error}'

            if attempt < maximum_attempts:
                delay = retry_delay_seconds*attempt
                tqdm.write(
                    f'SuperMAG request failed at {start.isoformat()} '
                    f'(attempt {attempt}/{maximum_attempts}); '
                    f'retrying in {delay:g} s')
                sleep(delay)
        else:
            raise RuntimeError(
                f'SuperMAG SML request failed at {start.isoformat()} after '
                f'{maximum_attempts} attempts: {failure}')
        time_chunks.append(data['tval'].to_numpy(dtype=np.int64))
        sml_chunks.append(data['SML'].to_numpy(dtype=float))

    time_seconds = np.concatenate(time_chunks)
    sml_nt = np.concatenate(sml_chunks)
    finite = np.isfinite(time_seconds) & np.isfinite(sml_nt)
    time_seconds = time_seconds[finite]
    sml_nt = sml_nt[finite]
    order = np.argsort(time_seconds)
    time_seconds = time_seconds[order]
    sml_nt = sml_nt[order]
    time_seconds, unique_id = np.unique(time_seconds, return_index=True)
    sml_nt = sml_nt[unique_id]
    return time_seconds, sml_nt


def load_or_download_sml(cache_path, start_ns, stop_ns,
                         userid=supermag_userid, refresh=False):
    """Load a covering SML cache, downloading the requested span if needed."""

    # One minute of padding supports centred derivatives at both map boundaries.
    required_start = int(start_ns//1_000_000_000)-60
    required_stop = int(np.ceil(stop_ns/1_000_000_000))+120
    cached = None if refresh else read_sml_cache(cache_path)
    if cached is not None:
        time_seconds, sml_nt, derivative = cached
        if time_seconds[0] <= required_start and time_seconds[-1] >= required_stop-60:
            return time_seconds, sml_nt, derivative

    time_seconds, sml_nt = download_sml(
        required_start, required_stop, userid=userid)
    write_sml_cache(cache_path, time_seconds, sml_nt)
    return read_sml_cache(cache_path)


def quiet_sml_mask(observation_time_ns, sml_time_seconds, sml_nt,
                   derivative_nt_per_minute, sml_limit_nt=80.0,
                   derivative_limit_nt_per_minute=10.0,
                   tolerance_seconds=60.0):
    """Match observations to nearest-minute SML and apply quiet/stable limits."""

    observation_time_ns = np.asarray(observation_time_ns, dtype=np.int64).ravel()
    sml_time_seconds = np.asarray(sml_time_seconds, dtype=np.int64)
    sml_nt = np.asarray(sml_nt, dtype=float)
    derivative = np.asarray(derivative_nt_per_minute, dtype=float)
    if not (sml_time_seconds.size == sml_nt.size == derivative.size):
        raise ValueError('SML arrays must have equal size')

    missing_time = np.iinfo(np.int64).min
    valid_time = observation_time_ns != missing_time
    observation_seconds = np.zeros(observation_time_ns.size, dtype=float)
    observation_seconds[valid_time] = (
        observation_time_ns[valid_time]/1_000_000_000)

    right = np.searchsorted(sml_time_seconds, observation_seconds)
    right = np.clip(right, 0, sml_time_seconds.size-1)
    left = np.clip(right-1, 0, sml_time_seconds.size-1)
    use_right = (
        np.abs(sml_time_seconds[right]-observation_seconds) <
        np.abs(sml_time_seconds[left]-observation_seconds))
    nearest = np.where(use_right, right, left)
    distance = np.abs(sml_time_seconds[nearest]-observation_seconds)
    matched = valid_time & (distance <= tolerance_seconds)
    quiet = (
        matched &
        (np.abs(sml_nt[nearest]) < sml_limit_nt) &
        (np.abs(derivative[nearest]) < derivative_limit_nt_per_minute))
    return quiet, matched


def read_checkpoint_samples(path, variable_name):
    """Read positions and one scalar prediction from a complete checkpoint."""

    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            return None

        required = {'time', 'latitude', 'longitude', variable_name}
        missing = required.difference(dataset.variables)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'{path} is missing required variables: {names}')

        variable = dataset.variables[variable_name]
        units = getattr(variable, 'units', None)
        if units != 'T':
            raise ValueError(
                f'{path}: {variable_name} must have units T, found {units!r}')

        latitude = np.asarray(
            np.ma.filled(dataset.variables['latitude'][:], np.nan),
            dtype=float)
        longitude = np.asarray(
            np.ma.filled(dataset.variables['longitude'][:], np.nan),
            dtype=float)
        values = np.asarray(
            np.ma.filled(variable[:], np.nan), dtype=float)
        missing_time = np.iinfo(np.int64).min
        time = np.asarray(
            np.ma.filled(dataset.variables['time'][:], missing_time),
            dtype=np.int64)

        if not (time.shape == latitude.shape == longitude.shape == values.shape):
            raise ValueError(f'{path}: coordinate and scalar shapes do not match')
        if values.ndim != 2 or not 1 <= values.shape[0] <= mem_count:
            raise ValueError(
                f'{path}: expected a MEM by sample scalar array with at most '
                f'{mem_count} MEMs, found shape {values.shape}')

        mem_number = np.broadcast_to(
            np.arange(1, values.shape[0]+1)[:, None], values.shape)

        if 'magnetic_local_time' in dataset.variables:
            mlt = np.asarray(
                np.ma.filled(
                    dataset.variables['magnetic_local_time'][:], np.nan),
                dtype=float)
        else:
            source_path = Path(str(getattr(dataset, 'source_path', '')))
            if not source_path.is_file():
                raise FileNotFoundError(
                    f'{path}: Level-2 source file does not exist: '
                    f'{source_path}')
            with Dataset(source_path) as source:
                magnetic = source.groups.get('MagneticCoords')
                if magnetic is None:
                    raise ValueError(
                        f'{source_path}: missing MagneticCoords group')
                mlt = np.full(values.shape, np.nan)
                for mem_id in range(values.shape[0]):
                    name = f'obs_magLT{mem_id+1}'
                    if name not in magnetic.variables:
                        raise ValueError(f'{source_path}: missing {name}')
                    source_mlt = np.asarray(
                        np.ma.filled(magnetic.variables[name][:], np.nan),
                        dtype=float)
                    if source_mlt.size < values.shape[1]:
                        raise ValueError(
                            f'{source_path}: {name} is shorter than checkpoint')
                    mlt[mem_id] = source_mlt[:values.shape[1]]
        if mlt.shape != values.shape:
            raise ValueError(
                f'{path}: magnetic local time shape does not match scalar data')

        metadata = {
            'satellite': str(
                getattr(dataset, 'satellite', 'unknown')).lower(),
            'checkpoint_schema': str(getattr(dataset, 'schema_version', 'unknown'))}
        return (
            time.ravel(), latitude.ravel(), longitude.ravel(), values.ravel(),
            mlt.ravel(), mem_number.ravel(), metadata)


def initialize_statistics(grid):
    return {
        'count': np.zeros(grid.shape, dtype=np.int64),
        'sum': np.zeros(grid.shape, dtype=np.float64),
        'sum_of_squares': np.zeros(grid.shape, dtype=np.float64)}


def bin_scalar_samples(grid, statistics, latitude, longitude, values,
                       minimum_latitude=50.0, return_binned=False):
    """Add scalar samples to sufficient statistics on a CS grid."""

    latitude = np.asarray(latitude, dtype=float).ravel()
    longitude = np.asarray(longitude, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if not (latitude.size == longitude.size == values.size):
        raise ValueError('latitude, longitude, and values must have equal size')

    finite = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(values)
    use = finite & (latitude >= minimum_latitude)
    input_count = np.count_nonzero(finite)
    if not np.any(use):
        if return_binned:
            return input_count, 0, np.array([], dtype=np.int32), np.array([])
        return input_count, 0

    xi, eta = grid.projection.geo2cube(longitude[use], latitude[use])
    xi = np.asarray(xi)
    eta = np.asarray(eta)

    xi_edges = grid.xi_mesh[0, :]
    eta_edges = grid.eta_mesh[:, 0]
    xi_id = np.searchsorted(xi_edges, xi, side='right')-1
    eta_id = np.searchsorted(eta_edges, eta, side='right')-1

    inside = (
        (xi >= xi_edges[0]) & (xi <= xi_edges[-1]) &
        (eta >= eta_edges[0]) & (eta <= eta_edges[-1]))
    if not np.any(inside):
        if return_binned:
            return input_count, 0, np.array([], dtype=np.int32), np.array([])
        return input_count, 0

    xi_id = np.clip(xi_id, 0, grid.shape[1]-1)
    eta_id = np.clip(eta_id, 0, grid.shape[0]-1)
    selected_values = values[use][inside]
    flat_id = (
        eta_id[inside]*grid.shape[1] + xi_id[inside]).astype(np.int32)
    size = grid.size

    statistics['count'] += np.bincount(
        flat_id, minlength=size).reshape(grid.shape)
    statistics['sum'] += np.bincount(
        flat_id, weights=selected_values,
        minlength=size).reshape(grid.shape)
    statistics['sum_of_squares'] += np.bincount(
        flat_id, weights=selected_values**2,
        minlength=size).reshape(grid.shape)
    if return_binned:
        return input_count, selected_values.size, flat_id, selected_values
    return input_count, selected_values.size


def initialize_footprint_statistics(grid):
    statistics = initialize_statistics(grid)
    statistics['coverage_weight'] = np.zeros(grid.shape, dtype=np.float64)
    statistics['contribution_count'] = np.zeros(grid.shape, dtype=np.int64)
    return statistics


def rasterize_checkpoint_footprints(
        grid, latitude, longitude, values, footprint_size_m=50_000.0,
        minimum_latitude=50.0):
    """Area-weight one checkpoint's square footprints onto the map grid.

    The footprint is aligned with the cubed-sphere xi/eta axes. All samples in
    one checkpoint are collapsed to one weighted value per cell so overlapping
    samples from the same inversion do not increase the checkpoint count.
    """

    latitude = np.asarray(latitude, dtype=float).ravel()
    longitude = np.asarray(longitude, dtype=float).ravel()
    values = np.asarray(values, dtype=float).ravel()
    if not (latitude.size == longitude.size == values.size):
        raise ValueError('latitude, longitude, and values must have equal size')
    if footprint_size_m <= 0:
        raise ValueError('footprint_size_m must be positive')

    finite = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(values)
    if not np.any(finite):
        empty_int = np.array([], dtype=np.int32)
        empty_float = np.array([], dtype=float)
        return empty_int, empty_float, empty_float, empty_int

    xi, eta = grid.projection.geo2cube(longitude[finite], latitude[finite])
    xi = np.asarray(xi, dtype=float)
    eta = np.asarray(eta, dtype=float)
    selected_values = values[finite]

    xi_edges = np.asarray(grid.xi_mesh[0, :], dtype=float)
    eta_edges = np.asarray(grid.eta_mesh[:, 0], dtype=float)
    cap = np.asarray(grid.lat >= minimum_latitude).ravel()
    half_width = np.arctan(footprint_size_m/(2*grid.R))
    footprint_area = (2*half_width)**2

    weighted_sum = np.zeros(grid.size, dtype=float)
    coverage_weight = np.zeros(grid.size, dtype=float)
    contribution_count = np.zeros(grid.size, dtype=np.int64)

    for center_xi, center_eta, value in zip(xi, eta, selected_values):
        xi_low = center_xi-half_width
        xi_high = center_xi+half_width
        eta_low = center_eta-half_width
        eta_high = center_eta+half_width

        xi_start = max(0, np.searchsorted(xi_edges, xi_low, side='right')-1)
        xi_stop = min(grid.shape[1], np.searchsorted(
            xi_edges, xi_high, side='left'))
        eta_start = max(0, np.searchsorted(eta_edges, eta_low, side='right')-1)
        eta_stop = min(grid.shape[0], np.searchsorted(
            eta_edges, eta_high, side='left'))

        for eta_id in range(eta_start, eta_stop):
            eta_overlap = max(
                0.0,
                min(eta_high, eta_edges[eta_id+1]) -
                max(eta_low, eta_edges[eta_id]))
            if eta_overlap == 0:
                continue

            for xi_id in range(xi_start, xi_stop):
                flat_id = eta_id*grid.shape[1] + xi_id
                if not cap[flat_id]:
                    continue

                xi_overlap = max(
                    0.0,
                    min(xi_high, xi_edges[xi_id+1]) -
                    max(xi_low, xi_edges[xi_id]))
                overlap_weight = xi_overlap*eta_overlap/footprint_area
                if overlap_weight == 0:
                    continue

                weighted_sum[flat_id] += overlap_weight*value
                coverage_weight[flat_id] += overlap_weight
                contribution_count[flat_id] += 1

    populated = coverage_weight > 0
    flat_ids = np.flatnonzero(populated).astype(np.int32)
    checkpoint_values = weighted_sum[populated]/coverage_weight[populated]
    return (
        flat_ids,
        checkpoint_values,
        coverage_weight[populated],
        contribution_count[populated].astype(np.int32))


def add_checkpoint_footprints(
        grid, statistics, latitude, longitude, values,
        footprint_size_m=50_000.0, minimum_latitude=50.0):
    """Add one area-weighted checkpoint map to cross-checkpoint statistics."""

    flat_ids, checkpoint_values, coverage_weight, contribution_count = (
        rasterize_checkpoint_footprints(
            grid, latitude, longitude, values,
            footprint_size_m=footprint_size_m,
            minimum_latitude=minimum_latitude))
    if flat_ids.size == 0:
        return flat_ids, checkpoint_values

    add_rasterized_footprints(
        statistics, flat_ids, checkpoint_values,
        coverage_weight, contribution_count)
    return flat_ids, checkpoint_values


def add_rasterized_footprints(
        statistics, flat_ids, checkpoint_values,
        coverage_weight, contribution_count):
    """Add an already-rasterized checkpoint estimate to map statistics."""

    flat_count = statistics['count'].ravel()
    flat_sum = statistics['sum'].ravel()
    flat_sum_of_squares = statistics['sum_of_squares'].ravel()
    flat_coverage = statistics['coverage_weight'].ravel()
    flat_contributions = statistics['contribution_count'].ravel()

    flat_count[flat_ids] += 1
    flat_sum[flat_ids] += checkpoint_values
    flat_sum_of_squares[flat_ids] += checkpoint_values**2
    flat_coverage[flat_ids] += coverage_weight
    flat_contributions[flat_ids] += contribution_count


def combine_mem_footprints(mem_results):
    """Combine separately rasterized MEMs into one checkpoint estimate."""

    populated = [result for result in mem_results if result[0].size]
    if not populated:
        empty_int = np.array([], dtype=np.int32)
        empty_float = np.array([], dtype=float)
        return empty_int, empty_float, empty_float, empty_int

    flat_ids = np.concatenate([result[0] for result in populated])
    values = np.concatenate([result[1] for result in populated])
    coverage = np.concatenate([result[2] for result in populated])
    contributions = np.concatenate([result[3] for result in populated])
    unique_ids, inverse = np.unique(flat_ids, return_inverse=True)
    total_coverage = np.bincount(inverse, weights=coverage)
    weighted_sum = np.bincount(inverse, weights=coverage*values)
    total_contributions = np.bincount(
        inverse, weights=contributions).astype(np.int32)
    combined_values = weighted_sum/total_coverage
    return (
        unique_ids.astype(np.int32), combined_values,
        total_coverage, total_contributions)


def unwrap_periodic(values, period=24.0):
    """Unwrap a continuous periodic track while preserving NaNs."""

    values = np.asarray(values, dtype=float)
    output = values.copy()
    finite = np.isfinite(values)
    if np.any(finite):
        angle = values[finite]*2*np.pi/period
        output[finite] = np.unwrap(angle)*period/(2*np.pi)
    return output


def combine_circular_footprints(mem_results, period=24.0):
    """Combine MEM footprint estimates using coverage-weighted directions."""

    populated = [result for result in mem_results if result[0].size]
    if not populated:
        return np.array([], dtype=np.int32), np.array([], dtype=float)

    flat_ids = np.concatenate([result[0] for result in populated])
    values = np.concatenate([result[1] for result in populated]) % period
    coverage = np.concatenate([result[2] for result in populated])
    unique_ids, inverse = np.unique(flat_ids, return_inverse=True)
    angle = values*2*np.pi/period
    east = np.bincount(inverse, weights=coverage*np.cos(angle))
    north = np.bincount(inverse, weights=coverage*np.sin(angle))
    combined = np.arctan2(north, east) % (2*np.pi)
    combined *= period/(2*np.pi)
    return unique_ids.astype(np.int32), combined


def calculate_median(flat_ids, values, shape):
    """Calculate the exact median in every populated flattened grid cell."""

    flat_ids = np.asarray(flat_ids)
    values = np.asarray(values, dtype=float)
    median = np.full(np.prod(shape), np.nan)
    if values.size == 0:
        return median.reshape(shape)

    order = np.argsort(flat_ids)
    sorted_ids = flat_ids[order]
    sorted_values = values[order]
    unique_ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)
    for cell_id, start, count in zip(unique_ids, starts, counts):
        median[cell_id] = np.median(sorted_values[start:start+count])
    return median.reshape(shape)


def calculate_circular_median(flat_ids, values, shape, period=24.0):
    """Calculate a wrap-safe median for each populated flattened cell."""

    flat_ids = np.asarray(flat_ids)
    values = np.asarray(values, dtype=float) % period
    median = np.full(np.prod(shape), np.nan)
    if values.size == 0:
        return median.reshape(shape)
    if flat_ids.size != values.size:
        raise ValueError('flat_ids and values must have equal size')

    order = np.argsort(flat_ids)
    sorted_ids = flat_ids[order]
    sorted_values = values[order]
    unique_ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)
    for cell_id, start, count in zip(unique_ids, starts, counts):
        cell_values = sorted_values[start:start+count]
        angle = cell_values*2*np.pi/period
        vector = np.mean(np.exp(1j*angle))
        reference = (
            cell_values[0] if np.abs(vector) < 1e-12
            else np.angle(vector)*period/(2*np.pi))
        offsets = (cell_values-reference+period/2) % period-period/2
        cell_median = (reference+np.median(offsets)) % period
        median[cell_id] = 0.0 if np.isclose(cell_median, period) else cell_median
    return median.reshape(shape)


def satellite_mem_id(satellite, mem_number):
    """Return the stable 0--11 identifier for one satellite/MEM stream."""

    satellite = str(satellite).lower()
    mem_number = int(mem_number)
    if satellite not in satellite_ids:
        raise ValueError(f'Unknown satellite identifier: {satellite!r}')
    if not 1 <= mem_number <= mem_count:
        raise ValueError(f'MEM number must be between 1 and {mem_count}')
    return satellite_ids.index(satellite)*mem_count+mem_number-1


def calculate_equal_mem_median(flat_ids, values, stream_ids, shape):
    """Give every represented satellite/MEM stream equal weight per cell.

    Every checkpoint estimate receives the inverse of its stream's checkpoint
    count in that cell. Each represented stream therefore has total weight one,
    while checkpoints remain equally weighted within their stream.
    """

    flat_ids = np.asarray(flat_ids, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    stream_ids = np.asarray(stream_ids, dtype=np.int64)
    if not (flat_ids.size == values.size == stream_ids.size):
        raise ValueError('flat_ids, values, and stream_ids must have equal size')

    size = int(np.prod(shape))
    represented_count = np.zeros(size, dtype=np.int16)
    equal_median = np.full(size, np.nan)
    if values.size == 0:
        return equal_median.reshape(shape), represented_count.reshape(shape)
    if np.any(flat_ids < 0) or np.any(flat_ids >= size):
        raise ValueError('flat cell ID is outside the requested shape')
    if np.any(stream_ids < 0) or np.any(
            stream_ids >= len(satellite_ids)*mem_count):
        raise ValueError('satellite/MEM stream ID is outside the valid range')

    stream_cell_id = stream_ids*size+flat_ids
    unique_stream_cells, inverse, stream_cell_counts = np.unique(
        stream_cell_id, return_inverse=True, return_counts=True)
    weights = 1/stream_cell_counts[inverse]
    represented_cell_ids = unique_stream_cells % size
    represented_count[:] = np.bincount(
        represented_cell_ids, minlength=size).astype(np.int16)

    order = np.lexsort((values, flat_ids))
    sorted_cells = flat_ids[order]
    sorted_values = values[order]
    sorted_weights = weights[order]
    cell_ids, starts, counts = np.unique(
        sorted_cells, return_index=True, return_counts=True)
    for cell_id, start, count in zip(cell_ids, starts, counts):
        cell_values = sorted_values[start:start+count]
        cell_weights = sorted_weights[start:start+count]
        cumulative = np.cumsum(cell_weights)
        half_weight = cumulative[-1]/2
        median_id = np.searchsorted(cumulative, half_weight, side='left')
        if (
                median_id+1 < count and
                np.isclose(cumulative[median_id], half_weight)):
            equal_median[cell_id] = (
                cell_values[median_id]+cell_values[median_id+1])/2
        else:
            equal_median[cell_id] = cell_values[median_id]
    return equal_median.reshape(shape), represented_count.reshape(shape)


def gaussian_filter_nan(values, sigma=1.0, valid_mask=None):
    """Apply a normalized Gaussian filter without treating missing cells as zero."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    numerator = gaussian_filter(
        np.where(finite, values, 0), sigma=sigma, mode='constant', cval=0)
    denominator = gaussian_filter(
        finite.astype(float), sigma=sigma, mode='constant', cval=0)
    filtered = np.full(values.shape, np.nan)
    supported = denominator > 0
    filtered[supported] = numerator[supported]/denominator[supported]
    if valid_mask is not None:
        filtered[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return filtered


def apply_coverage_threshold(values, count, minimum_count=10):
    """Mask map cells supported by fewer than the requested checkpoints."""

    if minimum_count < 0:
        raise ValueError('minimum_count must be non-negative')
    values = np.asarray(values, dtype=float)
    count = np.asarray(count)
    if values.shape != count.shape:
        raise ValueError('values and count must have the same shape')

    masked = values.copy()
    masked[count < minimum_count] = np.nan
    return masked


def finalize_statistics(statistics, grid, median,
                        minimum_latitude=50.0):
    """Calculate map products while retaining mergeable sufficient statistics."""

    count = statistics['count']
    total = statistics['sum']
    total_squared = statistics['sum_of_squares']
    cap = grid.lat >= minimum_latitude

    mean = np.full(grid.shape, np.nan)
    standard_deviation = np.full(grid.shape, np.nan)
    standard_error = np.full(grid.shape, np.nan)

    populated = count > 0
    mean[populated] = total[populated]/count[populated]

    repeated = count > 1
    centered_sum = np.zeros(grid.shape)
    centered_sum[repeated] = (
        total_squared[repeated] -
        total[repeated]**2/count[repeated])
    centered_sum[repeated] = np.maximum(centered_sum[repeated], 0)
    standard_deviation[repeated] = np.sqrt(
        centered_sum[repeated]/(count[repeated]-1))
    standard_error[repeated] = (
        standard_deviation[repeated]/np.sqrt(count[repeated]))

    return {
        **statistics,
        'mean': mean,
        'median': median,
        'standard_deviation': standard_deviation,
        'standard_error': standard_error,
        'cap_mask': cap}


#%% NetCDF output

def create_variable(dataset, name, dtype, dimensions, values,
                    units=None, long_name=None):
    variable = dataset.createVariable(
        name, dtype, dimensions, zlib=True, complevel=4, shuffle=True)
    variable[:] = values
    if units is not None:
        variable.units = units
    if long_name is not None:
        variable.long_name = long_name
    return variable


def write_map(path, grid, statistics, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')

    try:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            dataset.processing_complete = 0
            dataset.schema_version = map_schema_version
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            for name, value in metadata.items():
                setattr(dataset, name, value)

            dataset.createDimension('grid_y', grid.shape[0])
            dataset.createDimension('grid_x', grid.shape[1])
            dataset.createDimension('mesh_y', grid.lat_mesh.shape[0])
            dataset.createDimension('mesh_x', grid.lat_mesh.shape[1])

            for name, values, units in (
                    ('grid_latitude', grid.lat, 'degrees_north'),
                    ('grid_longitude', grid.lon, 'degrees_east'),
                    ('grid_xi', grid.xi, 'radians'),
                    ('grid_eta', grid.eta, 'radians')):
                create_variable(
                    dataset, name, 'f8', ('grid_y', 'grid_x'), values,
                    units=units)

            for name, values, units in (
                    ('mesh_latitude', grid.lat_mesh, 'degrees_north'),
                    ('mesh_longitude', grid.lon_mesh, 'degrees_east'),
                    ('mesh_xi', grid.xi_mesh, 'radians'),
                    ('mesh_eta', grid.eta_mesh, 'radians')):
                create_variable(
                    dataset, name, 'f8', ('mesh_y', 'mesh_x'), values,
                    units=units)

            create_variable(
                dataset, 'cap_mask', 'i1', ('grid_y', 'grid_x'),
                statistics['cap_mask'].astype(np.int8),
                long_name='cell centre is north of minimum latitude')
            create_variable(
                dataset, 'plot_coverage_mask', 'i1', ('grid_y', 'grid_x'),
                statistics['plot_coverage_mask'].astype(np.int8),
                long_name='cell satisfies minimum checkpoint coverage for plotting')
            create_variable(
                dataset, 'sample_count', 'i8', ('grid_y', 'grid_x'),
                statistics['count'], long_name='number of scalar samples')
            create_variable(
                dataset, 'scalar_sum', 'f8', ('grid_y', 'grid_x'),
                statistics['sum'], units='T',
                long_name='sum of scalar perturbation samples')
            create_variable(
                dataset, 'scalar_sum_of_squares', 'f8',
                ('grid_y', 'grid_x'), statistics['sum_of_squares'], units='T2',
                long_name='sum of squared scalar perturbation samples')
            create_variable(
                dataset, 'scalar_mean', 'f8', ('grid_y', 'grid_x'),
                statistics['mean'], units='T',
                long_name='mean scalar perturbation')
            create_variable(
                dataset, 'scalar_median', 'f8', ('grid_y', 'grid_x'),
                statistics['median'], units='T',
                long_name='median scalar perturbation')
            create_variable(
                dataset, 'scalar_median_gaussian_sigma1', 'f8',
                ('grid_y', 'grid_x'), statistics['median_gaussian_sigma1'],
                units='T',
                long_name='NaN-aware Gaussian-filtered median scalar perturbation')
            create_variable(
                dataset, 'scalar_standard_deviation', 'f8',
                ('grid_y', 'grid_x'), statistics['standard_deviation'], units='T',
                long_name='sample standard deviation of scalar perturbation')
            create_variable(
                dataset, 'scalar_standard_error', 'f8',
                ('grid_y', 'grid_x'), statistics['standard_error'], units='T',
                long_name='standard error of mean scalar perturbation')

            # Footprint-aware products first collapse all correlated samples
            # in one checkpoint to one estimate per cell. Their count and
            # standard error therefore refer to checkpoints, not samples.
            create_variable(
                dataset, 'footprint_checkpoint_count', 'i8',
                ('grid_y', 'grid_x'), statistics['footprint_count'],
                long_name='number of checkpoint estimates after footprint aggregation')
            create_variable(
                dataset, 'footprint_contribution_count', 'i8',
                ('grid_y', 'grid_x'),
                statistics['footprint_contribution_count'],
                long_name='number of sample-footprint overlaps')
            create_variable(
                dataset, 'footprint_coverage_weight', 'f8',
                ('grid_y', 'grid_x'), statistics['footprint_coverage_weight'],
                units='1',
                long_name='sum of fractional footprint overlap areas')
            create_variable(
                dataset, 'scalar_footprint_mean', 'f8',
                ('grid_y', 'grid_x'), statistics['footprint_mean'], units='T',
                long_name='mean of area-weighted checkpoint estimates')
            create_variable(
                dataset, 'scalar_footprint_median', 'f8',
                ('grid_y', 'grid_x'), statistics['footprint_median'], units='T',
                long_name='median of area-weighted checkpoint estimates')
            create_variable(
                dataset, 'scalar_footprint_equal_mem_median', 'f8',
                ('grid_y', 'grid_x'),
                statistics['footprint_equal_mem_median'], units='T',
                long_name=(
                    'median after equal weighting of represented distinct '
                    'satellite/MEM streams'))
            create_variable(
                dataset, 'footprint_represented_mem_count', 'i2',
                ('grid_y', 'grid_x'),
                statistics['footprint_represented_mem_count'],
                long_name=(
                    'number of distinct satellite/MEM streams represented'))
            create_variable(
                dataset, 'footprint_median_mlt', 'f8',
                ('grid_y', 'grid_x'), statistics['footprint_median_mlt'],
                units='hours',
                long_name=(
                    'circular median magnetic local time of checkpoint '
                    'footprint estimates'))
            create_variable(
                dataset, 'scalar_footprint_standard_deviation', 'f8',
                ('grid_y', 'grid_x'),
                statistics['footprint_standard_deviation'], units='T',
                long_name='checkpoint standard deviation after footprint aggregation')
            create_variable(
                dataset, 'scalar_footprint_standard_error', 'f8',
                ('grid_y', 'grid_x'),
                statistics['footprint_standard_error'], units='T',
                long_name='checkpoint standard error after footprint aggregation')

            dataset.processing_complete = 1

        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def plot_median(path, grid, median, minimum_latitude=50.0,
                title='Median scalar perturbation', color_limit=None):
    """Plot the binned median with polplot.Polarplot.plotimg()."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    median_nt = np.asarray(median)*1e9
    finite = np.isfinite(median_nt)
    if not np.any(finite):
        raise ValueError('Cannot plot a median map without populated cells')

    if color_limit is None:
        color_limit = np.nanpercentile(np.abs(median_nt[finite]), 98)
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0

    fig, ax = plt.subplots(figsize=(9, 9))
    polar = Polarplot(
        ax, minlat=minimum_latitude, sector='all',
        lt_label='geographic longitude [h]', lat_label='latitude')
    image = polar.plotimg(
        grid.lat, (grid.lon/15)%24, median_nt,
        crange=(-color_limit, color_limit), cmap='RdBu_r')
    polar.coastlines(
        north=True, resolution='50m', color='black', linewidth=0.6, zorder=5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label('median scalar perturbation [nT]')
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_coverage(path, grid, count, minimum_latitude=50.0,
                  title='Footprint/checkpoint coverage'):
    """Plot the number of checkpoint-level estimates contributing per cell."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = np.asarray(count)
    populated = count > 0
    if not np.any(populated):
        raise ValueError('Cannot plot coverage without populated cells')

    display_count = np.full(count.shape, np.nan)
    display_count[populated] = count[populated]
    maximum = np.max(count[populated])

    fig, ax = plt.subplots(figsize=(9, 9))
    polar = Polarplot(
        ax, minlat=minimum_latitude, sector='all',
        lt_label='geographic longitude [h]', lat_label='latitude')
    image = polar.plotimg(
        grid.lat, (grid.lon/15)%24, display_count,
        crange=(0, maximum), cmap='viridis')
    polar.coastlines(
        north=True, resolution='50m', color='black', linewidth=0.6, zorder=5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label('checkpoint estimates per cell')
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mlt(path, grid, median_mlt, minimum_latitude=50.0,
             title='Median magnetic local time'):
    """Plot circular median MLT with a cyclic 0--24 hour colour scale."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    median_mlt = np.asarray(median_mlt, dtype=float) % 24
    if not np.any(np.isfinite(median_mlt)):
        raise ValueError('Cannot plot MLT without populated cells')

    fig, ax = plt.subplots(figsize=(9, 9))
    polar = Polarplot(
        ax, minlat=minimum_latitude, sector='all',
        lt_label='geographic longitude [h]', lat_label='latitude')
    image = polar.plotimg(
        grid.lat, (grid.lon/15) % 24, median_mlt,
        crange=(0, 24), cmap='twilight_shifted')
    polar.coastlines(
        north=True, resolution='50m', color='black', linewidth=0.6, zorder=5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.04)
    colorbar.set_label('median magnetic local time [h]')
    colorbar.set_ticks(np.arange(0, 25, 3))
    ax.set_title(title)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


#%% Complete aggregation

def build_scalar_map(checkpoint_directory=input_base, output_path=output_file,
                     variable_name=scalar_variable,
                     minimum_latitude=minimum_latitude,
                     resolution_m=resolution_m,
                     footprint_size_m=footprint_size_m,
                     minimum_coverage_count=minimum_coverage_count,
                     all_data=False,
                     radius_m=earth_radius_m,
                     sml_cache_path=sml_cache_file,
                     supermag_userid=supermag_userid,
                     sml_limit_nt=sml_limit_nt,
                     derivative_limit_nt_per_minute=sml_derivative_limit_nt_per_minute,
                     refresh_sml=False,
                     plot_path=None,
                     footprint_plot_path=None,
                     equal_mem_plot_path=None,
                     mlt_plot_path=None,
                     coverage_plot_path=None,
                     show_progress=True):
    """Aggregate all complete orbit checkpoints into one scalar map."""

    if minimum_coverage_count < 0:
        raise ValueError('minimum_coverage_count must be non-negative')
    checkpoint_directory = Path(checkpoint_directory)
    output_path = Path(output_path)
    excluded_paths = {output_path.resolve()}
    if sml_cache_path is not None:
        sml_cache_path = Path(sml_cache_path)
        excluded_paths.add(sml_cache_path.resolve())
    paths = [
        path for path in discover_checkpoints(checkpoint_directory)
        if path.resolve() not in excluded_paths]

    start_ns, stop_ns, complete_time_files = checkpoint_time_range(
        paths, show_progress=show_progress)
    if all_data:
        sml_time = sml = sml_derivative = None
    else:
        if sml_cache_path is None:
            raise ValueError(
                'sml_cache_path is required for quiet-time filtering')
        sml_time, sml, sml_derivative = load_or_download_sml(
            sml_cache_path, start_ns, stop_ns,
            userid=supermag_userid, refresh=refresh_sml)

    grid = create_global_grid(
        minimum_latitude=minimum_latitude,
        resolution_m=resolution_m,
        radius_m=radius_m)
    statistics = initialize_statistics(grid)
    footprint_statistics = initialize_footprint_statistics(grid)

    processed_files = 0
    incomplete_files = 0
    input_samples = 0
    sml_matched_samples = 0
    selected_samples = 0
    binned_samples = 0
    satellite_file_counts = {}
    checkpoint_schemas = set()
    binned_id_chunks = []
    binned_value_chunks = []
    footprint_id_chunks = []
    footprint_value_chunks = []
    equal_mem_id_chunks = []
    equal_mem_value_chunks = []
    equal_mem_stream_chunks = []
    mlt_id_chunks = []
    mlt_value_chunks = []

    loop = tqdm(paths, desc='Binning scalar checkpoints', disable=not show_progress)
    for path in loop:
        loaded = read_checkpoint_samples(path, variable_name)
        if loaded is None:
            incomplete_files += 1
            continue

        time, latitude, longitude, values, mlt, mem_number, file_metadata = loaded
        if all_data:
            selected = np.ones(time.shape, dtype=bool)
            matched = np.zeros(time.shape, dtype=bool)
        else:
            selected, matched = quiet_sml_mask(
                time, sml_time, sml, sml_derivative,
                sml_limit_nt=sml_limit_nt,
                derivative_limit_nt_per_minute=derivative_limit_nt_per_minute,
                tolerance_seconds=sml_match_tolerance_seconds)
        finite_values = np.isfinite(values)
        input_samples += np.count_nonzero(finite_values)
        sml_matched_samples += np.count_nonzero(matched & finite_values)
        selected_samples += np.count_nonzero(selected & finite_values)

        _, used_count, flat_ids, binned_values = bin_scalar_samples(
            grid, statistics,
            latitude[selected], longitude[selected], values[selected],
            minimum_latitude=minimum_latitude, return_binned=True)
        binned_samples += used_count
        if used_count:
            binned_id_chunks.append(flat_ids)
            binned_value_chunks.append(binned_values)

        unwrapped_mlt = np.full(mlt.shape, np.nan)
        for current_mem in np.unique(mem_number):
            current = mem_number == current_mem
            unwrapped_mlt[current] = unwrap_periodic(mlt[current])

        mem_results = []
        mlt_mem_results = []
        for current_mem in np.unique(mem_number[selected]):
            selected_mem = selected & (mem_number == current_mem)
            result = rasterize_checkpoint_footprints(
                grid, latitude[selected_mem], longitude[selected_mem],
                values[selected_mem],
                footprint_size_m=footprint_size_m,
                minimum_latitude=minimum_latitude)
            mem_results.append(result)
            mem_ids, mem_values = result[:2]
            if mem_ids.size:
                stream_id = satellite_mem_id(
                    file_metadata['satellite'], current_mem)
                equal_mem_id_chunks.append(mem_ids)
                equal_mem_value_chunks.append(mem_values)
                equal_mem_stream_chunks.append(np.full(
                    mem_ids.size, stream_id, dtype=np.int8))
            mlt_mem_results.append(rasterize_checkpoint_footprints(
                grid, latitude[selected_mem], longitude[selected_mem],
                unwrapped_mlt[selected_mem],
                footprint_size_m=footprint_size_m,
                minimum_latitude=minimum_latitude))

        footprint_ids, footprint_values, footprint_weights, contributions = (
            combine_mem_footprints(mem_results))
        if footprint_ids.size:
            add_rasterized_footprints(
                footprint_statistics, footprint_ids, footprint_values,
                footprint_weights, contributions)
        if footprint_ids.size:
            footprint_id_chunks.append(footprint_ids)
            footprint_value_chunks.append(footprint_values)
        mlt_ids, checkpoint_mlt = combine_circular_footprints(mlt_mem_results)
        if mlt_ids.size:
            mlt_id_chunks.append(mlt_ids)
            mlt_value_chunks.append(checkpoint_mlt)
        processed_files += 1

        satellite = file_metadata['satellite']
        satellite_file_counts[satellite] = (
            satellite_file_counts.get(satellite, 0)+1)
        checkpoint_schemas.add(file_metadata['checkpoint_schema'])

        loop.set_postfix(files=processed_files, samples=binned_samples)

    if processed_files == 0:
        raise ValueError(f'No complete checkpoints found in {checkpoint_directory}')
    if binned_samples == 0:
        raise ValueError('No finite scalar samples fall within the map region')

    flat_ids = np.concatenate(binned_id_chunks)
    binned_values = np.concatenate(binned_value_chunks)
    median = calculate_median(flat_ids, binned_values, grid.shape)
    statistics = finalize_statistics(
        statistics, grid, median, minimum_latitude=minimum_latitude)
    statistics['median_gaussian_sigma1'] = gaussian_filter_nan(
        statistics['median'], sigma=1.0, valid_mask=statistics['cap_mask'])

    if not footprint_id_chunks:
        raise ValueError('No footprint overlaps fall within the map region')
    footprint_ids = np.concatenate(footprint_id_chunks)
    footprint_values = np.concatenate(footprint_value_chunks)
    footprint_median = calculate_median(
        footprint_ids, footprint_values, grid.shape)
    footprint_statistics = finalize_statistics(
        footprint_statistics, grid, footprint_median,
        minimum_latitude=minimum_latitude)
    statistics.update({
        'footprint_count': footprint_statistics['count'],
        'footprint_contribution_count': footprint_statistics['contribution_count'],
        'footprint_coverage_weight': footprint_statistics['coverage_weight'],
        'footprint_mean': footprint_statistics['mean'],
        'footprint_median': footprint_statistics['median'],
        'footprint_standard_deviation': footprint_statistics['standard_deviation'],
        'footprint_standard_error': footprint_statistics['standard_error']})
    equal_mem_ids = np.concatenate(equal_mem_id_chunks)
    equal_mem_values = np.concatenate(equal_mem_value_chunks)
    equal_mem_streams = np.concatenate(equal_mem_stream_chunks)
    equal_mem_median, represented_mem_count = calculate_equal_mem_median(
        equal_mem_ids, equal_mem_values, equal_mem_streams, grid.shape)
    if not mlt_id_chunks:
        raise ValueError(
            'No magnetic-local-time footprint overlaps fall within the map region')
    mlt_ids = np.concatenate(mlt_id_chunks)
    checkpoint_mlt = np.concatenate(mlt_value_chunks)
    median_mlt = calculate_circular_median(
        mlt_ids, checkpoint_mlt, grid.shape)
    statistics.update({
        'footprint_equal_mem_median': equal_mem_median,
        'footprint_represented_mem_count': represented_mem_count,
        'footprint_median_mlt': median_mlt})
    statistics['plot_coverage_mask'] = (
        statistics['cap_mask'] &
        (statistics['footprint_count'] >= minimum_coverage_count))
    metadata = {
        'source_directory': str(checkpoint_directory),
        'source_variable': variable_name,
        'minimum_latitude_degrees': float(minimum_latitude),
        'nominal_resolution_m': float(resolution_m),
        'footprint_size_m': float(footprint_size_m),
        'plot_minimum_coverage_count': int(minimum_coverage_count),
        'footprint_alignment': 'cubed-sphere xi/eta axes',
        'footprint_within_checkpoint_aggregation': 'overlap-area-weighted mean',
        'footprint_across_checkpoint_aggregation': 'equal-checkpoint median',
        'equal_mem_weighting': 1,
        'sphere_radius_m': float(radius_m),
        'sample_selection': 'all data' if all_data else 'quiet SML',
        'sml_filter_applied': int(not all_data),
        'supermag_sml_cache': '' if all_data else str(sml_cache_path),
        'sml_absolute_limit_nt': float(sml_limit_nt),
        'sml_absolute_derivative_limit_nt_per_minute': float(
            derivative_limit_nt_per_minute),
        'sml_match_tolerance_seconds': float(sml_match_tolerance_seconds),
        'median_gaussian_sigma_grid_cells': 1.0,
        'checkpoint_time_start_ns': int(start_ns),
        'checkpoint_time_stop_ns': int(stop_ns),
        'discovered_file_count': len(paths),
        'complete_time_file_count': complete_time_files,
        'processed_file_count': processed_files,
        'incomplete_file_count': incomplete_files,
        'finite_input_sample_count': input_samples,
        'sml_matched_sample_count': (
            -1 if all_data else sml_matched_samples),
        'selected_sample_count': selected_samples,
        'quiet_sample_count': -1 if all_data else selected_samples,
        'binned_sample_count': binned_samples,
        'footprint_checkpoint_cell_count': int(footprint_ids.size),
        'footprint_overlap_contribution_count': int(np.sum(
            footprint_statistics['contribution_count'])),
        'mlt_checkpoint_cell_count': int(mlt_ids.size),
        'satellite_file_counts': json.dumps(
            satellite_file_counts, sort_keys=True),
        'checkpoint_schema_versions': json.dumps(sorted(checkpoint_schemas))}
    metadata.update({
        'equal_mem_definition': 'distinct satellite/MEM stream',
        'equal_mem_aggregation': (
            'weighted checkpoint median with inverse checkpoint count '
            'per represented satellite/MEM stream'),
        'equal_mem_checkpoint_cell_count': int(equal_mem_ids.size)})
    write_map(output_path, grid, statistics, metadata)
    if any(path is not None for path in (
            plot_path, footprint_plot_path, equal_mem_plot_path)):
        plot_median_values = apply_coverage_threshold(
            statistics['median'], statistics['footprint_count'],
            minimum_count=minimum_coverage_count)
        plot_footprint_values = apply_coverage_threshold(
            statistics['footprint_median'], statistics['footprint_count'],
            minimum_count=minimum_coverage_count)
        if equal_mem_plot_path is not None:
            plot_equal_mem_values = apply_coverage_threshold(
                statistics['footprint_equal_mem_median'],
                statistics['footprint_count'],
                minimum_count=minimum_coverage_count)
        median_nt = plot_median_values*1e9
        color_limit = np.nanpercentile(
            np.abs(median_nt[np.isfinite(median_nt)]), 98)
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0
    if plot_path is not None:
        plot_median(
            plot_path, grid, plot_median_values,
            minimum_latitude=minimum_latitude,
            title=(
                f'Median {variable_name}, '
                f'coverage $\\geq$ {minimum_coverage_count}'),
            color_limit=color_limit)
    if footprint_plot_path is not None:
        plot_median(
            footprint_plot_path, grid,
            plot_footprint_values,
            minimum_latitude=minimum_latitude,
            title=(
                f'Median {variable_name}, '
                f'{footprint_size_m/1000:g} km footprint/checkpoint, '
                f'coverage $\\geq$ {minimum_coverage_count}'),
            color_limit=color_limit)
    if equal_mem_plot_path is not None:
        plot_median(
            equal_mem_plot_path, grid,
            plot_equal_mem_values,
            minimum_latitude=minimum_latitude,
            title=(
                f'Median {variable_name}, equal satellite/MEM weighting, '
                f'coverage $\\geq$ {minimum_coverage_count}'),
            color_limit=color_limit)
    if mlt_plot_path is not None:
        plot_mlt_values = apply_coverage_threshold(
            statistics['footprint_median_mlt'],
            statistics['footprint_count'],
            minimum_count=minimum_coverage_count)
        plot_mlt(
            mlt_plot_path, grid, plot_mlt_values,
            minimum_latitude=minimum_latitude,
            title=(
                f'Median MLT, {footprint_size_m/1000:g} km footprint, '
                f'coverage $\\geq$ {minimum_coverage_count}'))
    if coverage_plot_path is not None:
        plot_coverage(
            coverage_plot_path, grid,
            statistics['footprint_count'],
            minimum_latitude=minimum_latitude)
    return grid, statistics, metadata


#%% Command-line entry point

def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description='Bin EZIE lithosphere checkpoints on a northern CS grid')
    parser.add_argument('--input', type=Path, default=input_base,
                        help='root directory containing orbit checkpoints')
    parser.add_argument('--output', type=Path, default=output_file,
                        help='output NetCDF map')
    parser.add_argument('--plot', type=Path,
                        help='median PNG; defaults beside the output NetCDF')
    parser.add_argument('--footprint-plot', type=Path,
                        help='footprint/checkpoint median PNG')
    parser.add_argument('--coverage-plot', type=Path,
                        help='footprint/checkpoint cell-count PNG')
    parser.add_argument('--variable', default=scalar_variable,
                        help='checkpoint scalar variable to aggregate')
    parser.add_argument('--minimum-latitude', type=float,
                        default=minimum_latitude)
    parser.add_argument('--resolution-km', type=float,
                        default=resolution_m/1000)
    parser.add_argument('--footprint-km', type=float,
                        default=footprint_size_m/1000,
                        help='square FOV width used for area-overlap mapping')
    parser.add_argument('--minimum-coverage-count', type=nonnegative_integer,
                        default=minimum_coverage_count,
                        help='minimum checkpoint estimates required to plot a cell')
    parser.add_argument('--supermag-userid', default=supermag_userid)
    parser.add_argument('--sml-cache', type=Path, default=sml_cache_file)
    parser.add_argument('--sml-limit-nt', type=float, default=sml_limit_nt,
                        help='maximum absolute SML magnitude')
    parser.add_argument('--sml-derivative-limit', type=float,
                        default=sml_derivative_limit_nt_per_minute,
                        help='maximum absolute dSML/dt in nT/min')
    parser.add_argument('--refresh-sml', action='store_true',
                        help='redownload SML even if the cache covers the data')
    parser.add_argument(
        '--all-data', action='store_true',
        help='bypass SML filtering and include every finite observation')
    return parser.parse_args()


def main():
    args = parse_args()
    selection_suffix = '_all' if args.all_data else ''
    output_path = args.output
    if args.all_data and output_path == output_file:
        output_path = output_path.with_name(output_path.stem+'_all.nc')
    plot_path = args.plot
    if plot_path is None:
        plot_path = args.output.with_name(
            args.output.stem+f'_median{selection_suffix}.png')
    footprint_plot_path = args.footprint_plot
    if footprint_plot_path is None:
        if args.plot is None:
            footprint_plot_path = args.output.with_name(
                args.output.stem+
                f'_median_footprint_{args.footprint_km:g}km'
                f'{selection_suffix}.png')
        else:
            footprint_plot_path = plot_path.with_name(
                plot_path.stem+
                f'_footprint_{args.footprint_km:g}km'+plot_path.suffix)
    if args.plot is None and args.footprint_plot is None:
        equal_mem_plot_path = args.output.with_name(
            args.output.stem+
            f'_median_footprint_{args.footprint_km:g}km_equal_mem'
            f'{selection_suffix}.png')
    else:
        equal_mem_plot_path = footprint_plot_path.with_name(
            footprint_plot_path.stem+'_equal_mem'+footprint_plot_path.suffix)
    mlt_plot_path = args.output.with_name(
        args.output.stem+f'_median_mlt{selection_suffix}.png')
    coverage_plot_path = args.coverage_plot
    if coverage_plot_path is None:
        if args.plot is None:
            coverage_plot_path = args.output.with_name(
                args.output.stem+f'_median_coverage{selection_suffix}.png')
        else:
            coverage_plot_path = plot_path.with_name(
                plot_path.stem+'_coverage'+plot_path.suffix)
    _, _, metadata = build_scalar_map(
        checkpoint_directory=args.input,
        output_path=output_path,
        variable_name=args.variable,
        minimum_latitude=args.minimum_latitude,
        resolution_m=args.resolution_km*1000,
        footprint_size_m=args.footprint_km*1000,
        minimum_coverage_count=args.minimum_coverage_count,
        all_data=args.all_data,
        sml_cache_path=args.sml_cache,
        supermag_userid=args.supermag_userid,
        sml_limit_nt=args.sml_limit_nt,
        derivative_limit_nt_per_minute=args.sml_derivative_limit,
        refresh_sml=args.refresh_sml,
        plot_path=plot_path,
        footprint_plot_path=footprint_plot_path,
        equal_mem_plot_path=equal_mem_plot_path,
        mlt_plot_path=mlt_plot_path,
        coverage_plot_path=coverage_plot_path)
    message = (
        f"Wrote {output_path} from {metadata['processed_file_count']} files "
        f"and {metadata['binned_sample_count']} samples\n"
        f"Wrote {plot_path}\n"
        f"Wrote {footprint_plot_path}\n"
        f"Wrote {equal_mem_plot_path}\n"
        f"Wrote {mlt_plot_path}\n"
        f"Wrote {coverage_plot_path}")
    print(message)


if __name__ == '__main__':
    main()
