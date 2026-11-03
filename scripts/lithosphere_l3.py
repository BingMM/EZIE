#%% Import

import argparse
import gc
import json
import os
import subprocess
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import ezie
import numpy as np
from netCDF4 import Dataset
from secsy import CSgrid, CSprojection
from tqdm import tqdm

from ezie.utils import generate_ezie_data


#%% Configuration

def output_base_for_background(input_base, field_background):
    """Keep the legacy IGRF path and isolate every alternative background."""

    directory_name = (
        'EZIE_lithosphere_l3' if field_background == 'igrf'
        else f'EZIE_lithosphere_l3_{field_background}')
    return Path(input_base).parent / directory_name


base = Path('/media/bing/LaCie/EZIE_L2')

schema_version = '1.1'
map_params = None
l1 = 0.75
l2 = 2.5
#background = 'igrf'
background = 'chaos'
scalar_source = 'retrieved'
fit_mode = 'gradient'
workers = 3
blas_threads_per_worker = None

# Preserve the existing IGRF archive path, while every alternative background
# receives an independent output tree.
output_base = output_base_for_background(base, background)

blas_thread_variables = (
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'BLIS_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS',
    'NUMEXPR_NUM_THREADS')

required_checkpoint_variables = {
    'time', 'latitude', 'longitude', 'radius',
    'observed_dB', 'observed_variance',
    'predicted_dBe', 'predicted_dBn', 'predicted_dBu',
    'predicted_dB_linear', 'predicted_dB_exact',
    'track_offset', 'secs_amplitude',
    'grid_latitude', 'grid_longitude', 'grid_xi', 'grid_eta',
    'mesh_xi', 'mesh_eta',
    'projection_position', 'projection_orientation'}


#%% Provenance

def get_code_provenance():
    repo = Path(ezie.__file__).resolve().parents[1]

    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo,
            check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ['git', 'status', '--porcelain'], cwd=repo,
            check=True, capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit = 'unknown'
        dirty = True

    packages = {}
    for package in ('ezie', 'numpy', 'scipy', 'secsy', 'nandir',
                    'apexpy', 'netCDF4'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = 'unknown'

    return commit, dirty, packages


#%% Checkpoint helpers

def satellite_from_filename(file):
    marker = '_sv'
    start = file.stem.find(marker)
    if start == -1 or start+len(marker) >= len(file.stem):
        raise ValueError(f'Cannot determine satellite from {file.name}')
    return file.stem[start+len(marker)].lower()


def checkpoint_path(file):
    relative = file.relative_to(base).with_suffix('.nc')
    return output_base / relative


def checkpoint_is_complete(path, source_file, expected_background=None):
    if expected_background is None:
        expected_background = background
    if not path.is_file():
        return False

    try:
        with Dataset(path) as dataset:
            return (
                getattr(dataset, 'processing_complete', 0) == 1 and
                getattr(dataset, 'schema_version', '') == schema_version and
                getattr(dataset, 'source_name', '') == source_file.name and
                getattr(dataset, 'background', '') == expected_background and
                required_checkpoint_variables.issubset(dataset.variables))
    except OSError:
        return False


def checkpoint_value(dataset, name):
    """Restore a scalar attribute with the dtype used by the original grid."""

    value = getattr(dataset, name)
    dtype = np.dtype(getattr(dataset, f'{name}_dtype', 'float64'))
    return np.asarray(value, dtype=dtype)[()]


def reconstruct_grid(path):
    """Recreate the cubed-sphere grid stored in a result checkpoint."""

    with Dataset(path) as dataset:
        position_dtype = np.dtype(
            getattr(dataset, 'projection_position_dtype', 'float64'))
        orientation_dtype = np.dtype(
            getattr(dataset, 'projection_orientation_dtype', 'float64'))
        projection = CSprojection(
            np.asarray(dataset.variables['projection_position'][:],
                       dtype=position_dtype),
            np.asarray(dataset.variables['projection_orientation'][:],
                       dtype=orientation_dtype))
        edges = None
        if 'mesh_xi' in dataset.variables and 'mesh_eta' in dataset.variables:
            edges = (
                np.asarray(dataset.variables['mesh_xi'][0, :]),
                np.asarray(dataset.variables['mesh_eta'][:, 0]))
        return CSgrid(
            projection,
            checkpoint_value(dataset, 'grid_L_m'),
            checkpoint_value(dataset, 'grid_W_m'),
            checkpoint_value(dataset, 'grid_Lres_m'),
            checkpoint_value(dataset, 'grid_Wres_m'),
            edges=edges,
            wshift=checkpoint_value(dataset, 'grid_wshift'),
            R=checkpoint_value(dataset, 'RI_m'))


def datetime_to_ns(times):
    epoch = datetime(1970, 1, 1)
    values = []
    for time in times:
        if time.tzinfo is not None:
            time = time.astimezone(timezone.utc).replace(tzinfo=None)
        values.append(round((time-epoch).total_seconds()*1e9))
    return np.asarray(values, dtype=np.int64)


def stack_mem_values(mems, attribute, sample_count, fill_value=np.nan):
    values = np.full((len(mems), sample_count), fill_value, dtype=float)
    for i, mem in enumerate(mems):
        value = np.asarray(getattr(mem, attribute))
        values[i, :value.size] = value
    return values


def split_prediction(values, mems, sample_count):
    result = np.full((len(mems), sample_count), np.nan, dtype=float)
    start = 0
    for i, mem in enumerate(mems):
        stop = start + mem.nd
        result[i, :mem.nd] = values[start:stop]
        start = stop
    return result


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


def write_checkpoint(path, source_file, satellite, model, predictions,
                     timings, provenance):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')

    mems = model.data.mems
    nmem = len(mems)
    sample_count = max(mem.nd for mem in mems)
    grid = model.grid
    grid_y, grid_x = grid.shape
    mesh_y, mesh_x = grid.lat_mesh.shape

    time_values = np.full((nmem, sample_count), np.iinfo(np.int64).min,
                          dtype=np.int64)
    source_times = datetime_to_ns(model.data.dates)
    for i, mem in enumerate(mems):
        if source_times.size < mem.nd:
            raise ValueError('There are fewer timestamps than MEM observations')
        time_values[i, :mem.nd] = source_times[:mem.nd]

    commit, dirty, packages = provenance

    try:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            # File identity and processing configuration
            dataset.schema_version = schema_version
            dataset.processing_complete = 0
            dataset.source_name = source_file.name
            dataset.source_path = str(source_file)
            dataset.satellite = satellite
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            dataset.git_commit = commit
            dataset.git_dirty = int(dirty)
            dataset.package_versions = json.dumps(packages, sort_keys=True)
            dataset.fit_mode = model.fit_mode
            dataset.background = model.data.background or 'unknown'
            dataset.scalar_source = model.data.scalar_source or 'unknown'
            dataset.B0_model = model.B0_model
            dataset.l1 = model.l1
            dataset.l2 = model.l2
            dataset.center_mem = model.cm
            dataset.singularity_limit_m = model.slim
            # Store values as float64 plus their original numpy dtypes. The
            # dtype affects CSgrid's coordinate construction at metre scale.
            for name, value in (
                    ('grid_L_m', grid.L),
                    ('grid_W_m', grid.W),
                    ('grid_Lres_m', grid.Lres),
                    ('grid_Wres_m', grid.Wres),
                    ('grid_wshift', grid.wshift),
                    ('RI_m', model.RI)):
                setattr(dataset, name, float(value))
                setattr(dataset, f'{name}_dtype', np.asarray(value).dtype.str)

            for name, value in model.map_params.items():
                setattr(dataset, f'map_{name}', value)
            for name, value in timings.items():
                setattr(dataset, f'timing_{name}_s', value)

            # Dimensions
            dataset.createDimension('mem', nmem)
            dataset.createDimension('sample', sample_count)
            dataset.createDimension('grid_y', grid_y)
            dataset.createDimension('grid_x', grid_x)
            dataset.createDimension('mesh_y', mesh_y)
            dataset.createDimension('mesh_x', mesh_x)
            dataset.createDimension('horizontal_vector', 2)

            # Observation identity and input data
            create_variable(dataset, 'mem_number', 'i2', ('mem',),
                            np.arange(1, nmem+1))
            create_variable(dataset, 'sample_count', 'i4', ('mem',),
                            np.array([mem.nd for mem in mems]))
            time_variable = create_variable(
                dataset, 'time', 'i8', ('mem', 'sample'), time_values,
                units='nanoseconds since 1970-01-01 00:00:00 UTC',
                long_name='observation time')
            time_variable.missing_value = np.iinfo(np.int64).min

            observation_variables = {
                'latitude': ('lat', 'degrees_north'),
                'longitude': ('lon', 'degrees_east'),
                'radius': ('r', 'm'),
                'observed_dB': ('B', 'T'),
                'observed_variance': ('cov_mag', 'T2'),
                'background_Be': ('B0e', 'T'),
                'background_Bn': ('B0n', 'T'),
                'background_Bu': ('B0u', 'T'),
                'background_B': ('B0', 'T')}

            for name, (attribute, units) in observation_variables.items():
                values = stack_mem_values(mems, attribute, sample_count)
                create_variable(dataset, name, 'f8', ('mem', 'sample'),
                                values, units=units)

            # Reconstructed field at every observation location
            prediction_variables = {
                'predicted_dBe': ('Be', 'T'),
                'predicted_dBn': ('Bn', 'T'),
                'predicted_dBu': ('Bu', 'T'),
                'predicted_dB_linear': ('dB_linear', 'T'),
                'predicted_dB_exact': ('dB_exact', 'T'),
                'predicted_dB_with_offset': ('dB_with_offset', 'T'),
                'absolute_residual': ('absolute_residual', 'T')}

            for name, (key, units) in prediction_variables.items():
                values = split_prediction(
                    predictions[key], mems, sample_count)
                create_variable(dataset, name, 'f8', ('mem', 'sample'),
                                values, units=units)

            create_variable(dataset, 'track_offset', 'f8', ('mem',),
                            predictions['track_offsets'], units='T')

            # Fit-space diagnostics
            dataset.fit_observation_count = model.d.size
            dataset.weighted_residual_norm = predictions['weighted_residual_norm']
            dataset.normalized_weighted_residual = predictions['normalized_weighted_residual']

            # SECS coefficients and the complete source-grid definition
            create_variable(dataset, 'secs_amplitude', 'f8',
                            ('grid_y', 'grid_x'),
                            model.m.reshape(grid.shape),
                            long_name='SECS model coefficient')
            for name, values, units in (
                    ('grid_latitude', grid.lat, 'degrees_north'),
                    ('grid_longitude', grid.lon, 'degrees_east'),
                    ('grid_xi', grid.xi, 'radians'),
                    ('grid_eta', grid.eta, 'radians')):
                create_variable(dataset, name, 'f8', ('grid_y', 'grid_x'),
                                values, units=units)

            for name, values, units in (
                    ('mesh_latitude', grid.lat_mesh, 'degrees_north'),
                    ('mesh_longitude', grid.lon_mesh, 'degrees_east'),
                    ('mesh_xi', grid.xi_mesh, 'radians'),
                    ('mesh_eta', grid.eta_mesh, 'radians')):
                create_variable(dataset, name, 'f8', ('mesh_y', 'mesh_x'),
                                values, units=units)

            create_variable(dataset, 'projection_position', 'f8',
                            ('horizontal_vector',), grid.projection.position,
                            long_name='projection longitude and latitude')
            create_variable(dataset, 'projection_orientation', 'f8',
                            ('horizontal_vector',), grid.projection.orientation,
                            long_name='projection orientation vector')
            dataset.projection_position_dtype = np.asarray(
                grid.projection.position).dtype.str
            dataset.projection_orientation_dtype = np.asarray(
                grid.projection.orientation).dtype.str

            dataset.processing_complete = 1

        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def failure_record(source_file, error):
    return {
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source_file),
        'error_type': type(error).__name__,
        'error': str(error),
        'traceback': traceback.format_exc()}


def append_failure(log_path, record):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as stream:
        stream.write(json.dumps(record) + '\n')


#%% Process one inversion

def process_file(file, path, provenance):
    start_total = perf_counter()
    satellite = satellite_from_filename(file)

    start = perf_counter()
    data = generate_ezie_data(
        file, comp='F', background=background,
        scalar_source=scalar_source)
    load_seconds = perf_counter()-start

    model = ezie.Model(
        map_params=map_params, l1=l1, l2=l2,
        B0_model=background, fit_mode=fit_mode)
    model.add_data(data)

    start = perf_counter()
    m = model.m
    solve_seconds = perf_counter()-start

    # Use the cached linear scalar operator and evaluate the vector prediction
    # once for all MEM locations.
    start = perf_counter()
    dB_linear = model.G_absolute @ m
    Be, Bn, Bu = model.ev.get_B_predictions(
        lat=model.data.lat, lon=model.data.lon, r=model.data.r, ret=True)
    dB_exact = np.sqrt(
        (Be+model.data.B0e)**2 +
        (Bn+model.data.B0n)**2 +
        (Bu+model.data.B0u)**2) - model.data.B0

    offsets = np.repeat(
        model.track_offsets,
        [mem.nd for mem in model.data.mems])
    dB_with_offset = dB_linear + offsets
    absolute_residual = model.data.B - dB_with_offset

    fit_residual = model.G @ m - model.d
    weighted_residual_norm = model.weighted_residual_norm(fit_residual)
    normalized_weighted_residual = np.sqrt(
        weighted_residual_norm/model.d.size)
    prediction_seconds = perf_counter()-start

    predictions = {
        'Be': Be,
        'Bn': Bn,
        'Bu': Bu,
        'dB_linear': dB_linear,
        'dB_exact': dB_exact,
        'dB_with_offset': dB_with_offset,
        'absolute_residual': absolute_residual,
        'track_offsets': model.track_offsets,
        'weighted_residual_norm': weighted_residual_norm,
        'normalized_weighted_residual': normalized_weighted_residual}

    timings = {
        'load': load_seconds,
        'solve': solve_seconds,
        'prediction': prediction_seconds,
        'total': perf_counter()-start_total}

    write_checkpoint(
        path, file, satellite, model, predictions, timings, provenance)


def attempt_file(file, path, provenance):
    """Process one orbit and return a serializable failure record, if any."""

    try:
        process_file(file, path, provenance)
        return None
    except Exception as error:
        return failure_record(file, error)
    finally:
        # Model and Evaluator reference each other. Their multi-gigabyte
        # arrays otherwise remain alive until Python happens to run cyclic
        # garbage collection, which can be too late for a long batch.
        gc.collect()


#%% Parallel execution helpers

def positive_integer(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('value must be at least 1')
    return value


def worker_blas_threads(worker_count, requested_threads=None):
    if requested_threads is not None:
        return positive_integer(requested_threads)
    return max(1, (os.cpu_count() or 1)//worker_count)


@contextmanager
def blas_thread_environment(thread_count):
    """Set BLAS/OpenMP limits inherited by newly spawned worker processes."""

    previous = {name: os.environ.get(name) for name in blas_thread_variables}
    for name in blas_thread_variables:
        os.environ[name] = str(thread_count)

    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def update_counts(failure, failure_log, counts):
    if failure is None:
        counts['completed'] += 1
    else:
        append_failure(failure_log, failure)
        counts['failed'] += 1


def update_progress(loop, counts):
    loop.set_postfix(
        completed=counts['completed'],
        skipped=counts['skipped'],
        failed=counts['failed'])


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate restartable EZIE lithosphere checkpoints.')
    parser.add_argument(
        '--workers', type=positive_integer, default=workers,
        help='number of orbit processes (default: %(default)s)')
    parser.add_argument(
        '--blas-threads-per-worker', type=positive_integer,
        default=blas_threads_per_worker,
        help='BLAS threads in each worker; default divides CPUs among workers')
    return parser.parse_args()


#%% Run all available orbits

def main(worker_count=workers, requested_blas_threads=blas_threads_per_worker):
    worker_count = positive_integer(worker_count)
    files = sorted(base.rglob('*.nc4'))
    provenance = get_code_provenance()
    failure_log = output_base / 'failures.jsonl'

    pending = []
    counts = {'completed': 0, 'skipped': 0, 'failed': 0}
    for file in files:
        path = checkpoint_path(file)
        if checkpoint_is_complete(path, file):
            counts['skipped'] += 1
        else:
            pending.append((file, path))

    loop = tqdm(total=len(files))
    loop.update(counts['skipped'])
    update_progress(loop, counts)

    if worker_count == 1:
        for file, path in pending:
            failure = attempt_file(file, path, provenance)
            update_counts(failure, failure_log, counts)
            loop.update()
            update_progress(loop, counts)
        loop.close()
        return counts

    active_workers = min(worker_count, len(pending))
    if active_workers == 0:
        loop.close()
        return counts

    threads = worker_blas_threads(active_workers, requested_blas_threads)
    with blas_thread_environment(threads):
        with ProcessPoolExecutor(
                max_workers=active_workers,
                mp_context=get_context('spawn')) as executor:
            futures = {
                executor.submit(attempt_file, file, path, provenance): file
                for file, path in pending}

            for future in as_completed(futures):
                file = futures[future]
                try:
                    failure = future.result()
                except Exception as error:
                    failure = failure_record(file, error)
                update_counts(failure, failure_log, counts)
                loop.update()
                update_progress(loop, counts)

    loop.close()
    return counts


if __name__ == '__main__':
    arguments = parse_arguments()
    main(arguments.workers, arguments.blas_threads_per_worker)
