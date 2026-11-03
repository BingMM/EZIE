#%% Import

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm

if __package__:
    from .lithosphere_global_map import (
        checkpoint_time_range,
        create_variable,
        discover_checkpoints,
        load_or_download_sml,
        sml_cache_file,
        supermag_userid)
    from .lithosphere_sml_histograms import nearest_sml_indices
else:
    from lithosphere_global_map import (
        checkpoint_time_range,
        create_variable,
        discover_checkpoints,
        load_or_download_sml,
        sml_cache_file,
        supermag_userid)
    from lithosphere_sml_histograms import nearest_sml_indices


#%% Configuration

input_base = Path('/media/bing/LaCie/EZIE_lithosphere_l3')
output_base = input_base.parent / 'EZIE_lithosphere_debug_analysis'
output_file = output_base / 'corrected_observations_mlat_mlt.nc'

satellite_ids = ('a', 'b', 'c')
mem_count = 4
activity_labels = ('quiet', 'active')
direction_labels = ('ascending', 'descending')
statistics = ('quantile_10', 'median', 'quantile_90')

minimum_magnetic_latitude = 50.0
maximum_magnetic_latitude = 90.0
magnetic_latitude_step = 1.0
mlt_step_hours = 0.5
minimum_coverage_count = 10

quiet_sml_limit_nt = 80.0
quiet_derivative_limit_nt_per_minute = 10.0
active_sml_maximum_nt = -200.0
sml_match_tolerance_seconds = 60.0
schema_version = '1.0'


#%% Input loading

def local_orbit_direction(satellite_latitude):
    """Classify each sample from the local spacecraft-latitude derivative."""

    latitude = np.asarray(satellite_latitude, dtype=float)
    if latitude.ndim != 1 or latitude.size < 2:
        raise ValueError('satellite latitude must contain at least two samples')

    change = np.empty(latitude.size)
    change[0] = latitude[1]-latitude[0]
    change[-1] = latitude[-1]-latitude[-2]
    if latitude.size > 2:
        change[1:-1] = latitude[2:]-latitude[:-2]

    ascending = np.isfinite(change) & (change > 0)
    descending = np.isfinite(change) & (change < 0)
    return ascending, descending


def read_checkpoint_observations(path):
    """Load corrected scalar observations and their Level-2 magnetic geometry."""

    missing_time = np.iinfo(np.int64).min
    with Dataset(path) as checkpoint:
        if getattr(checkpoint, 'processing_complete', 0) != 1:
            return None

        required = {
            'time', 'sample_count', 'observed_dB', 'track_offset'}
        missing = required.difference(checkpoint.variables)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'{path} is missing required variables: {names}')

        observed = checkpoint.variables['observed_dB']
        offsets = checkpoint.variables['track_offset']
        if getattr(observed, 'units', None) != 'T':
            raise ValueError(f'{path}: observed_dB must have units T')
        if getattr(offsets, 'units', None) != 'T':
            raise ValueError(f'{path}: track_offset must have units T')

        time = np.asarray(
            np.ma.filled(checkpoint.variables['time'][:], missing_time),
            dtype=np.int64)
        sample_count = np.asarray(
            checkpoint.variables['sample_count'][:], dtype=int)
        observed_dB = np.asarray(
            np.ma.filled(observed[:], np.nan), dtype=float)
        track_offset = np.asarray(
            np.ma.filled(offsets[:], np.nan), dtype=float)
        if observed_dB.shape != time.shape:
            raise ValueError(f'{path}: observed_dB and time shapes do not match')
        if observed_dB.shape[0] != mem_count:
            raise ValueError(
                f'{path}: expected {mem_count} MEMs, '
                f'found {observed_dB.shape[0]}')
        if sample_count.shape != (mem_count,):
            raise ValueError(f'{path}: sample_count must have one value per MEM')
        if track_offset.shape != (mem_count,):
            raise ValueError(f'{path}: track_offset must have one value per MEM')

        source_path = Path(str(getattr(checkpoint, 'source_path', '')))
        satellite = str(getattr(checkpoint, 'satellite', '')).lower()
        metadata = {
            'background': str(getattr(checkpoint, 'background', 'unknown')),
            'scalar_source': str(
                getattr(checkpoint, 'scalar_source', 'unknown')),
            'checkpoint_schema': str(
                getattr(checkpoint, 'schema_version', 'unknown'))}

    if satellite not in satellite_ids:
        raise ValueError(f'{path}: unknown satellite {satellite!r}')
    if not source_path.is_file():
        raise FileNotFoundError(
            f'{path}: Level-2 source file does not exist: {source_path}')

    with Dataset(source_path) as source:
        magnetic = source.groups.get('MagneticCoords')
        geolocation = source.groups.get('Geolocation')
        if magnetic is None or geolocation is None:
            raise ValueError(
                f'{source_path}: missing MagneticCoords or Geolocation group')
        if 'sat_lat' not in geolocation.variables:
            raise ValueError(f'{source_path}: missing Geolocation/sat_lat')

        satellite_latitude = np.asarray(
            np.ma.filled(geolocation.variables['sat_lat'][:], np.nan),
            dtype=float)
        magnetic_latitude = []
        mlt = []
        for mem_number in range(1, mem_count+1):
            latitude_name = f'obs_maglat{mem_number}'
            mlt_name = f'obs_magLT{mem_number}'
            if latitude_name not in magnetic.variables:
                raise ValueError(f'{source_path}: missing {latitude_name}')
            if mlt_name not in magnetic.variables:
                raise ValueError(f'{source_path}: missing {mlt_name}')
            magnetic_latitude.append(np.asarray(
                np.ma.filled(magnetic.variables[latitude_name][:], np.nan),
                dtype=float))
            mlt.append(np.asarray(
                np.ma.filled(magnetic.variables[mlt_name][:], np.nan),
                dtype=float))

    corrected = observed_dB-track_offset[:, None]
    return {
        'satellite': satellite,
        'sample_count': sample_count,
        'time': time,
        'corrected_observed_dB': corrected,
        'magnetic_latitude': magnetic_latitude,
        'mlt': mlt,
        'satellite_latitude': satellite_latitude,
        'metadata': metadata}


#%% Magnetic-latitude and MLT binning

def create_bin_edges(
        minimum_latitude=minimum_magnetic_latitude,
        maximum_latitude=maximum_magnetic_latitude,
        latitude_step=magnetic_latitude_step,
        mlt_step=mlt_step_hours):
    if not minimum_latitude < maximum_latitude:
        raise ValueError('minimum latitude must be below maximum latitude')
    if latitude_step <= 0 or mlt_step <= 0:
        raise ValueError('bin steps must be positive')

    latitude_edges = np.arange(
        minimum_latitude, maximum_latitude+latitude_step/2, latitude_step)
    mlt_edges = np.arange(0, 24+mlt_step/2, mlt_step)
    return latitude_edges, mlt_edges


def collapse_checkpoint_bins(latitude, mlt, values, selected,
                             latitude_edges, mlt_edges):
    """Return one mean estimate per occupied bin for one checkpoint."""

    latitude = np.asarray(latitude, dtype=float)
    mlt = np.asarray(mlt, dtype=float) % 24
    values = np.asarray(values, dtype=float)
    selected = np.asarray(selected, dtype=bool)
    valid = (
        selected & np.isfinite(latitude) & np.isfinite(mlt) &
        np.isfinite(values) &
        (latitude >= latitude_edges[0]) &
        (latitude < latitude_edges[-1]))
    if not np.any(valid):
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=float)

    latitude_id = np.searchsorted(
        latitude_edges, latitude[valid], side='right')-1
    mlt_id = np.searchsorted(mlt_edges, mlt[valid], side='right')-1
    mlt_id = np.clip(mlt_id, 0, len(mlt_edges)-2)
    mlt_count = len(mlt_edges)-1
    flat_id = latitude_id*mlt_count+mlt_id

    unique_id, inverse = np.unique(flat_id, return_inverse=True)
    total = np.bincount(inverse, weights=values[valid])
    count = np.bincount(inverse)
    checkpoint_values = total/count
    return unique_id.astype(np.int32), checkpoint_values


def finalize_binned_statistics(flat_ids, values, shape):
    """Calculate count and exact 10th/median/90th checkpoint quantiles."""

    flat_ids = np.asarray(flat_ids)
    values = np.asarray(values, dtype=float)
    size = int(np.prod(shape))
    count = np.zeros(size, dtype=np.int64)
    output = {
        'quantile_10': np.full(size, np.nan),
        'median': np.full(size, np.nan),
        'quantile_90': np.full(size, np.nan)}
    if values.size == 0:
        return {
            'count': count.reshape(shape),
            **{name: result.reshape(shape) for name, result in output.items()}}

    order = np.argsort(flat_ids)
    sorted_ids = flat_ids[order]
    sorted_values = values[order]
    unique_ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)
    count[unique_ids] = counts
    for cell_id, start, cell_count in zip(unique_ids, starts, counts):
        cell_values = sorted_values[start:start+cell_count]
        output['quantile_10'][cell_id] = np.quantile(cell_values, 0.1)
        output['median'][cell_id] = np.median(cell_values)
        output['quantile_90'][cell_id] = np.quantile(cell_values, 0.9)

    return {
        'count': count.reshape(shape),
        **{name: result.reshape(shape) for name, result in output.items()}}


#%% Activity selection and archive aggregation

def activity_masks(
        observation_time_ns, sml_time_seconds, sml_nt, sml_derivative,
        quiet_limit_nt=quiet_sml_limit_nt,
        quiet_derivative_limit=quiet_derivative_limit_nt_per_minute,
        active_maximum_nt=active_sml_maximum_nt,
        tolerance_seconds=sml_match_tolerance_seconds):
    """Classify matched samples as quiet or active from SuperMAG SML."""

    nearest, matched = nearest_sml_indices(
        observation_time_ns, sml_time_seconds,
        tolerance_seconds=tolerance_seconds)
    matched_sml = np.asarray(sml_nt, dtype=float)[nearest]
    matched_derivative = np.asarray(sml_derivative, dtype=float)[nearest]
    quiet = (
        matched & np.isfinite(matched_sml) &
        np.isfinite(matched_derivative) &
        (np.abs(matched_sml) < quiet_limit_nt) &
        (np.abs(matched_derivative) < quiet_derivative_limit))
    active = (
        matched & np.isfinite(matched_sml) &
        (matched_sml <= active_maximum_nt))
    return {'quiet': quiet, 'active': active}, matched


def aggregate_debug_analysis(
        checkpoint_directory=input_base,
        sml_cache_path=sml_cache_file,
        userid=supermag_userid,
        refresh_sml=False,
        sml_data=None,
        minimum_latitude=minimum_magnetic_latitude,
        maximum_latitude=maximum_magnetic_latitude,
        latitude_step=magnetic_latitude_step,
        mlt_step=mlt_step_hours,
        quiet_limit_nt=quiet_sml_limit_nt,
        quiet_derivative_limit=quiet_derivative_limit_nt_per_minute,
        active_maximum_nt=active_sml_maximum_nt,
        tolerance_seconds=sml_match_tolerance_seconds,
        show_progress=True):
    """Aggregate corrected observations by satellite, MEM, activity, and pass."""

    paths = discover_checkpoints(checkpoint_directory)
    if not paths:
        raise ValueError(f'No checkpoints found in {checkpoint_directory}')
    latitude_edges, mlt_edges = create_bin_edges(
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        latitude_step=latitude_step,
        mlt_step=mlt_step)
    shape = (len(latitude_edges)-1, len(mlt_edges)-1)

    if sml_data is None:
        start_ns, stop_ns, _ = checkpoint_time_range(
            paths, show_progress=show_progress)
        sml_time, sml, sml_derivative = load_or_download_sml(
            sml_cache_path, start_ns, stop_ns,
            userid=userid, refresh=refresh_sml)
    else:
        sml_time, sml, sml_derivative = sml_data

    keys = [
        (satellite, mem_id, activity, direction)
        for satellite in satellite_ids
        for mem_id in range(mem_count)
        for activity in activity_labels
        for direction in direction_labels]
    id_chunks = {key: [] for key in keys}
    value_chunks = {key: [] for key in keys}

    processed_files = 0
    incomplete_files = 0
    matched_samples = 0
    quiet_samples = 0
    active_samples = 0
    selected_checkpoint_estimates = 0
    satellite_file_counts = {}
    backgrounds = set()
    scalar_sources = set()
    checkpoint_schemas = set()

    loop = tqdm(
        paths, desc='Binning corrected observations in MLAT/MLT',
        disable=not show_progress)
    for path in loop:
        loaded = read_checkpoint_observations(path)
        if loaded is None:
            incomplete_files += 1
            continue

        satellite = loaded['satellite']
        maximum_count = int(np.max(loaded['sample_count']))
        if loaded['satellite_latitude'].size < maximum_count:
            raise ValueError(
                f'{path}: Level-2 satellite latitude is shorter than checkpoint')
        ascending, descending = local_orbit_direction(
            loaded['satellite_latitude'][:maximum_count])
        direction_masks = {
            'ascending': ascending,
            'descending': descending}

        for mem_id in range(mem_count):
            sample_count = int(loaded['sample_count'][mem_id])
            latitude = loaded['magnetic_latitude'][mem_id]
            mlt = loaded['mlt'][mem_id]
            if latitude.size < sample_count or mlt.size < sample_count:
                raise ValueError(
                    f'{path}: Level-2 MEM {mem_id+1} geometry is too short')

            observation_time = loaded['time'][mem_id, :sample_count]
            corrected = loaded['corrected_observed_dB'][
                mem_id, :sample_count]
            activities, matched = activity_masks(
                observation_time, sml_time, sml, sml_derivative,
                quiet_limit_nt=quiet_limit_nt,
                quiet_derivative_limit=quiet_derivative_limit,
                active_maximum_nt=active_maximum_nt,
                tolerance_seconds=tolerance_seconds)
            matched_samples += np.count_nonzero(matched)
            quiet_samples += np.count_nonzero(activities['quiet'])
            active_samples += np.count_nonzero(activities['active'])

            for activity in activity_labels:
                for direction in direction_labels:
                    selected = (
                        activities[activity] &
                        direction_masks[direction][:sample_count])
                    flat_ids, checkpoint_values = collapse_checkpoint_bins(
                        latitude[:sample_count], mlt[:sample_count], corrected,
                        selected, latitude_edges, mlt_edges)
                    if flat_ids.size:
                        key = (satellite, mem_id, activity, direction)
                        id_chunks[key].append(flat_ids)
                        value_chunks[key].append(checkpoint_values)
                        selected_checkpoint_estimates += flat_ids.size

        processed_files += 1
        satellite_file_counts[satellite] = (
            satellite_file_counts.get(satellite, 0)+1)
        backgrounds.add(loaded['metadata']['background'])
        scalar_sources.add(loaded['metadata']['scalar_source'])
        checkpoint_schemas.add(loaded['metadata']['checkpoint_schema'])
        loop.set_postfix(files=processed_files)

    if processed_files == 0:
        raise ValueError('No complete checkpoints were processed')

    array_shape = (
        len(satellite_ids), mem_count,
        len(activity_labels), len(direction_labels), *shape)
    result = {
        'count': np.zeros(array_shape, dtype=np.int64),
        'quantile_10': np.full(array_shape, np.nan),
        'median': np.full(array_shape, np.nan),
        'quantile_90': np.full(array_shape, np.nan)}

    for satellite_id, satellite in enumerate(satellite_ids):
        for mem_id in range(mem_count):
            for activity_id, activity in enumerate(activity_labels):
                for direction_id, direction in enumerate(direction_labels):
                    key = (satellite, mem_id, activity, direction)
                    if id_chunks[key]:
                        flat_ids = np.concatenate(id_chunks[key])
                        values = np.concatenate(value_chunks[key])
                    else:
                        flat_ids = np.empty(0, dtype=np.int32)
                        values = np.empty(0, dtype=float)
                    finalized = finalize_binned_statistics(
                        flat_ids, values, shape)
                    index = (
                        satellite_id, mem_id, activity_id, direction_id)
                    for name in result:
                        result[name][index] = finalized[name]

    metadata = {
        'source_directory': str(checkpoint_directory),
        'supermag_sml_cache': str(sml_cache_path),
        'quiet_sml_absolute_limit_nt': float(quiet_limit_nt),
        'quiet_sml_derivative_limit_nt_per_minute': float(
            quiet_derivative_limit),
        'active_sml_maximum_nt': float(active_maximum_nt),
        'sml_match_tolerance_seconds': float(tolerance_seconds),
        'processed_file_count': processed_files,
        'incomplete_file_count': incomplete_files,
        'matched_sample_count': int(matched_samples),
        'quiet_sample_count': int(quiet_samples),
        'active_sample_count': int(active_samples),
        'checkpoint_bin_estimate_count': int(selected_checkpoint_estimates),
        'satellite_file_counts': json.dumps(
            satellite_file_counts, sort_keys=True),
        'backgrounds': json.dumps(sorted(backgrounds)),
        'scalar_sources': json.dumps(sorted(scalar_sources)),
        'checkpoint_schema_versions': json.dumps(sorted(checkpoint_schemas)),
        'within_checkpoint_aggregation': 'mean per MLAT/MLT bin',
        'across_checkpoint_aggregation': 'exact quantiles'}
    return latitude_edges, mlt_edges, result, metadata


#%% NetCDF output

def write_debug_analysis(path, latitude_edges, mlt_edges, result, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')

    latitude_centres = (latitude_edges[:-1]+latitude_edges[1:])/2
    mlt_centres = (mlt_edges[:-1]+mlt_edges[1:])/2
    dimensions = (
        'satellite', 'mem', 'activity', 'direction',
        'magnetic_latitude', 'mlt')

    try:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            dataset.processing_complete = 0
            dataset.schema_version = schema_version
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            dataset.activity_order = ','.join(activity_labels)
            dataset.direction_order = ','.join(direction_labels)
            for name, value in metadata.items():
                setattr(dataset, name, value)

            dataset.createDimension('satellite', len(satellite_ids))
            dataset.createDimension('mem', mem_count)
            dataset.createDimension('activity', len(activity_labels))
            dataset.createDimension('direction', len(direction_labels))
            dataset.createDimension(
                'magnetic_latitude', latitude_centres.size)
            dataset.createDimension(
                'magnetic_latitude_edge', latitude_edges.size)
            dataset.createDimension('mlt', mlt_centres.size)
            dataset.createDimension('mlt_edge', mlt_edges.size)

            satellite = dataset.createVariable(
                'satellite', 'S1', ('satellite',))
            satellite[:] = np.asarray(satellite_ids, dtype='S1')
            create_variable(
                dataset, 'mem_number', 'i2', ('mem',),
                np.arange(1, mem_count+1))
            create_variable(
                dataset, 'magnetic_latitude', 'f8',
                ('magnetic_latitude',), latitude_centres,
                units='degrees_north')
            create_variable(
                dataset, 'magnetic_latitude_edge', 'f8',
                ('magnetic_latitude_edge',), latitude_edges,
                units='degrees_north')
            create_variable(
                dataset, 'mlt', 'f8', ('mlt',), mlt_centres,
                units='hours')
            create_variable(
                dataset, 'mlt_edge', 'f8', ('mlt_edge',), mlt_edges,
                units='hours')
            create_variable(
                dataset, 'checkpoint_count', 'i8', dimensions,
                result['count'],
                long_name=(
                    'checkpoint estimates after within-checkpoint bin collapse'))
            for statistic in statistics:
                create_variable(
                    dataset, f'corrected_observed_dB_{statistic}',
                    'f8', dimensions, result[statistic], units='T',
                    long_name=(
                        f'{statistic} of observed_dB minus fitted track offset'))
            dataset.processing_complete = 1

        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def read_debug_analysis(path):
    """Load a completed analysis file so figures can be redesigned cheaply."""

    path = Path(path)
    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            raise ValueError(f'{path} is not a complete debug analysis')
        latitude_edges = np.asarray(dataset['magnetic_latitude_edge'][:])
        mlt_edges = np.asarray(dataset['mlt_edge'][:])
        result = {
            'count': np.asarray(dataset['checkpoint_count'][:]),
            'quantile_10': np.asarray(
                dataset['corrected_observed_dB_quantile_10'][:]),
            'median': np.asarray(
                dataset['corrected_observed_dB_median'][:]),
            'quantile_90': np.asarray(
                dataset['corrected_observed_dB_quantile_90'][:])}
        metadata = {
            name: dataset.getncattr(name) for name in dataset.ncattrs()}
    return latitude_edges, mlt_edges, result, metadata


#%% Diagnostic figures

def activity_title(activity, metadata):
    if activity == 'quiet':
        return (
            f"quiet: absolute SML below "
            f"{metadata['quiet_sml_absolute_limit_nt']:g} nT, "
            f"absolute dSML/dt below "
            f"{metadata['quiet_sml_derivative_limit_nt_per_minute']:g} nT/min")
    return (
        f"active: SML at or below "
        f"{metadata['active_sml_maximum_nt']:g} nT")


def midnight_plot_window(
        mlt_edges, *arrays, minimum_hour=-2.5, maximum_hour=5.5):
    """Reorder MLT cells onto a continuous window centred on midnight."""

    mlt_edges = np.asarray(mlt_edges, dtype=float)
    centres = (mlt_edges[:-1]+mlt_edges[1:])/2
    wrapped_centres = (centres+12) % 24-12
    if minimum_hour >= maximum_hour:
        raise ValueError('minimum MLT hour must be below maximum MLT hour')
    selected = np.flatnonzero(
        (wrapped_centres >= minimum_hour) &
        (wrapped_centres <= maximum_hour))
    selected = selected[np.argsort(wrapped_centres[selected])]
    if selected.size == 0:
        raise ValueError('No MLT cells lie inside the midnight plot window')

    step = np.median(np.diff(mlt_edges))
    plot_edges = np.concatenate((
        wrapped_centres[selected]-step/2,
        [wrapped_centres[selected[-1]]+step/2]))
    reordered = [np.take(array, selected, axis=-1) for array in arrays]
    return plot_edges, reordered


def _masked_midnight_field(
        mlt_edges, values, count, minimum_count,
        minimum_hour=-2.5, maximum_hour=5.5):
    display = np.where(count >= minimum_count, values*1e9, np.nan)
    plot_edges, (display, count) = midnight_plot_window(
        mlt_edges, display, count,
        minimum_hour=minimum_hour, maximum_hour=maximum_hour)
    return plot_edges, display, count


def _robust_color_limit(fields):
    finite_fields = [
        field[np.isfinite(field)].ravel()
        for field in fields if np.any(np.isfinite(field))]
    if not finite_fields:
        raise ValueError('No cells satisfy the coverage threshold')
    finite = np.concatenate(finite_fields)
    limit = np.percentile(np.abs(finite), 98)
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def plot_primary_comparison(
        path, latitude_edges, mlt_edges, result, metadata,
        minimum_count=minimum_coverage_count):
    """Compare quiet/active MEM 1 and MEM 3 on the supported nightside."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    direction_id = direction_labels.index('descending')
    columns = (
        ('quiet', 0), ('quiet', 2), ('active', 0), ('active', 2))
    fields = []
    counts = []
    plot_edges = None
    for activity, mem_id in columns:
        activity_id = activity_labels.index(activity)
        edge, display, count = _masked_midnight_field(
            mlt_edges,
            result['median'][:, mem_id, activity_id, direction_id],
            result['count'][:, mem_id, activity_id, direction_id],
            minimum_count)
        plot_edges = edge
        fields.append(display)
        counts.append(count)
    color_limit = _robust_color_limit(fields)

    fig, axes = plt.subplots(
        3, 4, figsize=(14, 10), sharex=True, sharey=True,
        constrained_layout=True)
    images = []
    for satellite_id, satellite in enumerate(satellite_ids):
        for column_id, (activity, mem_id) in enumerate(columns):
            ax = axes[satellite_id, column_id]
            image = ax.pcolormesh(
                plot_edges, latitude_edges, fields[column_id][satellite_id],
                cmap='RdBu_r', vmin=-color_limit, vmax=color_limit,
                shading='flat')
            images.append(image)
            supported = np.count_nonzero(
                counts[column_id][satellite_id] >= minimum_count)
            ax.text(
                0.02, 0.03, f'{supported} supported cells',
                transform=ax.transAxes, fontsize=8, color='0.25')
            ax.grid(color='0.8', linewidth=0.4)
            if satellite_id == 0:
                ax.set_title(f'{activity.capitalize()}, MEM {mem_id+1}')
        axes[satellite_id, 0].set_ylabel(
            f'Satellite {satellite.upper()}\nmagnetic latitude [deg]')

    for ax in axes[-1]:
        ax.set_xlabel('MLT relative to midnight [h]')
    colorbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.9)
    colorbar.set_label('median observed dB minus fitted offset [nT]')
    fig.suptitle(
        'Primary test: does MEM 3 track MEM 1 and strengthen with activity?\n'
        f'descending passes; at least {minimum_count} checkpoint estimates; '
        f'{activity_title("quiet", metadata)}; '
        f'{activity_title("active", metadata)}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mem3_quantiles(
        path, latitude_edges, mlt_edges, result, metadata, satellite_id,
        minimum_count=minimum_coverage_count):
    """Show whether the MEM-3 distribution shifts or only develops tails."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    direction_id = direction_labels.index('descending')
    mem_id = 2
    fields = {}
    plot_edges = None
    for activity_id, activity in enumerate(activity_labels):
        for statistic in statistics:
            edge, display, _ = _masked_midnight_field(
                mlt_edges,
                result[statistic][
                    satellite_id, mem_id, activity_id, direction_id],
                result['count'][
                    satellite_id, mem_id, activity_id, direction_id],
                minimum_count)
            plot_edges = edge
            fields[(activity, statistic)] = display
    color_limit = _robust_color_limit(list(fields.values()))

    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7), sharex=True, sharey=True,
        constrained_layout=True)
    labels = {
        'quantile_10': '10th percentile',
        'median': 'median',
        'quantile_90': '90th percentile'}
    images = []
    for activity_id, activity in enumerate(activity_labels):
        for statistic_id, statistic in enumerate(statistics):
            ax = axes[activity_id, statistic_id]
            image = ax.pcolormesh(
                plot_edges, latitude_edges, fields[(activity, statistic)],
                cmap='RdBu_r', vmin=-color_limit, vmax=color_limit,
                shading='flat')
            images.append(image)
            ax.grid(color='0.8', linewidth=0.4)
            if activity_id == 0:
                ax.set_title(labels[statistic])
        axes[activity_id, 0].set_ylabel(
            f'{activity.capitalize()}\nmagnetic latitude [deg]')
    for ax in axes[-1]:
        ax.set_xlabel('MLT relative to midnight [h]')
    colorbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.9)
    colorbar.set_label('observed dB minus fitted offset [nT]')
    satellite = satellite_ids[satellite_id].upper()
    fig.suptitle(
        f'Satellite {satellite}, MEM 3 distribution: quiet versus active SML\n'
        'A median shift is persistent; wider or asymmetric tails are '
        'intermittent\n'
        f'descending passes; at least {minimum_count} checkpoint estimates')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_direction_support(path, result):
    """Expose whether ascending and descending comparisons are supported."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f'{satellite.upper()}{mem_id+1}'
        for satellite in satellite_ids for mem_id in range(mem_count)]
    x = np.arange(len(labels))
    width = 0.38

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for activity_id, activity in enumerate(activity_labels):
        ax = axes[activity_id]
        support = result['count'][:, :, activity_id].sum(axis=(-2, -1))
        support = support.reshape(len(labels), len(direction_labels))
        ax.bar(
            x-width/2, support[:, 0], width,
            label='ascending', color='#4c78a8')
        ax.bar(
            x+width/2, support[:, 1], width,
            label='descending', color='#f58518')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylabel('checkpoint-cell estimates')
        ax.set_title(activity.capitalize())
        ax.grid(axis='y', color='0.85', linewidth=0.5)
    axes[0].legend()
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel('satellite and MEM number')
    fig.suptitle(
        'Pass-direction support (log-like scale)\n'
        'A field comparison is unreliable where one direction has little data')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_statistic_panels(
        path, latitude_edges, mlt_edges, values, count,
        activity, direction, statistic, metadata,
        minimum_count=minimum_coverage_count, color_limit=None):
    """Plot one MLAT/MLT statistic for all satellite/MEM combinations."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display = np.where(count >= minimum_count, values*1e9, np.nan)
    finite = display[np.isfinite(display)]
    if finite.size == 0:
        raise ValueError(
            f'No {activity}/{direction}/{statistic} cells satisfy coverage')
    if color_limit is None:
        color_limit = np.percentile(np.abs(finite), 98)
    if not np.isfinite(color_limit) or color_limit == 0:
        color_limit = 1.0

    fig, axes = plt.subplots(
        3, 4, figsize=(18, 13), sharex=True, sharey=True,
        constrained_layout=True)
    images = []
    for satellite_id, satellite in enumerate(satellite_ids):
        for mem_id in range(mem_count):
            ax = axes[satellite_id, mem_id]
            image = ax.pcolormesh(
                mlt_edges, latitude_edges,
                display[satellite_id, mem_id],
                cmap='RdBu_r', vmin=-color_limit, vmax=color_limit,
                shading='flat')
            ax.set_title(f'Satellite {satellite.upper()}, MEM {mem_id+1}')
            ax.set_xlim(0, 24)
            ax.set_ylim(latitude_edges[0], latitude_edges[-1])
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(color='0.75', linewidth=0.4)
            images.append(image)

    for ax in axes[-1, :]:
        ax.set_xlabel('magnetic local time [h]')
    for ax in axes[:, 0]:
        ax.set_ylabel('magnetic latitude [deg]')
    colorbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.88)
    labels = {
        'quantile_10': '10th percentile',
        'median': 'median',
        'quantile_90': '90th percentile'}
    colorbar.set_label(
        f"{labels[statistic]} observed dB minus fitted offset [nT]")
    fig.suptitle(
        f'{activity_title(activity, metadata)}; {direction}; '
        f'coverage >= {minimum_count}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_coverage_panels(
        path, latitude_edges, mlt_edges, count,
        activity, direction, metadata, maximum=None):
    """Plot checkpoint support for one activity/direction combination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    display = np.where(count > 0, count, np.nan)
    if maximum is None:
        maximum = np.nanmax(display)
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError(f'No {activity}/{direction} coverage is available')

    fig, axes = plt.subplots(
        3, 4, figsize=(18, 13), sharex=True, sharey=True,
        constrained_layout=True)
    images = []
    for satellite_id, satellite in enumerate(satellite_ids):
        for mem_id in range(mem_count):
            ax = axes[satellite_id, mem_id]
            image = ax.pcolormesh(
                mlt_edges, latitude_edges,
                display[satellite_id, mem_id],
                cmap='viridis', vmin=0, vmax=maximum, shading='flat')
            ax.set_title(f'Satellite {satellite.upper()}, MEM {mem_id+1}')
            ax.set_xlim(0, 24)
            ax.set_ylim(latitude_edges[0], latitude_edges[-1])
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(color='0.75', linewidth=0.4)
            images.append(image)

    for ax in axes[-1, :]:
        ax.set_xlabel('magnetic local time [h]')
    for ax in axes[:, 0]:
        ax.set_ylabel('magnetic latitude [deg]')
    colorbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label('checkpoint estimates per MLAT/MLT cell')
    fig.suptitle(f'{activity_title(activity, metadata)}; {direction}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_exhaustive_plots(
        output_directory, latitude_edges, mlt_edges, result, metadata,
        minimum_count=minimum_coverage_count):
    """Write percentile and coverage panels for every data split."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = []

    color_limits = {}
    for statistic in statistics:
        masked = np.where(
            result['count'] >= minimum_count,
            result[statistic]*1e9, np.nan)
        finite = masked[np.isfinite(masked)]
        if finite.size == 0:
            raise ValueError(
                f'No {statistic} cells satisfy the coverage threshold')
        color_limits[statistic] = np.percentile(np.abs(finite), 98)
    finite_counts = result['count'][result['count'] > 0]
    coverage_maximum = np.percentile(finite_counts, 98)

    for activity_id, activity in enumerate(activity_labels):
        for direction_id, direction in enumerate(direction_labels):
            index = (slice(None), slice(None), activity_id, direction_id)
            for statistic in statistics:
                short_name = {
                    'quantile_10': 'p10',
                    'median': 'median',
                    'quantile_90': 'p90'}[statistic]
                path = output_directory / (
                    f'corrected_observed_dB_{activity}_{direction}_'
                    f'{short_name}.png')
                plot_statistic_panels(
                    path, latitude_edges, mlt_edges,
                    result[statistic][index], result['count'][index],
                    activity, direction, statistic, metadata,
                    minimum_count=minimum_count,
                    color_limit=color_limits[statistic])
                paths.append(path)

            coverage_path = output_directory / (
                f'corrected_observed_dB_{activity}_{direction}_coverage.png')
            plot_coverage_panels(
                coverage_path, latitude_edges, mlt_edges,
                result['count'][index], activity, direction, metadata,
                maximum=coverage_maximum)
            paths.append(coverage_path)
    return paths


def write_focused_plots(
        output_directory, latitude_edges, mlt_edges, result, metadata,
        minimum_count=minimum_coverage_count):
    """Write the compact figures that address the scientific hypotheses."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = [
        output_directory/'comparison_mem1_mem3_quiet_active.png',
        output_directory/'pass_direction_support.png']
    plot_primary_comparison(
        paths[0], latitude_edges, mlt_edges, result, metadata,
        minimum_count=minimum_count)
    plot_direction_support(paths[1], result)

    for satellite_id, satellite in enumerate(satellite_ids):
        path = output_directory/f'mem3_quantiles_satellite_{satellite}.png'
        plot_mem3_quantiles(
            path, latitude_edges, mlt_edges, result, metadata, satellite_id,
            minimum_count=minimum_count)
        paths.append(path)
    return paths


#%% Complete workflow

def build_debug_analysis(
        checkpoint_directory=input_base,
        output_directory=output_base,
        output_path=output_file,
        sml_cache_path=sml_cache_file,
        userid=supermag_userid,
        refresh_sml=False,
        sml_data=None,
        minimum_latitude=minimum_magnetic_latitude,
        maximum_latitude=maximum_magnetic_latitude,
        latitude_step=magnetic_latitude_step,
        mlt_step=mlt_step_hours,
        quiet_limit_nt=quiet_sml_limit_nt,
        quiet_derivative_limit=quiet_derivative_limit_nt_per_minute,
        active_maximum_nt=active_sml_maximum_nt,
        tolerance_seconds=sml_match_tolerance_seconds,
        minimum_count=minimum_coverage_count,
        show_progress=True,
        write_plots=True,
        exhaustive_plots=False):
    latitude_edges, mlt_edges, result, metadata = aggregate_debug_analysis(
        checkpoint_directory=checkpoint_directory,
        sml_cache_path=sml_cache_path,
        userid=userid,
        refresh_sml=refresh_sml,
        sml_data=sml_data,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        latitude_step=latitude_step,
        mlt_step=mlt_step,
        quiet_limit_nt=quiet_limit_nt,
        quiet_derivative_limit=quiet_derivative_limit,
        active_maximum_nt=active_maximum_nt,
        tolerance_seconds=tolerance_seconds,
        show_progress=show_progress)
    write_debug_analysis(
        output_path, latitude_edges, mlt_edges, result, metadata)
    plot_paths = []
    if write_plots:
        if exhaustive_plots:
            plot_paths = write_exhaustive_plots(
                output_directory, latitude_edges, mlt_edges, result, metadata,
                minimum_count=minimum_count)
        else:
            plot_paths = write_focused_plots(
                output_directory, latitude_edges, mlt_edges, result, metadata,
                minimum_count=minimum_count)
    return latitude_edges, mlt_edges, result, metadata, plot_paths


#%% Command-line entry point

def positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return value


def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Analyze corrected EZIE observations in magnetic latitude and MLT'))
    parser.add_argument('--input', type=Path, default=input_base)
    parser.add_argument('--output-directory', type=Path, default=output_base)
    parser.add_argument('--output', type=Path, default=None,
                        help='NetCDF path; defaults beneath output-directory')
    parser.add_argument('--sml-cache', type=Path, default=sml_cache_file)
    parser.add_argument('--refresh-sml', action='store_true')
    parser.add_argument('--minimum-magnetic-latitude', type=float,
                        default=minimum_magnetic_latitude)
    parser.add_argument('--maximum-magnetic-latitude', type=float,
                        default=maximum_magnetic_latitude)
    parser.add_argument('--magnetic-latitude-step', type=positive_float,
                        default=magnetic_latitude_step)
    parser.add_argument('--mlt-step-hours', type=positive_float,
                        default=mlt_step_hours)
    parser.add_argument('--quiet-sml-limit-nt', type=positive_float,
                        default=quiet_sml_limit_nt)
    parser.add_argument('--quiet-derivative-limit', type=positive_float,
                        default=quiet_derivative_limit_nt_per_minute)
    parser.add_argument('--active-sml-maximum-nt', type=float,
                        default=active_sml_maximum_nt)
    parser.add_argument('--minimum-coverage-count', type=nonnegative_integer,
                        default=minimum_coverage_count)
    parser.add_argument(
        '--plot-existing', action='store_true',
        help='only regenerate focused figures from the existing NetCDF output')
    parser.add_argument(
        '--exhaustive-plots', action='store_true',
        help='write the former 16 activity/direction/statistic matrices')
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = (
        args.output if args.output is not None
        else args.output_directory/'corrected_observations_mlat_mlt.nc')
    if args.plot_existing:
        latitude_edges, mlt_edges, result, metadata = read_debug_analysis(
            output_path)
        if args.exhaustive_plots:
            plot_paths = write_exhaustive_plots(
                args.output_directory, latitude_edges, mlt_edges,
                result, metadata,
                minimum_count=args.minimum_coverage_count)
        else:
            plot_paths = write_focused_plots(
                args.output_directory, latitude_edges, mlt_edges,
                result, metadata,
                minimum_count=args.minimum_coverage_count)
        print(f'Wrote {len(plot_paths)} diagnostic figures from {output_path}')
        return

    _, _, _, metadata, plot_paths = build_debug_analysis(
        checkpoint_directory=args.input,
        output_directory=args.output_directory,
        output_path=output_path,
        sml_cache_path=args.sml_cache,
        refresh_sml=args.refresh_sml,
        minimum_latitude=args.minimum_magnetic_latitude,
        maximum_latitude=args.maximum_magnetic_latitude,
        latitude_step=args.magnetic_latitude_step,
        mlt_step=args.mlt_step_hours,
        quiet_limit_nt=args.quiet_sml_limit_nt,
        quiet_derivative_limit=args.quiet_derivative_limit,
        active_maximum_nt=args.active_sml_maximum_nt,
        minimum_count=args.minimum_coverage_count,
        exhaustive_plots=args.exhaustive_plots)
    print(f'Wrote {output_path} and {len(plot_paths)} diagnostic figures')
    print(
        f"Processed {metadata['processed_file_count']} checkpoints; "
        f"quiet samples {metadata['quiet_sample_count']:,}; "
        f"active samples {metadata['active_sample_count']:,}")


if __name__ == '__main__':
    main()
