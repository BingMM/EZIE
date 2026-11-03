#%% Import

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from polplot import Polarplot
from tqdm import tqdm

if __package__:
    from .lithosphere_global_map import (
        add_checkpoint_footprints,
        apply_coverage_threshold,
        calculate_median,
        create_global_grid,
        create_variable,
        discover_checkpoints,
        finalize_statistics,
        initialize_footprint_statistics)
else:
    from lithosphere_global_map import (
        add_checkpoint_footprints,
        apply_coverage_threshold,
        calculate_median,
        create_global_grid,
        create_variable,
        discover_checkpoints,
        finalize_statistics,
        initialize_footprint_statistics)


#%% Configuration

input_base = Path('/media/bing/LaCie/EZIE_lithosphere_l3')
output_base = input_base.parent / 'EZIE_lithosphere_mem_debug'
output_file = output_base / 'scalar_50km_mem_debug_all.nc'
median_plot_file = output_base / 'scalar_50km_mem_medians_all.png'
coverage_plot_file = output_base / 'scalar_50km_mem_coverage_all.png'
track_offset_plot_file = output_base / 'scalar_50km_track_offset_medians_all.png'
observed_dB_plot_file = output_base / 'scalar_50km_observed_dB_medians_all.png'
observed_dB_minus_offset_plot_file = (
    output_base / 'scalar_50km_observed_dB_minus_offset_medians_all.png')
observed_dB_minus_offset_p10_plot_file = (
    output_base / 'scalar_50km_observed_dB_minus_offset_p10_all.png')
observed_dB_minus_offset_p90_plot_file = (
    output_base / 'scalar_50km_observed_dB_minus_offset_p90_all.png')
background_B_plot_file = output_base / 'scalar_50km_background_B_medians_all.png'

scalar_variable = 'predicted_dB_exact'
minimum_latitude = 50.0
resolution_m = 50_000.0
footprint_size_m = 50_000.0
minimum_coverage_count = 10
earth_radius_m = 6_371_200.0
satellite_ids = ('a', 'b', 'c')
mem_count = 4
field_names = (
    'prediction', 'track_offset', 'observed_dB',
    'observed_dB_minus_offset', 'background_B')
debug_schema_version = '1.4'


#%% Checkpoint loading

def read_checkpoint_mems(path, variable_name=scalar_variable):
    """Read the mapped scalar quantities without applying time filtering."""

    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            return None

        required = {
            'latitude', 'longitude', variable_name,
            'track_offset', 'observed_dB', 'background_B'}
        missing = required.difference(dataset.variables)
        if missing:
            names = ', '.join(sorted(missing))
            raise ValueError(f'{path} is missing required variables: {names}')

        latitude = np.asarray(
            np.ma.filled(dataset.variables['latitude'][:], np.nan), dtype=float)
        longitude = np.asarray(
            np.ma.filled(dataset.variables['longitude'][:], np.nan), dtype=float)
        if latitude.ndim != 2:
            raise ValueError(f'{path}: expected MEM-by-sample arrays')
        if latitude.shape != longitude.shape:
            raise ValueError(f'{path}: coordinate shapes do not match')

        fields = {}
        for field, variable_name_in_file in (
                ('prediction', variable_name),
                ('observed_dB', 'observed_dB'),
                ('background_B', 'background_B')):
            variable = dataset.variables[variable_name_in_file]
            units = getattr(variable, 'units', None)
            if units != 'T':
                raise ValueError(
                    f'{path}: {variable_name_in_file} must have units T, '
                    f'found {units!r}')
            values = np.asarray(
                np.ma.filled(variable[:], np.nan), dtype=float)
            if values.shape != latitude.shape:
                raise ValueError(
                    f'{path}: {variable_name_in_file} and coordinate shapes '
                    'do not match')
            fields[field] = values

        offset_variable = dataset.variables['track_offset']
        offset_units = getattr(offset_variable, 'units', None)
        if offset_units != 'T':
            raise ValueError(
                f'{path}: track_offset must have units T, '
                f'found {offset_units!r}')
        offsets = np.asarray(
            np.ma.filled(offset_variable[:], np.nan), dtype=float)
        if offsets.shape != (latitude.shape[0],):
            raise ValueError(f'{path}: track_offset must have one value per MEM')
        fields['track_offset'] = np.broadcast_to(
            offsets[:, None], latitude.shape)
        fields['observed_dB_minus_offset'] = (
            fields['observed_dB'] - fields['track_offset'])

        metadata = {
            'background': str(getattr(dataset, 'background', 'unknown')),
            'checkpoint_schema': str(getattr(dataset, 'schema_version', 'unknown')),
            'satellite': str(getattr(dataset, 'satellite', 'unknown'))}
    return latitude, longitude, fields, metadata


def calculate_quantiles(flat_ids, values, shape, quantiles):
    """Calculate exact per-cell quantiles from checkpoint-level estimates."""

    flat_ids = np.asarray(flat_ids)
    values = np.asarray(values, dtype=float)
    quantiles = tuple(float(quantile) for quantile in quantiles)
    if any(quantile < 0 or quantile > 1 for quantile in quantiles):
        raise ValueError('quantiles must lie between zero and one')

    output = {
        quantile: np.full(np.prod(shape), np.nan)
        for quantile in quantiles}
    if values.size == 0:
        return {
            quantile: result.reshape(shape)
            for quantile, result in output.items()}

    order = np.argsort(flat_ids)
    sorted_ids = flat_ids[order]
    sorted_values = values[order]
    unique_ids, starts, counts = np.unique(
        sorted_ids, return_index=True, return_counts=True)
    for cell_id, start, count in zip(unique_ids, starts, counts):
        cell_values = sorted_values[start:start+count]
        for quantile in quantiles:
            output[quantile][cell_id] = np.quantile(cell_values, quantile)

    return {
        quantile: result.reshape(shape)
        for quantile, result in output.items()}


#%% MEM aggregation

def aggregate_mem_maps(
        checkpoint_directory=input_base, variable_name=scalar_variable,
        minimum_latitude=minimum_latitude, resolution_m=resolution_m,
        footprint_size_m=footprint_size_m, radius_m=earth_radius_m,
        minimum_coverage_count=minimum_coverage_count, show_progress=True):
    """Aggregate each satellite/MEM pair as an independent data stream."""

    if minimum_coverage_count < 0:
        raise ValueError('minimum_coverage_count must be non-negative')
    paths = discover_checkpoints(checkpoint_directory)
    grid = create_global_grid(
        minimum_latitude=minimum_latitude,
        resolution_m=resolution_m,
        radius_m=radius_m)

    statistics = {
        field: {
            satellite: [
                initialize_footprint_statistics(grid)
                for _ in range(mem_count)]
            for satellite in satellite_ids}
        for field in field_names}
    id_chunks = {
        field: {
            satellite: [[] for _ in range(mem_count)]
            for satellite in satellite_ids}
        for field in field_names}
    value_chunks = {
        field: {
            satellite: [[] for _ in range(mem_count)]
            for satellite in satellite_ids}
        for field in field_names}
    processed_files = 0
    incomplete_files = 0
    satellite_file_counts = {}
    checkpoint_schemas = set()
    backgrounds = set()

    loop = tqdm(
        paths, desc='Aggregating checkpoints by satellite/MEM',
        disable=not show_progress)
    for path in loop:
        loaded = read_checkpoint_mems(path, variable_name=variable_name)
        if loaded is None:
            incomplete_files += 1
            continue

        latitude, longitude, fields, file_metadata = loaded
        if latitude.shape[0] != mem_count:
            raise ValueError(
                f'{path}: expected {mem_count} MEMs, found {latitude.shape[0]}')
        satellite = file_metadata['satellite'].lower()
        if satellite not in satellite_ids:
            raise ValueError(
                f'{path}: expected satellite a, b, or c, found {satellite!r}')

        for field in field_names:
            for mem_id in range(latitude.shape[0]):
                flat_ids, checkpoint_values = add_checkpoint_footprints(
                    grid, statistics[field][satellite][mem_id],
                    latitude[mem_id], longitude[mem_id], fields[field][mem_id],
                    footprint_size_m=footprint_size_m,
                    minimum_latitude=minimum_latitude)
                if flat_ids.size:
                    id_chunks[field][satellite][mem_id].append(flat_ids)
                    value_chunks[field][satellite][mem_id].append(
                        checkpoint_values)

        processed_files += 1
        satellite_file_counts[satellite] = (
            satellite_file_counts.get(satellite, 0)+1)
        checkpoint_schemas.add(file_metadata['checkpoint_schema'])
        backgrounds.add(file_metadata['background'])
        loop.set_postfix(files=processed_files)

    if processed_files == 0:
        raise ValueError(f'No complete checkpoints found in {checkpoint_directory}')

    finalized = {
        field: {satellite: [] for satellite in satellite_ids}
        for field in field_names}
    for field in field_names:
        for satellite in satellite_ids:
            for mem_id in range(mem_count):
                if id_chunks[field][satellite][mem_id]:
                    flat_ids = np.concatenate(
                        id_chunks[field][satellite][mem_id])
                    values = np.concatenate(
                        value_chunks[field][satellite][mem_id])
                else:
                    flat_ids = np.empty(0, dtype=np.int32)
                    values = np.empty(0, dtype=float)
                median = calculate_median(flat_ids, values, grid.shape)
                result = finalize_statistics(
                    statistics[field][satellite][mem_id], grid, median,
                    minimum_latitude=minimum_latitude)
                if field == 'observed_dB_minus_offset':
                    quantiles = calculate_quantiles(
                        flat_ids, values, grid.shape, (0.1, 0.9))
                    result['quantile_10'] = quantiles[0.1]
                    result['quantile_90'] = quantiles[0.9]
                result['plot_coverage_mask'] = (
                    result['cap_mask'] &
                    (result['count'] >= minimum_coverage_count))
                finalized[field][satellite].append(result)

    metadata = {
        'source_directory': str(checkpoint_directory),
        'source_variable': variable_name,
        'mapped_fields': ','.join(field_names),
        'minimum_latitude_degrees': float(minimum_latitude),
        'nominal_resolution_m': float(resolution_m),
        'footprint_size_m': float(footprint_size_m),
        'plot_minimum_coverage_count': int(minimum_coverage_count),
        'processed_file_count': processed_files,
        'incomplete_file_count': incomplete_files,
        'satellite_file_counts': json.dumps(satellite_file_counts, sort_keys=True),
        'satellite_order': ','.join(satellite_ids),
        'mem_count_per_satellite': mem_count,
        'checkpoint_schema_versions': json.dumps(sorted(checkpoint_schemas)),
        'backgrounds': json.dumps(sorted(backgrounds)),
        'sml_filter_applied': 0}
    return grid, finalized, metadata


#%% Diagnostic output

def write_mem_debug_map(path, grid, field_statistics, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')

    try:
        with Dataset(temporary, 'w', format='NETCDF4') as dataset:
            dataset.processing_complete = 0
            dataset.schema_version = debug_schema_version
            dataset.created_utc = datetime.now(timezone.utc).isoformat()
            for name, value in metadata.items():
                setattr(dataset, name, value)

            dataset.createDimension('satellite', len(satellite_ids))
            dataset.createDimension('mem', mem_count)
            dataset.createDimension('grid_y', grid.shape[0])
            dataset.createDimension('grid_x', grid.shape[1])
            dataset.createDimension('mesh_y', grid.lat_mesh.shape[0])
            dataset.createDimension('mesh_x', grid.lat_mesh.shape[1])

            satellite_variable = dataset.createVariable(
                'satellite', 'S1', ('satellite',))
            satellite_variable[:] = np.asarray(satellite_ids, dtype='S1')
            satellite_variable.long_name = 'EZIE satellite identifier'
            create_variable(
                dataset, 'mem_number', 'i2', ('mem',),
                np.arange(1, mem_count+1))
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

            dimensions = ('satellite', 'mem', 'grid_y', 'grid_x')
            def stacked(statistics, name):
                return np.stack(
                    [[statistics[satellite][mem_id][name]
                      for mem_id in range(mem_count)]
                     for satellite in satellite_ids])

            prediction_statistics = field_statistics['prediction']
            create_variable(
                dataset, 'checkpoint_count', 'i8', dimensions,
                stacked(prediction_statistics, 'count'),
                long_name='checkpoint estimates per satellite, MEM, and cell')
            create_variable(
                dataset, 'footprint_contribution_count', 'i8', dimensions,
                stacked(prediction_statistics, 'contribution_count'),
                long_name=(
                    'sample-footprint overlaps per satellite, MEM, and cell'))
            create_variable(
                dataset, 'footprint_coverage_weight', 'f8', dimensions,
                stacked(prediction_statistics, 'coverage_weight'), units='1',
                long_name=(
                    'summed fractional footprint overlap per satellite, '
                    'MEM, and cell'))
            create_variable(
                dataset, 'plot_coverage_mask', 'i1', dimensions,
                stacked(
                    prediction_statistics,
                    'plot_coverage_mask').astype(np.int8))

            prefixes = {
                'prediction': 'scalar',
                'track_offset': 'track_offset',
                'observed_dB': 'observed_dB',
                'observed_dB_minus_offset': 'observed_dB_minus_offset',
                'background_B': 'background_B'}
            for field in field_names:
                statistics = field_statistics[field]
                prefix = prefixes[field]
                if field != 'prediction':
                    create_variable(
                        dataset, f'{prefix}_checkpoint_count', 'i8', dimensions,
                        stacked(statistics, 'count'),
                        long_name=(
                            f'checkpoint estimates for {field} per satellite, '
                            'MEM, and cell'))
                    create_variable(
                        dataset, f'{prefix}_plot_coverage_mask', 'i1',
                        dimensions,
                        stacked(
                            statistics,
                            'plot_coverage_mask').astype(np.int8))
                for statistic_name, output_name in (
                        ('mean', 'mean'),
                        ('median', 'median'),
                        ('standard_deviation', 'standard_deviation'),
                        ('standard_error', 'standard_error')):
                    create_variable(
                        dataset, f'{prefix}_{output_name}', 'f8', dimensions,
                        stacked(statistics, statistic_name), units='T')
                if field == 'observed_dB_minus_offset':
                    create_variable(
                        dataset, f'{prefix}_quantile_10', 'f8', dimensions,
                        stacked(statistics, 'quantile_10'), units='T')
                    create_variable(
                        dataset, f'{prefix}_quantile_90', 'f8', dimensions,
                        stacked(statistics, 'quantile_90'), units='T')
            dataset.processing_complete = 1

        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def plot_mem_statistic(
        path, grid, satellite_statistics,
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        statistic_name='median',
        colorbar_label='median scalar perturbation [nT]',
        cmap='RdBu_r', symmetric=True):
    """Plot one statistic for all 12 satellite/MEM pairs."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maps = {
        satellite: [
            apply_coverage_threshold(
                satellite_statistics[satellite][mem_id][statistic_name],
                satellite_statistics[satellite][mem_id]['count'],
                minimum_count=minimum_coverage_count)
            for mem_id in range(mem_count)]
        for satellite in satellite_ids}
    finite_values = np.concatenate([
        maps[satellite][mem_id][
            np.isfinite(maps[satellite][mem_id])]
        for satellite in satellite_ids for mem_id in range(mem_count)])
    if finite_values.size == 0:
        raise ValueError(
            f'No satellite/MEM {statistic_name} cells satisfy the '
            'coverage threshold')
    finite_values_nT = finite_values*1e9
    if symmetric:
        color_limit = np.percentile(np.abs(finite_values_nT), 98)
        if not np.isfinite(color_limit) or color_limit == 0:
            color_limit = 1.0
        color_range = (-color_limit, color_limit)
    else:
        color_range = tuple(np.percentile(finite_values_nT, (2, 98)))
        if not np.all(np.isfinite(color_range)):
            color_range = (0.0, 1.0)
        elif color_range[0] == color_range[1]:
            padding = max(abs(color_range[0])*0.01, 1.0)
            color_range = (
                color_range[0]-padding,
                color_range[1]+padding)

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    images = []
    for satellite_id, satellite in enumerate(satellite_ids):
        for mem_id in range(mem_count):
            ax = axes[satellite_id, mem_id]
            values = maps[satellite][mem_id]
            polar = Polarplot(
                ax, minlat=minimum_latitude, sector='all',
                lt_label='geographic longitude [h]', lat_label='latitude')
            image = polar.plotimg(
                grid.lat, (grid.lon/15)%24, values*1e9,
                crange=color_range, cmap=cmap)
            polar.coastlines(
                north=True, resolution='50m', color='black',
                linewidth=0.5, zorder=5)
            ax.set_title(f'Satellite {satellite.upper()}, MEM {mem_id+1}')
            images.append(image)
    colorbar = fig.colorbar(
        images[0], ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    colorbar.set_label(colorbar_label)
    fig.suptitle(
        f'No SML filtering; coverage $\\geq$ {minimum_coverage_count}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_mem_coverage(
        path, grid, satellite_statistics,
        minimum_latitude=minimum_latitude):
    """Plot coverage for all 12 satellite/MEM pairs with one count scale."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = max(
        np.max(satellite_statistics[satellite][mem_id]['count'])
        for satellite in satellite_ids for mem_id in range(mem_count))
    if maximum == 0:
        raise ValueError('No satellite/MEM coverage is available to plot')

    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    images = []
    for satellite_id, satellite in enumerate(satellite_ids):
        for mem_id in range(mem_count):
            ax = axes[satellite_id, mem_id]
            count = satellite_statistics[satellite][mem_id]['count']
            display_count = np.where(count > 0, count, np.nan)
            polar = Polarplot(
                ax, minlat=minimum_latitude, sector='all',
                lt_label='geographic longitude [h]', lat_label='latitude')
            image = polar.plotimg(
                grid.lat, (grid.lon/15)%24, display_count,
                crange=(0, maximum), cmap='viridis')
            polar.coastlines(
                north=True, resolution='50m', color='black',
                linewidth=0.5, zorder=5)
            ax.set_title(f'Satellite {satellite.upper()}, MEM {mem_id+1}')
            images.append(image)
    colorbar = fig.colorbar(
        images[0], ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
    colorbar.set_label('checkpoint estimates per cell')
    fig.suptitle('No SML filtering')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def build_mem_debug_maps(
        checkpoint_directory=input_base, output_path=output_file,
        median_plot_path=median_plot_file,
        coverage_plot_path=coverage_plot_file,
        track_offset_plot_path=track_offset_plot_file,
        observed_dB_plot_path=observed_dB_plot_file,
        observed_dB_minus_offset_plot_path=(
            observed_dB_minus_offset_plot_file),
        observed_dB_minus_offset_p10_plot_path=(
            observed_dB_minus_offset_p10_plot_file),
        observed_dB_minus_offset_p90_plot_path=(
            observed_dB_minus_offset_p90_plot_file),
        background_B_plot_path=background_B_plot_file,
        variable_name=scalar_variable,
        minimum_latitude=minimum_latitude,
        resolution_m=resolution_m,
        footprint_size_m=footprint_size_m,
        radius_m=earth_radius_m,
        minimum_coverage_count=minimum_coverage_count,
        show_progress=True):
    grid, field_statistics, metadata = aggregate_mem_maps(
        checkpoint_directory=checkpoint_directory,
        variable_name=variable_name,
        minimum_latitude=minimum_latitude,
        resolution_m=resolution_m,
        footprint_size_m=footprint_size_m,
        radius_m=radius_m,
        minimum_coverage_count=minimum_coverage_count,
        show_progress=show_progress)
    write_mem_debug_map(output_path, grid, field_statistics, metadata)
    plot_mem_statistic(
        median_plot_path, grid, field_statistics['prediction'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count)
    plot_mem_coverage(
        coverage_plot_path, grid, field_statistics['prediction'],
        minimum_latitude=minimum_latitude)
    plot_mem_statistic(
        track_offset_plot_path, grid, field_statistics['track_offset'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        colorbar_label='median fitted track offset [nT]')
    plot_mem_statistic(
        observed_dB_plot_path, grid, field_statistics['observed_dB'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        colorbar_label='median observed scalar perturbation [nT]')
    plot_mem_statistic(
        observed_dB_minus_offset_plot_path, grid,
        field_statistics['observed_dB_minus_offset'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        colorbar_label=(
            'median observed scalar perturbation minus fitted offset [nT]'))
    plot_mem_statistic(
        observed_dB_minus_offset_p10_plot_path, grid,
        field_statistics['observed_dB_minus_offset'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        statistic_name='quantile_10',
        colorbar_label=(
            '10th percentile observed scalar perturbation minus fitted '
            'offset [nT]'))
    plot_mem_statistic(
        observed_dB_minus_offset_p90_plot_path, grid,
        field_statistics['observed_dB_minus_offset'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        statistic_name='quantile_90',
        colorbar_label=(
            '90th percentile observed scalar perturbation minus fitted '
            'offset [nT]'))
    plot_mem_statistic(
        background_B_plot_path, grid, field_statistics['background_B'],
        minimum_latitude=minimum_latitude,
        minimum_coverage_count=minimum_coverage_count,
        colorbar_label='median background field magnitude [nT]',
        cmap='viridis', symmetric=False)
    return grid, field_statistics, metadata


#%% Command-line entry point

def nonnegative_integer(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create unfiltered per-satellite, per-MEM lithosphere maps'))
    parser.add_argument('--input', type=Path, default=input_base)
    parser.add_argument('--output', type=Path, default=output_file)
    parser.add_argument('--median-plot', type=Path, default=median_plot_file)
    parser.add_argument('--coverage-plot', type=Path, default=coverage_plot_file)
    parser.add_argument('--track-offset-plot', type=Path,
                        default=track_offset_plot_file)
    parser.add_argument('--observed-db-plot', type=Path,
                        default=observed_dB_plot_file)
    parser.add_argument('--observed-db-minus-offset-plot', type=Path,
                        default=observed_dB_minus_offset_plot_file)
    parser.add_argument('--observed-db-minus-offset-p10-plot', type=Path,
                        default=observed_dB_minus_offset_p10_plot_file)
    parser.add_argument('--observed-db-minus-offset-p90-plot', type=Path,
                        default=observed_dB_minus_offset_p90_plot_file)
    parser.add_argument('--background-b-plot', type=Path,
                        default=background_B_plot_file)
    parser.add_argument('--variable', default=scalar_variable)
    parser.add_argument('--minimum-latitude', type=float,
                        default=minimum_latitude)
    parser.add_argument('--resolution-km', type=float,
                        default=resolution_m/1000)
    parser.add_argument('--footprint-km', type=float,
                        default=footprint_size_m/1000)
    parser.add_argument('--minimum-coverage-count', type=nonnegative_integer,
                        default=minimum_coverage_count)
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, metadata = build_mem_debug_maps(
        checkpoint_directory=args.input,
        output_path=args.output,
        median_plot_path=args.median_plot,
        coverage_plot_path=args.coverage_plot,
        track_offset_plot_path=args.track_offset_plot,
        observed_dB_plot_path=args.observed_db_plot,
        observed_dB_minus_offset_plot_path=(
            args.observed_db_minus_offset_plot),
        observed_dB_minus_offset_p10_plot_path=(
            args.observed_db_minus_offset_p10_plot),
        observed_dB_minus_offset_p90_plot_path=(
            args.observed_db_minus_offset_p90_plot),
        background_B_plot_path=args.background_b_plot,
        variable_name=args.variable,
        minimum_latitude=args.minimum_latitude,
        resolution_m=args.resolution_km*1000,
        footprint_size_m=args.footprint_km*1000,
        minimum_coverage_count=args.minimum_coverage_count)
    paths = (
        args.output, args.median_plot, args.coverage_plot,
        args.track_offset_plot, args.observed_db_plot,
        args.observed_db_minus_offset_plot,
        args.observed_db_minus_offset_p10_plot,
        args.observed_db_minus_offset_p90_plot,
        args.background_b_plot)
    print(f"Wrote {', '.join(str(path) for path in paths)}")
    print(
        f"Processed {metadata['processed_file_count']} checkpoints "
        'without SML filtering')


if __name__ == '__main__':
    main()
