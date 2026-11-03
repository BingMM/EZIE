import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from scripts.lithosphere_global_map import (
    add_checkpoint_footprints,
    apply_coverage_threshold,
    bin_scalar_samples,
    build_scalar_map,
    calculate_circular_median,
    calculate_equal_mem_median,
    calculate_median,
    calculate_sml_derivative,
    create_global_grid,
    download_sml,
    finalize_statistics,
    gaussian_filter_nan,
    initialize_footprint_statistics,
    initialize_statistics,
    quiet_sml_mask,
    rasterize_checkpoint_footprints,
    write_sml_cache,
)


class LithosphereGlobalMapTest(unittest.TestCase):
    def test_circular_mlt_median_wraps_at_24_hours(self):
        median = calculate_circular_median(
            np.array([0, 0]), np.array([23.0, 1.0]), (1, 1))

        self.assertTrue(np.isclose(median[0, 0], 0.0, atol=1e-12))

    def test_sml_download_retries_failed_chunk(self):
        class Column:
            def __init__(self, values):
                self.values = np.asarray(values)

            def to_numpy(self, dtype):
                return self.values.astype(dtype)

        class Response:
            empty = False
            columns = {'tval', 'SML'}

            def __getitem__(self, name):
                return {
                    'tval': Column([0, 60]),
                    'SML': Column([-20.0, -30.0])}[name]

        with patch(
                'scripts.lithosphere_global_map.SuperMAGGetIndices',
                side_effect=[(0, None), (1, Response())]) as request:
            with patch('scripts.lithosphere_global_map.sleep') as pause:
                time, sml = download_sml(
                    0, 120, maximum_attempts=2, retry_delay_seconds=0)

        self.assertEqual(request.call_count, 2)
        pause.assert_called_once_with(0)
        np.testing.assert_array_equal(time, [0, 60])
        np.testing.assert_allclose(sml, [-20.0, -30.0])

    def test_all_data_bypasses_sml_filtering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_checkpoint(
                root/'a'/'first.nc', [1e-9, 3e-9], complete=1)
            output = root.parent/f'{root.name}_all.nc'
            footprint_plot = root.parent/f'{root.name}_footprint_all.png'

            try:
                _, _, metadata = build_scalar_map(
                    checkpoint_directory=root,
                    output_path=output,
                    all_data=True,
                    minimum_coverage_count=1,
                    sml_cache_path=None,
                    footprint_plot_path=footprint_plot,
                    show_progress=False)

                self.assertEqual(metadata['sample_selection'], 'all data')
                self.assertEqual(metadata['sml_filter_applied'], 0)
                self.assertEqual(metadata['selected_sample_count'], 2)
                self.assertEqual(metadata['quiet_sample_count'], -1)
                self.assertEqual(metadata['sml_matched_sample_count'], -1)
                with Dataset(output) as dataset:
                    self.assertEqual(dataset.sample_selection, 'all data')
                    self.assertEqual(np.sum(dataset['sample_count'][:]), 2)
                    self.assertIn(
                        'scalar_footprint_equal_mem_median', dataset.variables)
                self.assertTrue(footprint_plot.is_file())
            finally:
                output.unlink(missing_ok=True)
                footprint_plot.unlink(missing_ok=True)

    def test_equal_mem_median_is_not_dominated_by_checkpoint_count(self):
        flat_ids = np.zeros(4, dtype=int)
        values = np.array([-100.0, 0.0, 100.0, 10.0])
        stream_ids = np.array([0, 0, 0, 4])

        median, represented_count = calculate_equal_mem_median(
            flat_ids, values, stream_ids, (1, 1))

        np.testing.assert_allclose(median, 10.0)
        np.testing.assert_array_equal(represented_count, 2)

    def test_sml_filter_uses_absolute_level_and_derivative(self):
        minute = 60*1_000_000_000
        observation_time = np.arange(3, dtype=np.int64)*minute
        sml_time = np.arange(3, dtype=np.int64)*60
        sml = np.array([-50.0, -90.0, -60.0])
        derivative = np.array([5.0, 5.0, 15.0])

        quiet, matched = quiet_sml_mask(
            observation_time, sml_time, sml, derivative,
            sml_limit_nt=80.0, derivative_limit_nt_per_minute=10.0)

        np.testing.assert_array_equal(matched, [True, True, True])
        np.testing.assert_array_equal(quiet, [True, False, False])

    def test_gaussian_filter_preserves_constant_values_and_masks_cap(self):
        values = np.full((7, 7), np.nan)
        values[2:5, 2:5] = 4.0
        cap = np.ones(values.shape, dtype=bool)
        cap[0, :] = False

        filtered = gaussian_filter_nan(values, sigma=1.0, valid_mask=cap)

        np.testing.assert_allclose(filtered[np.isfinite(filtered)], 4.0)
        self.assertTrue(np.all(np.isnan(filtered[0, :])))

    def test_coverage_threshold_masks_under_supported_cells(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        count = np.array([[9, 10], [20, 0]])

        masked = apply_coverage_threshold(values, count, minimum_count=10)

        self.assertTrue(np.isnan(masked[0, 0]))
        self.assertTrue(np.isnan(masked[1, 1]))
        np.testing.assert_allclose(masked[[0, 1], [1, 0]], [2.0, 3.0])

    def test_sml_derivative_does_not_bridge_time_gaps(self):
        time = np.array([0, 60, 120, 600, 660])
        sml = np.array([-50.0, -55.0, -60.0, -60.0, -65.0])

        derivative = calculate_sml_derivative(time, sml)

        np.testing.assert_allclose(derivative[[0, 1, 4]], -5.0)
        self.assertTrue(np.isnan(derivative[2]))
        self.assertTrue(np.isnan(derivative[3]))

    def test_global_grid_covers_northern_cap_at_nominal_resolution(self):
        grid = create_global_grid(
            minimum_latitude=50.0, resolution_m=50_000.0)
        longitude = np.linspace(-180.0, 180.0, 721)
        latitude = np.full(longitude.shape, 50.0)
        xi, eta = grid.projection.geo2cube(longitude, latitude)

        self.assertTrue(np.all(xi >= grid.xi_min-1e-12))
        self.assertTrue(np.all(xi <= grid.xi_max+1e-12))
        self.assertTrue(np.all(eta >= grid.eta_min-1e-12))
        self.assertTrue(np.all(eta <= grid.eta_max+1e-12))
        self.assertAlmostEqual(grid.dxi*grid.R, 50_000.0, delta=500.0)

        statistics = initialize_statistics(grid)
        _, binned_count = bin_scalar_samples(
            grid, statistics, latitude, longitude, np.ones(longitude.size),
            minimum_latitude=50.0)
        self.assertEqual(binned_count, longitude.size)

    def test_scalar_statistics_are_binned_and_finalized(self):
        grid = create_global_grid(
            minimum_latitude=50.0, resolution_m=50_000.0)
        statistics = initialize_statistics(grid)
        latitude = np.array([70.0, 70.0, 49.0, np.nan])
        longitude = np.array([15.0, 15.0, 15.0, 15.0])
        values = np.array([1e-9, 3e-9, 100e-9, 5e-9])

        input_count, binned_count = bin_scalar_samples(
            grid, statistics, latitude, longitude, values,
            minimum_latitude=50.0)
        populated_id = np.flatnonzero(statistics['count'].ravel())[0]
        median = calculate_median(
            np.array([populated_id, populated_id]),
            np.array([1e-9, 3e-9]), grid.shape)
        result = finalize_statistics(
            statistics, grid, median, minimum_latitude=50.0)

        self.assertEqual(input_count, 3)
        self.assertEqual(binned_count, 2)
        self.assertEqual(np.sum(result['count']), 2)
        populated = result['count'] == 2
        np.testing.assert_allclose(result['mean'][populated], 2e-9)
        np.testing.assert_allclose(result['median'][populated], 2e-9)
        np.testing.assert_allclose(
            result['standard_deviation'][populated], np.sqrt(2)*1e-9)

    def test_square_footprint_is_allocated_by_cell_overlap(self):
        grid = create_global_grid(
            minimum_latitude=50.0, resolution_m=50_000.0)
        edge_id = grid.shape[0]//2+20
        xi = grid.xi_mesh[0, edge_id]
        eta = grid.eta_mesh[edge_id, 0]
        longitude, latitude = grid.projection.cube2geo(xi, eta)

        flat_ids, values, weights, contributions = (
            rasterize_checkpoint_footprints(
                grid, [latitude], [longitude], [2e-9],
                footprint_size_m=50_000.0,
                minimum_latitude=50.0))

        self.assertEqual(flat_ids.size, 4)
        np.testing.assert_allclose(values, 2e-9)
        np.testing.assert_allclose(weights, 0.25, atol=1e-12)
        np.testing.assert_array_equal(contributions, 1)

    def test_correlated_samples_become_one_checkpoint_estimate_per_cell(self):
        grid = create_global_grid(
            minimum_latitude=50.0, resolution_m=50_000.0)
        statistics = initialize_footprint_statistics(grid)
        cell_id = grid.shape[0]//2+20
        longitude, latitude = grid.projection.cube2geo(
            grid.xi[cell_id, cell_id], grid.eta[cell_id, cell_id])

        flat_ids, values = add_checkpoint_footprints(
            grid, statistics,
            latitude=[latitude, latitude], longitude=[longitude, longitude],
            values=[1e-9, 3e-9], footprint_size_m=10_000.0,
            minimum_latitude=50.0)

        self.assertEqual(flat_ids.size, 1)
        np.testing.assert_allclose(values, 2e-9)
        self.assertEqual(statistics['count'].ravel()[flat_ids[0]], 1)
        self.assertEqual(
            statistics['contribution_count'].ravel()[flat_ids[0]], 2)

    def test_complete_checkpoints_are_aggregated_to_netcdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_checkpoint(
                root/'a'/'first.nc', [1e-9, 3e-9], complete=1,
                mlt=[23.0, 1.0])
            self._write_checkpoint(
                root/'b'/'second.nc', [5e-9], complete=1, mlt=[5.0])
            self._write_checkpoint(root/'c'/'incomplete.nc', [100e-9], complete=0)
            output = root.parent/f'{root.name}_map.nc'
            plot = root.parent/f'{root.name}_median.png'
            footprint_plot = root.parent/f'{root.name}_median_footprint.png'
            equal_mem_plot = root.parent/f'{root.name}_median_equal_mem.png'
            mlt_plot = root.parent/f'{root.name}_median_mlt.png'
            coverage_plot = root.parent/f'{root.name}_coverage.png'
            sml_cache = root.parent/f'{root.name}_sml.nc'
            base_seconds = 1_700_000_000
            sml_time = np.arange(base_seconds-60, base_seconds+241, 60)
            write_sml_cache(
                sml_cache, sml_time, np.full(sml_time.shape, -50.0))

            try:
                _, _, metadata = build_scalar_map(
                    checkpoint_directory=root,
                    output_path=output,
                    minimum_latitude=50.0,
                    resolution_m=50_000.0,
                    minimum_coverage_count=2,
                    sml_cache_path=sml_cache,
                    plot_path=plot,
                    footprint_plot_path=footprint_plot,
                    equal_mem_plot_path=equal_mem_plot,
                    mlt_plot_path=mlt_plot,
                    coverage_plot_path=coverage_plot,
                    show_progress=False)

                self.assertEqual(metadata['processed_file_count'], 2)
                self.assertEqual(metadata['incomplete_file_count'], 1)
                self.assertEqual(metadata['binned_sample_count'], 3)
                with Dataset(output) as dataset:
                    self.assertEqual(dataset.processing_complete, 1)
                    self.assertEqual(np.sum(dataset['sample_count'][:]), 3)
                    populated = dataset['sample_count'][:] == 3
                    np.testing.assert_allclose(
                        dataset['scalar_mean'][:][populated], 3e-9)
                    np.testing.assert_allclose(
                        dataset['scalar_median'][:][populated], 3e-9)
                    self.assertIn(
                        'scalar_median_gaussian_sigma1', dataset.variables)
                    self.assertIn('scalar_footprint_median', dataset.variables)
                    self.assertIn(
                        'scalar_footprint_equal_mem_median', dataset.variables)
                    self.assertIn(
                        'footprint_represented_mem_count', dataset.variables)
                    self.assertEqual(dataset.plot_minimum_coverage_count, 2)
                    self.assertIn('plot_coverage_mask', dataset.variables)
                    footprint_count = dataset['footprint_checkpoint_count'][:]
                    footprint_populated = footprint_count == 2
                    self.assertTrue(np.any(footprint_populated))
                    np.testing.assert_allclose(
                        dataset['scalar_footprint_median'][:][footprint_populated],
                        3.5e-9)
                    np.testing.assert_allclose(
                        dataset['scalar_footprint_equal_mem_median'][:][
                            footprint_populated],
                        3.5e-9)
                    np.testing.assert_array_equal(
                        dataset['footprint_represented_mem_count'][:][
                            footprint_populated],
                        2)
                    np.testing.assert_allclose(
                        dataset['footprint_median_mlt'][:][footprint_populated],
                        2.5)
                self.assertTrue(plot.is_file())
                self.assertTrue(footprint_plot.is_file())
                self.assertTrue(equal_mem_plot.is_file())
                self.assertTrue(mlt_plot.is_file())
                self.assertTrue(coverage_plot.is_file())
            finally:
                output.unlink(missing_ok=True)
                plot.unlink(missing_ok=True)
                footprint_plot.unlink(missing_ok=True)
                equal_mem_plot.unlink(missing_ok=True)
                mlt_plot.unlink(missing_ok=True)
                coverage_plot.unlink(missing_ok=True)
                sml_cache.unlink(missing_ok=True)

    @staticmethod
    def _write_checkpoint(path, values, complete, mlt=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        values = np.asarray(values)
        with Dataset(path, 'w') as dataset:
            dataset.processing_complete = complete
            dataset.schema_version = '1.1'
            dataset.satellite = path.parent.name
            dataset.createDimension('mem', 1)
            dataset.createDimension('sample', values.size)
            latitude = dataset.createVariable(
                'latitude', 'f8', ('mem', 'sample'))
            longitude = dataset.createVariable(
                'longitude', 'f8', ('mem', 'sample'))
            scalar = dataset.createVariable(
                'predicted_dB_exact', 'f8', ('mem', 'sample'))
            magnetic_local_time = dataset.createVariable(
                'magnetic_local_time', 'f8', ('mem', 'sample'))
            latitude[:] = 70.0
            longitude[:] = 15.0
            scalar[:] = values.reshape(1, -1)
            magnetic_local_time[:] = (
                np.zeros(values.size) if mlt is None else mlt)
            scalar.units = 'T'
            magnetic_local_time.units = 'hours'
            time = dataset.createVariable('time', 'i8', ('mem', 'sample'))
            base_ns = 1_700_000_000*1_000_000_000
            time[:] = base_ns + np.arange(values.size)*10_000_000_000
            time.units = 'nanoseconds since 1970-01-01 00:00:00 UTC'


if __name__ == '__main__':
    unittest.main()
