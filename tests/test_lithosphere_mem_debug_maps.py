import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.lithosphere_mem_debug_maps import (
    build_mem_debug_maps,
    calculate_quantiles)


class LithosphereMemDebugMapTest(unittest.TestCase):
    def test_checkpoint_quantiles_are_calculated_per_cell(self):
        quantiles = calculate_quantiles(
            flat_ids=np.array([0, 0, 0, 1, 1]),
            values=np.array([0.0, 10.0, 20.0, -4.0, 6.0]),
            shape=(1, 3), quantiles=(0.1, 0.9))

        np.testing.assert_allclose(quantiles[0.1][0, :2], [2.0, -3.0])
        np.testing.assert_allclose(quantiles[0.9][0, :2], [18.0, 5.0])
        self.assertTrue(np.isnan(quantiles[0.1][0, 2]))
        self.assertTrue(np.isnan(quantiles[0.9][0, 2]))

    def test_satellite_mem_maps_are_aggregated_separately_without_sml(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for satellite_id, satellite in enumerate(('a', 'b', 'c')):
                satellite_offset = satellite_id*100e-9
                self._write_checkpoint(
                    root/satellite/'first.nc',
                    base_value=satellite_offset+1e-9,
                    satellite=satellite)
                self._write_checkpoint(
                    root/satellite/'second.nc',
                    base_value=satellite_offset+5e-9,
                    satellite=satellite)
            self._write_checkpoint(
                root/'incomplete.nc', base_value=500e-9,
                satellite='a', complete=0)
            output = root/'mem_debug.nc'
            median_plot = root/'mem_medians.png'
            coverage_plot = root/'mem_coverage.png'
            track_offset_plot = root/'track_offset.png'
            observed_dB_plot = root/'observed_dB.png'
            observed_dB_minus_offset_plot = (
                root/'observed_dB_minus_offset.png')
            observed_dB_minus_offset_p10_plot = (
                root/'observed_dB_minus_offset_p10.png')
            observed_dB_minus_offset_p90_plot = (
                root/'observed_dB_minus_offset_p90.png')
            background_B_plot = root/'background_B.png'

            _, statistics, metadata = build_mem_debug_maps(
                checkpoint_directory=root,
                output_path=output,
                median_plot_path=median_plot,
                coverage_plot_path=coverage_plot,
                track_offset_plot_path=track_offset_plot,
                observed_dB_plot_path=observed_dB_plot,
                observed_dB_minus_offset_plot_path=(
                    observed_dB_minus_offset_plot),
                observed_dB_minus_offset_p10_plot_path=(
                    observed_dB_minus_offset_p10_plot),
                observed_dB_minus_offset_p90_plot_path=(
                    observed_dB_minus_offset_p90_plot),
                background_B_plot_path=background_B_plot,
                footprint_size_m=10_000.0,
                minimum_coverage_count=2,
                show_progress=False)

            self.assertEqual(metadata['processed_file_count'], 6)
            self.assertEqual(metadata['incomplete_file_count'], 1)
            self.assertEqual(metadata['sml_filter_applied'], 0)
            self.assertEqual(
                tuple(statistics),
                ('prediction', 'track_offset', 'observed_dB',
                 'observed_dB_minus_offset', 'background_B'))
            for satellite_id, satellite in enumerate(('a', 'b', 'c')):
                self.assertEqual(len(statistics['prediction'][satellite]), 4)
                for mem_id, result in enumerate(
                        statistics['prediction'][satellite]):
                    populated = result['count'] == 2
                    self.assertTrue(np.any(populated))
                    expected = (
                        satellite_id*100e-9 + 4e-9 + mem_id*10e-9)
                    np.testing.assert_allclose(
                        result['median'][populated], expected)
                    np.testing.assert_allclose(
                        statistics['observed_dB'][satellite][mem_id]
                        ['median'][populated],
                        expected+100e-9)
                    np.testing.assert_allclose(
                        statistics['background_B'][satellite][mem_id]
                        ['median'][populated],
                        expected+40_000e-9)
                    np.testing.assert_allclose(
                        statistics['track_offset'][satellite][mem_id]
                        ['median'][populated],
                        satellite_id*100e-9 + 3e-9 + mem_id*10e-9)
                    np.testing.assert_allclose(
                        statistics['observed_dB_minus_offset'][satellite]
                        [mem_id]['median'][populated], 101e-9)
                    np.testing.assert_allclose(
                        statistics['observed_dB_minus_offset'][satellite]
                        [mem_id]['quantile_10'][populated], 101e-9)
                    np.testing.assert_allclose(
                        statistics['observed_dB_minus_offset'][satellite]
                        [mem_id]['quantile_90'][populated], 101e-9)

            with Dataset(output) as dataset:
                self.assertEqual(dataset.processing_complete, 1)
                self.assertEqual(dataset.sml_filter_applied, 0)
                np.testing.assert_array_equal(
                    dataset['satellite'][:], np.asarray(('a', 'b', 'c'),
                                                       dtype='S1'))
                self.assertEqual(dataset['scalar_median'].shape[:2], (3, 4))
                self.assertEqual(
                    dataset['track_offset_median'].shape[:2], (3, 4))
                self.assertEqual(
                    dataset['observed_dB_median'].shape[:2], (3, 4))
                self.assertEqual(
                    dataset['observed_dB_minus_offset_median'].shape[:2],
                    (3, 4))
                self.assertEqual(
                    dataset['observed_dB_minus_offset_quantile_10'].shape[:2],
                    (3, 4))
                self.assertEqual(
                    dataset['observed_dB_minus_offset_quantile_90'].shape[:2],
                    (3, 4))
                self.assertEqual(
                    dataset['background_B_median'].shape[:2], (3, 4))
            self.assertTrue(median_plot.is_file())
            self.assertTrue(coverage_plot.is_file())
            self.assertTrue(track_offset_plot.is_file())
            self.assertTrue(observed_dB_plot.is_file())
            self.assertTrue(observed_dB_minus_offset_plot.is_file())
            self.assertTrue(observed_dB_minus_offset_p10_plot.is_file())
            self.assertTrue(observed_dB_minus_offset_p90_plot.is_file())
            self.assertTrue(background_B_plot.is_file())

    @staticmethod
    def _write_checkpoint(path, base_value, satellite, complete=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        with Dataset(path, 'w') as dataset:
            dataset.processing_complete = complete
            dataset.schema_version = '1.1'
            dataset.background = 'igrf'
            dataset.satellite = satellite
            dataset.createDimension('mem', 4)
            dataset.createDimension('sample', 2)
            latitude = dataset.createVariable(
                'latitude', 'f8', ('mem', 'sample'))
            longitude = dataset.createVariable(
                'longitude', 'f8', ('mem', 'sample'))
            scalar = dataset.createVariable(
                'predicted_dB_exact', 'f8', ('mem', 'sample'))
            observed = dataset.createVariable(
                'observed_dB', 'f8', ('mem', 'sample'))
            background = dataset.createVariable(
                'background_B', 'f8', ('mem', 'sample'))
            track_offset = dataset.createVariable(
                'track_offset', 'f8', ('mem',))
            latitude[:] = 70.0
            longitude[:] = 15.0
            offsets = np.arange(4, dtype=float)[:, None]*10e-9
            sample_values = np.array([0.0, 2e-9])[None, :]
            scalar[:] = base_value + offsets + sample_values
            observed[:] = base_value + offsets + sample_values + 100e-9
            background[:] = base_value + offsets + sample_values + 40_000e-9
            track_offset[:] = base_value + offsets[:, 0]
            for variable in (scalar, observed, background, track_offset):
                variable.units = 'T'


if __name__ == '__main__':
    unittest.main()
