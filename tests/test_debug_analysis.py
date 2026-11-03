import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.debug_analysis import (
    aggregate_debug_analysis,
    midnight_plot_window,
    plot_statistic_panels,
    read_debug_analysis,
    write_debug_analysis)


class DebugAnalysisTest(unittest.TestCase):
    def test_activity_direction_and_quantiles_remain_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quiet_source = root/'quiet_source.nc4'
            active_source = root/'active_source.nc4'
            self._write_source(quiet_source)
            self._write_source(active_source)
            self._write_checkpoint(
                root/'quiet.nc', quiet_source, time_seconds=60,
                sample_offset_nt=0)
            self._write_checkpoint(
                root/'active.nc', active_source, time_seconds=120,
                sample_offset_nt=10)
            sml_data = (
                np.array([0, 60, 120, 180]),
                np.array([-20.0, -20.0, -300.0, -300.0]),
                np.zeros(4))

            latitude_edges, mlt_edges, result, metadata = (
                aggregate_debug_analysis(
                    checkpoint_directory=root,
                    sml_data=sml_data,
                    show_progress=False))

            quiet_ascending = result['count'][0, 0, 0, 0]
            active_descending = result['count'][0, 0, 1, 1]
            self.assertEqual(np.max(quiet_ascending), 1)
            self.assertEqual(np.max(active_descending), 1)
            self.assertEqual(metadata['quiet_sample_count'], 16)
            self.assertEqual(metadata['active_sample_count'], 16)

            quiet_values = result['median'][0, 0, 0, 0]
            active_values = result['median'][0, 0, 1, 1]
            np.testing.assert_allclose(
                quiet_values[quiet_ascending == 1], 1e-9)
            np.testing.assert_allclose(
                active_values[active_descending == 1], 16e-9)

            output = root/'analysis.nc'
            write_debug_analysis(
                output, latitude_edges, mlt_edges, result, metadata)
            with Dataset(output) as dataset:
                self.assertEqual(dataset.processing_complete, 1)
                self.assertEqual(
                    dataset['corrected_observed_dB_median'].shape[:4],
                    (3, 4, 2, 2))

            loaded_edges, loaded_mlt, loaded_result, loaded_metadata = (
                read_debug_analysis(output))
            np.testing.assert_array_equal(loaded_edges, latitude_edges)
            np.testing.assert_array_equal(loaded_mlt, mlt_edges)
            np.testing.assert_array_equal(
                loaded_result['count'], result['count'])
            self.assertEqual(
                loaded_metadata['processed_file_count'],
                metadata['processed_file_count'])

            figure = root/'quiet_ascending_median.png'
            plot_statistic_panels(
                figure, latitude_edges, mlt_edges,
                result['median'][:, :, 0, 0],
                result['count'][:, :, 0, 0],
                'quiet', 'ascending', 'median', metadata,
                minimum_count=1)
            self.assertTrue(figure.is_file())

    def test_midnight_plot_window_joins_late_and_early_mlt(self):
        mlt_edges = np.arange(0, 25, dtype=float)
        values = np.arange(24)

        plot_edges, (reordered,) = midnight_plot_window(
            mlt_edges, values, minimum_hour=-2.5, maximum_hour=2.5)

        np.testing.assert_array_equal(reordered, [21, 22, 23, 0, 1, 2])
        np.testing.assert_array_equal(plot_edges, np.arange(-3, 4))

    @staticmethod
    def _write_source(path):
        with Dataset(path, 'w') as dataset:
            dataset.createDimension('sample', 4)
            magnetic = dataset.createGroup('MagneticCoords')
            geolocation = dataset.createGroup('Geolocation')
            satellite_latitude = geolocation.createVariable(
                'sat_lat', 'f8', ('sample',))
            satellite_latitude[:] = [60.0, 61.0, 62.0, 61.0]
            for mem_number in range(1, 5):
                magnetic_latitude = magnetic.createVariable(
                    f'obs_maglat{mem_number}', 'f8', ('sample',))
                mlt = magnetic.createVariable(
                    f'obs_magLT{mem_number}', 'f8', ('sample',))
                magnetic_latitude[:] = [60.2, 60.2, 61.2, 61.2]
                mlt[:] = [1.2, 1.2, 2.2, 2.2]

    @staticmethod
    def _write_checkpoint(path, source_path, time_seconds, sample_offset_nt):
        with Dataset(path, 'w') as dataset:
            dataset.processing_complete = 1
            dataset.schema_version = '1.1'
            dataset.source_path = str(source_path)
            dataset.satellite = 'a'
            dataset.background = 'igrf'
            dataset.scalar_source = 'retrieved'
            dataset.createDimension('mem', 4)
            dataset.createDimension('sample', 4)
            time = dataset.createVariable(
                'time', 'i8', ('mem', 'sample'))
            sample_count = dataset.createVariable(
                'sample_count', 'i4', ('mem',))
            observed = dataset.createVariable(
                'observed_dB', 'f8', ('mem', 'sample'))
            offset = dataset.createVariable(
                'track_offset', 'f8', ('mem',))
            time[:] = time_seconds*1_000_000_000
            sample_count[:] = 4
            sample_values = (
                sample_offset_nt + np.array([0.0, 2.0, 4.0, 6.0]))*1e-9
            track_values = np.arange(4)*10e-9
            observed[:] = track_values[:, None]+sample_values[None, :]
            offset[:] = track_values
            observed.units = 'T'
            offset.units = 'T'


if __name__ == '__main__':
    unittest.main()
