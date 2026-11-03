import tempfile
import unittest
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from scripts.lithosphere_global_map import write_sml_cache
from scripts.lithosphere_sml_histograms import (
    build_sml_histograms,
    nearest_sml_indices,
)


class LithosphereSmlHistogramTest(unittest.TestCase):
    def test_nearest_sml_indices_apply_tolerance(self):
        minute = 60*1_000_000_000
        observation_time = np.array([10, 70, 500], dtype=np.int64)*1_000_000_000
        sml_time = np.array([0, 60, 120], dtype=np.int64)

        nearest, matched = nearest_sml_indices(
            observation_time, sml_time, tolerance_seconds=60)

        np.testing.assert_array_equal(nearest, [0, 1, 2])
        np.testing.assert_array_equal(matched, [True, True, False])

    def test_histograms_use_unique_data_covered_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_seconds = 1_700_000_000
            self._write_checkpoint(
                root/'a'/'first.nc', base_seconds, [0, 10, 70])
            self._write_checkpoint(
                root/'b'/'second.nc', base_seconds, [0, 10])
            self._write_checkpoint(
                root/'c'/'incomplete.nc', base_seconds, [120], complete=0)

            sml_time = np.arange(base_seconds-60, base_seconds+241, 60)
            sml = np.array([-40.0, -50.0, -60.0, -90.0, -70.0, -60.0])
            sml_cache = root/'sml.nc'
            write_sml_cache(sml_cache, sml_time, sml)
            output = root/'histograms.png'

            _, _, summary = build_sml_histograms(
                checkpoint_directory=root,
                output_path=output,
                sml_cache_path=sml_cache,
                show_progress=False,
                bins=5)

            self.assertTrue(output.is_file())
            self.assertEqual(summary['processed_file_count'], 2)
            self.assertEqual(summary['incomplete_file_count'], 1)
            self.assertEqual(summary['observation_count'], 5)
            self.assertEqual(summary['matched_observation_count'], 5)
            self.assertEqual(summary['unique_minute_count'], 2)

    @staticmethod
    def _write_checkpoint(path, base_seconds, offsets, complete=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.asarray(offsets, dtype=np.int64)
        with Dataset(path, 'w') as dataset:
            dataset.processing_complete = complete
            dataset.createDimension('sample', offsets.size)
            time = dataset.createVariable('time', 'i8', ('sample',))
            time[:] = (base_seconds+offsets)*1_000_000_000
            time.units = 'nanoseconds since 1970-01-01 00:00:00 UTC'


if __name__ == '__main__':
    unittest.main()
