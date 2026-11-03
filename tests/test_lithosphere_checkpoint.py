import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

import scripts.lithosphere_l3 as lithosphere_l3
from scripts.lithosphere_l3 import (
    checkpoint_is_complete,
    datetime_to_ns,
    output_base_for_background,
    reconstruct_grid,
    required_checkpoint_variables,
    satellite_from_filename,
    schema_version
)


class LithosphereCheckpointTest(unittest.TestCase):
    def test_backgrounds_use_independent_output_directories(self):
        input_base = Path('/data/EZIE_L2')

        self.assertEqual(
            output_base_for_background(input_base, 'igrf'),
            Path('/data/EZIE_lithosphere_l3'))
        self.assertEqual(
            output_base_for_background(input_base, 'chaos'),
            Path('/data/EZIE_lithosphere_l3_chaos'))

    def test_batch_collects_cycles_after_success_and_failure(self):
        class Progress:
            def update(self, count=1):
                pass

            def set_postfix(self, **kwargs):
                pass

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'one.nc4').touch()
            (root/'two.nc4').touch()

            with patch.object(lithosphere_l3, 'base', root), \
                    patch.object(lithosphere_l3, 'output_base', root/'output'), \
                    patch.object(lithosphere_l3, 'get_code_provenance', return_value=None), \
                    patch.object(lithosphere_l3, 'checkpoint_is_complete', return_value=False), \
                    patch.object(lithosphere_l3, 'process_file',
                                 side_effect=[None, RuntimeError('failed')]), \
                    patch.object(lithosphere_l3, 'append_failure') as append_failure, \
                    patch.object(lithosphere_l3.gc, 'collect') as collect, \
                    patch.object(lithosphere_l3, 'tqdm', return_value=Progress()):
                lithosphere_l3.main(worker_count=1)

            self.assertEqual(collect.call_count, 2)
            append_failure.assert_called_once()

    def test_parallel_batch_uses_requested_workers_and_blas_limit(self):
        class Progress:
            def update(self, count=1):
                pass

            def set_postfix(self, **kwargs):
                pass

            def close(self):
                pass

        class Future:
            def __init__(self, value):
                self.value = value

            def result(self):
                return self.value

        executor_settings = {}

        class Executor:
            def __init__(self, max_workers, mp_context):
                executor_settings['max_workers'] = max_workers
                executor_settings['start_method'] = mp_context.get_start_method()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, function, *args):
                return Future(function(*args))

        observed_blas_threads = []

        def process_file(*args):
            observed_blas_threads.append(os.environ['OPENBLAS_NUM_THREADS'])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'one.nc4').touch()
            (root/'two.nc4').touch()

            with patch.object(lithosphere_l3, 'base', root), \
                    patch.object(lithosphere_l3, 'output_base', root/'output'), \
                    patch.object(lithosphere_l3, 'get_code_provenance', return_value=None), \
                    patch.object(lithosphere_l3, 'checkpoint_is_complete', return_value=False), \
                    patch.object(lithosphere_l3, 'process_file', side_effect=process_file), \
                    patch.object(lithosphere_l3, 'ProcessPoolExecutor', Executor), \
                    patch.object(lithosphere_l3, 'as_completed', side_effect=lambda values: values), \
                    patch.object(lithosphere_l3.gc, 'collect'), \
                    patch.object(lithosphere_l3, 'tqdm', return_value=Progress()):
                counts = lithosphere_l3.main(
                    worker_count=3, requested_blas_threads=4)

        self.assertEqual(executor_settings['max_workers'], 2)
        self.assertEqual(executor_settings['start_method'], 'spawn')
        self.assertEqual(observed_blas_threads, ['4', '4'])
        self.assertEqual(counts, {'completed': 2, 'skipped': 0, 'failed': 0})

    def test_satellite_is_parsed_from_filename(self):
        file = Path('ezie_l2_20250515_022454_svb_v003_r000.nc4')

        self.assertEqual(satellite_from_filename(file), 'b')

    def test_datetime_conversion_preserves_utc(self):
        times = [datetime(1970, 1, 1),
                 datetime(1970, 1, 1, 0, 0, 1, tzinfo=timezone.utc)]

        np.testing.assert_array_equal(
            datetime_to_ns(times), np.array([0, 1_000_000_000]))

    def test_complete_checkpoint_requires_identity_and_variables(self):
        source = Path('ezie_l2_20250515_022454_sva_v003_r000.nc4')

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.nc'
            with Dataset(output, 'w') as dataset:
                dataset.processing_complete = 1
                dataset.schema_version = schema_version
                dataset.source_name = source.name
                dataset.background = 'igrf'
                dataset.createDimension('one', 1)
                for name in required_checkpoint_variables:
                    variable = dataset.createVariable(name, 'f8', ('one',))
                    variable[:] = 0

            self.assertTrue(checkpoint_is_complete(
                output, source, expected_background='igrf'))

            self.assertFalse(checkpoint_is_complete(
                output, source, expected_background='chaos'))

            with Dataset(output, 'a') as dataset:
                dataset.processing_complete = 0

            self.assertFalse(checkpoint_is_complete(output, source))

    def test_grid_can_be_reconstructed_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.nc'
            with Dataset(output, 'w') as dataset:
                dataset.grid_L_m = 2_000_000.0
                dataset.grid_L_m_dtype = '<f8'
                dataset.grid_W_m = 5_771_787.125
                dataset.grid_W_m_dtype = '<f4'
                dataset.grid_Lres_m = 20_000.0
                dataset.grid_Lres_m_dtype = '<f8'
                dataset.grid_Wres_m = 40_000.0
                dataset.grid_Wres_m_dtype = '<f8'
                dataset.grid_wshift = 0.0
                dataset.grid_wshift_dtype = '<i8'
                dataset.RI_m = 6_481_200.0
                dataset.RI_m_dtype = '<f8'
                dataset.projection_position_dtype = '<f4'
                dataset.projection_orientation_dtype = '<f8'
                dataset.createDimension('horizontal_vector', 2)
                position = dataset.createVariable(
                    'projection_position', 'f8', ('horizontal_vector',))
                orientation = dataset.createVariable(
                    'projection_orientation', 'f8', ('horizontal_vector',))
                position[:] = [-43.54090118, 66.70948029]
                orientation[:] = [-0.93283741, 0.36029762]

            grid = reconstruct_grid(output)

            self.assertEqual(grid.shape, (117, 96))
            self.assertEqual(grid.lat_mesh.shape, (118, 97))
            self.assertEqual(np.asarray(grid.W).dtype, np.dtype('float32'))
            self.assertEqual(grid.W, np.float32(5_771_787.125))


if __name__ == '__main__':
    unittest.main()
