import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix, issparse

from ezie.data import Data
from ezie.model import Model
from ezie.utils import generate_ezie_data


class DataRadiusTest(unittest.TestCase):
    def test_scalar_radius_is_applied_to_every_mem(self):
        lat = [np.array([65.0, 66.0]), np.array([67.0, 68.0])]
        lon = [np.array([10.0, 11.0]), np.array([12.0, 13.0])]

        data = Data(datetime(2026, 1, 1), lat, lon, 6451.2e3)

        for mem in data.mems:
            np.testing.assert_allclose(mem.r, np.full(2, 6451.2e3))

        self.assertEqual(data.dates, [datetime(2026, 1, 1)])

    def test_generated_data_retains_full_time_interval(self):
        times = [datetime(2026, 1, 1, 0, 0),
                 datetime(2026, 1, 1, 0, 1),
                 datetime(2026, 1, 1, 0, 2)]
        extracted = {'time': times,
                     'time_mid': times[1],
                     'altitude': np.full(3, 80.0),
                     'lat1': np.array([65.0, 66.0, 67.0]),
                     'lon1': np.array([10.0, 11.0, 12.0]),
                     'dBe1': np.ones(3),
                     'cov1': np.repeat(np.eye(3)[None, :, :], 3, axis=0)}

        data = generate_ezie_data(extracted, comp='E', mems=1)

        self.assertEqual(data.dates, times)
        self.assertEqual(data.date, times[1])

    @patch('ezie.utils.load_file')
    def test_generated_data_forwards_chaos_background_to_file_loader(self, load_file):
        times = [datetime(2026, 1, 1, 0, 0), datetime(2026, 1, 1, 0, 1)]
        load_file.return_value = {
            'time': times,
            'altitude': np.full(2, 80.0),
            'lat1': np.array([65.0, 66.0]),
            'lon1': np.array([10.0, 11.0]),
            'dBe1': np.ones(2),
            'cov1': np.repeat(np.eye(3)[None, :, :], 2, axis=0),
            'background': 'chaos'
        }
        chaos_model = object()

        data = generate_ezie_data(
            'event.nc4', comp='E', mems=1,
            background='chaos', chaos_model=chaos_model)

        load_file.assert_called_once_with(
            'event.nc4', background='chaos', chaos_model=chaos_model,
            scalar_source='debiased')
        np.testing.assert_allclose(data.mems[0].Be, np.ones(2))

    def test_extracted_data_cannot_be_referenced_without_source_fields(self):
        times = [datetime(2026, 1, 1)]
        extracted = {
            'time': times,
            'altitude': np.array([80.0]),
            'background': 'igrf'
        }

        with self.assertRaisesRegex(ValueError, "already been extracted"):
            generate_ezie_data(extracted, comp='E', mems=1, background='chaos')


class ModelDataResetTest(unittest.TestCase):
    def test_add_data_invalidates_every_downstream_cache(self):
        model = Model(l1=None, l2=2.5)

        # Simulate an automatically selected l1 and populated downstream caches.
        model._l1 = 0.75
        model._grid = object()
        model._L = object()
        model._LTL = object()
        model._reg_sparse = object()
        model._reg_terms = object()
        model._G = object()
        model._G_absolute = object()
        model._GTG = object()
        model._GTd = object()
        model._Q = object()
        model._Q_absolute = object()
        model._Q_factor = object()
        model._Qinv = object()
        model._D = object()
        model._solver = object()
        model._track_offsets = object()
        model._ev = object()
        model._resolution = object()

        data = object()
        model.add_data(data)

        self.assertIs(model.data, data)
        self.assertIsNone(model._grid)
        self.assertIsNone(model._L)
        self.assertIsNone(model._LTL)
        self.assertIsNone(model._reg_sparse)
        self.assertIsNone(model._reg_terms)
        self.assertIsNone(model._G)
        self.assertIsNone(model._G_absolute)
        self.assertIsNone(model._GTG)
        self.assertIsNone(model._GTd)
        self.assertIsNone(model._Q)
        self.assertIsNone(model._Q_absolute)
        self.assertIsNone(model._Q_factor)
        self.assertIsNone(model._Qinv)
        self.assertIsNone(model._D)
        self.assertIsNone(model._solver)
        self.assertIsNone(model._track_offsets)
        self.assertIsNone(model._ev)
        self.assertIsNone(model._resolution)
        self.assertIsNone(model._l1)
        self.assertEqual(model._l2, 2.5)


class ModelGridPrecisionTest(unittest.TestCase):
    @patch('ezie.model.CSgrid')
    @patch('ezie.model.CSprojection')
    def test_float32_track_is_promoted_before_grid_construction(
            self, projection, grid):
        lat = np.array([60.0, 65.0, 70.0, 75.0], dtype=np.float32)
        lon = np.array([-30.0, -20.0, -10.0, 0.0], dtype=np.float32)
        expected_grid = object()
        grid.return_value = expected_grid

        result = Model(l1=0.75, l2=2.5).create_grid(lat, lon)

        position, orientation = projection.call_args.args
        grid_width = grid.call_args.args[2]
        self.assertEqual(np.asarray(position).dtype, np.dtype('float64'))
        self.assertEqual(np.asarray(orientation).dtype, np.dtype('float64'))
        self.assertEqual(np.asarray(grid_width).dtype, np.dtype('float64'))
        self.assertIs(result, expected_grid)


class ModelSparseRegularizationTest(unittest.TestCase):
    def test_derivative_and_ltl_remain_sparse_until_regularization_assembly(self):
        Le = np.array([[1.0, -1.0, 0.0],
                       [0.0, 1.0, -1.0],
                       [-1.0, 0.0, 1.0]])
        Ln = np.array([[0.0, 1.0, -1.0],
                       [-1.0, 0.0, 1.0],
                       [1.0, -1.0, 0.0]])

        class Grid:
            lat = np.array([70.0, 71.0, 72.0])
            lon = np.array([10.0, 11.0, 12.0])

            def get_Le_Ln(self, return_sparse=False):
                if not return_sparse:
                    raise AssertionError('dense derivative path was requested')
                return csr_matrix(Le), csr_matrix(Ln)

        class Apex:
            def basevectors_qd(self, lat, lon, height, coords):
                f1 = np.array([[1.0, 1.0, 1.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]])
                return f1, np.zeros_like(f1)

        model = Model(l1=0.75, l2=2.5)
        model._grid = Grid()
        model._apx = Apex()
        model._GTG = np.diag([2.0, 3.0, 4.0])

        expected_ltl = Le.T @ Le
        np.testing.assert_allclose(model.L.toarray(), Le)
        np.testing.assert_allclose(model.LTL.toarray(), expected_ltl)
        self.assertTrue(issparse(model.L))
        self.assertTrue(issparse(model.LTL))

        ridge = 10**model.l1 * model.gtg_mag
        smoothness = 10**model.l2 * model.gtg_mag / np.median(
            np.diag(expected_ltl))
        expected_reg = ridge*np.eye(3) + smoothness*expected_ltl
        np.testing.assert_allclose(model.reg_sparse.toarray(), expected_reg)
        np.testing.assert_allclose(model.reg, expected_reg)
        self.assertTrue(issparse(model.reg_sparse))
        self.assertIsInstance(model.reg, np.ndarray)


if __name__ == '__main__':
    unittest.main()
