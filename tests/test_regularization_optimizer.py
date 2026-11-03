import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.sparse import eye

from ezie.regularization_optimizer import (
    RegularizationOptimizer,
    calc_gini,
)


class RegularizationOptimizerWoodburyTest(unittest.TestCase):
    def test_parametric_curvature_accepts_nonmonotonic_residual_norm(self):
        optimizer = RegularizationOptimizer.__new__(RegularizationOptimizer)
        optimizer.l1_fit = np.linspace(0.0, 1.0, 50)
        optimizer._rnorm = (optimizer.l1_fit-0.5)**2
        optimizer._mnorm = -optimizer.l1_fit**2

        selected_id = optimizer.max_curvature()

        self.assertIsNotNone(selected_id)
        self.assertEqual(optimizer.rnorm_fit.shape, (1000,))
        self.assertEqual(optimizer.curvature.shape, (1000,))

    @patch('ezie.regularization_optimizer.find_peaks')
    def test_parametric_curvature_selects_last_peak(self, find_peaks_mock):
        find_peaks_mock.return_value = (np.array([200, 800]), {})
        optimizer = RegularizationOptimizer.__new__(RegularizationOptimizer)
        optimizer.l1_fit = np.linspace(0.0, 1.0, 50)
        optimizer._rnorm = optimizer.l1_fit
        optimizer._mnorm = optimizer.l1_fit**2

        selected_id = optimizer.max_curvature()

        expected_l1 = optimizer.curvature_l1[800]
        expected_id = np.argmin(abs(optimizer.l1_fit-expected_l1))
        self.assertEqual(selected_id, expected_id)

    @patch('ezie.regularization_optimizer.BandedWoodburySolver')
    def test_single_resolution_row_uses_banded_woodbury(self, solver_class):
        row = np.array([0.1, 0.7, 0.2])
        solver_class.return_value.resolution_rows.return_value = row[None, :]

        model = SimpleNamespace(
            _l1=0.75,
            _l2=2.5,
            G=np.ones((2, 3)),
            Q=np.eye(2),
            reg_sparse=eye(3, format='csr'),
            grid=SimpleNamespace(size=3))
        model.reset_regularization = unittest.mock.Mock()
        model.reset_solution = unittest.mock.Mock()

        optimizer = RegularizationOptimizer.__new__(RegularizationOptimizer)
        optimizer.model = model
        result = optimizer.compute_gini_for_pair(1.0, 3.0, idx=1)

        solver_class.assert_called_once_with(
            G=model.G,
            covariance=model.Q,
            regularization=model.reg_sparse)
        solver_class.return_value.resolution_rows.assert_called_once_with([1])
        self.assertAlmostEqual(result, calc_gini(row))
        self.assertEqual(
            model.reset_regularization.call_args_list,
            [unittest.mock.call(1.0, 3.0),
             unittest.mock.call(0.75, 2.5)])
        self.assertEqual(model.reset_solution.call_count, 2)


if __name__ == '__main__':
    unittest.main()
