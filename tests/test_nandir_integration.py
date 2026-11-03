import unittest

import numpy as np
from nandir import QuadraticRegularization
from scipy.linalg import cho_factor, cho_solve

from ezie.model import Model


class NandirIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.GTG = np.array([[6.0, 1.0, 0.5],
                             [1.0, 5.0, 0.25],
                             [0.5, 0.25, 4.0]])
        self.GTd = np.array([2.0, -1.0, 0.5])
        self.reg = np.array([[0.5, 0.0, 0.0],
                             [0.0, 0.75, 0.1],
                             [0.0, 0.1, 1.0]])

        self.model = Model.__new__(Model)
        self.model._GTG = self.GTG
        self.model._GTd = self.GTd
        self.model._reg = self.reg
        self.model._reg_terms = (QuadraticRegularization(self.reg),)
        self.model.reset_solution()

    def test_solver_matches_previous_cholesky_path(self):
        c_factor = cho_factor(self.GTG + self.reg, overwrite_a=False, check_finite=False)
        expected_m = cho_solve(c_factor, self.GTd, check_finite=False)
        expected_Cmpost = cho_solve(c_factor, np.eye(self.GTG.shape[0]), check_finite=False)
        expected_Rm = expected_Cmpost @ self.GTG

        np.testing.assert_allclose(self.model.m, expected_m, rtol=1e-13, atol=1e-15)
        self.assertIsNone(self.model.solver._Cmpost)

        np.testing.assert_allclose(self.model.Cmpost, expected_Cmpost, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(self.model.Rm, expected_Rm, rtol=1e-13, atol=1e-15)

    def test_solution_reset_replaces_solver(self):
        original_solver = self.model.solver

        self.model.reset_solution()

        self.assertIsNot(self.model.solver, original_solver)


if __name__ == '__main__':
    unittest.main()
