import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ezie.evaluator import Evaluator, _prediction_std, _transform_horizontal_G


class UncertaintyPropagationTest(unittest.TestCase):
    def setUp(self):
        self.Cmpost = np.array([[2.0, 0.6],
                                [0.6, 1.0]])
        self.Ge = np.array([[1.0, 0.2],
                            [0.5, -0.1]])
        self.Gn = np.array([[0.3, 1.0],
                            [-0.4, 0.8]])
        self.Gu = np.array([[0.1, -0.2],
                            [0.7, 0.3]])

    def test_horizontal_operator_rotation_matches_covariance_rotation(self):
        angle = np.deg2rad(np.array([30.0, -20.0]))
        first_e = np.cos(angle)
        first_n = np.sin(angle)
        second_e = -np.sin(angle)
        second_n = np.cos(angle)

        G_first, G_second = _transform_horizontal_G(
            self.Ge, self.Gn, first_e, first_n, second_e, second_n)

        for i in range(2):
            G_geo = np.vstack((self.Ge[i], self.Gn[i]))
            covariance_geo = G_geo @ self.Cmpost @ G_geo.T
            transform = np.array([[first_e[i], first_n[i]],
                                  [second_e[i], second_n[i]]])
            covariance_rotated = transform @ covariance_geo @ transform.T

            self.assertAlmostEqual(_prediction_std(G_first, self.Cmpost)[i],
                                   np.sqrt(covariance_rotated[0, 0]))
            self.assertAlmostEqual(_prediction_std(G_second, self.Cmpost)[i],
                                   np.sqrt(covariance_rotated[1, 1]))

    @patch('ezie.evaluator.get_SECS_B_G_matrices')
    def test_magnitude_uncertainty_uses_direction_and_cross_covariance(self, get_G):
        get_G.return_value = self.Ge, self.Gn, self.Gu
        grid = SimpleNamespace(lat=np.array([[65.0, 66.0]]),
                               lon=np.array([[10.0, 11.0]]),
                               R=6481.2e3)
        model = SimpleNamespace(grid=grid, slim=10e3, Cmpost=self.Cmpost)
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.model = model
        evaluator.calc_main_field = lambda lat, lon, r, ret: (
            np.array([5.0, 5.0]),
            np.array([3.0, 0.0]),
            np.array([4.0, 0.0]),
            np.array([0.0, 5.0]),
            None)

        zeros = np.zeros(2)
        actual = evaluator.get_B_magnitude_u_predictions(
            np.array([65.0, 66.0]), np.array([10.0, 11.0]), 6451.2e3,
            dB=(zeros, zeros, zeros))

        G_magnitude = np.vstack((0.6*self.Ge[0] + 0.8*self.Gn[0],
                                 self.Gu[1]))
        expected = _prediction_std(G_magnitude, self.Cmpost)
        np.testing.assert_allclose(actual, expected)

    @patch('ezie.evaluator.get_SECS_J_G_matrices')
    def test_qd_current_uncertainty_rotates_operators(self, get_G):
        get_G.return_value = self.Ge, self.Gn
        grid = SimpleNamespace(lat=np.array([[65.0, 66.0]]),
                               lon=np.array([[10.0, 11.0]]),
                               lat_mesh=np.array([[65.0, 66.0]]),
                               lon_mesh=np.array([[10.0, 11.0]]),
                               R=6481.2e3)
        model = SimpleNamespace(grid=grid, slim=10e3, Cmpost=self.Cmpost,
                                map_params={'HI': 110e3})
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.model = model
        evaluator.vec_geo2qd = lambda lat, lon, h, east, north: (-north, east)

        first_u, second_u = evaluator.get_J_qd_u_predictions(h=110)

        np.testing.assert_allclose(first_u, _prediction_std(-self.Gn, self.Cmpost))
        np.testing.assert_allclose(second_u, _prediction_std(self.Ge, self.Cmpost))


if __name__ == '__main__':
    unittest.main()
