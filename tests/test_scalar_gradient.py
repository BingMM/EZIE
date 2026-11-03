import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ezie.data import Data
from ezie.model import Model


def make_scalar_data(track_values, track_variances=None):
    nmem = len(track_values)
    if track_variances is None:
        track_variances = [np.ones(len(values)) for values in track_values]

    lat = [np.linspace(65+i, 66+i, len(values)) for i, values in enumerate(track_values)]
    lon = [np.linspace(10+i, 11+i, len(values)) for i, values in enumerate(track_values)]
    radius = [np.full(len(values), 6451.2e3) for values in track_values]
    B0e = [np.zeros(len(values)) for values in track_values]
    B0n = [np.full(len(values), 3e-5) for values in track_values]
    B0u = [np.full(len(values), -4e-5) for values in track_values]
    B0 = [np.full(len(values), 5e-5) for values in track_values]

    return Data(
        date=datetime(2026, 1, 1), lat=lat, lon=lon, r=radius,
        B=[np.asarray(values, dtype=float) for values in track_values],
        cov_mag=[np.asarray(values, dtype=float) for values in track_variances],
        B0e=B0e, B0n=B0n, B0u=B0u, B0=B0, background='chaos')


class ScalarGradientTest(unittest.TestCase):
    def test_difference_operator_does_not_cross_mem_boundaries(self):
        data = make_scalar_data([
            np.array([1.0, 2.0, 4.0]),
            np.array([10.0, 13.0])
        ])
        model = Model(fit_mode='gradient', l1=0, l2=0)
        model.add_data(data)

        expected_D = np.array([
            [-1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -1.0, 1.0]
        ])

        np.testing.assert_allclose(model.D.toarray(), expected_D)
        np.testing.assert_allclose(model.d, np.array([1.0, 2.0, 3.0]))

    def test_gradient_covariance_retains_adjacent_correlations(self):
        variances = [np.array([1.0, 4.0, 9.0]), np.array([16.0, 25.0])]
        data = make_scalar_data([np.zeros(3), np.zeros(2)], variances)
        model = Model(fit_mode='gradient', l1=0, l2=0)
        model.add_data(data)

        D = model.D.toarray()
        expected = D @ np.diag(np.hstack(variances)) @ D.T
        np.testing.assert_allclose(model.Q, expected)
        self.assertEqual(model.Q[0, 1], -4.0)
        self.assertEqual(model.Q[1, 2], 0.0)

    def test_normal_equations_are_invariant_to_per_mem_offsets(self):
        values = [np.array([1.0, 2.0, 4.0]), np.array([3.0, 5.0, 8.0])]
        shifted = [values[0]+100.0, values[1]-50.0]
        G_absolute = np.array([
            [1.0, 0.0], [1.0, 1.0], [1.0, 3.0],
            [0.0, 1.0], [1.0, 1.0], [3.0, 1.0]
        ])

        models = []
        for tracks in (values, shifted):
            model = Model(fit_mode='gradient', l1=0, l2=0)
            model.add_data(make_scalar_data(tracks))
            model._G_absolute = G_absolute
            models.append(model)

        np.testing.assert_allclose(models[0].d, models[1].d)
        np.testing.assert_allclose(models[0].GTG, models[1].GTG)
        np.testing.assert_allclose(models[0].GTd, models[1].GTd)

    def test_offsets_are_recovered_conditional_on_model(self):
        G_absolute = np.array([
            [1.0, 0.0], [1.0, 1.0], [1.0, 2.0],
            [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]
        ])
        m = np.array([2.0, -1.0])
        offsets = np.array([4.0, -3.0])
        prediction = G_absolute @ m
        tracks = [prediction[:3]+offsets[0], prediction[3:]+offsets[1]]

        model = Model(fit_mode='gradient', l1=0, l2=0)
        model.add_data(make_scalar_data(tracks))
        model._G_absolute = G_absolute
        model._solver = SimpleNamespace(m=m)

        np.testing.assert_allclose(model.track_offsets, offsets)

    @patch('ezie.model.get_SECS_B_G_matrices')
    def test_scalar_operator_uses_background_not_vector_observations(self, get_G):
        data = make_scalar_data([np.zeros(2)])
        Ge = np.array([[1.0], [2.0]])
        Gn = np.array([[3.0], [4.0]])
        Gu = np.array([[5.0], [6.0]])
        get_G.return_value = Ge, Gn, Gu

        model = Model(fit_mode='gradient', l1=0, l2=0)
        model.add_data(data)
        model._grid = SimpleNamespace(
            lat=np.array([70.0]), lon=np.array([20.0]), R=model.RI)

        # Background unit vector is (0, 0.6, -0.8).
        expected = 0.6*Gn - 0.8*Gu
        np.testing.assert_allclose(model.G_absolute, expected)

    def test_gradient_mode_rejects_vector_observations(self):
        data = Data(
            date=datetime(2026, 1, 1),
            lat=[np.array([65.0, 66.0])],
            lon=[np.array([10.0, 11.0])],
            r=[np.full(2, 6451.2e3)],
            Be=[np.zeros(2)], cov_ee=[np.ones(2)])
        model = Model(fit_mode='gradient')

        with self.assertRaisesRegex(ValueError, 'scalar-only'):
            model.add_data(data)


if __name__ == '__main__':
    unittest.main()
