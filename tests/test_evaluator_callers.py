import unittest
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from ezie.validation import Validation
from ezie.visualization import Plotter


class FakeEvaluator:
    def __init__(self):
        self.mean_calls = 0
        self.uncertainty_calls = 0
        self.magnitude_uncertainty_calls = 0

    def get_B_predictions(self, lat, lon, r, ret=False):
        self.mean_calls += 1
        values = np.ones_like(lat, dtype=float)
        return values, 2*values, 3*values

    def get_B_u_predictions(self, lat, lon, r, ret=False):
        self.uncertainty_calls += 1
        values = np.full_like(lat, 0.1, dtype=float)
        return values, 2*values, 3*values

    def get_B_magnitude_u_predictions(self, lat, lon, r, dB=None):
        self.magnitude_uncertainty_calls += 1
        return np.full_like(lat, 0.1, dtype=float)

    def calc_main_field(self, lat, lon, r, ret=False):
        values = np.ones_like(lat, dtype=float)
        return values, values, values, values, None


class EvaluatorCallerTest(unittest.TestCase):
    def setUp(self):
        self.ev = FakeEvaluator()

    def test_data_model_plot_requests_mean_and_uncertainty_separately(self):
        mem = SimpleNamespace(lat=np.array([65.0, 66.0]),
                              lon=np.array([10.0, 11.0]),
                              r=np.array([6451.2e3, 6451.2e3]),
                              Be=np.array([1.0, 1.1]),
                              cov_ee=np.array([0.01, 0.01]))
        data = SimpleNamespace(mems=[mem], comp='e', date='test')
        plotter = Plotter(model=SimpleNamespace(data=data, ev=self.ev))

        fig, _ = plotter.plot_data_model_comparison()
        plt.close(fig)

        self.assertEqual(self.ev.mean_calls, 1)
        self.assertEqual(self.ev.uncertainty_calls, 1)

    def test_scalar_plot_requests_directional_magnitude_uncertainty(self):
        mem = SimpleNamespace(lat=np.array([65.0, 66.0]),
                              lon=np.array([10.0, 11.0]),
                              r=np.array([6451.2e3, 6451.2e3]),
                              B=np.array([1.0, 1.1]),
                              cov_mag=np.array([0.01, 0.01]))
        data = SimpleNamespace(mems=[mem], comp='s', date='test')
        plotter = Plotter(model=SimpleNamespace(data=data, ev=self.ev))

        fig, _ = plotter.plot_data_model_comparison()
        plt.close(fig)

        self.assertEqual(self.ev.mean_calls, 1)
        self.assertEqual(self.ev.uncertainty_calls, 0)
        self.assertEqual(self.ev.magnitude_uncertainty_calls, 1)

    def test_validation_requests_mean_and_uncertainty_separately(self):
        validation = Validation.__new__(Validation)
        validation.ev = self.ev
        validation._lat_grid = np.array([65.0, 66.0])
        validation._lon_grid = np.array([10.0, 11.0])

        validation.ev_sm()

        self.assertEqual(self.ev.mean_calls, 1)
        self.assertEqual(self.ev.uncertainty_calls, 1)
        np.testing.assert_allclose(validation._Be_ev, np.ones(2))
        np.testing.assert_allclose(validation._Be_std_ev, np.full(2, 0.1))


if __name__ == '__main__':
    unittest.main()
