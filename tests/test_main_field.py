import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ezie.evaluator import Evaluator
from ezie.utils import CHAOS_MODEL_PATH


class ChaosMainFieldTest(unittest.TestCase):
    def test_model_path_is_relative_to_repository(self):
        expected = Path(__file__).resolve().parents[1] / 'data' / 'CHAOS-8.4.mat'

        self.assertEqual(CHAOS_MODEL_PATH, expected)
        self.assertTrue(CHAOS_MODEL_PATH.is_file())

    @patch('ezie.evaluator.calc_chaos_internal_field_spherical')
    def test_chaos_uses_full_time_and_complete_internal_field(self, calc_chaos):
        calc_chaos.return_value = (
            np.array([30.0]), np.array([20.0]), np.array([10.0]))
        date = datetime(2025, 5, 15, 2, 24, 54, 123456)
        model = SimpleNamespace(B0_model='chaos',
                                map_params={'RE': 6371.2e3},
                                data=SimpleNamespace(date=date))
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.model = model

        _, Be0, Bn0, Bu0, _ = evaluator.calc_main_field(
            np.array([70.0]), np.array([20.0]), np.array([6451.2e3]), ret=True)

        args = calc_chaos.call_args.args
        self.assertEqual(args[0], date)
        self.assertTrue(calc_chaos.call_args.kwargs['include_static'])
        np.testing.assert_allclose(Be0, np.array([10e-9]))
        np.testing.assert_allclose(Bn0, np.array([-20e-9]))
        np.testing.assert_allclose(Bu0, np.array([30e-9]))


if __name__ == '__main__':
    unittest.main()
