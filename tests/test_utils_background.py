import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import chaosmagpy as cp
import numpy as np

from ezie.utils import (
    CHAOS_MODEL_PATH,
    NT_TO_T,
    calc_chaos_internal_field_geodetic,
    calc_chaos_internal_field_spherical,
    extract_data,
    generate_ezie_data,
    load_chaos_model
)


class Variable:
    def __init__(self, values):
        self.values = np.ma.array(values)

    def __getitem__(self, key):
        return self.values[key]


def make_dataset():
    n = 2
    geolocation = {
        'reference_altitude': Variable([80.0, 80.0])
    }
    retrieved = {}
    ancillary = {}

    for mem in range(1, 5):
        geolocation[f'obs_lat{mem}'] = Variable([70.0, 71.0])
        geolocation[f'obs_lon{mem}'] = Variable([20.0, 21.0])

        retrieved[f'dbgeoe{mem}'] = Variable([1.0, 2.0])
        retrieved[f'dbgeon{mem}'] = Variable([3.0, 4.0])
        retrieved[f'dbgeod{mem}'] = Variable([5.0, 6.0])
        retrieved[f'retrieved_dbmag{mem}'] = Variable([10.0, 20.0])
        retrieved[f'debiased_dbmag{mem}'] = Variable([-5.0, 5.0])
        retrieved[f'retrieved_b_tot{mem}'] = Variable([100.0, 200.0])
        retrieved[f'retrieved_b_tot_err{mem}'] = Variable([2.0, 3.0])
        retrieved[f'cov_ned{mem}'] = Variable(np.repeat(np.eye(3)[None, :, :], n, axis=0))

        ancillary[f'anc_model_bgeoe{mem}'] = Variable([10.0, 20.0])
        ancillary[f'anc_model_bgeon{mem}'] = Variable([30.0, 40.0])
        ancillary[f'anc_model_bgeod{mem}'] = Variable([50.0, 60.0])
        ancillary[f'anc_model_bgeo{mem}'] = Variable([90.0, 180.0])

    return MagicMock(
        SpaceVehicle='A',
        __getitem__=lambda self, key: {
            'Time': {'time_utc': Variable(['2025-05-15T02:24:54', '2025-05-15T02:24:55'])},
            'Geolocation': geolocation,
            'RetrievedParameters': retrieved,
            'Ancillary': ancillary
        }[key])


class ChaosUtilityTest(unittest.TestCase):
    @patch('ezie.utils.cp.CHAOS.from_mat')
    def test_model_is_loaded_only_once(self, from_mat):
        from_mat.return_value = object()
        load_chaos_model.cache_clear()

        try:
            first = load_chaos_model()
            second = load_chaos_model()
        finally:
            load_chaos_model.cache_clear()

        self.assertIs(first, second)
        from_mat.assert_called_once_with(CHAOS_MODEL_PATH)

    def test_spherical_field_adds_static_internal_but_no_external_sources(self):
        model = MagicMock()
        model.synth_values_tdep.return_value = (
            np.array([1.0]), np.array([2.0]), np.array([3.0]))
        model.synth_values_static.return_value = (
            np.array([10.0]), np.array([20.0]), np.array([30.0]))
        date = datetime(2025, 5, 15, 2, 24, 54, 123456)

        result = calc_chaos_internal_field_spherical(
            date, np.array([6450.0]), np.array([20.0]), np.array([10.0]),
            include_static=True, chaos_model=model)

        np.testing.assert_allclose(result, np.array([[11.0], [22.0], [33.0]]))
        args = model.synth_values_tdep.call_args.args
        self.assertAlmostEqual(args[0], cp.data_utils.mjd2000(date))
        model.synth_values_static.assert_called_once()
        model.synth_values_gsm.assert_not_called()
        model.synth_values_sm.assert_not_called()

    def test_spherical_core_field_omits_static_field(self):
        model = MagicMock()
        model.synth_values_tdep.return_value = (1.0, 2.0, 3.0)

        result = calc_chaos_internal_field_spherical(
            datetime(2025, 5, 15), 6450.0, 20.0, 10.0,
            include_static=False, chaos_model=model)

        self.assertEqual(result, (1.0, 2.0, 3.0))
        model.synth_values_static.assert_not_called()

    @patch('ezie.utils.calc_chaos_internal_field_spherical')
    @patch('ezie.utils.cp.coordinate_utils.geo_to_gg')
    @patch('ezie.utils.cp.coordinate_utils.gg_to_geo')
    def test_geodetic_field_converts_coordinates_and_components(
            self, gg_to_geo, geo_to_gg, calc_spherical):
        gg_to_geo.return_value = np.array([6450.0]), np.array([21.0])
        calc_spherical.return_value = (
            np.array([30.0]), np.array([20.0]), np.array([10.0]))
        geo_to_gg.return_value = (
            np.array([80.0]), np.array([20.0]),
            np.array([15.0]), np.array([25.0]))

        Be, Bn, Bu = calc_chaos_internal_field_geodetic(
            [datetime(2025, 5, 15)], np.array([70.0]), np.array([10.0]),
            np.array([80.0]), include_static=True, chaos_model='model')

        np.testing.assert_allclose(Be, np.array([10.0]))
        np.testing.assert_allclose(Bn, np.array([15.0]))
        np.testing.assert_allclose(Bu, np.array([-25.0]))
        gg_to_geo.assert_called_once()
        geo_to_gg.assert_called_once()
        self.assertTrue(calc_spherical.call_args.kwargs['include_static'])
        self.assertEqual(calc_spherical.call_args.kwargs['chaos_model'], 'model')


class ExtractDataBackgroundTest(unittest.TestCase):
    @patch('ezie.utils.calc_chaos_internal_field_geodetic')
    def test_default_uses_debiased_igrf_relative_scalar_data(self, calc_chaos):
        data = extract_data(make_dataset())

        self.assertEqual(data['background'], 'igrf')
        self.assertEqual(data['scalar_source'], 'debiased')
        np.testing.assert_allclose(data['dBe1'], np.array([1.0, 2.0])*NT_TO_T)
        np.testing.assert_allclose(data['dBn1'], np.array([3.0, 4.0])*NT_TO_T)
        np.testing.assert_allclose(data['dBu1'], -np.array([5.0, 6.0])*NT_TO_T)
        np.testing.assert_allclose(data['dB1'], np.array([-5.0, 5.0])*NT_TO_T)
        calc_chaos.assert_not_called()

    def test_retrieved_scalar_source_retains_non_debiased_level(self):
        data = extract_data(make_dataset(), scalar_source='retrieved')

        self.assertEqual(data['scalar_source'], 'retrieved')
        np.testing.assert_allclose(data['dB1'], np.array([10.0, 20.0])*NT_TO_T)

        ezie_data = generate_ezie_data(data, comp='F', mems=1)
        self.assertEqual(ezie_data.scalar_source, 'retrieved')
        np.testing.assert_allclose(ezie_data.mems[0].B, np.array([10.0, 20.0])*NT_TO_T)

    @patch('ezie.utils.calc_chaos_internal_field_geodetic')
    def test_chaos_replaces_igrf_for_vector_and_scalar_data(self, calc_chaos):
        chaos_model = object()
        chaos_Be = np.array([7.0, 17.0])
        chaos_Bn = np.array([25.0, 35.0])
        chaos_Bu = np.array([-45.0, -55.0])
        calc_chaos.return_value = chaos_Be, chaos_Bn, chaos_Bu

        data = extract_data(make_dataset(), background='chaos', chaos_model=chaos_model)

        self.assertEqual(data['background'], 'chaos')
        np.testing.assert_allclose(data['dBe1'], np.array([4.0, 5.0])*NT_TO_T)
        np.testing.assert_allclose(data['dBn1'], np.array([8.0, 9.0])*NT_TO_T)
        np.testing.assert_allclose(data['dBu1'], -np.array([10.0, 11.0])*NT_TO_T)

        chaos_magnitude = np.sqrt(chaos_Be**2 + chaos_Bn**2 + chaos_Bu**2)
        np.testing.assert_allclose(
            data['dB1'],
            (np.array([-5.0, 5.0]) + np.array([90.0, 180.0]) - chaos_magnitude)*NT_TO_T)

        self.assertEqual(calc_chaos.call_count, 4)
        self.assertTrue(calc_chaos.call_args.kwargs['include_static'])
        self.assertIs(calc_chaos.call_args.kwargs['chaos_model'], chaos_model)

        expected_cov = np.repeat(np.eye(3)[None, :, :], 2, axis=0)*NT_TO_T**2
        np.testing.assert_allclose(data['cov1'], expected_cov)
        np.testing.assert_allclose(data['mag_var1'], np.array([4.0, 9.0])*NT_TO_T**2)

    def test_invalid_background_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_data(make_dataset(), background='external')

    def test_invalid_scalar_source_is_rejected(self):
        with self.assertRaises(ValueError):
            extract_data(make_dataset(), scalar_source='total')

    def test_scalar_data_retains_background_field_for_forward_model(self):
        extracted = extract_data(make_dataset())

        data = generate_ezie_data(extracted, comp='F', mems=1)
        mem = data.mems[0]

        self.assertEqual(data.background, 'igrf')
        self.assertEqual(data.scalar_source, 'debiased')
        np.testing.assert_allclose(mem.B0e, np.array([10.0, 20.0])*NT_TO_T)
        np.testing.assert_allclose(mem.B0n, np.array([30.0, 40.0])*NT_TO_T)
        np.testing.assert_allclose(mem.B0u, -np.array([50.0, 60.0])*NT_TO_T)
        np.testing.assert_allclose(mem.B0, np.array([90.0, 180.0])*NT_TO_T)


if __name__ == '__main__':
    unittest.main()
