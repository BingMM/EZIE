from datetime import datetime
from collections.abc import Mapping
from functools import lru_cache
from numbers import Integral
from os import PathLike
from pathlib import Path

import chaosmagpy as cp
import numpy as np
from netCDF4 import Dataset

from .data import Data


NT_TO_T = 1e-9
CHAOS_MODEL_PATH = Path(__file__).resolve().parents[1] / 'data' / 'CHAOS-8.4.mat'


#%% Internal geomagnetic field

@lru_cache(maxsize=None)
def load_chaos_model(path=CHAOS_MODEL_PATH):
    """Load and cache a CHAOS model from a MAT file."""

    return cp.CHAOS.from_mat(Path(path))


def calc_chaos_internal_field_spherical(time, radius, theta, lon,
                                         include_static=False, chaos_model=None):
    """Evaluate the internal CHAOS field in geocentric spherical coordinates.

    Parameters are broadcast by ChaosMagPy. ``radius`` and the returned field
    components use kilometers and nanotesla, respectively. The components are
    radial, southward, and eastward. External GSM and SM sources are never
    included.
    """

    if chaos_model is None:
        chaos_model = load_chaos_model()

    time_mjd2000 = cp.data_utils.mjd2000(time)
    B_radius, B_theta, B_phi = chaos_model.synth_values_tdep(
        time_mjd2000, radius, theta, lon)

    if include_static:
        B_radius_static, B_theta_static, B_phi_static = chaos_model.synth_values_static(
            radius, theta, lon)
        B_radius = B_radius + B_radius_static
        B_theta = B_theta + B_theta_static
        B_phi = B_phi + B_phi_static

    return B_radius, B_theta, B_phi


def calc_chaos_internal_field_geodetic(time, lat, lon, altitude,
                                        include_static=True, chaos_model=None):
    """Evaluate the internal CHAOS field as geodetic east, north, and up.

    ``lat`` and ``altitude`` are WGS84 geodetic latitude in degrees and height
    in kilometers. Returned components are in nanotesla. By default the field
    contains both the time-dependent core field and the static lithospheric
    field, but no external sources.
    """

    lat = np.asarray(lat)
    lon = np.asarray(lon)
    altitude = np.asarray(altitude)

    radius, theta = cp.coordinate_utils.gg_to_geo(altitude, 90-lat)
    B_radius, B_theta, Be = calc_chaos_internal_field_spherical(
        time, radius, theta, lon,
        include_static=include_static, chaos_model=chaos_model)
    _, _, Bn, Bd = cp.coordinate_utils.geo_to_gg(
        radius, theta, B_radius, B_theta)

    return Be, Bn, -Bd


#%% Post-launch Level-2 input

def _normalize_background(background):
    if not isinstance(background, str):
        raise TypeError("background must be 'igrf' or 'chaos'")

    background = background.lower()
    if background not in ('igrf', 'chaos'):
        raise ValueError("background must be 'igrf' or 'chaos'")

    return background


def _normalize_scalar_source(scalar_source):
    if not isinstance(scalar_source, str):
        raise TypeError("scalar_source must be 'debiased' or 'retrieved'")

    scalar_source = scalar_source.lower()
    if scalar_source not in ('debiased', 'retrieved'):
        raise ValueError("scalar_source must be 'debiased' or 'retrieved'")

    return scalar_source


def extract_data(dataset, background='igrf', chaos_model=None,
                 scalar_source='debiased'):
    """Extract fields from an open post-launch EZIE Level-2 dataset.

    Parameters
    ----------
    dataset
        Open post-launch EZIE Level-2 NetCDF dataset.
    background : {'igrf', 'chaos'}, optional
        Reference field for the returned perturbations. ``'igrf'`` preserves
        the Level-2 product. ``'chaos'`` adds back the exact ancillary IGRF
        field stored in the product and subtracts the complete internal CHAOS
        field (time-dependent core plus static lithosphere). External CHAOS
        sources are excluded.
    chaos_model, optional
        Preloaded ``chaosmagpy.CHAOS`` model. Used only when
        ``background='chaos'``. When omitted in CHAOS mode, the repository
        model is loaded and cached automatically.
    scalar_source : {'debiased', 'retrieved'}, optional
        Level-2 scalar perturbation to use. The default is the explicitly
        debiased ``debiased_dbmag#`` product. ``'retrieved'`` selects the
        non-debiased ``retrieved_dbmag#`` product.
    """

    background = _normalize_background(background)
    scalar_source = _normalize_scalar_source(scalar_source)

    data = {'sat': dataset.SpaceVehicle,
            'time': [datetime.fromisoformat(t) for t in dataset['Time']['time_utc'][:]],
            'altitude': dataset['Geolocation']['reference_altitude'][:].filled(np.nan),
            'background': background,
            'scalar_source': scalar_source}
    data['n'] = len(data['time'])
    data['time_mid'] = data['time'][data['n']//2]

    if background == 'chaos' and chaos_model is None:
        chaos_model = load_chaos_model()

    # Convert magnetic observations from nT to T at the input boundary.
    for i in range(4):
        mem = i+1
        lat = dataset['Geolocation'][f'obs_lat{mem}'][:].filled(np.nan)
        lon = dataset['Geolocation'][f'obs_lon{mem}'][:].filled(np.nan)
        dBe = dataset['RetrievedParameters'][f'dbgeoe{mem}'][:].filled(np.nan)
        dBn = dataset['RetrievedParameters'][f'dbgeon{mem}'][:].filled(np.nan)
        dBu = -dataset['RetrievedParameters'][f'dbgeod{mem}'][:].filled(np.nan)

        data[f'lat{mem}'] = lat
        data[f'lon{mem}'] = lon
    
        igrf_Be = dataset['Ancillary'][f'anc_model_bgeoe{mem}'][:].filled(np.nan)
        igrf_Bn = dataset['Ancillary'][f'anc_model_bgeon{mem}'][:].filled(np.nan)
        igrf_Bu = -dataset['Ancillary'][f'anc_model_bgeod{mem}'][:].filled(np.nan)
        igrf_B = dataset['Ancillary'][f'anc_model_bgeo{mem}'][:].filled(np.nan)
        dB_igrf = dataset['RetrievedParameters'][f'{scalar_source}_dbmag{mem}'][:].filled(np.nan)
        if background == 'chaos':
            chaos_Be, chaos_Bn, chaos_Bu = calc_chaos_internal_field_geodetic(data['time'], lat, lon, data['altitude'], include_static=True, chaos_model=chaos_model)

            dBe = dBe + igrf_Be - chaos_Be
            dBn = dBn + igrf_Bn - chaos_Bn
            dBu = dBu + igrf_Bu - chaos_Bu
            B_background = np.sqrt(chaos_Be**2 + chaos_Bn**2 + chaos_Bu**2)
            data[f'Be{mem}'] = chaos_Be * NT_TO_T
            data[f'Bn{mem}'] = chaos_Bn * NT_TO_T
            data[f'Bu{mem}'] = chaos_Bu * NT_TO_T
            data[f'B{mem}'] = B_background * NT_TO_T
        else:
            B_background = igrf_B
            data[f'Be{mem}'] = igrf_Be * NT_TO_T
            data[f'Bn{mem}'] = igrf_Bn * NT_TO_T
            data[f'Bu{mem}'] = igrf_Bu * NT_TO_T
            data[f'B{mem}'] = B_background * NT_TO_T

        data[f'dBe{mem}'] = dBe * NT_TO_T
        data[f'dBn{mem}'] = dBn * NT_TO_T
        data[f'dBu{mem}'] = dBu * NT_TO_T
        data[f'dB{mem}'] = (dB_igrf + igrf_B - B_background) * NT_TO_T

        # Transform covariance from NED to ENU and from nT² to T².
        T_ned_to_enu = np.array([[0, 1, 0],
                                 [1, 0, 0],
                                 [0, 0, -1]])
        cov_ned = dataset['RetrievedParameters'][f'cov_ned{mem}'][:].filled(np.nan) * NT_TO_T**2
        data[f'cov{mem}'] = T_ned_to_enu @ cov_ned @ T_ned_to_enu.T

        B_total_error = dataset['RetrievedParameters'][f'retrieved_b_tot_err{mem}'][:].filled(np.nan)
        data[f'mag_var{mem}'] = B_total_error**2 * NT_TO_T**2
        
        ### Testing
        
        Be = data[f'Be{mem}'] + data[f'dBe{mem}']
        Bn = data[f'Bn{mem}'] + data[f'dBn{mem}']
        Bu = data[f'Bu{mem}'] + data[f'dBu{mem}']
        B = np.sqrt(Be**2 + Bn**2 + Bu**2)
        be = Be / B
        bn = Bn / B
        bu = Bu / B
        
        offset = data[f'dB{mem}'] + data[f'B{mem}'] - B
        
        data[f'dBe_c{mem}'] = Be + offset*be - data[f'Be{mem}']
        data[f'dBn_c{mem}'] = Bn + offset*bn - data[f'Bn{mem}']
        data[f'dBu_c{mem}'] = Bu + offset*bu - data[f'Bu{mem}']
        
    return data


def load_nc4_file(file):
    """Open an EZIE NetCDF4 file."""

    return Dataset(file)


def load_file(file, background='igrf', chaos_model=None,
              scalar_source='debiased'):
    """Load and extract one post-launch EZIE Level-2 file."""

    with load_nc4_file(file) as dataset:
        return extract_data(dataset, background=background,
                            chaos_model=chaos_model,
                            scalar_source=scalar_source)


def load_files(files, background='igrf', chaos_model=None,
               scalar_source='debiased'):
    """Load and extract a sequence of post-launch EZIE Level-2 files."""

    return [load_file(file, background=background, chaos_model=chaos_model,
                      scalar_source=scalar_source) for file in files]


#%% EZIE data object

def _normalize_comp(comp):
    if not isinstance(comp, str):
        raise TypeError("comp must be a string")

    comp = comp.upper()
    if comp == 'ALL':
        return 'ENU'
    if comp in ('F', 'MAG', 'MAGNITUDE'):
        return 'F'
    if not comp:
        raise ValueError("comp cannot be empty")
    if len(comp) != len(set(comp)):
        raise ValueError("comp cannot contain repeated components")

    invalid = set(comp) - set('ENU')
    if invalid:
        raise ValueError("comp must be 'all', 'magnitude', or a combination of 'E', 'N', and 'U'")

    return comp


def _prepare_data(data, background=None, chaos_model=None,
                  scalar_source=None):
    if isinstance(data, (str, PathLike)):
        if background is None:
            background = 'igrf'
        if scalar_source is None:
            scalar_source = 'debiased'
        scalar_source = _normalize_scalar_source(scalar_source)
        return load_file(data, background=background, chaos_model=chaos_model,
                         scalar_source=scalar_source)

    if isinstance(data, Mapping):
        if background is not None:
            requested_background = _normalize_background(background)
            extracted_background = data.get('background', 'igrf')
            if requested_background != extracted_background:
                raise ValueError(
                    f"data have already been extracted with background='{extracted_background}'; "
                    f"cannot apply background='{requested_background}' without the source dataset")
        if scalar_source is not None:
            scalar_source = _normalize_scalar_source(scalar_source)
            extracted_scalar_source = data.get('scalar_source')
            if (extracted_scalar_source is not None and
                    scalar_source != extracted_scalar_source):
                raise ValueError(
                    f"data have already been extracted with scalar_source='{extracted_scalar_source}'; "
                    f"cannot apply scalar_source='{scalar_source}' without the source dataset")
        return data

    if hasattr(data, 'groups') and 'Time' in data.groups:
        if background is None:
            background = 'igrf'
        if scalar_source is None:
            scalar_source = 'debiased'
        scalar_source = _normalize_scalar_source(scalar_source)
        return extract_data(data, background=background,
                            chaos_model=chaos_model,
                            scalar_source=scalar_source)

    raise TypeError("data must be a file path, an open NetCDF dataset, or extracted EZIE data")


def _normalize_mems(mems):
    if mems is None:
        return [1, 2, 3, 4]
    if isinstance(mems, Integral) and not isinstance(mems, bool):
        mems = [mems]
    else:
        try:
            mems = list(mems)
        except TypeError as error:
            raise TypeError("mems must be an integer or an iterable of integers") from error

    if not mems:
        raise ValueError("mems cannot be empty")
    if not all(isinstance(mem, Integral) and not isinstance(mem, bool) for mem in mems):
        raise TypeError("mems must contain only integers")

    mems = [int(mem) for mem in mems]
    if len(mems) != len(set(mems)):
        raise ValueError("mems cannot contain repeated MEM numbers")
    if any(mem not in (1, 2, 3, 4) for mem in mems):
        raise ValueError("mems can contain only MEM numbers 1, 2, 3, and 4")

    return mems


def generate_ezie_data(data, comp='all', mems=None, background=None,
                       chaos_model=None, offset_corrected=False,
                       scalar_source=None):
    """Generate an ``ezie.Data`` object from post-launch Level-2 data.

    Parameters
    ----------
    data
        A NetCDF file path, an open NetCDF dataset, or data returned by
        :func:`extract_data` or :func:`load_file`.
    comp : str, optional
        Magnetic components to include. ``'all'`` and ``'ENU'`` select all
        three vector components. Any non-repeating combination of ``'E'``,
        ``'N'``, and ``'U'`` is accepted. ``'F'``, ``'mag'``, or
        ``'magnitude'`` selects scalar magnetic-magnitude perturbations.
    mems : int or iterable of int, optional
        MEM units to include, in the requested order. Each value must be unique
        and between 1 and 4. The default is ``[1, 2, 3, 4]``.
    background : {'igrf', 'chaos'}, optional
        Reference field to apply when ``data`` is a file path or open NetCDF
        dataset. The default is the Level-2 IGRF reference. For an already
        extracted mapping, omit this argument to preserve its existing
        reference, or supply the matching value as a consistency check.
    chaos_model, optional
        Preloaded ``chaosmagpy.CHAOS`` model forwarded to
        :func:`extract_data` when ``background='chaos'``.
    scalar_source : {'debiased', 'retrieved'}, optional
        Level-2 scalar product to use when reading a file or open dataset.
        The default is ``'debiased'`` for those inputs. For an extracted
        mapping, omission preserves its existing source; an explicit value
        must match the source selected during extraction.
    """

    data = _prepare_data(data, background=background,
                         chaos_model=chaos_model,
                         scalar_source=scalar_source)
    comp = _normalize_comp(comp)
    mems = _normalize_mems(mems)

    kwargs = {'date': data['time'],
              'lat': [data[f'lat{i}'] for i in mems],
              'lon': [data[f'lon{i}'] for i in mems],
              'r': [6371.2e3 + data['altitude']*1e3 for i in mems],
              'background': data.get('background'),
              'scalar_source': data.get('scalar_source')}

    background_fields = ('Be', 'Bn', 'Bu', 'B')
    if all(f'{field}{i}' in data for field in background_fields for i in mems):
        kwargs['B0e'] = [data[f'Be{i}'] for i in mems]
        kwargs['B0n'] = [data[f'Bn{i}'] for i in mems]
        kwargs['B0u'] = [data[f'Bu{i}'] for i in mems]
        kwargs['B0'] = [data[f'B{i}'] for i in mems]

    if comp == 'F':
        kwargs['B'] = [data[f'dB{i}'] for i in mems]
        kwargs['cov_mag'] = [data[f'mag_var{i}'] for i in mems]
        return Data(**kwargs)

    if 'E' in comp:
        if offset_corrected:
            kwargs['Be'] = [data[f'dBe_c{i}'] for i in mems]
        else:
            kwargs['Be'] = [data[f'dBe{i}'] for i in mems]
        kwargs['cov_ee'] = [data[f'cov{i}'][:, 0, 0] for i in mems]
    if 'N' in comp:
        if offset_corrected:
            kwargs['Bn'] = [data[f'dBn_c{i}'] for i in mems]
        else:
            kwargs['Bn'] = [data[f'dBn{i}'] for i in mems]        
        kwargs['cov_nn'] = [data[f'cov{i}'][:, 1, 1] for i in mems]
    if 'U' in comp:
        if offset_corrected:
            kwargs['Bu'] = [data[f'dBu_c{i}'] for i in mems]
        else:
            kwargs['Bu'] = [data[f'dBu{i}'] for i in mems]
        kwargs['cov_uu'] = [data[f'cov{i}'][:, 2, 2] for i in mems]

    if 'E' in comp and 'N' in comp:
        kwargs['cov_en'] = [data[f'cov{i}'][:, 0, 1] for i in mems]
    if 'E' in comp and 'U' in comp:
        kwargs['cov_eu'] = [data[f'cov{i}'][:, 0, 2] for i in mems]
    if 'N' in comp and 'U' in comp:
        kwargs['cov_nu'] = [data[f'cov{i}'][:, 1, 2] for i in mems]

    return Data(**kwargs)
