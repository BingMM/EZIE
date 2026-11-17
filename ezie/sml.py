"""Read cached SuperMAG SML and match observation times without downloads."""

from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def read_sml_cache(path):
    path = Path(path)
    if not path.is_file():
        return None
    with Dataset(path) as dataset:
        if getattr(dataset, 'processing_complete', 0) != 1:
            return None
        required = {'time', 'SML', 'dSML_dt'}
        if not required.issubset(dataset.variables):
            return None
        return (
            np.asarray(dataset.variables['time'][:], dtype=np.int64),
            np.asarray(dataset.variables['SML'][:], dtype=float),
            np.asarray(dataset.variables['dSML_dt'][:], dtype=float))


def nearest_sml_indices(observation_time_ns, sml_time_seconds,
                        tolerance_seconds=60.0):
    """Return nearest SML indices and the match mask for observation times."""

    observation_time_ns = np.asarray(observation_time_ns, dtype=np.int64).ravel()
    sml_time_seconds = np.asarray(sml_time_seconds, dtype=np.int64).ravel()
    if sml_time_seconds.size == 0 or np.any(np.diff(sml_time_seconds) <= 0):
        raise ValueError('SML time must be a non-empty increasing array')

    observation_seconds = observation_time_ns/1_000_000_000
    right = np.searchsorted(sml_time_seconds, observation_seconds)
    right = np.clip(right, 0, sml_time_seconds.size-1)
    left = np.clip(right-1, 0, sml_time_seconds.size-1)
    use_right = (
        np.abs(sml_time_seconds[right]-observation_seconds) <
        np.abs(sml_time_seconds[left]-observation_seconds))
    nearest = np.where(use_right, right, left)
    matched = (
        np.abs(sml_time_seconds[nearest]-observation_seconds) <=
        tolerance_seconds)
    return nearest, matched
