# CHAOS Level-2 background replacement

Last reviewed: 2026-10-22

## Result

`ezie.utils.extract_data()` now accepts `background='chaos'` in addition to
the backward-compatible default `background='igrf'`. The correction is

`dB_CHAOS = dB_L2 + B_IGRF_L2 - B_CHAOS`.

The stored ancillary IGRF N/E/D components are used rather than a newly
evaluated IGRF model. CHAOS is evaluated at every observation time and WGS84
MEM position. Its time-dependent internal and static lithospheric fields are
included; GSM and SM external fields are excluded. Scalar perturbations use
the retrieved total-field magnitude minus the synthesized CHAOS magnitude.

The deterministic reference-field replacement does not alter the Level-2
measurement covariance. Uncertainty in either background model is not added.

`generate_ezie_data()` exposes the same `background` and optional
`chaos_model` arguments when given a file path or open dataset. An extracted
mapping retains its existing background and cannot be relabeled without the
source fields required to perform the correction.

## Code organization

CHAOS path resolution, cached loading, spherical synthesis, and WGS84
geodetic synthesis are centralized in `ezie.utils`. The evaluator imports the
shared spherical function but requests `include_static=False`, retaining its
core-only linearization-field semantics.

## Verification

- `python -m unittest discover -s tests -v`: 22 tests passed.
- `python -m compileall -q ezie tests`: passed.
- A 240-sample EZIE-A Level-2 event produced finite IGRF- and CHAOS-relative
  vector and scalar data.
- For MEM 1, maximum absolute corrections were 54.4 nT east, 70.5 nT north,
  107.4 nT up, and 102.8 nT scalar.
- Vector covariance and scalar variance were unchanged exactly.
- Direct vector and scalar `generate_ezie_data(path, background='chaos')`
  construction succeeded with 240 finite samples.

## Next scientific check

Run otherwise identical inversions with IGRF- and CHAOS-relative inputs on
several events and compare inferred current structure. Treat background-model
uncertainty as a separate future covariance question.
