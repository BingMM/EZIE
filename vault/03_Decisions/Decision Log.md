# Decision Log

Last reviewed: 2026-10-22

## 2025-11-03 — Use the BSD 3-Clause license

The historical log records the license change, and the repository contains a
BSD 3-Clause `LICENSE` file.

## 2026-10-21 — Adopt the standard repository project vault

The repository root `AGENTS.md` is the machine-readable operational contract.
The repository-root `vault/` is the single live technical-memory system. The
earlier `log/EZIE/` notes and images were migrated into the vault as historical
evidence; local Obsidian state is ignored.

The migration does not imply new scientific progress. Portfolio priority and
the next scientific target remain user decisions.

## 2026-10-21 — Preserve the pre-launch code as a release baseline

Publish the scientifically unchanged, documentation-cleaned code as the
archival pre-release `v0.1.0-prelaunch`. Develop compatibility with real
flight data on a separate long-lived `flight-data` branch.

The pre-release must state that the code predates launch, is not validated
against real flight data, is not an official EZIE Level-3 producer, has
incomplete dependency metadata, and has no automated regression suite. The
first branch milestone is to characterize representative real Level-2 data
before changing the Level-3 inversion.

## 2026-10-22 — Use component-aware post-launch Data generation

`ezie.utils.generate_ezie_data()` is the boundary between post-launch Level-2
inputs and the internal `Data` representation. It accepts file paths, open
NetCDF datasets, or extracted mappings. The default is all vector components;
unique E/N/U subsets and scalar `F` magnitude are explicit alternatives. Any
unique non-empty ordered subset of MEMs 1–4 may be selected, with all four as
the default. Omitted components must also omit every associated covariance
term. Internal magnetic units remain tesla and tesla squared.

## 2026-10-22 — Use nandir behind the precomputed normal-equation boundary

Keep EZIE responsible for data ingestion, units, geometry, SECS design
matrices, observation covariance, regularization construction and selection,
evaluation, plotting, and validation. Delegate the solve, cached factorization,
posterior model covariance, and model resolution to `nandir.Solver` using
EZIE's existing `GTG`, `GTd`, and assembled regularization matrix.

Preserve the public `Model.m`, `Model.Cmpost`, `Model.Rm`, and
`Model.c_factor` interfaces. Pin nandir to the reviewed upstream commit in
`pyproject.toml`. Moving observation weighting or regularization selection into
nandir is explicitly outside this first integration step.

## 2026-10-22 — Invalidate all downstream state when data changes

`Model.add_data()` preserves the configured grid parameters, center MEM, and
fixed-versus-automatic regularization intent, but invalidates every object
derived from the observations: grid, regularization machinery, design and
normal equations, solver products, evaluator products, and resolution. This
prevents a reused `Model` from silently combining a new event with cached state
from an earlier event.

## 2026-10-22 — Propagate posterior uncertainty through prediction operators

Coordinate transformations of posterior uncertainty must act on the SECS
prediction matrices before contraction with `Cmpost`; standard deviations are
not vector components. Scalar-magnitude uncertainty uses the directional
operator along the predicted total magnetic field. Both approaches retain the
posterior component cross-correlations without constructing full field-space
covariance matrices.

## 2026-10-22 — Use the CHAOS time-dependent internal field as the reference

The CHAOS main-field option deliberately calls only
`CHAOS.synth_values_tdep()`. This is the time-dependent internal field and does
not include the external GSM or SM contributions. The separate static internal
lithospheric field is also omitted, matching the intended core/main-field role
and the closest CHAOS analogue to the IGRF option. Use the complete event
timestamp when converting to MJD2000.

## 2026-10-22 — Support a full internal CHAOS background for Level-2 data

Keep the operational Level-2 IGRF-relative perturbations as the default input.
When explicitly requested, replace that reference component-wise using
`dB_CHAOS = dB_L2 + B_IGRF_L2 - B_CHAOS`. Use the ancillary IGRF vector stored
in the Level-2 product so the operational subtraction is reversed exactly.

For this data correction, the CHAOS field contains both the time-dependent
core field and the static lithospheric field, but no external GSM or SM field.
Evaluate it at every observation time and WGS84 MEM position, using explicit
geodetic-to-geocentric coordinate and component transformations. Scalar data
use `F_retrieved - |B_CHAOS|`. Measurement covariance is unchanged; background
model uncertainty is not yet represented.

This does not change the separate evaluator decision above: the evaluator's
linearization field continues to use only the time-dependent core field.

## 2026-10-23 — Base the next inversion extension on scalar gradients

Do not use the delivered three-component perturbations as inversion input
while their unexplained rotation remains unresolved. Use the rotation-invariant
scalar perturbation instead.

The next inversion formulation should fit along-track scalar contrasts or an
equivalent nuisance-offset projection, with one constant baseline parameter
per MEM. Apply the observation transformation consistently to the scalar data,
SECS scalar forward operator, and covariance. The solved SECS coefficients
still define the full east, north, and up perturbation field; gradients are the
observation space, not the model representation.

Track gradients do not determine absolute baselines or model modes that look
constant along every observed track. Report recovered offsets as quantities
conditional on the regularized SECS solution, and assess resolution of the
broad-scale field explicitly.

## 2026-10-23 — Include the lithospheric field in evaluator CHAOS backgrounds

Supersede the 2026-10-22 evaluator-specific core-only choice. When
`B0_model='chaos'`, `Evaluator.calc_main_field()` uses the complete internal
CHAOS field: time-dependent core plus static lithosphere. External GSM and SM
sources remain excluded. This aligns the evaluator's intended background
content with CHAOS-referenced Level-2 extraction, although stored per-sample
backgrounds remain preferable at observation locations because extraction
performs the explicit WGS84 coordinate conversion.

## 2026-10-23 — Default scalar inversions to the debiased Level-2 product

Use `debiased_dbmag#` by default for scalar data in both absolute and gradient
inversions. Preserve `retrieved_dbmag#` as an explicit diagnostic option via
`scalar_source='retrieved'`. When changing from IGRF to CHAOS, select the
Level-2 scalar source first, add the ancillary IGRF magnitude, and subtract the
complete internal CHAOS magnitude.

## 2026-10-23 — Use the retrieved scalar as input to gradient fits

Keep `debiased_dbmag#` as the default scalar product, including for ordinary
absolute scalar inversions. For the explicit gradient workflow, select
`scalar_source='retrieved'` so the original Level-2 track constants remain
visible at the input boundary and are removed transparently by the model's
nuisance-offset transformation. Treat an absolute inversion of the debiased
product as a comparison. Both routes discard essentially the same four
constant track modes; neither restores their physical information.
