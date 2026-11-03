# Model lifecycle and evaluator callers

Date: 2026-10-22

## Implemented

- Updated `Plotter.plot_data_model_comparison()` and `Validation.ev_sm()` to
  request mean and posterior uncertainty from their separate evaluator APIs.
- Normalized the plot axes to shape `(nmem, ncomp)`, including the valid
  one-MEM, one-component case.
- Made `Model.add_data()` invalidate the grid, regularization machinery,
  design and normal equations, nandir solver, evaluator, and resolution.
- Preserved whether each regularization parameter was configured explicitly or
  should be selected automatically when a model is reused with new data.
- Corrected scalar radius expansion in `Data` from the reversed list
  multiplication to one radius value per MEM.
- Preserved the complete post-launch observation time sequence in `Data.dates`
  for validation while retaining the midpoint in `Data.date`.

## Verification

Seven standard-library tests pass in the dedicated EZIE environment. They cover
nandir equivalence and reset, split evaluator callers, data-change cache
invalidation, scalar radius and observation-interval handling, and the
one-by-one plot axes case.

A real flight-data E-component, MEM-1 reduced-grid data/model comparison also
completed. Its axes had shape `(1, 1)` and the model coefficients were finite.

## Uncertainty-propagation analysis

Standard deviations are not vector components and must not be passed through a
coordinate-vector rotation directly. For a local linear transform `A`, the
covariance transforms as `A C A.T`. Equivalently, transform the relevant SECS
prediction matrices and calculate their posterior variances using `Cmpost`.

For scalar magnitude `F = |B0 + dB| - |B0|`, first-order posterior variance is
`u.T C_B u`, where `u` is the unit vector along `B0 + dB`. This includes all
component cross-covariances. The present root-sum-square of component standard
deviations is instead `sqrt(trace(C_B))` and is not the uncertainty of `F`.

## Uncertainty propagation implemented

The remaining issue was resolved in the same session:

- QD magnetic-field/current and cubed-sphere current operators are transformed
  pointwise before calculating posterior standard deviations.
- Scalar-magnitude uncertainty uses
  `G_F = u_E G_E + u_N G_N + u_U G_U`, evaluated along the predicted total
  magnetic field.
- Tiny negative variances from floating-point roundoff are clipped to zero
  before taking square roots.
- Scalar comparison no longer computes unused vector-component uncertainties.

Synthetic tests explicitly reproduce `A C A.T` and directional scalar
covariance results. Eleven focused tests pass in total. Reduced real-data scalar
comparison and solution plots complete with finite scalar, QD magnetic, and
QD/cubed-sphere current uncertainties.

## CHAOS reference field

The absolute CHAOS MAT path was replaced with a path derived from
`ezie/evaluator.py`, and the parsed model is cached once per process. The MJD
conversion now consumes the complete midpoint `datetime` instead of truncating
to midnight.

ChaosMagPy 0.15 documents `synth_values_tdep()` as the time-dependent internal
field. The code intentionally does not call the external `synth_values_gsm()`
or `synth_values_sm()` methods and does not add the separate static internal
lithospheric model. Two focused tests protect the relative path and source/time
selection. All 13 tests pass. A real flight-data evaluation started from
`/tmp` returned finite main-field magnitudes near 53,000 nT.
