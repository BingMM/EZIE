# Scalar-gradient inversion design

Last reviewed: 2026-10-26

## Scientific constraint

The user reports an unexplained rotation in the Level-2 three-component
perturbations that cannot currently be accounted for. The next inversion
extension must therefore use the rotation-invariant scalar field rather than
E/N/U observations.

## Target formulation

For MEM track `k`, represent the scalar data as

`d_k = Gs_k m + a_k + error`,

where `m` is the existing SECS model, `Gs_k` is its scalar magnetic forward
operator, and `a_k` is one constant offset for the track. There are four
offsets for one spacecraft file, rather than one offset per vector component.

An explicit first-difference operator `D` gives

- `d_fit = D d`
- `G_fit = D Gs`
- `Q_fit = D Q D.T`

with separate blocks for every MEM and no differences across gaps. An
equivalent covariance-weighted nuisance projection may be numerically cleaner
and permits the four offsets to be recovered after solving.

The inversion does not integrate a measured gradient into a field. It solves
the SECS coefficients directly against scalar gradients, then evaluates
`Ge m`, `Gn m`, and `Gu m` to retrieve the perturbation vector everywhere.

## Identifiability

Scalar gradients contain no information about a constant level on each MEM.
Recovered offsets are therefore conditional on the fitted, regularized SECS
field. Broad-scale model modes whose scalar signature is approximately
constant along all tracks will also be weakly constrained and must be exposed
through resolution or sensitivity diagnostics.

## Initial implementation choice

Start with the current linearized scalar SECS forward operator and full
covariance propagation. Keep exact nonlinear scalar forward modelling as a
later Gauss-Newton extension if the neglected second-order magnetic term is
large relative to the data uncertainty.

## Implemented result

`Model(fit_mode='gradient')` now provides the scalar-only gradient path while
the default `fit_mode='absolute'` preserves the existing workflow. Gradient
mode rejects vector observations, retains raw scalar perturbations as
`d_absolute`, constructs one sparse first-difference block per MEM, and exposes
the transformed `d`, `G`, and `Q` to the existing solver and regularization
machinery.

The fit covariance is `D Q D.T`, including the negative covariance between
adjacent differences. It is factored once with Cholesky and applied through
solves rather than an explicit inverse for normal-equation construction and
L-curve residuals. The nandir integration remains unchanged.

`Data` and `MEM` now retain the per-sample IGRF or CHAOS background vector and
magnitude. Gradient mode uses its unit direction to project the SECS vector
forward operator into scalar observation space. These background components
are deterministic model values and are distinct from the excluded Level-2
retrieved components.

`model.track_offsets` returns one variance-weighted scalar offset per MEM,
conditional on the regularized solution. The standard data/model comparison
adds those offsets in absolute scalar space, and
`plot_gradient_data_model_comparison()` displays the actual first differences.

## Verification

- Thirty-one focused tests pass, including offset invariance, MEM-boundary
  separation, correlated covariance propagation, conditional offset recovery,
  scalar-only enforcement, background projection, and cache invalidation.
- Compilation passes.
- A real 240-sample event with full internal CHAOS background produced 960
  absolute scalar observations and 956 fit differences.
- A reduced grid produced 54 finite SECS coefficients, finite retrieved E/N/U
  perturbations, and conditional offsets of about +148.2, -26.8, -40.9, and
  +254.4 nT for MEMs 1-4.
- The backward-compatible absolute scalar mode also completed on the same
  reduced 54-coefficient grid with finite coefficients.
- Absolute-with-offset and gradient-space comparison plots both completed.

## Remaining limitation

The first implementation separates the four MEMs but differences every pair
of adjacent samples within each MEM. It must be extended with segment breaks
before use on files containing missing samples or anomalously large time gaps.

## Full-grid comparison with the absolute scalar inversion

At the default 11,232-coefficient resolution for the 2025-05-15 02:24:54
event, fixed `l1=0.75` and `l2=2.5` gave normalized weighted residual norms of
1.027 for the absolute scalar fit and 1.011 for the gradient fit. After adding
the four conditional gradient offsets, the two linear scalar predictions at
the data locations had correlation 0.9986 and an RMS difference of 10.2 nT.

This agreement does not imply agreement of the retrieved vector field. At the
data locations, absolute-versus-gradient E/N/U correlations were 0.46, 0.90,
and 0.74. Across the full 80 km map they were 0.37, 0.78, and 0.65, with RMS
map differences of 121, 77, and 181 nT. The gradient map also had substantially
lower E/N/U RMS amplitudes. The result is consistent with the gradient fit
removing one scalar constant mode per MEM: the fitted offsets can vertically
align the scalar tracks for comparison, but they are nuisance parameters and
do not enter `plot_solution_cs()`. The absolute inversion instead has to map
those track levels into its SECS field. Scalar along-track observations also
do not uniquely determine all three vector components or the field between
tracks.

The shared plotting levels are fixed, so the visual difference is not caused
by independent color normalization. The same numerical `l1` and `l2` values
do not guarantee equal scale-dependent regularization because differencing
changes the spectrum of the data sensitivity matrix even though its median
diagonal scale is normalized. The user decided not to investigate this
regularization distinction at this point.

A consistency issue was also found: absolute scalar mode recomputes its
background through the evaluator, whereas gradient mode uses the complete
background stored in `Data` and the manual comparison uses that stored
background for both models. The evaluator now includes the lithospheric field,
but its spherical coordinate path is not identical to extraction's explicit
WGS84 conversion. On this event, the resulting background directions differ
by a median 0.21 degrees and at most 0.34 degrees. This does not explain the
large map difference, but both scalar modes should still use the same stored
background at observation locations.

## Level-2 raw and debiased scalar variables

The Level-2 file explicitly contains both `retrieved_dbmag#` and
`debiased_dbmag#`. On the checked event, every debiased MEM series has zero
mean, and `retrieved_dbmag# - debiased_dbmag#` is exactly constant within each
MEM: +178.165, -5.414, -37.603, and +277.722 nT. Also,
`retrieved_b_tot# - anc_model_bgeo#` reproduces `retrieved_dbmag#` to about
0.006 nT. This identifies the previously observed scalar/vector magnitude
offsets as the explicit Level-2 debiasing operation rather than an unexplained
numerical discrepancy.

The scalar loader now uses `debiased_dbmag#` by default for both absolute and
gradient inversions. The non-debiased product is available explicitly through
`scalar_source='retrieved'`. Background replacement is applied after selecting
the scalar source: the selected IGRF-relative perturbation is added to the
ancillary IGRF magnitude and the requested background magnitude is subtracted.

Since `D(d+c) = Dd`, starting from the Level-2 debiased scalar or adding its
constant back and then differencing produces the same fitted model, provided
the removed bias is exactly constant. On the real event, small Level-2 rounding
variations of roughly 0.001 nT led to only a `1.46e-6` relative difference
between reduced-grid gradient coefficient vectors and a `5.9e-5` nT RMS
difference in their absolute scalar predictions. The recovered
`track_offsets` shift with the chosen absolute input and are not independent
evidence for which level is physical.

This makes the gradient result a natural reconstruction of the variation that
is invariant to the documented Level-2 debiasing choice. It still cannot
recover any real broad-scale scalar signal removed with those constants. The
raw absolute solution and offset-invariant gradient solution should therefore
be retained as two interpretations until the Level-2 debiasing rationale or
independent absolute observations establish which baseline is scientifically
appropriate.

The user subsequently found virtually no difference between the absolute fit
of the debiased scalar and the gradient fit. For transparency, the intended
gradient workflow will use `scalar_source='retrieved'`, leaving the Level-2
constants present at ingestion and letting the explicit model transformation
remove them. The absolute debiased inversion remains the comparison path.

## Regularization-optimizer compatibility

Gradient mode supplies the optimizer with the same public quantities as
absolute mode: transformed `G` and `d`, the correctly transformed covariance
through `weighted_residual_norm`, `GTG`, `GTd`, and the resulting model
resolution. There is no theoretical or interface reason that the Gini/ridge
search should reject gradient observations.

A reduced real-event run confirmed that the Gini/ridge stage completes, but
automatic final L-curve selection raises `ValueError: x must be strictly
increasing if s = 0`. The residual norm was non-monotonic along the fitted
two-parameter ridge (10 non-positive steps out of 49). The equivalent reduced
absolute scalar run also failed (12 non-positive steps), showing that this is
not gradient-specific. The current code fits `mnorm` as a single-valued spline
of `rnorm`; a robust future repair should calculate curvature parametrically
along the regularization path rather than sorting the residual coordinates.
The user has deferred regularization work for now.

The user subsequently reported two visible knees in a completed optimization
run and clarified the published method's purpose: the Gini ridge explicitly
reduces the two regularization parameters to the one-dimensional family
`(l1, l2_ridge(l1))`. Multiple knees do not invalidate that reduction. Even a
classical one-parameter Tikhonov L-curve can have more than one local curvature
maximum when different spectral or spatial scales transition at different
regularization strengths.

Two cases must be distinguished. A monotone residual/model-norm curve with two
corners can represent genuine competing regularization scales. A curve that
folds because residual norm reverses direction instead indicates that the
estimated ridge switches branches or changes the balance between the two
penalties enough to weaken some model directions. The current ridge search
selects the Gini maximum independently at each `l1`, with no continuity
penalty, and then fits a spline; this can manufacture the second case. Ridge
continuation or a smooth-path penalty can suppress branch hopping, but no
defensible ridge constraint can guarantee a single physical knee.

For the user's completed run,
`np.count_nonzero(np.diff(model.regOpt.rnorm) <= 0)` returned zero. Its two
knees therefore occur on a monotonic residual path and are not caused by an
L-curve fold. Continuous ridge tracing is retained as a later idea rather than
the present task.

Before the 2026-10-26 change, `plot_L_curve()` always drew a second star labelled
`Concave offset` at hard-coded index 3. It was not a knee detected from the
curve. The obsolete offset result, property, and plot marker have now been
removed.

## Parametric-curvature implementation

On 2026-10-26, `max_curvature()` was changed to fit `rnorm(l1)` and
`mnorm(l1)` separately and evaluate the signed parametric curvature

`(r' m'' - m' r'') / (r'^2 + m'^2)^(3/2)`.

This avoids the former strictly monotonic-`rnorm` requirement. Negative signed
curvature is excluded as concave. Local positive-curvature peaks are ordered by
increasing `l1`; if more than one remains, the last is selected. If no local
peak is found but positive curvature exists, the largest positive value is the
fallback. The unused `stable_spline_mask` and misspelled `covex_mask` were
removed. Two focused tests cover a non-monotonic residual coordinate and the
last-peak rule; the full 40-test suite passes. A real-event ridge/L-curve rerun
remains necessary.
