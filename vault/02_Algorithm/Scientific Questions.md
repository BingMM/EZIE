# Scientific Questions

Last reviewed: 2026-10-21

## Inversion structure

**Confirmed from code:** `Model` builds a cubed-sphere SECS grid, forms
magnetic design matrices through `secsy`, weights observations with scalar or
dense covariance, and solves a regularized inverse problem. Evaluation and
resolution are separated into dedicated classes.

## Component-dependent resolution

**Historical hypothesis:** the 2025-11-03 diagnostic reported poor resolution
when solving with a single east, north, or up component; east+north and
east+up performed well, north+up poorly, and all three well. This needs a
reproducible metric, frozen input, and regression test before it is treated as
a current result.

## Regularization selection

**Confirmed from code:** the optimizer contains spline L-curve and grid-search
machinery.

**Historical hypothesis:** knee selection was unstable because of the
admissible-curve choice and spline boundary derivatives. The final notes say a
permanent offset of five points was already in use, but do not demonstrate
robustness across cases.

## Observation altitude

**Open scientific question:** historical work questioned both a fixed 80 km
height and the height of minimum retrieved temperature. A temperature-weighted
average over roughly 60–95 km produced a steadier 80–82 km effective height in
one exploration, but no validated forward-model prescription is recorded.

## Ground and multi-satellite information

**Confirmed from code:** SuperMAG observations can be fetched and used for
validation plots.

**Not yet established:** assimilation of ground magnetometers into the solve,
confidence regions, and the joint geometry/semantics for satellites A, B, and
C.
