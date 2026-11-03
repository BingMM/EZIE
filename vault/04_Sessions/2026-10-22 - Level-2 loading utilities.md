# Level-2 loading utilities

Date: 2026-10-22

## Change

Moved the reusable post-launch Level-2 ingestion functions from
`scripts/Bscalar_inversion.py` into `ezie/utils.py`:

- `extract_data`
- `load_nc4_file`
- `load_file`
- `load_files`

The script now imports `load_file` from the package utility module. The utility
owns the nT-to-T and nT-squared-to-T-squared conversions and the NED-to-ENU
covariance transformation. `load_file` uses a context manager so the NetCDF
dataset also closes if extraction fails.

## Verification

Using `ezie_l2_20250515_022454_sva_v003_r000.nc4`:

- the loader returned EZIE-A with 240 samples;
- `load_files` returned one extracted record for a one-file input;
- `dBe1` ranged from approximately `-1.133e-6` to `1.215e-6` T;
- covariance-derived standard deviations ranged from approximately
  `1.190e-7` to `4.202e-7` T;
- the transformed ENU east variance matched the source NED east variance after
  the `1e-18` conversion;
- both the utility module and analysis script passed syntax parsing.

## Component-aware Data generation

Added `generate_ezie_data(data, comp='all')` to `ezie/utils.py`. It accepts a
file path, an open NetCDF dataset, or an already extracted mapping. `all` and
`ENU` select all vector components; any unique combination of E, N, and U is
accepted in any order. Repeated and unknown components are rejected. `F`,
`mag`, and `magnitude` select the scalar-magnitude path; scalar and vector
inputs cannot be mixed because the current model covariance logic does not
support that combination.

The optional `mems` argument selects any unique, non-empty combination of MEM
numbers 1 through 4 and preserves the requested order. Its default is all four
MEMs. A single integer is accepted for a one-MEM selection.

Vector subsets include only their selected observations, variances, and
pairwise covariance terms. `Model.generate_Q_dense()` was generalized to
assemble the covariance matrix from the selected component set rather than
assuming ENU.

For scalar data, the extracted perturbation is
`retrieved_b_tot - anc_model_bgeo`, and its variance is
`retrieved_b_tot_err**2`. Main-field calculations were converted from the nT
returned by IGRF/CHAOS to internal tesla. Scalar design-matrix projection was
also corrected to evaluate the main-field unit vector at observation
coordinates and broadcast it across model columns.

Verification covered all single and paired vector subsets, arbitrary ENU
ordering, all three input routes, duplicate rejection, and covariance assembly
for EN. A reduced-grid scalar inversion completed with a 960-element data
vector, a `(960, 207)` design matrix, and 207 finite model coefficients.
All 15 non-empty MEM subsets were also constructed successfully. An ordered
`mems=[4, 2]`, `comp='EN'` case produced a 960-element data vector and a finite
`(960, 960)` covariance matrix; invalid, repeated, and empty MEM selections
were rejected.

## Evaluation performance diagnosis

Accessing `model.ev.Be` currently calls `get_B_predictions()`, which computes
all three mean magnetic components and all three posterior standard-deviation
maps. The uncertainty calculation requests `model.Cmpost`, explicitly forms
the dense inverse of `GTG + reg`, and evaluates expressions of the form
`diag(G Cmpost G.T)`.

For the current full-resolution event, there are 11,232 model parameters. One
dense model covariance or one square evaluation matrix occupies about 0.94 GiB
in float64; the three magnetic design matrices occupy about 2.82 GiB before
intermediates. This is not optimized for a mean-only `Be` request.

The appropriate refactor is to separate mean predictions from uncertainty
predictions. Exact diagonal prediction variances can then be computed from the
existing Cholesky factor in chunks using
`diag(G A^-1 G.T) = column_norm(L^-1 G.T)^2`, avoiding the explicit posterior
covariance and full predicted covariance matrices. Full-grid exact uncertainty
will remain computationally expensive even after reducing memory use.

The user subsequently separated mean and uncertainty evaluation for both
magnetic field and current density. The retained dense-`Cmpost` path now
computes each requested diagonal as `einsum('ij,ij->i', G @ Cmpost, G)`, so it
does not form the full evaluation-point covariance `G @ Cmpost @ G.T`.
Verification matched the former expression to approximately `1.8e-15` on a
synthetic case and `1.7e-23` on a reduced EZIE case. Chunked Cholesky evaluation
remains a possible later alternative rather than part of this change.
