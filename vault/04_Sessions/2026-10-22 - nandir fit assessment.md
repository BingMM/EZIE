# nandir fit assessment

Date: 2026-10-22

## Question

Assess whether the separate `nandir` project, which generalizes linear inverse
problems, is a useful architectural fit for EZIE. This was an EZIE assessment;
no `nandir` or EZIE solver integration was requested or made.

## Confirmed algebraic fit

The core systems are the same. EZIE solves a weighted regularized normal
equation and nandir exposes the generic form

`(G.T W G + R) m = G.T W d`.

EZIE's two scaled regularization terms map directly to two nandir
`QuadraticRegularization` objects:

- identity with `lreg=10**l1` and `scale=True`;
- `LTL` with `lreg=10**l2` and `scale=True`.

Using a real post-launch event with E/N components from MEM 1 and a reduced
EZIE grid (`G` shape 480 by 207), a nandir solver initialized from EZIE's
precomputed `GTG` and `GTd` reproduced the current implementation with these
relative differences:

- regularization matrix: `1.18e-16`;
- model estimate: `3.89e-15`;
- posterior covariance: `5.35e-15`;
- model resolution matrix: `3.78e-15`.

All compared outputs were finite. This establishes numerical compatibility to
machine precision for the tested case.

## Architectural boundary

The natural boundary is below `ezie.Model`: nandir can own regularization
assembly, factorization reuse, solving, posterior model covariance, and model
resolution. EZIE should retain Level-2 ingestion, component and MEM selection,
units and coordinate conventions, SECS grid and design-matrix construction,
observation-covariance semantics, scientific construction of `LTL`,
two-parameter regularization selection, field/current evaluation, plotting,
and validation.

Passing precomputed `GTG` and `GTd` is the lowest-risk first integration. It
reproduces current results, but does not remove EZIE's present dense inversion
of observation covariance `Q`. A deeper path through nandir's `G`, `d`, and
weights requires first establishing sparse precision or whitening semantics,
because nandir consumes a precision matrix or weights while EZIE currently
stores covariance.

Nandir's generic regularization-path helper is one-dimensional and its exposed
data-misfit norm is unweighted, so neither directly replaces EZIE's current
two-parameter, covariance-weighted L-curve selection. Prediction uncertainty
`diag(Gp Cmpost Gp.T)` also remains an EZIE evaluation concern. Nandir's
least-squares fallback after Cholesky failure may change failure behavior and
should be an explicit adoption decision.

## Recommendation

If integration is pursued, do it in stages:

1. Add an equivalence test that protects the current EZIE solve, posterior
   covariance, and resolution results.
2. Delegate only the precomputed-normal-equation solve to nandir.
3. Benchmark full-size runtime and memory.
4. Consider moving observation weighting only after covariance/precision and
   sparse or whitening behavior are validated.

The current environment cannot run nandir's focused test suite because
`pytest` is not installed. The direct numerical comparison above therefore
provides useful evidence, but is not a substitute for an automated integration
test.

## Integration outcome

The user selected the lowest-risk route. EZIE now creates a lazy
`nandir.Solver` from its existing `GTG`, `GTd`, and fully assembled
regularization matrix. The solver backs `Model.m`, `Model.Cmpost`, and
`Model.Rm`; `Model.c_factor` forwards nandir's cached Cholesky factor for
compatibility with the existing regularization optimizer.

Nandir is pinned in `pyproject.toml` to upstream commit
`e452b5c6a4f7340143b15ce552ae79447490cee0`. A new standard-library unit test
compares the delegated outputs with the former SciPy Cholesky calculation and
checks that `reset_solution()` creates a fresh solver. Both tests pass in the
dedicated EZIE environment.

The real-data E/N, MEM-1 reduced-grid verification produced relative
differences of `3.788e-15` for the model, `3.617e-15` for posterior covariance,
and `3.584e-15` for model resolution. The `Model` and
`RegularizationOptimizer` public interaction remains unchanged.

The local nandir checkout was then installed in editable mode in the dedicated
EZIE environment. The same two tests pass without a temporary `PYTHONPATH`.
