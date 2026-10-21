# Current State

Last reviewed: 2026-10-21

Branch reviewed: `main`

Baseline commit reviewed: `6e4a22874d03ab062904ca4ccd5c4721c4d16866`

## Verified repository state

- **Confirmed by the user:** this is the most modern open-source EZIE data
  codebase, designed before launch to ingest Level-2 data and produce Level-3
  equivalent-current estimates.
- **Confirmed:** `main` matched `origin/main` at `6e4a228` before the
  documentation migration. This checkpoint adds root instructions and the
  standard vault without changing scientific code.
- **Confirmed:** the package exposes `Model`, `Data`, `MEM`,
  `RegularizationOptimizer`, `Plotter`, `Evaluator`, `Resolution`, and
  `Validation` through `ezie.__init__`.
- **Confirmed:** the code contains scalar and vector magnetic-observation
  paths, dense or diagonal observation covariance, regularized SECS solves,
  resolution machinery, and SuperMAG comparison support.
- **Confirmed:** `python -m compileall -q ezie scripts` passed on 2026-10-21.
- **Confirmed:** `import ezie` failed in the current environment because
  `kneed` is unavailable. `pyproject.toml` declares no runtime dependencies
  even though the package imports several external scientific libraries.
- **Confirmed:** there is no tracked automated test suite. The two scripts are
  interactive workflows with hard-coded local paths rather than tests.

## Last recorded technical position

The final 2025 session notes record SuperMAG validation safeguards and code
cleanup as completed. They retain the following unresolved work:

- ground magnetometers as inversion inputs, likely through a separate data
  representation;
- confidence-region support, if needed;
- coordinated handling of satellites A, B, and C after representative input
  examples become available;
- further cleanup and documentation.

Historical notes also report unstable automatic L-curve knee selection and
strong component dependence in resolution experiments. These observations
have not been reproduced against the current checkout and remain hypotheses.

## Risks and gaps

- The environment is not reproducible from package metadata because runtime
  dependencies are absent from `pyproject.toml`.
- There are no regression tests for component ordering, covariance layout,
  coordinate frames, regularization, resolution, or validation behavior.
- The scientific meaning of an effective observation altitude remains open;
  a fixed 80 km assumption and a temperature-weighted alternative were
  discussed historically but not established here.
- The repository includes versioned sample data, generated Python caches, and
  package metadata. Their cleanup or release policy has not been decided.
- A legacy session note contained likely access tokens. They are redacted in
  the live vault, but may remain in Git history and should be treated as
  compromised if still valid.

## Current project status

Portfolio status and priority are unknown. The user has registered the project
and confirmed that the cleaned pre-launch snapshot should be published as an
archival pre-release, followed by a separate `flight-data` branch for adapting
the implementation to real flight data. The first representative-data
validation milestone remains to be specified.
