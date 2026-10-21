# Current State

Last reviewed: 2026-10-21

Branch reviewed: `flight-data`

Release commit reviewed: `9508bf9595b4a2ad70b0a9fc3e75d9176c8ad59f`

## Verified repository state

- **Confirmed by the user:** this is the most modern open-source EZIE data
  codebase, designed before launch to ingest Level-2 data and produce Level-3
  equivalent-current estimates.
- **Confirmed:** `v0.1.0-prelaunch` points to documentation-only commit
  `9508bf9`. Relative to the prior scientific-code commit `6e4a228`, it adds
  root instructions and the standard vault without changing scientific or
  package files.
- **Confirmed:** `main`, `v0.1.0-prelaunch`, and the starting point of
  `flight-data` have been pushed to `origin`.
- **Confirmed:** GitHub Release `v0.1.0-prelaunch` was published on 2026-10-21
  as a non-draft pre-release with the intended limitations and release notes.
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
and confirmed the archival pre-release plus branch strategy. The cleaned tag
and hosted pre-release plus the `flight-data` branch are published. The first
representative-data validation milestone remains outstanding.
