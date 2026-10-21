# Handoff - Latest

Last reviewed: 2026-10-21

Branch: `flight-data`

Pre-launch release commit: `9508bf9595b4a2ad70b0a9fc3e75d9176c8ad59f`

## Summary

The repository now has standard root agent instructions and a structured
technical vault. The useful 2025 Obsidian log and figures were migrated into
`04_Sessions/`. This was a documentation architecture change only; no package
logic or scientific result was changed.

Commit `9508bf9` is pushed on `main` and tagged `v0.1.0-prelaunch` on GitHub.
The `flight-data` branch was created from that commit and is also pushed. The
hosted GitHub pre-release was published and verified through GitHub's public
API. Exact release notes are recorded in the 2026-10-21 session note.

The package structurally compiles, but it is not reproducible from
`pyproject.toml`, has no automated tests, and does not import in the current
environment because `kneed` is missing.

## Next actionable step

Identify a representative real Level-2 file and characterize its schema,
units, coordinate frames, quality flags, covariance, missing-data behavior,
and spacecraft/MEM identifiers before changing the Level-3 inversion.

Establish a reproducible dependency environment and minimal regression
protection before changing pre-launch scientific assumptions.

## Blockers and risks

- Current activity, priority, collaborator commitments, and deadlines are not
  recorded.
- Real flight data may differ from the pre-launch examples in schema,
  calibration, quality flags, covariance, geometry, or missing-data behavior;
  these differences have not yet been inventoried.
- Package dependencies are undeclared and the current environment lacks
  `kneed`.
- No automated regression suite exists.
- Historical scientific observations have not been reproduced against this
  checkout.
- Likely access tokens were removed from a live historical note, but the
  original strings may remain in Git history. Rotate or revoke them if they
  could still be valid.

## Verification

- Passed: `python -m compileall -q ezie scripts`
- Failed as expected in the current environment:
  `python -c 'import ezie; print(ezie.__all__)'`
- Failure: `ModuleNotFoundError: No module named 'kneed'`
- Automated tests: none tracked

## Portfolio impact

- Central update needed: No
- Changes: None outstanding; the central card records the published hosted
  pre-release and branch.
- Sync summary: EZIE is the most modern open-source L2-to-L3 codebase, but it
  predates launch. The cleaned hosted pre-release and separate `flight-data`
  adaptation branch are published; the first real-data milestone and project
  priority remain outstanding.
