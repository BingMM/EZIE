# Handoff - Latest

Last reviewed: 2026-10-21

Branch: `main`

Baseline commit: `6e4a22874d03ab062904ca4ccd5c4721c4d16866`

## Summary

The repository now has standard root agent instructions and a structured
technical vault. The useful 2025 Obsidian log and figures were migrated into
`04_Sessions/`. This was a documentation architecture change only; no package
logic or scientific result was changed.

The package structurally compiles, but it is not reproducible from
`pyproject.toml`, has no automated tests, and does not import in the current
environment because `kneed` is missing.

## Next actionable step

Execute the confirmed baseline and branch strategy:

1. keep commit `6e4a228` identifiable as the exact pre-launch code baseline;
2. commit the documentation-only agent/vault migration without changing the
   scientific code;
3. create a long-lived `flight-data` branch for real-data adaptation;
4. make the first milestone ingestion and characterization of a representative
   real Level-2 file before modifying the Level-3 inversion.

The strategy was confirmed by the user on 2026-10-21. After creating the
release and branch, establish a reproducible dependency environment and
minimal regression protection before changing pre-launch scientific
assumptions.

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
- Changes: Record the archival pre-release and `flight-data` branch after they
  are published.
- Sync summary: EZIE is the most modern open-source L2-to-L3 codebase, but it
  predates launch. An archival pre-release and separate `flight-data`
  adaptation branch are confirmed; portfolio priority remains unknown.
