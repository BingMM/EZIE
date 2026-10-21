# 2026-10-21 — Pre-launch release and flight-data branch

## Completed

- Migrated the existing technical log into the standard repository vault.
- Added the repository-root `AGENTS.md`.
- Redacted likely legacy access strings from the live historical note.
- Committed the documentation-only checkpoint as `9508bf9` on `main`.
- Created and pushed annotated tag `v0.1.0-prelaunch` at `9508bf9`.
- Created and pushed `flight-data` from the tagged commit.

No files under `ezie/`, `scripts/`, or `data/`, and no package metadata,
README, or license files changed in the release commit.

## Hosted release status

Published by the user on 2026-10-21:

- URL: `https://github.com/BingMM/EZIE/releases/tag/v0.1.0-prelaunch`
- GitHub release ID: `419252779`
- Draft: No
- Pre-release: Yes
- Published at: `2026-10-21T07:06:18Z`

The public GitHub API confirmed the intended tag, title, and release body.

## Release title

EZIE v0.1.0-prelaunch — pre-launch L2-to-L3 reference

## Release notes

This archival pre-release preserves the most modern open-source EZIE
Level-2-to-Level-3 processing code developed before the mission launched.

The code ingests EZIE Level-2 magnetic-field observations and estimates
Level-3 equivalent ionospheric electric currents using SECS. This snapshot is
intended as a reproducible reference for comparison with the forthcoming
real-flight-data adaptation on the `flight-data` branch.

Important limitations:

- developed before launch and not validated against real EZIE flight data;
- not an official EZIE Level-3 production system;
- runtime dependencies are not completely declared in `pyproject.toml`;
- no automated regression test suite is included;
- historical scientific questions remain around regularization stability,
  effective observation altitude, component-dependent resolution, ground
  magnetometer assimilation, and multi-satellite handling.

Scientific code baseline before the documentation checkpoint: `6e4a228`.
Release tag commit, including cleaned documentation and project memory:
`9508bf9`.

## Verification

- Remote `main` contains `9508bf9`.
- Remote tag `v0.1.0-prelaunch` points to `9508bf9`.
- Remote branch `flight-data` was created from `9508bf9`.
- The hosted release is public, non-draft, and marked as a pre-release.
- The release commit was checked to contain documentation/vault changes only.
- All Python source files parsed successfully.
- Structural compilation passed before release preparation.
- Package import remains environment-blocked by missing `kneed`.

## Security note

Likely legacy access strings are redacted in the release tree. They may remain
in older public Git history and should be treated as compromised if still
valid.
