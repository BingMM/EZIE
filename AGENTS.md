# EZIE repository instructions

## Purpose

This repository contains research code for estimating equivalent ionospheric
electric currents from EZIE Level-2 magnetic-field observations with
Spherical Elementary Current Systems (SECS). It is an exploratory scientific
codebase, not the official producer of EZIE Level-3 data.

The main tracks are:

- `ezie/`: reusable data, inversion, regularization, evaluation, resolution,
  plotting, and SuperMAG-validation components;
- `scripts/`: interactive example and diagnostic workflows;
- `data/`: versioned sample inputs needed by the current examples;
- `vault/`: durable technical memory and project handoff.

Ignored `Frank_*` and `Rafael_*` directories are local analysis/output trees.
Do not treat them as authoritative or modify them during routine work.

## Startup reading order

Before substantive work:

1. inspect the live branch, `HEAD`, and tracked worktree status;
2. read `vault/START HERE - AI Onboarding.md`;
3. read `vault/01_Project/Project Brief.md`;
4. read `vault/01_Project/Current State.md`;
5. read `vault/05_Handoff/Handoff - Latest.md`;
6. read relevant algorithm, decision, and dated session notes only as needed.

## Source-of-truth order

When sources disagree, use:

1. live code, Git state, focused tests, and regenerated results;
2. `Current State.md` and `Handoff - Latest.md`;
3. decision and algorithm notes;
4. dated session history.

Treat historical scientific observations as hypotheses unless they are
reproduced against the current code and data.

## Scientific and safety rules

- Preserve the distinction between an equivalent-current research estimate
  and an official EZIE Level-3 product.
- Do not silently change coordinate frames, component ordering, observation
  altitude, units, covariance layout, or regularization semantics.
- Validate scalar and vector observation paths separately. Historical notes
  show strong component-dependent reconstruction behavior.
- Do not run SuperMAG downloads or other network workflows without confirming
  the required credentials, time range, and expected data volume. Never store
  credentials, signed URLs, or access tokens in code or the vault.
- Preserve versioned sample data unless a replacement and migration have been
  explicitly agreed. Do not add generated figures, large external datasets,
  caches, or local result trees merely because they are useful in one run.
- Leave ignored collaborator/output directories untouched unless they are the
  explicit task target.
- Keep scientific workflows linear and editor-friendly, using `#%%` sections
  and short navigational comments where useful.

## Verification

There is currently no automated test suite and package dependencies are not
declared completely in `pyproject.toml`.

Safe structural check:

```bash
python -m compileall -q ezie scripts
```

Environment-dependent import check:

```bash
python -c 'import ezie; print(ezie.__all__)'
```

Run example inversions only after checking their hard-coded paths, input
files, dependency environment, and output locations. Do not use them as
unattended smoke tests: they may be scientifically expensive and may access
external services.

## Memory checkpoints

After a verified implementation, reproduced scientific result, durable
decision, changed blocker, or changed next action:

1. add a dated note under `vault/04_Sessions/` only when the evidence merits
   a historical record;
2. rewrite `vault/01_Project/Current State.md` when verified state changed;
3. append only durable choices to `vault/03_Decisions/Decision Log.md`;
4. replace the operational content of
   `vault/05_Handoff/Handoff - Latest.md`;
5. refresh its `Portfolio impact` section, even when no central update is
   needed.

Do not write raw logs or transient speculation into live memory. Preserve the
division of ownership: this repository holds technical continuity, while the
central second brain holds priority, deadlines, and cross-project decisions.
