# Vault Conventions

Last reviewed: 2026-10-21

## Ownership

- `01_Project/Project Brief.md`: stable purpose, scope, and code map.
- `01_Project/Current State.md`: authoritative verified snapshot.
- `02_Algorithm/`: durable scientific and algorithmic interpretation.
- `03_Decisions/Decision Log.md`: durable decisions with rationale.
- `04_Sessions/`: dated historical evidence, including the migrated 2025 log.
- `05_Handoff/Handoff - Latest.md`: concise operational continuation point.

## Evidence labels

- **Confirmed**: supported by live code, Git, a focused check, a reproduced
  result, or a direct user statement.
- **Hypothesis**: plausible but not adequately verified against current code.
- **Superseded**: retained to explain history but no longer current.

Prefer links over repeating the same account in several notes. Replace stale
live state rather than appending a diary. Never store credentials, signed
download URLs, or private access tokens in the vault.
