# Decision Log

Last reviewed: 2026-10-21

## 2025-11-03 — Use the BSD 3-Clause license

The historical log records the license change, and the repository contains a
BSD 3-Clause `LICENSE` file.

## 2026-10-21 — Adopt the standard repository project vault

The repository root `AGENTS.md` is the machine-readable operational contract.
The repository-root `vault/` is the single live technical-memory system. The
earlier `log/EZIE/` notes and images were migrated into the vault as historical
evidence; local Obsidian state is ignored.

The migration does not imply new scientific progress. Portfolio priority and
the next scientific target remain user decisions.

## 2026-10-21 — Preserve the pre-launch code as a release baseline

Publish the scientifically unchanged, documentation-cleaned code as the
archival pre-release `v0.1.0-prelaunch`. Develop compatibility with real
flight data on a separate long-lived `flight-data` branch.

The pre-release must state that the code predates launch, is not validated
against real flight data, is not an official EZIE Level-3 producer, has
incomplete dependency metadata, and has no automated regression suite. The
first branch milestone is to characterize representative real Level-2 data
before changing the Level-3 inversion.
