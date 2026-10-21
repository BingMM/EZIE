# Project Brief

Last reviewed: 2026-10-21

## Purpose

EZIE is a three-CubeSat mission that remotely senses the mesospheric magnetic
field using four MEM measurement tracks per spacecraft. This repository is a
research implementation designed to ingest EZIE Level-2 magnetic-field data
and produce Level-3 equivalent ionospheric electric-current estimates with
SECS.

It is explicitly not the official repository for generating EZIE Level-3
products.

## Development context

**Confirmed by the user on 2026-10-21:** this is the most modern open-source
codebase for handling EZIE data. Its current implementation and assumptions
were developed before the mission launched. Real flight data may therefore
require substantial adaptation while the pre-launch implementation remains a
valuable reproducible reference.

## Scope

The repository currently covers:

- ingestion and organization of MEM magnetic observations and covariance;
- scalar and vector observation handling;
- SECS grid construction and regularized current inversion;
- automatic regularization-parameter exploration;
- current and magnetic-field evaluation;
- spatial-resolution diagnostics;
- plotting and comparison with SuperMAG ground magnetometers.

Confidence regions, ground-magnetometer assimilation, and coordinated use of
multiple EZIE satellites were recorded as future questions, not verified
capabilities.

## Code map

- `ezie/data.py` and `ezie/mem.py`: observation containers.
- `ezie/model.py`: grid, covariance, design matrix, solve, and component
  orchestration.
- `ezie/regularization_optimizer.py`: L-curve and grid-search machinery.
- `ezie/evaluator.py`: equivalent-current and magnetic-field evaluation,
  including background-field models.
- `ezie/resolution.py`: resolution matrices and point-spread diagnostics.
- `ezie/validation.py` and `ezie/supermag_api.py`: SuperMAG inventory, data,
  and comparison support.
- `ezie/visualization.py`: standard diagnostic plots.
- `scripts/`: interactive scalar-inversion and resolution examples.
- `data/`: versioned sample L1/L2-style inputs, station metadata, and a CHAOS
  model file.

## Scientific context

The inverse problem follows the SECS approach represented by `secsy`. The main
scientific uncertainties retained from the 2025 work concern stable
regularization selection, the observation-altitude representation,
component-dependent resolving power, use of ground magnetometers, and the
eventual multi-satellite configuration.
