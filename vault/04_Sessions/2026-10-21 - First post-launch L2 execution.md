# First post-launch L2 execution

Date: 2026-10-21

## Scope

Diagnosed the first attempt to run `scripts/Bscalar_inversion.py` against
`ezie_l2_20250515_004708_sva_v003_r000.nc4` from the post-launch Level-2
archive. The user's working script changes were preserved.

## Reproduced findings

- The new file exposes four 225-sample MEM tracks. The geographic positions,
  vector magnetic disturbances, and `cov_ned` arrays used by the script are
  finite and unmasked. `reference_altitude` is uniformly 80 km.
- Calling `Model()` without explicit regularization reaches
  `RegularizationOptimizer.automate_m_id()` and fails because it accesses
  `self.grid.mems`. MEM objects belong to `self.data.mems`.
- Git history confirms this is a pre-launch regression. Before commit
  `6e4a228`, the same calculation correctly iterated over
  `self.model.data.mems`; a convenience-property refactor accidentally changed
  it to `self.grid.mems`.
- The regression was repaired by iterating over `self.data.mems`. Against the
  representative real file, automatic selection now sees all four MEM tracks
  and returns model index 5533, which is valid for the 10,944-cell grid.
- Supplying the historical `l1=0.75, l2=2.5` values bypasses automatic
  regularization, as the pre-launch scalar example did.
- With explicit regularization, construction of `LTL` stops in
  `secsy.CSprojection.local2geo_enu_rotation()`. The grid is geometrically
  coherent but reaches 89.829 degrees geographic latitude. At one grid cell,
  nominally zero vertical/horizontal rotation terms reach approximately
  `1.6e-6`, just above the dependency's `1e-6` assertion tolerance.
- Running the same explicit-regularization workflow with Python assertions
  disabled completes. The solution has shape `(10944,)` and all coefficients
  are finite. This is diagnostic evidence only; disabling assertions globally
  is not a production fix and does not validate the scientific result.
- The user confirmed that a different post-launch event completes without the
  near-pole assertion. The pole failure is therefore event/grid specific, not
  a universal failure of the post-launch data path.
- Unit tracing confirms that `Data` and the SECS design matrices use SI units:
  magnetic perturbations must be in tesla and magnetic covariance entries in
  tesla squared. `Plotter.plot_solution()` converts predicted fields back to
  nT with `1e9`. The Level-2 fields are labeled nT; the covariance variable
  lacks a units attribute, but its diagonal square roots are physically
  consistent with uncertainties in nT, so its entries are covariance in nT
  squared. The required input conversions are therefore `dB * 1e-9` and
  `cov_ned * 1e-18`.

## Interpretation

The displayed polar geometry can be correct while the derivative operator
fails: plotting uses mapped positions, whereas `get_Le_Ln()` also constructs a
geographic east/north basis. That basis becomes numerically ill-conditioned
near the geographic pole. The present failure is a tolerance/coordinate-basis
edge case rather than evidence that the plotted grid is malformed.

## Next technical step

Apply the SI conversions and reassess the solution and uncertainty plots. Keep
the near-pole rotation issue as an event-specific robustness task; do not use
`python -O` as the operational solution.
