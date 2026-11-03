# Current State

Last reviewed: 2026-10-29

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
- **Confirmed:** structural compilation passes. Importing EZIE in the default
  Python still fails because scientific runtime dependencies are undeclared;
  the dedicated EZIE environment imports it successfully.
- **Confirmed:** `pyproject.toml` now pins the reviewed nandir upstream commit,
  but the package's other scientific runtime dependencies remain undeclared.
- **Confirmed:** a focused standard-library `unittest` now protects nandir
  delegation against the former SciPy Cholesky results. Broader component,
  covariance, geometry, regularization, and validation coverage is absent.
- **Confirmed on representative post-launch L2 data:** the revised scalar
  workflow reads four finite, unmasked 225-sample MEM tracks and constructs the
  expected polar grid. A pre-launch `self.grid.mems` automatic-regularization
  regression was repaired to use `self.data.mems` and now produces valid model
  index 5533. With explicit regularization, a near-pole `secsy` rotation
  assertion still stops the workflow; disabling assertions for diagnosis only
  produces 10,944 finite solution coefficients.
- **Confirmed by the user:** a different post-launch event runs without the
  near-pole assertion, making that failure event/grid specific.
- **Confirmed from code and the pre-launch example:** internal magnetic-field
  units are tesla and covariance units are tesla squared; plotting converts
  predictions to nT. Raw Level-2 nT fields and nT-squared covariance must be
  scaled by `1e-9` and `1e-18`, respectively.
- **Confirmed:** reusable post-launch NetCDF extraction and loading now live in
  `ezie/utils.py`. The working script imports `load_file`; the utility performs
  SI conversion, NED-to-ENU covariance transformation, and safe dataset
  closure. It was verified against a 240-sample EZIE-A file.
- **Confirmed:** `generate_ezie_data()` accepts paths, open NetCDF datasets, or
  extracted mappings and supports scalar magnitude or any non-repeating E/N/U
  subset. It also accepts any unique non-empty ordered selection of MEMs 1–4,
  defaulting to all four. Dense covariance assembly now follows the selected
  components. The scalar `F` path has consistent tesla units and
  observation-point main-field projection and completed a finite reduced-grid
  inversion.
- **Confirmed:** mean and uncertainty evaluation are now separated for magnetic
  field and current density. Diagonal prediction uncertainties use
  `einsum('ij,ij->i', G @ Cmpost, G)` and no longer materialize the full
  `G Cmpost G.T` matrix. The result matches the previous calculation to machine
  precision. `Cmpost` itself remains a cached dense matrix.
- **Confirmed:** EZIE now delegates its precomputed-normal-equation solve,
  cached factorization, posterior covariance, and model resolution to
  `nandir.Solver`. Existing `Model` properties remain compatible. On a
  real-data E/N, MEM-1 reduced-grid case, the integrated path reproduced the
  former solution, posterior covariance, and resolution matrix with relative
  differences between `3.58e-15` and `3.79e-15`.
- **Confirmed:** the local nandir checkout is installed in editable mode in the
  dedicated EZIE environment, and the integration tests pass there without a
  temporary `PYTHONPATH`.
- **Confirmed:** mean and uncertainty evaluator callers in visualization and
  validation now use the split APIs correctly. The data/model plot also
  normalizes Matplotlib axes for every one/many MEM-component combination.
- **Confirmed:** `Model.add_data()` invalidates all data-dependent caches while
  preserving configured fixed or automatic regularization intent. Scalar
  radius input to `Data` is replicated correctly across MEMs. Generated data
  retains the complete observation interval for validation while using its
  midpoint as `Data.date`. Seven focused tests pass, and a real one-MEM,
  one-component flight-data comparison plot completes with finite model
  coefficients.
- **Confirmed:** quasi-dipole magnetic/current and cubed-sphere current
  uncertainties now transform the prediction matrices before posterior contraction,
  equivalent to pointwise `A C A.T`. Scalar-magnitude comparison uncertainty
  uses the directional derivative along the predicted total field and therefore
  includes all posterior component correlations. Eleven focused tests pass; both
  affected plots also complete with finite uncertainties on reduced real data.
- **Confirmed:** shared CHAOS-8.4 loading and internal-field synthesis now live
  in `ezie.utils`; the repository-relative model is cached after first load.
  The evaluator imports this functionality and requests the time-dependent
  core plus static lithospheric field while excluding external sources.
  `extract_data()` additionally
  supports an explicit `background='chaos'` mode that reverses the stored
  ancillary IGRF subtraction and removes the CHAOS core plus static
  lithospheric field at every WGS84 observation position and timestamp. It
  excludes external GSM/SM fields, uses the exact CHAOS magnitude for scalar
  data, and leaves measurement covariance unchanged. `generate_ezie_data()`
  forwards the background choice for file and open-dataset inputs. Thirty-one
  focused tests pass. On a 240-sample flight event all corrected fields were finite; the
  largest absolute MEM-1 component correction was about 107.4 nT.
- **Confirmed on the same 240-sample event:** the delivered total field and the
  magnitude reconstructed from ancillary IGRF plus retrieved N/E/D differ by
  an almost exactly constant value on each MEM track: +178.16, -5.41, -37.60,
  and +277.72 nT for MEMs 1-4. After removing those constants, the discrepancy
  has only about 0.0012 nT standard deviation and 0.0023 nT maximum absolute
  deviation. The source is now identified: the file contains both
  `retrieved_dbmag#` and zero-mean `debiased_dbmag#`, and their exact constant
  difference is the four values above. `retrieved_b_tot# - anc_model_bgeo#`
  reproduces the non-debiased `retrieved_dbmag#` to within about 0.006 nT.
  This establishes that the component/scalar level difference reflects an
  explicit Level-2 debiasing step, but not which level is physically correct.
- **Confirmed by the user:** replacing the vector magnitude with the delivered
  scalar magnitude while retaining the vector direction did not improve the
  inversion. This correction constrains only one field-aligned combination of
  three possible component offsets and therefore does not resolve arbitrary
  per-track E/N/U baselines. The experimental corrected vectors also require
  transformed covariance before they are statistically valid input.
- **Confirmed by the user:** the Level-2 three-component perturbations contain
  an unexplained rotation that cannot currently be accounted for. The next
  inversion extension will therefore use only rotation-invariant scalar data.
  It should fit along-track scalar gradients with one constant offset per MEM,
  while retaining SECS coefficients as the model so the full perturbation
  vector can still be evaluated after the solve.
- **Confirmed:** `Model(fit_mode='gradient')` implements that scalar-only path.
  It constructs sparse, separate first-difference blocks for every MEM;
  transforms scalar data, the scalar SECS operator, and covariance; applies the
  correlated fit covariance through cached Cholesky solves; and retains the
  existing nandir normal-equation boundary. `track_offsets` recovers one
  variance-weighted scalar offset per MEM conditional on the fitted model.
  Per-sample CHAOS/IGRF background vectors are retained in `Data` solely to
  project the SECS forward field into scalar observation space; no retrieved
  Level-2 vector component enters the gradient inversion.
- **Confirmed on the 240-sample event:** a reduced 54-coefficient gradient
  solve used 956 differences from 960 scalar observations, returned finite
  coefficients and retrieved E/N/U fields, and produced conditional offsets
  of about +148.2, -26.8, -40.9, and +254.4 nT. Both absolute-with-offset and
  gradient-space comparison plots completed. Thirty-one focused tests pass.
- **Confirmed:** `scripts/lithosphere_l3.py` is now a restartable all-orbit
  processor. It writes one compressed, atomic NetCDF4 checkpoint per input,
  skips validated complete outputs, and records failures with tracebacks. Each
  checkpoint retains input identity and provenance; observation data,
  background, predictions, offsets, and residual diagnostics; the SECS model;
  and the full grid/projection definition. A representative full-resolution
  checkpoint was 0.542 MB with finite results, and `reconstruct_grid()`
  reproduced all stored grid and mesh coordinates exactly. Native-grid field
  evaluation is deliberately deferred because it would roughly triple the
  archive runtime and peak near 21.4 GB. Thirty-six focused tests pass.
- **Confirmed:** all-orbit output is background-isolated. The existing IGRF
  archive remains at `EZIE_lithosphere_l3`, while setting
  `background='chaos'` automatically selects `EZIE_lithosphere_l3_chaos`.
  Checkpoint validation now also requires the stored background to match the
  requested background, preventing an IGRF result from being silently accepted
  during a CHAOS run. Fifty-three focused tests pass.
- **Confirmed:** `scripts/lithosphere_global_map.py` now streams all complete
  orbit checkpoints into a north-polar cubed-sphere map. The default grid
  covers the Northern Hemisphere down to 50 degrees at nominal 50 km
  resolution and aggregates `predicted_dB_exact`. It saves mergeable count,
  sum, and sum-of-squares fields together with mean, exact median, sample
  standard deviation, standard error, grid geometry, source counts, and
  provenance in an atomic NetCDF file. The command-line workflow also renders
  the median in nT with `Polarplot.plotimg()` and a robust symmetric color
  range and overlays 50 m coastlines. A NaN-aware Gaussian-filtered statistic
  with sigma one grid cell remains stored in NetCDF but is no longer plotted.
  A second product area-weights 50-by-50-km square FOVs into
  overlapping cells, collapses all samples from one checkpoint/inversion to
  one weighted value per cell, and takes an equal-checkpoint median. The FOVs
  are explicitly aligned to the global cubed-sphere xi/eta axes. NetCDF stores
  the footprint median/mean/checkpoint standard deviation and error together
  with checkpoint count, raw overlap count, and summed fractional coverage.
  A 960-sample real checkpoint rasterized to 474 checkpoint-cell estimates and
  3,202 sample-cell overlaps in 0.012 s. The raw centre-bin, ordinary footprint,
  and equal-satellite/MEM-weighted footprint maps use `RdBu_r` and common color
  limits. A coverage map uses `viridis` to show the footprint-aware
  checkpoint count in every populated cell, with the same coastlines and map
  geometry.
- **Confirmed:** all three magnetic-field maps now mask cells with fewer than
  ten footprint/checkpoint estimates by default. The threshold is adjustable
  through `--minimum-coverage-count`; the coverage-count map remains unmasked
  so weakly sampled regions stay visible. NetCDF schema 1.2 records the chosen
  threshold and a `plot_coverage_mask` without deleting the underlying map
  statistics. Fifty-four focused tests pass.
- **Confirmed:** `lithosphere_global_map.py` now always produces both the
  ordinary and equal-MEM-weighted footprint products. Satellite/MEM identity is
  retained as 12 distinct streams. In every footprint cell, each checkpoint estimate is
  weighted by the inverse number of checkpoints from its stream, giving every
  represented stream total weight one before calculating the weighted median.
  The legacy equal-checkpoint footprint median remains unchanged and is still
  written for comparison. NetCDF schema 1.3 adds
  `scalar_footprint_equal_mem_median` and
  `footprint_represented_mem_count`; the command writes a separately named
  `_equal_mem` PNG with the same color limits as the legacy map. A synthetic
  unequal-coverage test recovers the equal-stream result, and splitting then
  recombining all four MEMs from a real checkpoint reproduces the legacy cell
  IDs, values, coverage weights, and contribution counts exactly.
- **Confirmed:** `lithosphere_global_map.py --all-data` now bypasses SuperMAG
  matching and both quiet-time thresholds explicitly. With default paths it
  writes `scalar_50km_all.nc` and PNG names ending in `_all`, preserving the
  quiet-time products. Both ordinary and equal-weighted footprint maps are
  produced automatically.
  Schema 1.3 records `sample_selection`, `sml_filter_applied`, and
  `selected_sample_count`; SML-specific counts are marked not applicable for
  an all-data run. A synthetic end-to-end test confirms that no SML cache is
  required and every finite observation is retained. The initial all-data-only
  plotting path incorrectly accessed the equal-MEM field conditionally and
  failed after writing the NetCDF; equal-MEM aggregation is now unconditional,
  and the exact all-data standard-plot regression passes.
- **Confirmed from the all-data coverage and footprint-median maps:** nested
- **Confirmed:** the global-map workflow now also writes a median magnetic
  local-time map. It reads `MagneticCoords/obs_magLT1...4` from each
  checkpoint's original Level-2 source (or a checkpoint-local MLT variable),
  footprint-aggregates each track after unwrapping the 24-hour seam, and takes
  a circular median across checkpoints. The NetCDF field
  `footprint_median_mlt` and cyclic `twilight_shifted` PNG both use the range
  `[0, 24)` hours and the normal minimum-coverage mask. The complete suite has
  62 passing tests, including an explicit 23/1 MLT wrap case.
- **Confirmed from the all-data coverage and footprint-median maps:** nested
  high-count rings near the pole coincide with several sharp red/blue median
  transitions. Because coverage counts checkpoint-level estimates after
  within-checkpoint collapse, these rings are not merely repeated 10-second
  samples. The earlier turning-latitude summary incorrectly pooled MEM number
  across satellites; EZIE has 12 distinct satellite/MEM tracks, not four.
  Therefore its four pooled percentiles do not resolve the ring geometry.
  Changing track populations remain a plausible source, but that explanation
  is provisional until all 12 combinations are inspected separately.
- **Confirmed:** `scripts/lithosphere_mem_debug_maps.py` implements the next
  diagnostic without applying any SML selection. It keys the data by satellite
  and MEM, performs the same 50-km footprint and within-checkpoint collapse
  independently for all 12 A/B/C × MEM 1–4 combinations. The same aggregation
  now maps the reconstructed scalar perturbation, fitted track offset,
  `observed_dB`, derived `observed_dB - track_offset`, and `background_B`.
  The derived quantity is explicitly not the original Level-2 debiased product,
  which is absent from retrieved-source checkpoints. The latter four fields add
  3-by-4 median figures in nT. An atomic NetCDF stores mean, median, standard
  deviation, standard error, and field-specific support. Products go to the
  separate `/media/bing/LaCie/EZIE_lithosphere_mem_debug` directory. The default
  ten-checkpoint threshold affects only the median panels; coverage remains
  unmasked. Focused tests confirm that satellites, quantities, and quantiles
  are not mixed and bring the complete suite to 57 passing tests.
  The production figures have not yet been generated or scientifically
  interpreted.
- **Confirmed from generated diagnostics:** the current per-track debug product
  is the complete IGRF archive (4,647 checkpoints: A 1,589, B 1,533, C 1,525).
  Some individual reconstructed-field panels contain broad curved bands near
  their inner coverage boundaries, but they do not unambiguously reproduce the
  distinct concentric rings in the combined map. Track-population mixing
  therefore remains a viable explanation. The fitted-offset maps do not show
  corresponding sharp rings, while the IGRF background-magnitude maps are
  spatially smooth and plausible. The
  current quiet-time CHAOS aggregate also shows clear rings, but it is only
  3,459/4,647 complete: A and B are complete and C has 337 checkpoints. This
  weakens background choice and fitted offsets as primary explanations, while
  leaving population mixing, track/gradient-inversion geometry, and structure
  already present in offset-corrected observations unresolved.
- **Confirmed:** the positive poleward and negative equatorward MEM-3 bands are
  present in both reconstructed predictions and `observed_dB - track_offset`,
  so the inversion is reproducing an input-data structure rather than creating
  it. An IGRF finite-difference calculation at 3,110 sampled MEM-3 locations
  gives median absolute gradients of about 130 nT/degree latitude and
  25 nT/km altitude. A 100–150 nT absolute error therefore corresponds
  typically to 0.8–1.2 degrees (85–130 km) latitude error or 4–6 km altitude
  error. MEM 3 has about 55-degree Earth incidence; a 4–6 km height error moves
  its footprint only about 6–9 km horizontally. A height/reference-layer
  problem is therefore more plausible than a literal 100-km horizontal
  geolocation error, but a uniform height bias would be mostly removed by the
  fitted track offset and cannot alone explain the paired residual bands.
- **Inferred, not confirmed:** the full magnetic-coordinate comparison weakens
  a simple MEM-3-only geolocation-error explanation. MEM 1 contains the same
  broad low-latitude-negative/high-latitude-positive reversal, especially for
  active SML, while MEM 3 amplifies it strongly and consistently across all
  three satellites. This is compatible with a physical activity-dependent
  signal plus an angle-dependent MEM-3 gain or retrieval bias; it does not yet
  separate those contributions. The quiet MEM-3 median retains a weaker
  paired structure, so a purely activity-driven explanation is also
  insufficient. A controlled reference-altitude/view-angle test remains the
  next independent discriminator.
- **Confirmed:** `scripts/debug_analysis.py` implements the first four
  discriminating tests for `observed_dB - track_offset`. It bins all 12
  satellite/MEM streams at 1-degree magnetic latitude by 0.5-hour MLT,
  separates quiet (`abs(SML) < 80 nT` and `abs(dSML/dt) < 10 nT/min`) from
  active (`SML <= -200 nT`) samples, and separates locally ascending from
  descending spacecraft motion. Repeated samples within one checkpoint/bin
  collapse to one mean before exact cross-checkpoint 10th-percentile, median,
  and 90th-percentile calculation. The full archive result is stored under
  `/media/bing/LaCie/EZIE_lithosphere_debug_analysis`. The original 16
  exhaustive matrices obscured the comparison by plotting all 24 MLT hours
  and severely under-supported ascending subsets. The default output is now
  five focused figures: one descending-pass MEM-1/MEM-3 quiet/active median
  comparison with MLT unwrapped around midnight, one direction-support chart,
  and one MEM-3 quiet/active percentile comparison per satellite. Descending
  checkpoint-cell support exceeds ascending support by 77 times for quiet data
  and 44 times for active data, so the direction-dependent field test is not
  presently supported. `--plot-existing` regenerates figures from the NetCDF
  without reopening checkpoints; `--exhaustive-plots` retains the former
  output as an option. The complete suite contains 58 passing tests.
- **Confirmed:** the debug workflow now stores and plots exact per-cell 10th
  and 90th percentiles of `observed_dB - track_offset`, using the same 50-km
  footprint, within-checkpoint collapse, coverage threshold, and no-SML sample
  selection as the median. The suite contains 58 passing tests.
- **Confirmed:** the downstream map now time-matches every checkpoint sample
  to the nearest one-minute SuperMAG SML value and retains only samples with
  `abs(SML) < 80 nT` and `abs(dSML/dt) < 10 nT/min`. The derivative is centred
  and invalidated across cadence gaps. A complete 2025-05-15 through
  2025-08-31 cache contains 156,960 consecutive one-minute samples with no
  gaps; 34.1% of its minutes pass both criteria. The cache is reused when it
  covers the checkpoint snapshot and otherwise refreshed. Fifty-four focused
  EZIE tests pass.
- **Confirmed:** the complete 156,960-minute SML cache was found intact in the
  LaCie trash after the map output directory was cleared and was copied back to
  `/media/bing/LaCie/EZIE_lithosphere_maps/supermag_sml.nc`; the trash copy was
  preserved. Its coverage was verified against the current checkpoint span, so
  quiet-map reruns do not contact SuperMAG. If a future rebuild is necessary,
  each failed seven-day request is now retried up to three times with bounded
  backoff. The retry regression and all 61 tests pass.
- **Confirmed:** `scripts/lithosphere_sml_histograms.py` plots side-by-side SML
  and `dSML/dt` histograms for the times actually covered by complete
  checkpoints. It counts unique matched one-minute SML bins rather than
  repeatedly weighting the same index value by four MEMs and 10-second sample
  cadence, and marks the ±80 nT and ±10 nT/min selection limits. A real run
  over 4,292 completed checkpoints found 26,724 unique covered minutes, of
  which 35.4% pass both limits, using the existing cache without downloading.
  Fifty-four focused tests pass.
- **Confirmed during the first batch run:** all 2,016 failures observed at
  64.2% completion came from the same near-pole `secsy` rotation assertion.
  The projection centre inherited float32 Level-2 coordinates; nominally zero
  vertical-horizontal rotation terms reached `1.4e-6` to `1.8e-5`. Casting
  the projection definition to float64 reduced the error to `6.5e-15`–`7.5e-14`
  for 16 sampled failed A/B files. The failures therefore reflect coordinate
  precision rather than corrupt data or failed inversion solves. Grid creation
  now promotes track coordinates before all projection arithmetic. The
  original failing event completed a full-resolution inversion with finite
  coefficients and predictions after the fix.
- **Confirmed from kernel logs and a focused lifetime probe:** repeated
  `lithosphere_l3.py` deaths are system OOM kills, not ordinary Python
  exceptions. Three killed Python processes had about 55.9–58.5 GB anonymous
  RSS. `Model._ev -> Evaluator.model -> Model` forms a strong reference cycle;
  after deleting both local names, a probe kept both objects alive until
  `gc.collect()` reclaimed the cycle. Since each model owns multi-gigabyte
  matrices and NumPy buffers contribute little to Python's cyclic-GC trigger,
  several completed inversions can accumulate before collection. The
  restart-and-continue behavior is consistent with this mechanism.
  `lithosphere_l3.py` now calls `gc.collect()` in a `finally` block after every
  attempted orbit, including exceptions, so completed and failed model cycles
  are reclaimed before the next large inversion.
- **Confirmed:** `lithosphere_l3.py` now supports configurable process-level
  orbit parallelism through `--workers`. The default remains one worker.
  Parallel workers use the safe `spawn` context, return serializable failure
  records for parent-only logging, retain per-orbit cyclic collection, and
  receive explicit BLAS/OpenMP thread limits. Unless overridden with
  `--blas-threads-per-worker`, available logical CPUs are divided across the
  active workers. A real two-process spawn smoke test completed and the full
  48-test suite passes. In the resumed production run, the user reports that
  two workers reduce the checkpoint completion cadence from about seven to
  4–5 seconds per orbit. Total RAM remains just below 25% on the 62-GiB
  machine; CPU use has a floor near 50% and spikes above 90%. This supports a
  controlled three-worker trial, although transient peak RSS and swap still
  need monitoring before changing the default.
- **Confirmed by the user:** the three-worker production trial also keeps RAM
  low and does not substantially change the aggregate CPU profile. No
  three-worker completion cadence has yet been recorded. The next informative
  benchmark is four workers with the automatic eight BLAS threads per worker,
  followed by higher worker counts only while measuring aggregate checkpoint
  throughput, peak RSS, and swap.
- **Confirmed:** the magnetic-east derivative `L` and `L.T @ L` now remain
  sparse through regularization construction. Only the final matrix required
  by the existing direct solver is densified, scaled in place, and given its
  ridge diagonal. On a 10,944-coefficient real event, `L.T @ L` density was
  0.0822%, its construction took 0.043 s, and complete checkpoint processing
  took 7.11 s with 6.28 GB peak memory. The model matched the saved dense
  baseline to `2.84e-15` in relative norm and predictions agreed below
  `2.2e-13` nT. The restarted production run sustains about 9 seconds per
  inversion, compared with roughly 15 seconds before the change. Thirty-seven
  focused tests pass.
- **Confirmed by the user:** full exact posterior model covariance `Cmpost`
  and model resolution `Rm` remain required capabilities. A matrix-free
  iterative mean solver would make those products require roughly one
  iterative solve per model-space column and is therefore not the preferred
  primary backend. Further performance work should preserve the direct,
  factorization-backed posterior path.
- **Confirmed:** nandir now applies original sparse regularization terms and
  computes exact resolution as `Rm = I - Cmpost @ R`. The EZIE Gini search
  uses this identity for its single resolution row per parameter pair. On the
  10,944-coefficient event, the resolution-row multiplication fell from
  0.0172 s to 0.00063 s with `1.07e-13` relative row difference and
  `1.78e-15` Gini difference. Dense assembly and Cholesky factorization still
  took 3.37 s, so repeated factorization—not `Rm` multiplication—dominates the
  observed 5.3 s per Gini iteration.
- **Confirmed by production retest:** the revised Gini search runs at about
  3 s per parameter pair. For the 10,944-parameter system, regularization and
  dense-system assembly were measured near 0.52 s combined, while Cholesky
  remained dominant and varied with concurrent BLAS load. Each pair changes
  the full 0.892-GiB system, so its factorization cannot be reused directly.
- **Confirmed:** the Gini search now uses nandir's exact banded Woodbury solver
  for its one required resolution row. On a representative full-grid pair it
  took 1.64 s versus 3.28 s for dense Cholesky. Across all four search-domain
  corners the speed-up was 1.08–3.31 times, with model and resolution-row
  differences near `1e-14` and `9e-14` and Gini differences below `2.5e-12`.
  The final inversion still uses the direct solver and retains full exact
  `Cmpost` and `Rm`. Forty-eight focused EZIE tests and all fourteen nandir
  test functions pass.
- **Confirmed:** L-curve curvature is now evaluated parametrically along
  increasing `l1`, using separate cubic splines for residual and model norm and
  the signed curvature of `(rnorm(l1), mnorm(l1))`. This removes the former
  requirement that `rnorm` be strictly monotonic. Negative signed curvature is
  excluded; when multiple positive-curvature peaks remain, the peak at the
  largest `l1` is selected. The unused hard-coded stability mask and obsolete
  `offset` result/plot marker were removed.
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

- The environment is not reproducible from package metadata because only the
  new nandir dependency is declared; the other runtime dependencies remain
  absent from `pyproject.toml`.
- The new solver-equivalence test does not cover component ordering, covariance
  layout, coordinate frames, regularization selection, or validation behavior.
- The scientific meaning of an effective observation altitude remains open;
  a fixed 80 km assumption and a temperature-weighted alternative were
  discussed historically but not established here.
- The repository includes versioned sample data, generated Python caches, and
  package metadata. Their cleanup or release policy has not been decided.
- A legacy session note contained likely access tokens. They are redacted in
  the live vault, but may remain in Git history and should be treated as
  compromised if still valid.
- The tagged pre-launch contained an automatic-regularization regression
  introduced in commit `6e4a228`; it is repaired on `flight-data` and verified
  through successful model-index selection on the representative file.
- The first real-data grid reaches 89.829 degrees geographic latitude. Its
  positions plot coherently, but construction of geographic east/north
  derivative operators hits a `secsy` numerical assertion near the pole. A
  different event avoids this failure.
- The public data containers do not document or enforce units, making silent
  nT/T and covariance/std-dev mistakes possible.
- Scalar and vector Level-2 products can have large relative per-MEM zero-level
  offsets even though their along-track variations agree. Absolute-data
  inversions can map those offsets into broad equivalent-current structure.
- A scalar-gradient inversion cannot determine the four track baselines or
  broad model modes that appear constant along the tracks without help from
  regularization or independent absolute observations.
- A default-grid comparison found nearly identical fitted scalar signatures
  after restoring gradient offsets (correlation 0.9986), but only moderate
  agreement between the retrieved E/N/U maps. The smoother gradient map is
  consistent with excluding track-constant nuisance modes, but is not by
  itself evidence that the map is more accurate; scalar data leave vector and
  between-track structure underdetermined.
- Scalar loading now defaults to `debiased_dbmag#` for both absolute and
  gradient inversions. The non-debiased `retrieved_dbmag#` remains available
  through `scalar_source='retrieved'`. Adding its constant back cannot change
  the gradient solution because `D(d+c)=Dd`, but it does change the absolute
  scalar inversion.
- Absolute scalar mode still recomputes the CHAOS background rather than using
  the exact per-sample background retained by `Data`. The evaluator now
  includes both core and lithospheric internal fields, but it treats supplied
  latitude and radius as geocentric; extraction evaluates the stored field via
  an explicit WGS84 geodetic conversion. Stored fields should therefore remain
  the observation-point source of truth.
- The initial difference operator separates MEM tracks but does not yet split
  a track at missing samples or anomalously large time gaps. Representative
  files used so far are finite and regularly sampled; broader ingestion must
  add segment boundaries before gradient fitting.
- Gradient mode is structurally compatible with the regularization optimizer:
  its transformed `G`, `d`, covariance-weighted residual, `GTG`, and resolution
  matrix use the same interfaces. A reduced real-event run exposed
  non-monotonic residual norms in both gradient and absolute scalar modes. The
  former `mnorm(rnorm)` spline failed in that case; the new `l1`-parameterized
  curvature removes that structural limitation but still needs a real-event
  rerun to validate the selected knee.
- Exact full-grid posterior uncertainty remains intrinsically expensive because
  `Cmpost` and `G @ Cmpost` are dense. Chunked Cholesky solves remain a possible
  alternative if repeated dense-covariance reuse does not justify the memory.
- A full-grid batch benchmark for `lithosphere_l3.py` found 4,647 input files,
  about 9.94 s per fixed-regularization gradient solve, and 7.6 GB peak memory
  for an 11,232-coefficient grid. Dense `LTL` construction was the largest
  timed stage at 3.83 s, followed by the 2.33 s direct solve and 1.71 s scalar
  design construction. The derivative matrices are structurally sparse but
  currently materialized as dense arrays. Controlled multiprocessing is now
  available with per-worker BLAS limits, but its full-grid speed-up and peak
  memory remain unmeasured; one solve previously used roughly 15 threads.
- Nandir is pinned from GitHub rather than a package-index release. EZIE still
  constructs dense `GTG` and `GTd` and handles covariance before delegation
  for the final direct solve; the Gini Woodbury path instead passes `G`, `Q`,
  and sparse regularization. Nandir's one-parameter path helper does not
  replace EZIE's two-parameter L-curve search.
- The new `BandedWoodburySolver` currently comes from the local editable nandir
  checkout. EZIE's Git dependency still points to commit `e452b5c`; nandir must
  be committed and the pin updated before this integration is reproducible in
  a fresh environment.

## Current project status

Portfolio status and priority are unknown. The user has registered the project
and confirmed the archival pre-release plus branch strategy. The cleaned tag
and hosted pre-release plus the `flight-data` branch are published. The first
representative-data validation milestone remains outstanding.
