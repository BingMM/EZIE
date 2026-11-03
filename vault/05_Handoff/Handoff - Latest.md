# Handoff - Latest

Last reviewed: 2026-10-29

Branch: `flight-data`

Pre-launch release commit: `9508bf9595b4a2ad70b0a9fc3e75d9176c8ad59f`

## Summary

The first post-launch Level-2 execution has been reproduced using
`ezie_l2_20250515_004708_sva_v003_r000.nc4`. The revised loader exposes four
finite, unmasked 225-sample MEM tracks and constructs a geometrically coherent
polar grid.

Commit `9508bf9` is pushed on `main` and tagged `v0.1.0-prelaunch` on GitHub.
The `flight-data` branch was created from that commit and is also pushed. The
hosted GitHub pre-release was published and verified through GitHub's public
API. Exact release notes are recorded in the 2026-10-21 session note.

Two independent issues were identified. The pre-launch automatic-regularization
regression introduced by commit `6e4a228` was repaired by changing
`self.grid.mems` to `self.data.mems`; it now sees all four tracks and selects
valid model index 5533. With explicit regularization, the workflow still stops
at a `secsy` rotation assertion because one grid cell lies at 89.829 degrees
geographic latitude and nominally zero coupling terms reach roughly `1.6e-6`,
just above the `1e-6` tolerance. Disabling assertions for diagnosis lets the
solve complete with 10,944 finite coefficients; this is not an operational fix
or scientific validation.

The user confirmed that another event avoids the polar assertion. Unit tracing
also identified the cause of an unusable solution plot: the package expects
magnetic perturbations in tesla and covariance entries in tesla squared, while
the Level-2 values were passed numerically as nT and nT squared. The plotter
multiplies predictions by `1e9` to display nT, amplifying the mismatch.

The post-launch extraction and file-loading helpers have now been moved from
the analysis script into `ezie/utils.py`. The utility applies the required SI
and coordinate conversions and was verified against the working 240-sample
EZIE-A event.

`generate_ezie_data(data, comp='all')` is now also in the utility module. It
accepts a path, open dataset, or extracted mapping; supports scalar magnitude
and every unique E/N/U subset; and omits unselected observations and covariance
terms. Its `mems` argument supports every non-empty unique ordered subset of
MEMs 1–4 and defaults to all four. Supporting work generalized covariance
assembly and repaired latent scalar-path unit, coordinate-count, and
broadcasting errors. A reduced-grid scalar solve completed with finite
coefficients.

Mean and uncertainty evaluation have now been separated for magnetic field and
current density. The dense cached-`Cmpost` uncertainty path was optimized to
calculate `diag(G Cmpost G.T)` with an elementwise `einsum` contraction after
`G @ Cmpost`, avoiding the full evaluation-point covariance. It reproduces the
old expression to machine precision.

Nandir is now integrated at the precomputed-normal-equation boundary. EZIE
continues to construct `GTG`, `GTd`, and its complete regularization matrix;
`nandir.Solver` owns the cached factorization, model solve, posterior model
covariance, and model resolution. The existing `Model.m`, `Model.Cmpost`,
`Model.Rm`, and `Model.c_factor` interfaces are preserved. Nandir is pinned to
reviewed upstream commit `e452b5c6a4f7340143b15ce552ae79447490cee0`.

A focused `unittest` passes both delegation equivalence and cache-reset checks.
On the real-data E/N, MEM-1 reduced grid, the integrated model, covariance, and
resolution outputs differ from the former Cholesky implementation by only
`3.58e-15` to `3.79e-15` in relative norm.

The local nandir checkout is installed in editable mode in the dedicated EZIE
environment. The tests pass there without adding nandir to `PYTHONPATH`.

The evaluator API split is now propagated to the data/model plot and SuperMAG
validation. `Model.add_data()` invalidates all data-dependent caches while
retaining configured fixed-versus-automatic regularization behavior, and
scalar radius input to `Data` is repaired. Plot axes are normalized for all
one/many MEM-component combinations. Post-launch `Data` now retains the full
observation time sequence required by validation and continues to use its
midpoint as the representative date.

Posterior uncertainty propagation is now covariance-aware. QD magnetic and
current plus cubed-sphere current prediction matrices are transformed before
contraction with `Cmpost`, and scalar-magnitude uncertainty uses a directional
prediction matrix along the predicted total field. This preserves
cross-correlations without materializing full field-space covariance matrices.

Shared CHAOS-8.4 loading and internal-field synthesis now live in
`ezie.utils`; the repository-relative model is cached after first load. The
evaluator imports the shared spherical helper and now includes both the
time-dependent core and static lithospheric internal fields, while continuing
to exclude external GSM and SM sources.

Level-2 extraction now accepts `background='igrf'` (the unchanged default) or
`background='chaos'`. CHAOS mode reverses the exact ancillary IGRF vector in
the file and subtracts the time-dependent plus static internal CHAOS field at
each timestamp and WGS84 MEM position. It excludes external sources, computes
scalar perturbations against the CHAOS magnitude, and retains the supplied
measurement covariance. A 240-sample real event produced finite corrected
fields and unchanged covariance; the maximum absolute MEM-1 component
correction was about 107.4 nT. `generate_ezie_data()` forwards `background`
and an optional preloaded `chaos_model` for direct file or open-dataset input;
already-extracted mappings are protected against incompatible relabeling.

A scalar/vector consistency check on the 240-sample event found that
`retrieved_b_tot` minus the magnitude of ancillary IGRF plus retrieved N/E/D
is constant to about 0.0023 nT within each track, but the four track constants
are +178.16, -5.41, -37.60, and +277.72 nT. The two products therefore retain
the same along-track variation to numerical precision but have different
per-MEM zero levels. The source has now been identified in the file:
`debiased_dbmag#` is `retrieved_dbmag#` with its constant MEM mean removed,
and `retrieved_b_tot# - anc_model_bgeo#` matches the non-debiased variable to
about 0.006 nT. The offset is therefore an explicit Level-2 debiasing product,
although which zero level is physically correct remains unknown.
The user tested forcing the vector magnitude to the delivered scalar magnitude;
this did not improve the inversion. That construction fixes only the
field-aligned scalar discrepancy, not arbitrary offsets in all three vector
components, and its covariance has not been propagated.

The user reports an unexplained rotation in the Level-2 three-component data
that cannot currently be accounted for. Vector components are therefore out
of scope as inversion observations for the next extension. The agreed target
is a scalar-gradient observation model with one constant offset per MEM; the
SECS coefficients remain the model and continue to produce a full perturbation
vector after the solve.

That path is now implemented as `Model(fit_mode='gradient')`. It accepts only
scalar `Data`, builds a sparse blockwise first-difference operator with one
block per MEM, propagates `d`, the scalar SECS design matrix, and covariance,
and uses cached Cholesky covariance solves before the unchanged nandir
normal-equation boundary. `Data` retains per-sample background vectors from
IGRF or CHAOS for scalar projection; these are deterministic model fields, not
the suspect retrieved vector observations. Conditional per-MEM offsets are
available as `model.track_offsets`.

Scalar loading now defaults to `debiased_dbmag#` for both absolute and
gradient inversions. `scalar_source='retrieved'` explicitly selects the
non-debiased `retrieved_dbmag#`. Adding the Level-2 constant back to the
debiased data and then applying the same difference operator is algebraically
identical and cannot materially change the gradient `m`, but it changes the
absolute scalar inversion.

For transparent scientific use, the user will explicitly select
`scalar_source='retrieved'` with `fit_mode='gradient'`: the raw Level-2
constants then enter the workflow and are visibly eliminated as nuisance
offsets by EZIE. The default debiased scalar remains appropriate for the
ordinary absolute comparison. The two solutions are virtually identical on
the event inspected so far.

A real reduced-grid run on the 240-sample event completed with 956 gradient
observations, 54 finite SECS coefficients, finite retrieved E/N/U fields, and
offsets of approximately +148.2, -26.8, -40.9, and +254.4 nT. Absolute-space
plots include the fitted offsets, and a separate gradient comparison displays
the quantities actually fitted.

## Next actionable step

The restartable all-orbit checkpoint workflow is implemented in
`scripts/lithosphere_l3.py`. It stores both cached linear and exact nonlinear
scalar predictions, as well as E/N/U at observation locations, so that choice
does not need to be made before the long run. Each atomic NetCDF4 result also
stores `m` and an exactly reconstructable source grid; native-grid E/N/U is
deferred to selective post-processing because evaluating it for every orbit
would add about 25 hours and raise peak memory near 21.4 GB. The next step is
to run the batch and inspect `failures.jsonl` plus output counts. Substantial
further acceleration requires a separately validated sparse, matrix-free mean
solver; the current full grid takes about 9.94 s and 7.6 GB per orbit.

`scripts/lithosphere_global_map.py` now provides the first downstream map
stage. It takes a snapshot of all complete checkpoint files and streams their
`predicted_dB_exact` samples onto a north-polar cubed-sphere grid extending to
50 degrees latitude at nominal 50 km resolution. It writes atomic NetCDF with
count, sum, sum of squares, mean, exact median, sample standard deviation,
standard error, grid geometry, and input provenance. The command-line run also
writes a polar PNG of the median in nT using `Polarplot.plotimg()`, with a
symmetric 98th-percentile color range and geographic coastlines. It also stores
but no longer plots a NaN-aware Gaussian-filtered median with sigma one grid
cell; missing bins are normalized out rather than treated as zero, and the
result remains masked outside the latitude cap. Exact medians require retaining binned values
and cell IDs during this downstream run, still modest relative to an inversion.
The default grid is 178 by 178 cells; a read-only check against a
real completed checkpoint assigned 873 of its 964 finite samples north of 50
degrees before quiet-time filtering. Run the aggregate after the current batch
finishes to capture the complete archive snapshot.

The same run now produces a median-MLT diagnostic. MLT comes from the original
Level-2 `MagneticCoords/obs_magLT1...4` arrays referenced by each checkpoint,
unless a checkpoint-local `magnetic_local_time` variable is present. Track MLT
is unwrapped before footprint averaging, then checkpoint estimates are reduced
with a circular median, so midnight-adjacent values such as 23 and 1 hours map
to 0 rather than 12. The result is stored as `footprint_median_mlt` and plotted
with a cyclic colormap and the standard coverage threshold. The complete suite
passes 62 tests.

The map now also contains a footprint/checkpoint aggregation requested after
track striping was seen in the raw centre-bin median. Every prediction is
treated as the centre of a 50-by-50-km square aligned with the cubed-sphere
xi/eta axes. Its fractional overlap with each cell is calculated explicitly.
Overlapping samples from one checkpoint are first combined into one
area-weighted value per cell, after which the median is taken across checkpoint
values with equal checkpoint weight. This avoids interpreting the approximately
15-km along-track sampling as independent 50-km observations. The original raw
median is preserved, and ordinary plus equal-satellite/MEM-weighted footprint
PNGs are produced with the same `RdBu_r` color limits.

A fourth PNG now maps `footprint_checkpoint_count`: the number of independent
checkpoint-level estimates contributing to each populated cell after the
within-checkpoint collapse. It uses a linear `viridis` count scale and the same
polar geometry and coastlines. The default filename appends `_coverage` to the
ordinary median plot stem, and `--coverage-plot` can override it.

Field plotting now requires at least ten footprint/checkpoint estimates per
cell by default. The raw centre-bin median and both footprint medians are
masked by the same support criterion before
color-limit calculation and rendering. `--minimum-coverage-count` changes the
threshold. The coverage map itself remains unmasked, and schema 1.2 stores both
`plot_minimum_coverage_count` and `plot_coverage_mask`; full underlying
statistics remain available even where the plotted field is hidden.

Coverage-driven changes in the mixture of MEM tracks are now tested by default.
The implementation treats the 12 satellite/MEM
combinations as distinct streams. Within each footprint cell, checkpoint
estimates receive inverse stream-count weights, so every represented stream
has total weight one in the weighted median regardless of its number of
checkpoints. The legacy equal-checkpoint median is still calculated and stored.
Schema 1.3 writes
`scalar_footprint_equal_mem_median` and
`footprint_represented_mem_count`, plus a separate `_equal_mem` PNG sharing the
legacy map's color limits. A synthetic unequal-coverage case gives the expected
equal-stream median. A real four-MEM checkpoint also confirms that the new
per-MEM rasterization, when recombined, reproduces the legacy footprint values,
weights, and counts exactly.

The global-map command now also has an explicit `--all-data` selection. It
bypasses SuperMAG loading and both SML thresholds rather than approximating an
unfiltered run with large numerical limits. Default outputs are isolated as
`scalar_50km_all.nc` and PNG filenames ending in `_all`, so the existing quiet
map is not overwritten. Both ordinary and equal-stream footprint maps are
produced automatically. NetCDF metadata records the
selection mode and marks SML-specific counts as not applicable. The synthetic
end-to-end test exercises all-data standard plotting without an SML cache. The
first implementation failed only after writing its complete NetCDF because the
plot block requested an absent equal-MEM field. Equal-MEM calculation is now
unconditional, the enabling/path options and sigma-one plot are removed, and
the exact regression passes.

The all-data coverage map exposes nested polar rings, and several sharp
blue/red transitions in the footprint median align with them. Since the map
already collapses each checkpoint to one estimate per cell, this is not simply
the 15-km sample cadence. The earlier 310-checkpoint turning-latitude analysis
pooled equal MEM numbers across satellites, but MEM 1 on A, B, and C represents
three different tracks. Its four pooled percentile sets therefore do not prove
that there are only four turning envelopes. Changing track populations remain
a plausible source of the rings, but this is provisional until the 12 distinct
satellite/MEM combinations have been inspected.

`scripts/lithosphere_mem_debug_maps.py` now provides that diagnostic. It does
not load or filter on SML. Instead it keys each checkpoint by its satellite
attribute and repeats the 50-km footprint allocation and within-checkpoint
collapse independently for all 12 A/B/C × MEM 1–4 combinations. One atomic
NetCDF stores dimensions `(satellite, mem, grid_y, grid_x)`. The workflow maps
five fields in the same pass: reconstructed scalar perturbation, fitted track
offset, `observed_dB`, derived `observed_dB - track_offset`, and `background_B`.
The derived field is not the Level-2 debiased measurement; that separate source
variable was not saved in retrieved-source checkpoints. All fields use the same
footprint, within-checkpoint collapse, cross-checkpoint median, and coverage
threshold. Six shared-scale 3-by-4 figures show the five field medians in nT and unmasked
checkpoint coverage. Perturbations and offsets use `RdBu_r`; the positive
background magnitude uses a robust sequential `viridis` scale. Products are
isolated in `/media/bing/LaCie/EZIE_lithosphere_mem_debug`. The added diagnostic
figures are `scalar_50km_track_offset_medians_all.png`,
`scalar_50km_observed_dB_medians_all.png`, and
`scalar_50km_background_B_medians_all.png`, together with
`scalar_50km_observed_dB_minus_offset_medians_all.png` for the derived field.
The default ten-checkpoint support
threshold is applied only to median panels and is adjustable with
`--minimum-coverage-count`. The next step is to run these products and
determine whether each field transition is present within an individual track,
in the observations/background/offsets, or only after tracks are combined.

The first generated debug set uses the complete IGRF archive: 4,647 checkpoints
(A 1,589, B 1,533, C 1,525). Some individual satellite/MEM panels have broad
curved bands near their inner coverage boundaries, but they do not clearly
reproduce the combined map's concentric rings. Cross-track population mixing
therefore remains viable. The fitted-offset panels do
not contain matching sharp rings, and `background_B` is smooth and physically
plausible. A separate quiet-time CHAOS aggregate at 3,459 checkpoints also has
clear rings; that snapshot contains all available A and B files but only 337 C
files, so its combined morphology is still compositionally incomplete. The
updated debug script's `observed_dB_minus_offset` map is the next discriminator:
rings there would implicate the input data or sampling geometry. If individual
corrected-observation and prediction panels remain ring-free, but the combined
map has rings, changing track populations would be implicated. Rings appearing
only in individual reconstructed predictions would instead implicate the
inversion or operator geometry. A like-for-like IGRF/CHAOS background
comparison remains to be made.

The paired MEM-3 bands occur in both the reconstructed prediction and
`observed_dB - track_offset`, which moves their origin upstream of the
inversion. A first-order IGRF sensitivity calculation used 3,110 sampled MEM-3
locations. Absolute field-magnitude gradients had 10th/median/90th percentiles
of 25/130/256 nT per degree latitude and 21.7/25.2/28.4 nT per kilometre
altitude. At the median gradient, 100–150 nT requires 0.8–1.2 degrees latitude
(about 85–130 km) or 4–6 km altitude. MEM 3's Level-2 Earth-incidence angle is
about 55 degrees, so the latter height error implies only 6–9 km horizontal
footprint displacement. This favors an effective emission-height or
angle-dependent retrieval problem over a literal large latitude error, but a
constant height error would mostly enter the fitted offset; the paired residual
bands require spatial or geometry dependence.

The current evidence favors, but does not establish, an angle-dependent
retrieval/geolocation artifact over an electrojet: the bands repeat in MEM 3
on all satellites, MEM 3 has the largest incidence angle, their geometry is
geographic/coverage-boundary-like, and a similar large-scale pattern remains
under the strict `abs(SML) < 40 nT` selection. The absence of rings in some
MEM-1 medians is not a sufficient control because fitting one constant per
track, taking a long-term median, and binning geographically can suppress a
broad or intermittent physical electrojet. The discriminating analysis is to
bin `observed_dB - track_offset` by magnetic latitude and MLT, split by SML and
ascending/descending geometry, compare simultaneous MEMs, and test pierce
points/backgrounds at several reference altitudes.

`scripts/debug_analysis.py` implements the first four items. It reads magnetic
latitude, MLT, and spacecraft latitude from each checkpoint's original Level-2
source and combines them with the checkpoint's `observed_dB`, fitted offset,
time, satellite, and MEM identity. Defaults are 1-degree MLAT by 0.5-hour MLT,
quiet `abs(SML) < 80 nT` plus `abs(dSML/dt) < 10 nT/min`, and active
`SML <= -200 nT`. Ascending/descending is classified locally from spacecraft
latitude so samples on either side of a polar pass do not share one orbit-wide
label. Each checkpoint contributes at most one mean estimate to an MLAT/MLT
cell before exact cross-checkpoint 10th percentile, median, and 90th percentile
statistics. The atomic NetCDF goes to
`/media/bing/LaCie/EZIE_lithosphere_debug_analysis`. The full-archive run
showed that the former 16 three-by-four matrices were a poor visual encoding:
the data occupy only the nightside hours around midnight, midnight was split
across the two ends of the x-axis, and the ascending subsets were nearly
empty. Default plotting now unwraps MLT around midnight and writes five focused
figures: a descending-pass quiet/active MEM-1-versus-MEM-3 median comparison,
a direction-support chart, and one quiet/active MEM-3 percentile comparison
per satellite. The exhaustive matrices remain available with
`--exhaustive-plots`; `--plot-existing` regenerates plots from the NetCDF
without repeating the archive aggregation. Descending checkpoint-cell support
is 77 times the ascending support for quiet samples and 44 times for active
samples, so no scientific direction comparison should be made from this
archive. The focused median comparison shows the broad latitude sign reversal
in MEM 1 as well as MEM 3, particularly under active SML, while MEM 3 strongly
amplifies it across all satellites. This weakens a simple MEM-3-only
geolocation error but remains compatible with physical structure plus an
angle-dependent retrieval gain or bias. The complete suite contains 58
passing tests.

The debug script now calculates exact per-cell 10th and 90th percentiles for
`observed_dB - track_offset` after the same checkpoint-footprint collapse used
for its median. It writes `observed_dB_minus_offset_quantile_10` and
`observed_dB_minus_offset_quantile_90` to NetCDF and adds
`scalar_50km_observed_dB_minus_offset_p10_all.png` and
`scalar_50km_observed_dB_minus_offset_p90_all.png`. The complete suite contains
57 passing tests.

The NetCDF schema is now 1.1 and stores footprint checkpoint count, sample-cell
overlap count, summed fractional coverage, mean, median, checkpoint standard
deviation, and checkpoint standard error. The legacy sample-based standard
error is retained for comparison but should not be interpreted as independent
sample uncertainty. A real 960-sample checkpoint produced 474 checkpoint-cell
estimates from 3,202 overlaps in 0.012 s. Checkpoint estimates may still be
temporally correlated, so a later uncertainty analysis should use orbit/time
blocks rather than treating even the checkpoint count as strictly independent.

The map workflow now applies the requested quiet-time selection before
binning: `abs(SML) < 80 nT` and `abs(dSML/dt) < 10 nT/min`. Observation times
are matched to the nearest SuperMAG minute with a 60-second maximum tolerance;
`dSML/dt` is the centred derivative in nT/min and is invalidated next to time
gaps. The confirmed `mih008` account was used to cache all 156,960 consecutive
one-minute SML values from 2025-05-15 00:00 through 2025-08-31 23:59 UTC at
`/media/bing/LaCie/EZIE_lithosphere_maps/supermag_sml.nc`. There are no cadence
gaps, and 53,501 minutes (34.1%) satisfy both thresholds. The script reuses a
covering cache and downloads again only if the completed checkpoint span grows
beyond it or `--refresh-sml` is given.

The cache disappeared from the active output directory when that directory was
cleared, causing a quiet-map run to attempt a fresh download; SuperMAG returned
a failed response on the sixth seven-day chunk. The intact cache was located in
the LaCie trash, verified as complete and gap-free, and copied back without
removing the trash copy. Future downloads retry each failed chunk up to three
times with bounded backoff. The retry regression and complete 61-test suite
pass.

`scripts/lithosphere_sml_histograms.py` now provides a direct diagnostic of
that selection. It scans complete checkpoint times once, matches them to the
cached one-minute index, deduplicates repeated MEM and 10-second observations
to unique SML minutes, and plots SML plus `dSML/dt` with dashed ±80 nT and ±10
nT/min limits. The default output is
`/media/bing/LaCie/EZIE_lithosphere_maps/sml_histograms.png`; bin count and paths
are configurable from the command line. A real read-only archive run processed
4,292 complete files, found 26,724 unique data-covered minutes, and measured a
35.4% joint quiet fraction without downloading new SuperMAG data.

The first production run exposed a systematic blocker: at a 64.2% snapshot,
all 2,016 failures were the same `secsy` near-pole rotation assertion. The
projection centre inherited float32 Level-2 coordinates. On 16 sampled failed
A/B files, casting the projection parameters to float64 reduced the offending
nominally zero terms from `1.4e-6`–`1.8e-5` to `6.5e-15`–`7.5e-14`. This is a
precision defect, not evidence of bad files. The minimal float64
projection-construction fix is now implemented, and the original failing event
completed a full-resolution inversion with finite coefficients and
predictions. The next operational action is a restart that skips existing
complete checkpoints and retries files without outputs.
Preserve or rotate the old failure manifest so historical failures remain
distinguishable from failures after the fix.

The restarted batch has also been killed intermittently by system memory
exhaustion. Kernel logs record three Python OOM kills at approximately 55.9,
58.4, and 58.5 GB anonymous RSS. A focused lifetime test identified a concrete
retention mechanism: every processed orbit evaluates `model.ev`, creating the
strong cycle `Model._ev -> Evaluator.model -> Model`. Deleting the local model
and evaluator names did not release them; `gc.collect()` immediately reclaimed
the cycle. Large NumPy buffers do not advance cyclic-GC thresholds in
proportion to their size, so multiple multi-gigabyte models can survive long
enough to exhaust RAM. The lowest-risk repair is explicit collection after
every `process_file()` returns, optionally combined with `model.reset_ev()` to
break the cycle before return. That repair is now implemented: `gc.collect()`
runs in the batch loop's `finally` block after every successful or failed
attempt. A focused test covers both paths. Verify current RSS across multiple
full-grid orbits before considering allocator-specific measures such as
`malloc_trim`.

The batch runner now also supports configurable process parallelism. The
default remains one worker; `--workers 2` or a larger value distributes
independent pending orbits across spawned processes. Each process performs its
own post-orbit `gc.collect()`. Failures are returned as plain records and only
the parent appends `failures.jsonl`, avoiding concurrent writes. Parallel runs
set BLAS/OpenMP thread limits before spawning; by default logical CPUs are
divided among active workers, with an explicit override available through
`--blas-threads-per-worker`. A real two-process spawn smoke test passed, but a
longer production observation is still needed to establish peak RSS and swap
behavior before changing the one-worker default.

The first production two-worker observation is now available. The user reports
4–5 seconds per completed orbit versus about seven seconds with one worker,
total RAM just below 25% of 62 GiB, and CPU utilization ranging from a floor
near 50% to spikes above 90%. The progress-bar cadence measures completed
checkpoints rather than each worker's individual solve time. The remaining
headroom makes a three-worker trial reasonable; the automatic limit will assign
10 BLAS threads per worker on the 32-logical-CPU host. Monitor peak rather than
average RAM and confirm that swap remains near zero.

Background-specific production archives are now isolated automatically. The
legacy IGRF configuration keeps the existing
`/media/bing/LaCie/EZIE_lithosphere_l3` path; changing the configuration to
`background='chaos'` selects
`/media/bing/LaCie/EZIE_lithosphere_l3_chaos`. Completion checks require the
checkpoint's stored `background` attribute to match, so an IGRF checkpoint can
no longer be reused as a CHAOS result. The downstream global-map script still
defaults to the IGRF archive; a CHAOS map run must pass the CHAOS input and a
separate output path.

The user reports that three workers still keep RAM low and do not materially
change the observed CPU profile. There is no numerical minimum BLAS thread
count: one thread remains valid, and fewer threads per process can improve
aggregate throughput when independent dense factorizations scale poorly. No
three-worker completion cadence has yet been reported. Test four workers next
(automatically eight BLAS threads each), then judge further increases from
aggregate checkpoint cadence, peak RSS, and swap rather than per-worker solve
duration or average RAM.

The first performance optimization is also complete. EZIE now retains the
`secsy` magnetic-east derivative and `L.T @ L` as sparse matrices and densifies
only once for the unchanged direct solver. On the formerly failing
10,944-coefficient event, the complete checkpoint workflow took 7.11 s and
6.28 GB peak memory. The sparse result matched the saved dense model to
`2.84e-15` relative norm and its linear scalar prediction to better than
`2.2e-13` nT. The sparse `L` is now the required foundation for an optional
matrix-free mean solver.

The restarted production batch reports about 9 seconds per inversion rather
than the previous 15 seconds, a sustained improvement of roughly 40%.

The user requires full exact `Cmpost` and `Rm`, so the proposed matrix-free
iterative solver is no longer the preferred primary optimization. Although
those quantities exist as operators, materializing them iteratively would
require approximately one solve per model-space column and defeat the intended
speed-up. Preserve the current direct factorization path; future optimization
should target exact posterior construction and exact resolution algebra.

Exact resolution algebra is now optimized in nandir as
`Rm = I - Cmpost @ R`, preserving sparse regularization. EZIE's Gini map uses
the same identity for its one resolution row per parameter pair. The row
operation dropped from 0.0172 s to 0.00063 s on a 10,944-coefficient event and
the Gini value agreed to `1.78e-15`. This is not the main explanation for the
reported 5.3 s per Gini iteration: dense assembly and Cholesky factorization
alone took 3.37 s. The next regularization-optimizer speed-up must address
repeated exact factorization while retaining full posterior products.

The production retest reports about 3 s per Gini parameter pair, down from
5.3 s. Separated timing confirms that dense Cholesky remains the limiting
operation: regularization plus system assembly was about 0.52 s, while the
10,944-square factorization dominated and varied with concurrent BLAS load.
The matrix is 0.892 GiB and each `(l1, l2)` pair changes it, preventing direct
reuse of the previous Cholesky factor.

The exact data-space alternative is now implemented. Nandir's
`BandedWoodburySolver` exploits the seven-diagonal, bandwidth-192
regularization and factors an 896-square data system rather than the
10,944-square dense model system. EZIE uses it only for the one resolution row
needed by each Gini pair. A representative pair fell from 3.28 s to 1.64 s;
the four search-domain corners showed 1.08–3.31-times speed-ups and Gini
differences below `2.5e-12`. The final selected inversion remains direct, with
full exact `Cmpost` and `Rm` unchanged.

The ridge fit remains based on the single maximum-Gini `l2` value at each
`l1`. The default `l2_steps` is now 11. Retest the complete ridge search and
confirm that the selected regularization path is stable at the odd grid size.

The integration currently depends on the local editable nandir checkout.
Before release or installation elsewhere, commit the nandir implementation and
update EZIE's existing `e452b5c` Git pin to that reviewed commit.

Benchmark the revised mean and uncertainty paths at full resolution, then
reassess solutions and data/model comparison for selected components and
scalar magnitude. Revisit chunked Cholesky uncertainty only if dense `Cmpost`
reuse does not justify its memory cost.

Benchmark the nandir-backed path at full resolution. Do not move observation
weighting until covariance-versus-precision and sparse or whitened
representations have been validated.

Compare IGRF- and CHAOS-referenced inversions on representative events to
measure how much the lithospheric correction changes the inferred equivalent
current. Background-model uncertainty is not yet included in observation
covariance.

Benchmark the scalar-gradient path at the intended grid resolution and compare
its current solution, residuals, and resolution with the absolute scalar
inversion. Quantify broad-scale modes lost with the four baselines. Before
using files with missing samples or time gaps, split the difference operator
at those segment boundaries.

The first default-grid comparison now shows that the two scalar predictions
are almost identical after restoring conditional track offsets, while their
retrieved vector maps differ strongly. The user has explicitly deferred a
regularization comparison. The immediate consistency task is therefore to make
absolute scalar mode use the stored per-sample background just as gradient
mode does; adding the lithospheric field to the evaluator removed the
core-versus-static content mismatch, but not the evaluator-versus-stored
coordinate-path distinction. Resolution or held-out-MEM diagnostics remain
useful later because fit curves alone cannot distinguish the vector maps.

Gradient mode requires no special regularization-optimizer formulation: the
optimizer sees the transformed model through the usual normal-equation and
weighted-residual interfaces. The former final-selection implementation fitted
`mnorm` directly as a function of `rnorm` and therefore failed when the
residual coordinate was non-monotonic. Curvature is now calculated
parametrically from separate `rnorm(l1)` and `mnorm(l1)` cubic splines. Negative
signed curvature is excluded. If more than one positive-curvature peak is
present, the peak at the largest `l1` is selected, regardless of which peak is
largest. The unused `stable_spline_mask`/`covex_mask` code has been removed.
The obsolete hard-coded `offset` result and its `Concave offset` plot marker
have also been removed. Rerun both the previously folded case and the observed
two-knee case to validate the new selection.

## Blockers and risks

- Current activity, priority, collaborator commitments, and deadlines are not
  recorded.
- Real flight data may differ from the pre-launch examples in schema,
  calibration, quality flags, covariance, geometry, or missing-data behavior;
  these differences have not yet been inventoried.
- Nandir is now declared and pinned, but the remaining scientific package
  dependencies are undeclared. The default Python lacks `kneed` and `ppigrf`.
- Focused tests cover nandir delegation, data-change cache invalidation, scalar
  radius handling, and the split evaluator callers. Broader scientific
  regression coverage remains absent.
- Nandir's generic one-parameter path and unweighted exposed misfit do not
  replace EZIE's two-parameter weighted L-curve workflow.
- Historical scientific observations have not been reproduced against this
  checkout.
- The tagged pre-launch automatic-regularization path contains the
  `self.grid.mems` regression; the `flight-data` checkout now repairs it.
- The representative real-data grid reaches the geographic-pole coordinate
  singularity closely enough to trigger a strict `secsy` rotation assertion.
- Large track-dependent scalar/vector zero-level differences are present in
  the checked event. Their source is the explicit Level-2 debiasing step, but
  the physically correct absolute level is not yet established.
- The gradient operator currently separates MEMs but assumes every adjacent
  sample within a MEM belongs to one continuous, valid segment.
- Input units are implicit rather than validated: fields must be tesla and
  covariance must be tesla squared.
- Likely access tokens were removed from a live historical note, but the
  original strings may remain in Git history. Rotate or revoke them if they
  could still be valid.

## Verification

- Passed: `python -m compileall -q ezie tests`
- Passed in the dedicated EZIE environment with nandir installed editable:
  `python -m unittest discover -s tests -v` (31 tests)
- Passed: real-data E/N, MEM-1 reduced-grid equivalence for `m`, `Cmpost`, and
  `Rm`, with relative differences below `3.8e-15`.
- Passed: real-data E, MEM-1 reduced-grid data/model comparison, including the
  one-by-one Matplotlib axes case, with finite model coefficients.
- Passed: reduced real-data scalar comparison and solution plots with finite
  directional magnitude and QD magnetic/current uncertainties; cubed-sphere
  current uncertainty was also finite.
- Passed: a CHAOS-8.4 real-data evaluation launched from `/tmp`, confirming the
  working-directory-independent path, full timestamp, and finite internal field.
- Passed: `background='chaos'` extraction of a 240-sample real Level-2 event;
  all vector and scalar perturbations were finite, covariance and scalar
  variance were unchanged, and the maximum absolute MEM-1 vector correction
  was about 107.4 nT.
- Passed: direct `generate_ezie_data(path, background='chaos')` construction
  for both ENU vector and scalar `F` data on the same 240-sample event.
- Passed: real-event default debiased and explicit retrieved scalar loading
  with CHAOS referencing. The restored MEM constants match the Level-2 values;
  reduced-grid gradient solutions differ by only `1.46e-6` in relative model
  norm, consistent with roughly 0.001 nT product rounding.
- Passed: reduced real-data scalar-gradient solve and both comparison plots;
  956 correlated differences produced 54 finite SECS coefficients, finite
  retrieved vector fields, and four finite conditional offsets.
- Passed: backward-compatible absolute scalar solve on the same reduced grid
  after the shared covariance-solve refactor.
- Passed: restartable full-resolution lithosphere checkpoint generation for a
  real 240-sample event; the 0.542 MB file had finite model/prediction arrays,
  complete provenance, and an exactly reconstructed source grid.
- Passed in the dedicated EZIE environment:
  `python -m unittest discover -s tests -v` (54 tests).
- Passed: full-resolution inversion and checkpoint of the original near-pole
  failure after promoting projection arithmetic to float64.
- Passed: full-resolution sparse-regularization equivalence against the saved
  dense baseline (`2.84e-15` relative model difference).
- Passed: banded Woodbury Gini-row equivalence across all four optimizer-domain
  corners; absolute Gini differences were below `2.5e-12`.
- Passed: parametric curvature with a non-monotonic residual coordinate and
  selection of the last positive-curvature peak when multiple peaks exist.
- Passed: northern-cap grid coverage at 50 degrees, scalar sufficient-statistic
  and exact-median binning, incomplete-checkpoint skipping, atomic aggregate
  NetCDF output, coastline-overlaid `Polarplot.plotimg()` PNG generation, and
  NaN-aware one-cell Gaussian filtering.
- Passed: absolute SML/derivative filtering and derivative invalidation across
  index gaps. A real checkpoint matched all 964 finite samples to the cache.
- Passed: explicit cyclic collection after both successful and failed batch
  attempts.
- Passed: two-worker dispatch, spawned-process selection, active-worker
  capping, BLAS-thread propagation, parent-only result handling, and a real
  two-process spawn smoke test returning serializable worker failures.
- Passed: exact quarter-cell allocation for a 50-km square footprint centred
  on a grid intersection; correlated samples in one checkpoint collapse to one
  estimate per cell; footprint variables and the third polar PNG are written
  end to end. A real checkpoint rasterization was finite and took 0.012 s.
- Passed: nearest-minute SML matching and unique-minute histogram aggregation;
  a real 4,292-checkpoint run wrote both histograms from 26,724 covered minutes.
- Passed: IGRF/CHAOS output-directory separation and rejection of a complete
  checkpoint whose stored background does not match the requested run.
- Passed: coverage-threshold masking at the inclusive ten-count boundary,
  threshold metadata/mask output, and rendering of all four polar products.
- Passed: no-SML, independent satellite/MEM footprint/checkpoint aggregation,
  `(3, 4, grid_y, grid_x)` NetCDF output, and all eight 12-panel diagnostic
  figures. The tests use deliberately separated A/B/C and field values plus
  known per-cell distributions to detect satellite, quantity, or quantile
  mixing; the complete suite contains 57 passing tests.
- Passed: magnetic-latitude/MLT aggregation of offset-corrected observations,
  quiet/active and ascending/descending separation, checkpoint-level collapse,
  exact 10th/median/90th percentiles, atomic NetCDF output, and a representative
  12-panel figure. A real checkpoint matched the existing SML cache.
- Default Python import remains blocked by undeclared `kneed` and `ppigrf`.

## Portfolio impact

- Central update needed: Yes
- Changes: representative post-launch L2 ingestion reached the inversion. The
  pre-launch automatic-regularization regression is repaired, another event
  avoids the pole assertion, an nT-to-SI input mismatch was identified, and
  nandir now backs the solve at the precomputed-normal-equation boundary. The
  all-orbit lithosphere run is now restartable and archives self-contained
  per-inversion checkpoints without paying the native-grid evaluation cost.
- Sync summary: the first real-data execution now has concrete evidence, but
  it is not yet scientifically validated. Project priority remains unrecorded.
