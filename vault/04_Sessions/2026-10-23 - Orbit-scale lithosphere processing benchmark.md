# Orbit-scale lithosphere processing benchmark

Last reviewed: 2026-10-29

## Scope

Review, timing, and implementation of a restartable `scripts/lithosphere_l3.py`
batch workflow. The archive contained 4,647 unique version-003, revision-000
files across spacecraft A, B, and C.

## Representative full-grid timing

For `ezie_l2_20250515_022454_sva_v003_r000.nc4`, the gradient model used 960
scalar samples and an 11,232-coefficient grid. With fixed `l1=0.75` and
`l2=2.5`:

- Level-2 loading: 0.05 s
- scalar SECS design matrix: 1.71 s
- transformed covariance factorization: 0.01 s
- `GTG`: 0.62 s
- dense `LTL`: 3.83 s
- dense regularization assembly: 0.43 s
- final dense solve: 2.33 s
- total lazy solve path: about 9.94 s

The process reached about 7.6 GB resident memory and used roughly 15 CPU cores
through threaded numerical libraries. One 11,232-square float64 matrix is
about 0.94 GiB. Serial inversion of all files would therefore require roughly
13 hours before post-solve evaluation and failure overhead.

## Prediction path

The fitted linear scalar prediction is already available as
`model.G_absolute @ model.m`; it took 0.003 s. Rebuilding vector SECS operators
at all observation locations took 1.68 s. On the checked orbit, reconstructing
the exact total-field magnitude from the vector prediction differed from the
linear scalar prediction by 0.206 nT RMS and 0.926 nT maximum, over a modeled
range of approximately -341 to +236 nT.

Using the cached scalar prediction would save about 1.7 s per orbit but changes
the reported quantity by the small second-order magnitude term. If exact
magnitude is required, evaluate all MEM locations in one call and split the
result afterward.

Native-grid vector evaluation is not a cheap addition. Evaluating E/N/U at
11,446 cubed-sphere mesh points took 19.46 s after an 8.61 s solve and raised
peak resident memory to about 21.4 GB. Repeating that for every file would add
roughly 25 hours to the batch. The preferred checkpoint therefore stores the
SECS coefficient vector together with the complete source-grid definition;
native-grid fields can then be generated later, selectively or on a coarser
common statistical grid, without repeating the inversion.

One self-contained NetCDF/HDF5 checkpoint per input file is the recommended
restart boundary. It should contain the input identity and processing
configuration; observation times, coordinates, scalar input, covariance and
background; fitted E/N/U and both linear and exact scalar predictions at data
locations; conditional track offsets and residual summaries; `m`; source-grid
center and mesh coordinates; projection position/orientation; radii and map
parameters; and code/dependency provenance. Dense posterior covariance,
normal matrices, and native-grid predictions should not be stored by default.
Write through a temporary file followed by an atomic rename, skip validated
completed outputs on restart, and keep an auditable failure manifest.

## Implemented checkpoint workflow

`scripts/lithosphere_l3.py` now processes one input at a time and writes one
compressed NetCDF4 checkpoint beneath `EZIE_lithosphere_l3`, preserving the
input directory layout. A checkpoint is written through a temporary file and
atomically renamed only after `processing_complete=1`; complete compatible
files are skipped on restart, while exceptions and tracebacks are appended to
`failures.jsonl`.

Each file stores satellite and source identity, all observation times and
positions, the retrieved scalar data and variance, IGRF background components,
predicted E/N/U at observation locations, linear and exact scalar predictions,
offset-restored predictions and residuals, per-MEM offsets, weighted residual
summaries, the SECS coefficients, full source-grid center and edge coordinates,
projection and grid parameters, regularization settings, timings, Git state,
and dependency versions. Native-grid field predictions and dense inverse
matrices are intentionally omitted.

`reconstruct_grid(path)` recreates the `CSgrid` directly from the saved edge
coordinates and projection. A representative full-resolution event produced a
0.542 MB checkpoint with finite coefficients and predictions; every recreated
grid and mesh coordinate matched the stored arrays exactly. The full
standard-library suite passes with 36 tests.

## First batch-run failure diagnosis

At 64.2% processed, 968 checkpoints had completed and 2,016 files had failed.
All failures were the same `secsy` assertion in
`CSprojection.local2geo_enu_rotation()` while constructing `get_Le_Ln()`; they
were not loading, covariance, or solver failures. The processed A files split
into 554 successes and 1,035 failures, while the partial B run had 414
successes and 981 failures at the diagnostic snapshot.

The grid centre inherited float32 latitude/longitude from Level-2. Near the
geographic pole, using that float32 centre produced nominally zero
vertical-horizontal rotation terms of about `1.4e-6` to `1.8e-5` in a sample
of 16 failed A/B files, above `secsy`'s fixed `1e-6` assertion. Rebuilding the
same projections after casting their positions and orientations to float64
reduced the terms to approximately `6.5e-15` to `7.5e-14`. Horizontal
rotation orthogonality was already accurate to about `1e-7`. This identifies
a coordinate-precision defect rather than invalid orbit data. Grid construction
now promotes latitude and longitude to float64 before calculating the
projection, track orientation, and grid width. The original failing event
`ezie_l2_20250515_004708_sva_v003_r000.nc4` subsequently completed a
full-resolution inversion and checkpoint with finite coefficients and
predictions. Completed checkpoints can remain because the script skips them
and the sub-metre grid difference is negligible relative to 20–40 km cells.
Preserve or rotate the original failure log so resolved historical failures
are not mistaken for current failures.

## Repeated-process OOM diagnosis

The restarted workflow sometimes terminated without recording a per-file
exception and continued normally after a process restart. Kernel logs confirm
three global OOM kills of Python processes: approximately 58.4 GB anonymous
RSS on 2026-10-23, 58.5 GB on 2026-10-25, and 55.9 GB on 2026-10-26. This is
far above the isolated 6–8 GB per-orbit peak and demonstrates cumulative
retention in the long-lived process.

`process_file()` accesses `model.ev`, which creates a strong cycle:
`Model._ev` references an `Evaluator`, while `Evaluator.model` references the
same `Model`. A focused weak-reference probe showed that deleting both local
names left both objects alive; an explicit `gc.collect()` then reclaimed the
cycle. Each retained model keeps its large `G`, `GTG`, Cholesky/system matrix,
grid, data, and prediction machinery reachable. NumPy array bytes do not drive
Python's generational collection thresholds proportionally, so automatic cycle
collection may occur too late for these unusually large objects.

The first repair is now implemented: `gc.collect()` runs in a `finally` block
after every completed or failed orbit, after `process_file()` has returned or
unwound. A focused test verifies collection on both paths. Monitor current RSS
over several full-grid iterations after restarting. Native allocator trimming
should only be considered if objects are confirmed reclaimed but RSS still
grows.

The production archive is now separated by background model. IGRF retains the
existing unsuffixed `EZIE_lithosphere_l3` directory for compatibility, while a
CHAOS configuration automatically writes to `EZIE_lithosphere_l3_chaos`.
Checkpoint completeness now includes the stored background attribute, so an
existing IGRF product cannot cause a CHAOS orbit to be skipped. The downstream
map defaults remain IGRF-specific and require explicit CHAOS input/output paths
when comparing backgrounds. The complete suite contains 53 passing tests.

## Configurable orbit workers

The user subsequently observed stable total RAM near 25% and CPU averaging
about 50%, with inversion spikes, while throughput remained just below seven
seconds per orbit. The batch script now accepts `--workers N`; one worker
remains the safe default. Independent files run in spawned processes, each
retaining explicit post-orbit cyclic collection. The parent alone updates the
progress bar and failure manifest.

To prevent every Cholesky factorization from requesting all CPU threads, the
parent sets BLAS/OpenMP thread limits before spawning workers. The automatic
limit is `logical CPUs // active workers`; `--blas-threads-per-worker` can
override it for controlled benchmarks. The environment is restored in the
parent afterward. A focused test covers worker-count capping and thread-limit
propagation, a real two-process spawn smoke test returned worker results, and
the full suite now contains 48 passing tests. Production speed-up and peak RSS
have not yet been measured, so this is an available experimental mode rather
than a new default.

## Sparse regularization implementation

EZIE now requests `CSgrid.get_Le_Ln(return_sparse=True)` and retains the
magnetic-east derivative `L` and `L.T @ L` as CSR matrices. The existing
direct solver still receives a dense regularization matrix, but that matrix is
allocated only once: the sparse smoothness matrix is converted, scaled in
place, and the ridge term is added directly to its diagonal. The
regularization optimizer's model-norm contraction was updated to support the
sparse matrix without changing its definition.

On the formerly failing full-resolution A event with 10,944 coefficients, `L`
had 43,772 nonzeros and `L.T @ L` had 98,476 nonzeros, a density of 0.0822%.
Constructing `L.T @ L` took 0.043 s and dense regularization assembly took
0.122 s. The mean-solution path completed in 5.86 s with 5.12 GB peak resident
memory; the complete prediction-and-checkpoint workflow took 7.11 s with 6.28
GB peak memory. The sparse model matched the saved dense baseline to
`2.84e-15` in relative model norm. Linear scalar predictions differed by only
`7.73e-14` nT RMS and `2.12e-13` nT maximum. All 37 focused tests pass.

Sparse `L` could support a matrix-free mean solver, but full exact `Cmpost` and
`Rm` are user requirements. The direct factorization therefore remains the
primary path.

After restarting the production batch with the sparse implementation, the
observed sustained processing time was about 9 seconds per inversion, down
from about 15 seconds. The production improvement is therefore approximately
40%; the difference from the isolated 7.11-second benchmark is consistent
with event-size variation, disk activity, and sustained system load.

Exact resolution now uses `Rm = I - Cmpost @ R`, with nandir applying the
original sparse regularization terms. EZIE's Gini map uses the same identity
for its one required resolution row per regularization pair. On the
10,944-coefficient event, the old dense `posterior_column @ GTG` took 0.0172 s
and the sparse complement took 0.00063 s. Rows agreed to `1.07e-13` relative
norm and their Gini indices to `1.78e-15`. Assembly and Cholesky factorization
took 3.37 s, so this exact optimization will only modestly affect the reported
5.3 s per Gini pair; repeated factorization is the actual bottleneck.

The user's production retest subsequently reported about 3 s per Gini pair.
The resolution-row algebra alone cannot explain the full change because its
measured saving was only 0.0166 s; sparse in-place assembly, cache state, and
sustained BLAS load also affect the observed rate. A separated full-grid probe
measured 0.195 s to assemble regularization, 0.321 s to add the dense system,
and a load-sensitive 5.71 s for Cholesky while another production process was
active. The 10,944-square float64 system occupies 0.892 GiB, and its Cholesky
factorization requires order `n^3/3`, approximately `4.4e11` floating-point
operations per regularization pair.

An exact Woodbury data-space formulation was tested because the event had only
896 gradient observations versus 10,944 model coefficients. Generic sparse LU
was slower than dense Cholesky: its 896 solves of `R X = G.T` took 6.08 s. The
regularization, however, had only seven populated diagonals and bandwidth 192.
Using banded Cholesky reduced a representative complete Gini pair to 1.64 s
from 3.28 s, a 1.99-times speed-up. Across the four corners of the optimizer's
`(l1, l2)` domain, speed-ups ranged from 1.08 to 3.31. The Woodbury model and
resolution row agreed with dense Cholesky to about `1e-14` and `9e-14`; Gini
differences remained below `2.5e-12`.

The verified banded formulation is now implemented as nandir's
`BandedWoodburySolver` and used only for the single resolution row required by
EZIE's Gini search. The final selected inversion remains on the established
direct solver, so full exact `Cmpost` and `Rm` behavior is unchanged.

## Main optimization opportunity

File loading and sparse regularization construction are now negligible. The
dominant remaining cost is dense system assembly and Cholesky factorization,
including one new factorization for every regularization pair. A mean-only
matrix-free solver would avoid this work but conflicts with the requirement
for full exact posterior covariance and resolution. Further acceleration must
therefore preserve those products, for example through an exact data-space or
other factorization strategy rather than an iterative-only backend.

The current dense solve already consumes approximately 15 hardware threads.
Orbit-level multiprocessing must therefore use the implemented per-worker
thread limits and will multiply the per-process memory footprint. The initial
two-worker result is favorable, but longer peak-memory monitoring is required
before parallel deployment becomes the default.

The first production test with two workers reports a 4–5 second checkpoint
completion cadence, improved from about seven seconds with one worker. Total
RAM sits just below 25% of the 62-GiB host, while aggregate CPU utilization has
a floor near 50% and spikes above 90%. Because parallel progress reports
completed files, this cadence is aggregate throughput rather than the duration
of either individual inversion. The observed headroom supports a controlled
three-worker test; automatic thread division will use 10 BLAS threads per
worker on this 32-logical-CPU system. Peak RSS and swap, not average RAM alone,
remain the stop criteria.

The three-worker trial also reports low RAM and approximately the same
aggregate CPU profile, although no new completion cadence was recorded. BLAS
has no correctness-related minimum thread count; even one thread is valid.
Because large Cholesky factorizations usually have diminishing parallel
scaling, more independent workers with fewer threads may improve aggregate
throughput. Four workers with the automatic eight threads each is the next
controlled point. At higher worker counts, simultaneous per-process memory
peaks are likely to become the constraint before a four-thread BLAS floor is.

For the final statistical map, online count/sum/sum-of-squares bins would avoid
retaining every orbit's latitude, longitude, and prediction arrays in the
`out` dictionary. The script's bare `except` should also be replaced by logged
`Exception` handling before a multi-hour run so failed orbits are auditable.

## Northern scalar aggregate map

`scripts/lithosphere_global_map.py` implements the downstream statistical map.
It takes a stable snapshot of completed `.nc` checkpoints and streams one file
at a time, avoiding an archive-sized in-memory table. The default field is the
nonlinear reconstructed scalar perturbation `predicted_dB_exact`.

The target is a north-polar `CSgrid` covering all longitudes down to 50 degrees
latitude with nominal 50 km cells. The resulting square grid is 178 by 178;
samples south of 50 degrees are excluded and a cap mask identifies cell centres
within the requested region. The atomic NetCDF output retains count, sum, sum
of squares, mean, exact median, sample standard deviation, standard error, the
full centre and edge geometry, and source-file/sample metadata. The sufficient
statistics allow means and variances to be recomputed or merged without
reopening every orbit. Exact median calculation additionally retains the
binned values and 32-bit cell IDs during this downstream pass, which remains
modest compared with the inversion memory footprint.

The command-line workflow then uses `Polarplot.plotimg()` to save the median
map in nT beside the NetCDF file. Its color limits are symmetric around zero
and use the 98th percentile of the finite absolute medians to prevent a few
extreme bins from flattening the map. Geographic 50 m coastlines are overlaid.
A NetCDF variable retains the NaN-aware Gaussian-filtered median with sigma one
grid cell, but the separate sigma-one PNG has been removed.

Visible track striping motivated a third, physically informed aggregation.
Each prediction now represents a 50-by-50-km square FOV aligned with the
global cubed-sphere xi/eta axes. Exact projected overlap fractions distribute
the value among intersected cells. All overlapping samples from one checkpoint
are then collapsed to a single area-weighted cell estimate before the code
takes an equal-checkpoint median. Thus the roughly 15-km sample spacing does
not increase the statistical count for a cell several times within the same
inversion. The raw centre-bin product is retained for direct comparison.

Schema 1.1 adds footprint checkpoint count, sample-cell overlap count, summed
fractional coverage, mean, exact median, checkpoint standard deviation, and
checkpoint standard error. The command-line workflow writes a third footprint
PNG, uses `RdBu_r` for all maps, and accepts `--footprint-km`. The legacy raw
sample standard error remains in the file but is not an independent-sample
uncertainty. A real 960-sample checkpoint produced 474 footprint/checkpoint
cell estimates from 3,202 overlaps in 0.012 s; all outputs were finite.

The workflow now also renders a fourth coverage PNG from
`footprint_checkpoint_count`. This is the number of checkpoint-level cell
estimates after correlated samples within each inversion have been collapsed,
not the inflated raw 10-second sample count. The map uses a linear `viridis`
scale, coastlines, and the same polar grid; its path is configurable through
`--coverage-plot`.

All three magnetic-field plots now require at least ten footprint/checkpoint
estimates per cell by default. This is an inclusive threshold and is adjustable
with `--minimum-coverage-count`. It is applied only to rendered field values;
the coverage map remains complete and the underlying statistics are retained.
NetCDF schema 1.2 records the threshold and the resulting
`plot_coverage_mask`. A focused boundary test brings the complete suite to 54
passing tests.

Schema 1.3 adds a coverage-composition control that is now always calculated.
The 12 satellite/MEM combinations remain distinct through footprint
aggregation. Every checkpoint estimate is assigned the
inverse of its stream's checkpoint count in that cell, giving every represented
stream total weight one in an exact weighted median. The ordinary
equal-checkpoint footprint median is retained for a direct before/after
comparison. The NetCDF adds `scalar_footprint_equal_mem_median` and
`footprint_represented_mem_count`, and the command writes a separate
`_equal_mem` PNG using the legacy map's color limits. The former equal-MEM CLI
options have been removed; both footprint products are standard output.

A synthetic cell with three checkpoints in one stream and one in another
confirms that checkpoint abundance cannot dominate the equalized median. On a
real four-MEM checkpoint, separately rasterizing and recombining the MEMs gives
identical cell IDs, footprint values, coverage weights, and contribution counts
to the original all-MEM rasterization. The complete suite contains 60 passing
tests.

An explicit `--all-data` mode now bypasses SML loading, matching, and filtering
altogether. With default paths it writes `scalar_50km_all.nc` plus the
historical PNG names ending in `_all`, rather than overwriting the quiet-time
map. Both footprint products are produced automatically. Schema 1.3 records whether the
SML filter was applied and the number of selected finite observations; SML
match and quiet counts are marked not applicable in all-data output. A
synthetic run verifies all-data plotting with no SML cache. The first real
all-data run completed aggregation and wrote a valid NetCDF, then failed during
plotting because the optional equal-MEM field was accessed when absent. The
equal-MEM product is now always calculated, the associated CLI options and
sigma-one PNG are removed, and the regression passes.

## Polar coverage-ring diagnosis

The global-map run also writes median MLT per footprint cell. It obtains the
four MEM MLT tracks from `MagneticCoords/obs_magLT1...4` in the original
Level-2 file named by checkpoint provenance, while supporting an embedded
`magnetic_local_time` variable for self-contained or synthetic checkpoints.
The footprint calculation unwraps each track across midnight, combines MEM
directions on the unit circle, and uses a circular median across checkpoint
estimates. Outputs are `footprint_median_mlt` in NetCDF and an automatic cyclic
MLT PNG; 24-hour numerical seam values are normalized to zero. The full suite
passes 62 tests.

The all-data footprint coverage map contains several high-count rings near the
pole, and visible red/blue boundaries in the footprint median overlap parts of
those rings. A stratified sample of 310 completed checkpoints initially gave
5th/median/95th-percentile maximum latitudes of 74.96/80.85/82.30° for pooled
MEM 1, 73.67/78.82/79.99° for pooled MEM 2, 71.40/75.76/76.69° for pooled MEM
3, and 76.65/84.04/86.74° for pooled MEM 4. This grouping was incomplete: MEM
numbers are satellite-specific, so the analysis combined three distinct tracks
under every MEM number. These percentiles cannot identify the individual ring
envelopes; the correct diagnostic has 12 satellite/MEM groups.

The footprint map already reduces every checkpoint to at most one estimate per
cell, so the rings do not come from counting adjacent 10-second observations as
independent. Track turning geometry and a changing set of contributors remain
plausible explanations, but a coverage threshold cannot correct that kind of
compositional change. Aligned median transitions are therefore suspect until
the archive is split into the 12 satellite/MEM tracks.

That diagnostic is now implemented in
`scripts/lithosphere_mem_debug_maps.py`. It deliberately bypasses SML loading
and filtering, keys each stream by satellite plus MEM, and applies the same
50-km footprint overlap and within-checkpoint collapse used by the combined
map. It writes one atomic NetCDF with `(satellite, mem, grid_y, grid_x)` fields,
plus shared-scale 3-by-4 median and coverage figures. It now carries the
reconstructed scalar perturbation, fitted track offset, `observed_dB`, derived
`observed_dB - track_offset`, and `background_B` through that same aggregation
in one pass. The derived quantity is not the Level-2 debiased product, which is
not present in retrieved-source checkpoints. Four additional figures map these
diagnostic quantities in nT, while the NetCDF retains their means,
medians, standard deviations, standard errors, and field-specific counts. The
products are isolated under `/media/bing/LaCie/EZIE_lithosphere_mem_debug`.
The default ten-count threshold masks only the median panels; the coverage
panels retain all populated cells. A synthetic six-checkpoint test assigns
distinct values to A, B, and C and verifies all 12 tracks and five fields,
skips incomplete checkpoints, and writes the NetCDF plus eight figures. The
complete suite has 57 passing tests.

The first real debug run processed the complete IGRF archive of 4,647
checkpoints (A 1,589, B 1,533, C 1,525). Some individual reconstructed-field
panels have broad curved bands near their inner coverage boundaries, but these
are not unambiguously the concentric rings visible in the combined map. The
fitted-offset panels do not show matching sharp rings and the
background-magnitude panels are smooth. A quiet-time CHAOS aggregate made from
3,459 checkpoints
also shows clear rings, but its satellite mix is incomplete: A and B are
complete while C contributes only 337 checkpoints. These observations weaken
both background choice and fitted offsets as primary explanations, but do not
separate cross-track composition from inversion geometry. The derived
`observed_dB_minus_offset` diagnostic should distinguish rings already present
in the corrected observations from structures introduced by the inversion.

That comparison shows the paired positive-poleward/negative-equatorward MEM-3
bands in both corrected observations and reconstructed predictions, placing the
structure upstream of the inversion. At 3,110 sampled MEM-3 locations, IGRF
field-magnitude finite differences give median sensitivities of about
130 nT/degree latitude and 25 nT/km altitude. A 100–150 nT absolute discrepancy
thus requires roughly 0.8–1.2 degrees latitude or 4–6 km altitude at a typical
location. MEM 3's roughly 55-degree Earth-incidence angle converts the latter
to only 6–9 km horizontal footprint displacement. An effective emission-height
or view-angle retrieval error is more plausible than a roughly 100-km literal
horizontal geolocation error, although a uniform height bias would largely be
removed by the fitted track offset.

Schema 1.4 of the debug aggregate adds exact 10th and 90th percentiles of
`observed_dB - track_offset`, calculated across checkpoint-level cell estimates
after the existing footprint collapse. Two additional 3-by-4 figures use the
same support threshold and no-SML selection as the median. A focused quantile
test and the end-to-end output test bring the full suite to 57 passing tests.

`scripts/debug_analysis.py` adds the magnetic-coordinate discrimination stage.
For every satellite/MEM it maps `observed_dB - track_offset` on a configurable
MLAT/MLT grid (default 1 degree by 0.5 hour), separates quiet and active SML
conditions, and splits samples by local ascending/descending spacecraft
latitude. Within-checkpoint samples in one cell are averaged before checkpoints
receive equal weight in exact 10th-percentile, median, and 90th-percentile
statistics. Defaults define quiet as `abs(SML) < 80 nT` with
`abs(dSML/dt) < 10 nT/min` and active as `SML <= -200 nT`; all are CLI options.
The script writes one atomic NetCDF to the separate
`EZIE_lithosphere_debug_analysis` directory. The full-archive run exposed why
the initial 16 three-by-four figures were not interpretable: nearly all
supported cells lie within a few hours of midnight, the 0/24 boundary split
that sector across both ends of each panel, and descending checkpoint-cell
support exceeds ascending support by factors of 77 for quiet data and 44 for
active data. The default output was therefore replaced by five focused
figures: a nightside descending-pass quiet/active MEM-1-versus-MEM-3 median
comparison, a direction-support chart, and one MEM-3 percentile comparison per
satellite. `--plot-existing` regenerates these from the saved NetCDF;
`--exhaustive-plots` preserves the former matrices when explicitly requested.

The focused result shows the broad low-latitude-negative and
high-latitude-positive reversal in MEM 1 as well as MEM 3, especially during
active SML, but MEM 3 has substantially larger amplitude on all satellites.
The quiet MEM-3 median retains a weaker paired structure. This evidence does
not support a simple MEM-3-only geolocation error or a purely activity-driven
electrojet; it is compatible with physical activity dependence combined with
an angle-dependent MEM-3 retrieval gain or bias. The archive cannot test
ascending-versus-descending consistency because its directional support is too
imbalanced. A synthetic end-to-end test, saved-NetCDF reload test, midnight
unwrap test, and full suite pass; the repository suite now contains 58 tests.

Eight focused tests cover cap-edge coverage, statistics and exact median,
quiet-time filtering, incomplete files, NetCDF output, Gaussian filtering, and
polar PNG generation, plus exact footprint overlap and within-checkpoint
collapse. The complete
50-test suite passes. A real checkpoint from the
running archive was read without mutation: all 964 scalar samples were finite
and the pre-filter geometry check assigned 873 north of 50 degrees to the
global grid. The full archive aggregate should be run after the producer
finishes so its file snapshot is complete.

## SuperMAG quiet-time selection

The map now matches each checkpoint observation time to the nearest minute of
SuperMAG SML and retains samples only when `abs(SML) < 80 nT` and
`abs(dSML/dt) < 10 nT/min`. The derivative is calculated with centred finite
differences in nT/min and marked invalid beside any cadence gap larger than 90
seconds. Matching uses a maximum separation of 60 seconds.

The confirmed SuperMAG account retrieved the complete source-archive interval
from 2025-05-15 00:00 through 2025-08-31 23:59 UTC. The atomic cache at
`/media/bing/LaCie/EZIE_lithosphere_maps/supermag_sml.nc` contains 156,960
consecutive one-minute values with no gaps. Of these, 53,501 minutes (34.1%)
pass both thresholds. The first checked orbit matched all 964 finite samples;
none passed the quiet filter because that orbit occurred during disturbed
conditions. Two additional focused tests cover the absolute thresholds and
gap-safe derivative; the later Gaussian-map and batch-collection tests bring
the complete suite to 48 tests.

The active cache was later removed with the map output directory, and the next
quiet run failed when SuperMAG returned an unsuccessful response on download
chunk six. The original complete, gap-free cache was found in the LaCie trash
and copied back while preserving the trash copy. Cache coverage was verified
against the current all-data map's checkpoint interval. The downloader now
retries a failed seven-day chunk up to three times with 5- and 10-second
backoff before raising. The new retry regression and all 61 tests pass.

## SML coverage histograms

`scripts/lithosphere_sml_histograms.py` plots the SML and `dSML/dt`
distributions at times covered by complete inversion checkpoints. Because each
one-minute index value is otherwise repeated across four MEMs and several
10-second samples, the histogram counts unique matched SML minutes rather than
raw observations. Dashed lines show the ±80 nT and ±10 nT/min map-selection
limits, and the title reports the fraction satisfying both.

The implementation reads every checkpoint only once, reuses the existing SML
cache, and downloads only if the cache no longer covers the checkpoint span.
A real run over the current 4,292 complete checkpoints found 26,724 unique
data-covered minutes and a 35.4% joint quiet fraction. No network request was
needed. Two focused tests cover nearest-minute tolerance and deduplication;
the complete EZIE suite now contains 52 passing tests.
