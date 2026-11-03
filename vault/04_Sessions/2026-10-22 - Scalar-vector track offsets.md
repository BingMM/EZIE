# Scalar-vector track offsets

Last reviewed: 2026-10-22

## Question

The delivered scalar perturbation appeared inconsistent with the magnitude of
the retrieved vector components. The norm of the perturbation vector is not a
valid comparison: the consistent vector-derived scalar is

`|B_IGRF + dB_vector| - |B_IGRF|`.

## Reproduced result

For `ezie_l2_20250515_022454_sva_v003_r000.nc4`, compare the delivered
`retrieved_b_tot` directly with
`|B_IGRF_vector + dB_vector|`. Their difference is nearly constant within each
MEM track:

- MEM 1: +178.164 nT
- MEM 2: -5.414 nT
- MEM 3: -37.602 nT
- MEM 4: +277.722 nT

After removing each track's median difference, the residual standard deviation
is about 0.0011-0.0012 nT and the largest absolute residual is about 0.0023 nT.
The ancillary IGRF magnitude agrees with the norm of its N/E/D components to
within 0.004 nT.

## Interpretation

The scalar and vector products contain effectively identical along-track
variation in this event but different per-MEM zero levels. This supports an
offset-aware inversion, but does not identify whether the scalar or vector
absolute level is correct. The pattern must be checked across more events.

The scalar error is also almost identical to the vector covariance projected
onto the total-field direction: median pairs for MEMs 1-4 are approximately
131.2/131.3, 149.2/149.3, 137.2/136.9, and 135.7/137.2 nT. This is evidence
that the scalar and vector estimates are strongly related and should not be
treated as independent observations without retrieval-level covariance
information.

It is algebraically possible to combine the scalar magnitude with the vector
direction using `B_hybrid = F_retrieved * B_vector / |B_vector|`. This selects
the scalar absolute zero as authoritative. For this event it changes the
median ENU perturbations by about (-13.6, +32.2, -174.7) nT on MEM 1 and
(-21.4, +41.7, -273.6) nT on MEM 4. Because this introduces large mostly
vertical corrections, it should not be adopted without establishing which
product has the physically meaningful zero level and propagating the joint
scalar-vector uncertainty.

An offset-aware solve should either estimate constant nuisance parameters per
MEM/component or apply an equivalent contrast projection. If adjacent
differences are used explicitly, the data, design matrix, and covariance must
all be transformed; adjacent differences are correlated.

## Trial outcome

The user tested the magnitude-direction fusion and reported that it did not
improve the inversion. This is consistent with the correction removing only
one field-aligned combination of the possible E/N/U offsets. Two transverse
offset combinations per MEM remain unconstrained by scalar magnitude. The
experimental path also retains the original vector covariance, which is not
the covariance of the nonlinearly rescaled vector.

The saved `looking_at_the_data.py` currently calls `generate_ezie_data()`
without its experimental `offset_corrected=True` flag, so its present default
execution uses the original components; the reported trial may have been run
interactively or before the flag was reverted.
