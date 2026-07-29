Goals for future implementation 
---------------------------------
1. Better config file that can batch process many samples using the same reductionn parameters without selecting each one individually
2. Output format to follow through all configuration parameters. Give an option for ORSO specific details.
3. Track a


Stitching — failure modes and potential fixes
----------------------------------------------
Audit of the current implementation (`src/pxr_reduce/reduction.py`:
`mark_stitch_points`, `compute_scale_factors`, `_fit_scale_factor`,
`apply_scaling`, `finalize`). Legend: [silent] = wrong result, no error;
[dropped] = data removed with only a logged warning; [loud] = raises.
Observe all of these with `loader.diagnose_stitches()` and the
`--diagnostics` counts-vs-theta plot (num_stitch_points, failed flag, scale,
segment alignment).

### Failure modes

Detection (`mark_stitch_points`)
- [silent] Missed boundary: a boundary is marked only on a `sam_th` back-step
  (> `stitch_theta_backstep`, 0.001 deg) OR a change in a watched column
  (`hos, exposure, slits_vert, slits_horz`). A condition change in any other
  motor with no angle back-step is never marked -> two segments left at their
  own I0 levels, unscaled -> a step in R.
- [silent] First reflectivity frame of a scan is never a boundary (by design),
  so a stitch coinciding with it is not caught.
- [dropped] Spurious boundary: with `stitch_condition_tol = 0`, any watched
  readback jitter marks a boundary; if it has no real overlap it becomes a
  failed stitch (below) and masks the rest of the scan.

Overlap matching (`compute_scale_factors`) — main failure surface
- [dropped] Empty overlap -> FAILED stitch. Overlap needs matching angles on
  both sides; the set empties via (a) exact-angle matching (identical `sam_th`
  only; a slightly-off return sweep -> no match), (b) the `stitch_cutoff`
  (1.003) spot/dark filter dropping all faint high-angle overlap points, or
  (c) all overlap points saturated. On empty overlap the loop sets
  `failed_stitch_mask` from that index and BREAKS, abandoning every later
  boundary in the scan; with `drop_failed_stitch=True` the whole tail is then
  dropped in finalize. One early failure nukes the rest of the scan.
- [silent] Wrong points matched when settling repeats have jittery (non-equal)
  `sam_th`: the `repeat` counter under-counts and post-frames are mis-classified.

The fit (`_fit_scale_factor`)
- [silent] No goodness-of-fit check: unweighted through-origin least squares
  (`post = scale*pre`); if overlap points disagree it still returns a scale.
- [silent] Single overlap point: only warns; `scale_err` comes back 0.0
  (covariance inestimable) — misleadingly precise.
- [silent] Ill-conditioned / negative inputs (near-zero or negative overlap R
  from dark over-subtraction) -> scale blows up or goes negative/zero.
- [silent] Fit ignores `R_err` (noisy overlap point weighted like a clean one).

Accumulation & application (`apply_scaling`)
- [silent] Cumulative propagation: scales multiply forward, so a wrong early
  scale offsets the entire rest of the curve; no global re-fit.
- [dropped] scale <= 0 silently zeros/negates R -> removed by the `R > 0`
  filter in finalize.

Scan segmentation coupling
- [silent] Pooled multi-scan samples not split: if `sam_th` does not jump more
  than `new_scan_marker` (15 deg) between pooled scans, they are treated as one
  scan and stitched across separate measurements (different I0). Reflectivity
  max angles are often < 15 deg, so tune `new_scan_marker` down.

Not a failure
- A bad I0 makes absolute R wrong but does not break stitching (the scale is a
  ratio, so I0 cancels within a scan).

### Potential actions (rough priority)

1. Don't `break` on a failed stitch — skip that boundary, keep going, and mask
   only its own segment (or seed from the condition ratio), so one bad overlap
   doesn't discard the rest of the scan.
2. Tolerance-based / interpolated angle matching instead of exact `sam_th`
   equality (match nearest angle within a tolerance, or interpolate R(q) of one
   segment onto the other's overlap angles).
3. Robust, weighted scale estimate: weighted geometric mean / log-ratio of
   per-angle ratios with outlier rejection, instead of unweighted curve_fit;
   report a real single-point uncertainty rather than 0.0.
4. Fit-quality / sanity check: flag stitches whose overlap points disagree
   beyond tolerance, or whose fitted scale diverges from the expected condition
   ratio (e.g. exposure change should give ~1 after normalization).
5. Guard degenerate scales (<= 0, non-finite) and surface them instead of
   silently dropping points.
6. Widen / make configurable the set of stitch-trigger conditions; consider a
   small `stitch_condition_tol` default to reject readback jitter.
7. Consider a global (all-segment) least-squares stitch to avoid cumulative
   error growth from sequential pairwise scaling.
8. Ensure pooled multi-scan samples segment correctly (lower/auto
   `new_scan_marker`, or segment on scan-ID rather than only `sam_th` jumps). 