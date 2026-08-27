# Walk index: which simulations were run, from where, and where their plots live

This is a navigation doc, not a findings doc — it inventories every random-walk simulation
run for the RORB perturbation-validation work, grouped by *what kind of state it started
from* (GEMM-native vs. organoid-derived, pooled-population vs. single-real-cell,
archetype-average vs. true dynamically-discovered attractor). For the actual scientific
conclusions, see `FINDINGS.md` (full detail) and `SUMMARY.md` (condensed). Scripts referenced
below live in `network-inference-DIRECT-NET/claude_analysis/04_rorb_perturbation_validation/`.

All walks use the same fitted rules (`6667/rules/rules_6667.txt`), 4000 steps x 100
iterations, and (except where noted) measure position/occupancy against GEMM's own fixed
8-archetype reference frame.

## 1. GEMM-native archetype walks (the original basin set)

**Origin: GEMM.** Starting points are GEMM's own filtered attractor basins
(`6667/attractors/attractors_threshold_0.5`, `filtered=True`) — pooled-population walks
(every basin member seeded, not one representative cell). Only 3 of the 5 filtered basins
were walked and aggregated: `Arc_3`/Secretory, `Arc_5`/NE1, `Arc_6`/NE2. (`Generalist_NE`
and `Generalist_nonNE` were never aggregated this way — documented as a future option, not
executed, since the leaf-basin `perturbation_stats.csv` mechanism already covers
"does knockdown destabilize this basin" for them.)

- **Perturbation**: RORA_RORB knockdown vs. unperturbed.
- **Raw walk data**: `6667/walks/long_walks/4000_step_walks/Arc_{3,5,6}_radius_4_percentages*.csv`
  (`_RORA_RORB_kd` suffix = perturbed).
- **Plots**: same directory — `Arc_{3,5,6}_radius_4_['RORA_RORB_kd']_{percentages,reached}.pdf`.
- **Hexagon (RadViz) plot**: only `Arc_5` got this treatment —
  `hexagon_plots/walks/radviz_archetype_projection_gemm_Arc_5*.png` (plain, `_T3`, `_with_ASCL1` variants).

## 2. Organoid population walks, under GEMM's rules (`run_organoid_perturbation_walks.py`)

**Origin: organoid.** Starting points are real `organoid_shGFP` cells, binarized and grouped
by organoid's *own* `predicted.id` label, then collapsed to one representative discrete
state per label (`bb.tl.find_avg_states`). 4 of organoid's labels were walked: `Generalist NE`,
`Intermediate`, `Neuroendocrine1`, `Neuroendocrine2`. Measured against GEMM's fixed 5-basin
`attractor_dict` — i.e., organoid states walked forward under GEMM's dynamics, scored on
GEMM's own basin grid.

- **Perturbation**: RORA_RORB knockdown vs. unperturbed.
- **Raw walk data**: `6667/organoid_seeded/walks/long_walks/4000_step_walks/<idx>/` — idx-to-label
  lookup in `6667/organoid_seeded/average_states_idx_<label>.txt` / `average_states.txt`.
- **Occupancy summary (all 4 groups)**: `organoid_seeded_walk_basin_occupancy.csv`.
- **Axis-position plots (3 of 4 groups only — `Intermediate` has occupancy numbers but no
  dedicated plot)**:
  - Pseudotime-calibrated: `axis_position_plots/pseudotime_calibrated_axis/walk_pseudotime_axis_{Generalist_NE,Neuroendocrine1,Neuroendocrine2}*.png`
  - Raw Hamming %, Generalist_NE/nonNE poles: `axis_position_plots/raw_hamming_axis/walk_axis_position_{Generalist_NE,Neuroendocrine1,Neuroendocrine2}*.png`
  - Raw %, NE1/nonNE1 poles (no calibration): `axis_position_plots/ne1_nonne1_raw_axis/walk_axis_position_ne1nonne1_raw_{Generalist_NE,Neuroendocrine1,Neuroendocrine2}*.png`
- **Temporal-distance plots (3 of 4 groups)**: `temporal_distance_plots/walk_temporal_distance_{Generalist_NE,Neuroendocrine1,Neuroendocrine2}_combined*.png`
  (`walk_temporal_ordering.csv` / `_with_ASCL1.csv` hold the underlying step-of-closest-approach stats).
- **Hexagon (RadViz) plot**: only `Neuroendocrine1` got this treatment —
  `hexagon_plots/walks/radviz_archetype_projection_Neuroendocrine1*.png` (plain, `_T3`, `_with_ASCL1` variants).
- **Companion, not a walk**: `organoid_rorb_archetype_shift.csv` /
  `organoid_predicted_id_proportions_by_condition.csv` independently check the same claim
  in organoid's real (non-simulated) shGFP/shRORB1/shRORB2 data.

## 3. Single real-cell "new start" walks — archetype AVERAGE states (`run_new_start_walks.py`, `plot_new_start_*.py`)

**Mixed origin — see per-row.** Every one of these starts is a literal real cell (Hamming
distance 0 to the archetype's average state, for the GEMM ones), not a synthesized point —
but the average state itself is *not* a dynamically verified attractor (see §5).

| Start | Origin | idx | Perturbation | Plot prefix / folder |
|---|---|---|---|---|
| GEMM NE1 (`Arc_5` avg) | GEMM | `8914049663948766` | RORA_RORB knockdown | `gemm_ne1_start/{hexagon,axis,axis_raw,temporal}_GEMM_NE1.*` |
| GEMM `Generalist_nonNE` avg | GEMM | `1222330852650596` | RORA_RORB overexpression | `gemm_nonne_start_oe/*_GEMM_Generalist_nonNE.*` |
| GEMM `Arc_4`/nonNE1 avg | GEMM | `3474165076405860` | RORA_RORB overexpression | `gemm_nonne_start_oe/*_GEMM_Arc4_nonNE1.*` |
| GEMM `Arc_2`/nonNE2 avg | GEMM | `3474165059628640` | RORA_RORB overexpression | `gemm_nonne_start_oe/*_GEMM_Arc2_nonNE2.*` |
| Organoid real shRORB1+shRORB2 `Generalist nonNE` cells, pooled | **organoid** | `1371881672649956` | RORA_RORB overexpression | `organoid_shrorb_nonne_oe/*_organoid_shRORB_Generalist_nonNE.*` |

- **Raw walk data**: the 4 GEMM starts are under
  `6667/new_starts_seeded/walks/long_walks/4000_step_walks/<idx>/`; the organoid start
  shares the org-seeded root, `6667/organoid_seeded/walks/long_walks/4000_step_walks/1371881672649956/`
  (same pooling mechanism as §2, just a different — perturbed — organoid population).
- Each folder has up to 4 plot types: `hexagon_*` (RadViz), `axis_*` (pseudotime-calibrated),
  `axis_*_raw` (uncalibrated %, NE1/nonNE1 poles), `temporal_*` (Hamming-distance-vs-step).
  The organoid start has hexagon/axis/axis_raw/temporal but no separate "raw" axis for GEMM
  Generalist_nonNE variants beyond what's listed — check each folder's file listing directly
  for exact coverage per start.

## 4. True-attractor-seeded walks (this session — `*_true_attractor.py` scripts)

**Origin: GEMM, both entries.** Motivated by confirming that §3's average-state starts
weren't artificially unstable (see §5) — reruns two of §3's starts from a *bobaT-discovered,
dynamically verified* attractor state (0 of 53 genes with >50% flip probability — a genuine
fixed point) instead of the archetype's average state.

| Start | idx | Relation to avg state | Perturbation | Plot prefix |
|---|---|---|---|---|
| GEMM NE1 true attractor | `7788149757105118` | Hamming distance 2 from NE1 avg (`8914049663948766`) | RORA_RORB knockdown | `gemm_ne1_start/*_true_attractor.*` |
| GEMM nonNE true attractor (shared candidate — see below) | `95331434180192` | Distance 3 from `Generalist_nonNE` avg; distance 7 from `Arc_4` avg | RORA_RORB overexpression | `gemm_nonne_start_oe/*_true_attractor.*` |

- This nonNE state is a true-attractor candidate shared across `Generalist_nonNE` (filtered
  attractor), `Arc_4`/nonNE1 (unfiltered candidate), and `Arc_2`/nonNE2 (unfiltered
  candidate) — i.e. there may be only 1-2 truly distinct nonNE attractors despite 3
  archetype labels.
- **Raw walk data**: same root as §1's GEMM-native walks,
  `6667/walks/long_walks/4000_step_walks/<idx>/` (not `new_starts_seeded/`).
- **Result**: both true-attractor reruns reproduce the same qualitative and largely
  quantitative shift as their average-state counterparts in §3 — the earlier findings
  aren't an artifact of starting from an unstable point.
- **Gap**: no true-attractor rerun exists for §3's organoid-seeded start (row 5) — its
  average state was never checked for / replaced with a genuine fixed point.

## 5. Flip-probability diagnostic — not a walk (`plot_flip_probability_by_archetype.py`)

**Origin: GEMM (real cells' own binarized states).** A static stability check, no
simulation involved: every real GEMM cell's own discrete state is scored with
`bb.utils.get_flip_probs` against the fitted rules. Two metrics, per cell and per
archetype:
- `mean_flip_prob` — average flip probability across all 53 genes (never exactly 0, even
  at true fixed points, because individual gene-rule probabilities have a nonzero floor).
- `n_genes_flip_gt_0.5` — the strict, correct fixed-point count (0 = genuine attractor).
  Only 30/8908 real cells (0.3%) are true fixed points by this criterion.

This is what motivated §4: archetype AVERAGE states have 2-15 unstable genes by this
metric (not genuine attractors), while bobaT's own discovered attractors have exactly 0.

- **Outputs**: `flip_probability_diagnostic/flip_probability_{by_archetype,strict_by_archetype}.{png,pdf}`,
  `_summary.csv` (per-archetype stats), `flip_probability_per_cell.csv` (all 8908 cells).

## Slide deck

`figures-slides/GRN_benchmarking_summary.pptx`, "ATTRACTOR STABILITY DIAGNOSTIC" slides draw
from §5; "TRUE ATTRACTOR VALIDATION" slides draw from §4 (hexagon plots only — the axis/raw/
temporal true-attractor plots exist as files but weren't added to the deck).
