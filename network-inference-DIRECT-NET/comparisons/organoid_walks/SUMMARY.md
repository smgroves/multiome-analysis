# Headline summary: RORB perturbation-walk validation

Full detail in `FINDINGS.md`. This is the condensed version.
Companion doc: `../domain_shift_diagnostic/SUMMARY.md` covers why organoid/mets_compiled
validate poorly against GEMM's fitted rules in the first place.

## 1. RORB knockdown: direction transfers at the archetype level, even though absolute values don't

*(Archetype names used below: `Arc_1`=Intermediate, `Arc_2`=nonNE2, `Arc_3`=Secretory,
`Arc_4`=nonNE1, `Arc_5`=NE1, `Arc_6`=NE2.)*

- Direct single-gene test (does GEMM's rule predict the right *sign* of RORB knockdown's
  effect on each of its 9 direct target genes in organoid) was weak and not significant.
- But at the **archetype/state-space level**, seeding BoBa-T's actual perturbation
  simulation (not the validation machinery) with organoid's own real cells and walking
  forward under GEMM's fitted rules reproduces the right direction: NE-basin occupancy
  collapses under RORB knockdown (e.g. 10.5%→0.4%), proportionally similar to GEMM's own
  native simulation.
- Independently, organoid's real experimental data shows a substantial, statistically
  robust, dose-like shift from NE identity toward Intermediate identity across
  shGFP→shRORB1→shRORB2 (`Generalist NE` 75%→67%→64%, `Intermediate` 5%→14%→29%),
  confirmed against a clean negative control.
- Follow-up: measured Hamming distance to all 8 archetype average states directly
  (including the 3 excluded from the usable basin set, e.g. `Arc_1`="Intermediate",
  confirmed real) instead of relying on fixed-radius basin occupancy. Placing every
  archetype on the shared NE↔nonNE combinatorial axis (% agreement with `Generalist_NE`
  on the 46 genes that actually differ between `Generalist_NE`/`Generalist_nonNE`) shows
  all 8 fall on one ordered spectrum: `Generalist_NE` 100% → `Arc_6` 96% → `Arc_5` 91% →
  **`Arc_1`/"Intermediate" 65%** → `Arc_3` 59% → `Arc_4` 7% → `Arc_2` 4% →
  `Generalist_nonNE` 0%. Knockdown moves cells significantly *along this exact axis*,
  toward the nonNE end — it just travels past `Arc_1`'s 65% way-point all the way to
  `Generalist_nonNE`/`Arc_2`/`Arc_4` (~95-100%) within the 4000-step simulation. **Right
  axis, right direction — the simulation just doesn't stop at "Intermediate," it runs
  further**, plausibly because an unconstrained long walk approximates the asymptotic
  fate of sustained knockdown, while the real, finite shRNA experiment captures a partial,
  earlier snapshot of the same trajectory (consistent with (C)'s real shift being
  substantial but not complete). This also explains the original puzzle: a point 65%
  along a 46-bit path is still far from both ends in raw Hamming terms even though it's a
  genuine, correctly-ordered interpolation — a continuous PCA/UMAP embedding just
  compresses that same path into 1-2 dimensions where "65% along" looks visually central.
- **Directly confirmed by tracking the walk trajectories over time**: closest approach to
  `Arc_1`/Intermediate happens significantly *earlier* than closest approach to the nonNE
  cluster, in 74-89% of individual walks across all three NE-starting organoid
  populations (Wilcoxon p down to 2e-14). The mean distance-vs-step curve shows the shape
  directly — distance to Intermediate dips to a minimum early (~step 900), then rises
  again as the walk continues past it toward nonNE. Same shape in the unperturbed walk,
  much smaller amplitude — knockdown doesn't create the ordered NE→Intermediate→nonNE
  passage, it amplifies it enough to actually complete the transit.
- **Visualized directly as a single trajectory** by collapsing the network's 46-gene
  NE↔nonNE axis to one number per step (100=`Generalist_NE`, 0=`Generalist_nonNE`,
  Intermediate at 65). All three NE-starting organoid populations show the same picture:
  unperturbed walks drift toward, and settle almost exactly at, the Intermediate way
  -point; knockdown walks cross straight through it and continue on toward the nonNE side
  (`walk_axis_position_*.png`).

## Bottom line for the paper

BoBa-T's rules encode context-specific regulatory logic, not a context-independent causal
law — this is a real, reportable finding, not a defect. The method generalizes well to
new in-vivo tumors (allografts) but should not be expected to reproduce absolute
expression values in organoid culture. It *does*, however, correctly reproduce the
*direction* of a targeted perturbation's effect at the archetype level even where absolute
prediction fails — a meaningfully different and more defensible claim than "predicts
organoid expression," and one with direct, positive experimental support.
