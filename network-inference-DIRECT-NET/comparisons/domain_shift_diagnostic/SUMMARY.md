# Headline summary: BoBa-T hyperparameters, domain shift, and RORB perturbation validation

Full detail in `FINDINGS.md` (11 sections) and `../../BoBa-T_hyperparameters.md`. This is
the condensed version.

## 1. The organoid null-condition shortfall is not a fittable hyperparameter

- Raising `node_normalization` (0.3→0.4, run `6668`) did **not** fix organoid_shGFP's poor
  external validation — it made things slightly worse across every external set tested.
  Ruled out the "stronger null, needs more data to resolve" hypothesis.
- Capping regulators-per-gene (run `6669`) and a prototype "global-reference"
  normalization scheme (run `6670`) were tested too: the cap was neutral (no cost, no
  benefit — in-sample fit identical to baseline); the normalization prototype correctly
  targets the right mechanism in principle but hit a real numerical failure mode when
  applied across datasets (predictions can saturate to a constant, making R² blow up).
  Neither moves the organoid shortfall.

## 2. The real driver: genuine cross-context rewiring, confirmed directly (not a statistical artifact)

- Initial read ("in vitro vs in vivo") was **wrong** — checked against all 33 of 6667's
  scored external samples (allografts, human tumors, organoid, mets_compiled), organoid's
  numbers are unremarkable; some human tumor and allograft samples validate just as
  poorly.
- What actually predicts validation quality: how much of a sample's own variance sits on
  **GEMM's specific fitted identity axes** (r=0.65 across all 33 samples) — not how much
  variance it has, and not how far its average expression state is from training data
  (distance-based metrics: r=-0.09 to -0.49, much weaker).
- **Decisive test**: grouping cells by their *exact* combinatorial regulator state (removing
  any population-mixture confound) shows organoid's disagreement with GEMM's rules is
  **real rewiring, not a compositional artifact** — GEMM's own held-out data agrees with
  itself to within 0.03-0.06; organoid disagrees by ~0.40 even conditioning on the identical
  regulator combination. Consistent with known biology: TF *activity* depends on
  cofactor/chromatin context that BoBa-T's regulator-expression values can't see, and that
  context differs between an intact in vivo tumor and organoid culture.
- Extended to the full network (53 genes x 33 samples): per-sample transferability
  correlates with mean R² at **r=0.96**, the single best predictor found. Even the most
  transferable real (non-self-loop) gene only reaches 63% agreement across this
  deliberately heterogeneous population — a realistic ceiling, not a bug.

## 3. organoid_shGFP is not "flat" — it has real biology that just doesn't align to GEMM's specific axes

- organoid_shGFP's own dominant variance (65% of it) is real, biologically coherent
  lineage heterogeneity (NE/Intermediate/ASCL1 identity), not noise — it just doesn't
  project onto GEMM's specific PCA-defined identity combination, the same underlying
  phenomenon as the rewiring finding above.

## 4. RORB knockdown: direction transfers at the archetype level, even though absolute values don't

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
