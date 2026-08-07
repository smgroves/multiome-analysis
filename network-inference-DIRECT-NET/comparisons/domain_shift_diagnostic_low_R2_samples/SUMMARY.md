# Headline summary: BoBa-T hyperparameters and domain shift

Full detail in `FINDINGS.md` and `../../claude_analysis/BoBa-T_hyperparameters.md`.
This is the condensed version. *(Scripts referenced below live under
`network-inference-DIRECT-NET/claude_analysis/`, organized by phase.)*
Companion doc: `../organoid_walks/SUMMARY.md` covers RORB perturbation-walk validation.

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

