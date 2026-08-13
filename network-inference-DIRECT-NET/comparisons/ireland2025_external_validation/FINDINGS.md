# External validation of 6667 on Ireland et al. 2025 (Zenodo 15857303)

New external-validation samples from ["Basal cell of origin resolves neuroendocrine–tuft
lineage plasticity in cancer"](https://zenodo.org/records/15857303) (Ireland et al. 2025,
Nature) — a separate mouse SCLC lineage-plasticity paper using the same RPM/RPMA/RPR2
genetically-engineered models already referenced elsewhere in this project. Scoped to the
record's 5 AnnData (h5ad) files (of 16 total; the 11 Seurat/SCE `.rds` files were left out
of this round per direct user decision). Pipeline, gene panel, and env conventions follow
`claude_analysis/05_preprocessing/preprocess_zenodo_ireland2025.py` and
`claude_analysis/01_normalization_sweep/main_external_validation_organoid_mets.py`.

## 1. Preprocessing

All 5 files share an identical scanpy-derived schema (QC columns `n_genes_by_counts`/
`total_counts`/`pct_counts_mito`; `layers["counts"]` reliably genuine raw integer counts in
every file, `X` was NOT reliably raw — sometimes already log-transformed depending on the
file — confirmed by value inspection, not layer name, per this project's standing rule).
Gene panel was pulled directly from the 6667 network's own 53 nodes (`rules_6667.txt`), not
the older, broader Direct_net/FigR_DORC_TF/extra_genes union — simpler and sufficient, per
user feedback during planning. **All 5 datasets matched 54/54 genes (53 nodes + RORA/RORB
separately) — zero missing genes in any of them.**

| dataset | cells (post-QC) | condition column | condition groups |
|---|---|---|---|
| `cgrp_k5` | 27,674 | `Cre` | K5 (20,046) / CGRP (7,628) |
| `organoid_celltag` | 9,089 | `Genotype` | RPMA (6,131) / WT (1,767) / RPM (1,191) |
| `tbo_allograft_5khvg` | 4,435 | `GenoCT` | RPM_CTpostCre (2,836) / RPM_CTpreCre (1,599) |
| `rpr2_allograft` | 19,364 | `Genotype` | RPM (10,572) / RPR2 (8,792) |
| `celltag_fate_dpt` | 26,618 | `Genotype` | RPM (16,458) / RPMA (10,160) |

100% of cells passed QC (`pct_counts_mito<20`, `n_genes_by_counts>200`) in every dataset —
already pre-filtered upstream. MAGIC-imputed output sanity-checked: no NaNs, RORA_RORB
present everywhere, and even `organoid_celltag`'s near-zero pre-MAGIC ASCL1 signal (only 4
distinct values — that dataset is entirely basal/hillock/secretory lineage, no NE
population, so this is real biology not a bug) became properly continuous post-MAGIC
(8,112 distinct values).

## 2. Validation results (brcd 6667, norm=0.3)

| sample | mean R² | mean AUC | mean F1 | mean accuracy |
|---|---|---|---|---|
| `celltag_fate_dpt` | **0.330** | **0.789** | 0.744 | 0.756 |
| `celltag_fate_dpt_RPMA` | 0.138 | 0.720 | 0.657 | 0.689 |
| `celltag_fate_dpt_RPM` | 0.059 | 0.680 | 0.630 | 0.656 |
| `organoid_celltag` | 0.308 | 0.786 | 0.696 | 0.741 |
| `organoid_celltag_WT` | 0.315 | 0.790 | 0.720 | 0.741 |
| `organoid_celltag_RPM` | 0.266 | 0.767 | 0.690 | 0.724 |
| `organoid_celltag_RPMA` | 0.229 | 0.753 | 0.678 | 0.715 |
| `tbo_allograft_5khvg_RPM_CTpreCre` | 0.207 | 0.740 | 0.686 | 0.709 |
| `tbo_allograft_5khvg` | 0.129 | 0.705 | 0.652 | 0.678 |
| `rpr2_allograft` | 0.135 | 0.695 | 0.641 | 0.673 |
| `rpr2_allograft_RPM` | 0.104 | 0.696 | 0.639 | 0.672 |
| `rpr2_allograft_RPR2` | 0.103 | 0.693 | 0.648 | 0.668 |
| `cgrp_k5_K5` | 0.103 | 0.704 | 0.659 | 0.679 |
| `cgrp_k5` | 0.088 | 0.697 | 0.650 | 0.670 |
| `cgrp_k5_CGRP` | 0.087 | 0.694 | 0.647 | 0.668 |
| `tbo_allograft_5khvg_RPM_CTpostCre` | 0.086 | 0.691 | 0.641 | 0.666 |

All 16 samples score comfortably within the range of the existing 33-sample population
(mean R² spans ~0.02–0.55 there); none are degenerate. Notably, `celltag_fate_dpt` and
`organoid_celltag` (both R²≈0.3, AUC≈0.79) validate substantially **better** than the
already-scored `organoid_shGFP`/`mets_compiled` from the earlier Zenodo dataset (R²≈0.02),
despite `organoid_celltag` being, like the earlier organoid, an in-vitro culture system —
consistent with this project's established finding that in-vitro-vs-in-vivo status is not
itself the driver of validation quality (`domain_shift_diagnostic_and_organoid_walks/
FINDINGS.md` §4).

Outputs: `6667/validation/external_validation/<name>/` per sample (`summary_stats_fixed.csv`
via `fixed_get_sklearn_metrics` — plain `get_sklearn_metrics` mislabels `RORA_RORB` as
`RORA`, see that module's docstring).

## 3. Missing-gene marginalization: not exercised this round

Since all 5 datasets matched 54/54 network genes, the marginalize-vs-zero-fill comparison
built in `claude_analysis/06_marginal_rule_validation/` (`marginalize_rules.py`,
self-tested; `main_external_validation_marginal.py`, template ready) had no live case to run
on. Reusable if a future external dataset is missing a network gene.

## 4. Is poor extrapolation explained by batch effects? (`claude_analysis/07_batch_effect_diagnostics/`)

**Quick kNN batch-mixing test, all 33+5 samples**: correlated a per-dataset "batch
separation score" (how much more often a cell's k-nearest-neighbors share its own dataset
vs. chance) against mean R²/AUC. Result: **r=+0.62 for both** — the *opposite* sign a
"batch effects hurt validation" story predicts. More-separated datasets validate *better*,
not worse. Likely mostly re-detecting the already-known restriction-of-range/diversity
effect (`domain_shift_diagnostic_and_organoid_walks/FINDINGS.md` §6, r=0.58, same
direction) rather than an independent batch signal. `cgrp_k5` itself is unremarkable here
(37x separation, typical of the allograft/human-tumor population).

**Housekeeping-gene check** (12 genes not expected to track cell-state biology: Actb,
Gapdh, Tbp, B2m, Ppia, Rplp0, Ywhaz, Sdha, Hprt, Ubc, Polr2a, Tubb5 — extracted from raw
full-transcriptome sources for the 19 datasets where that's available: GEMM, organoid,
the 5 Ireland sets, all 12 allografts; human tumors/mets_compiled excluded, no raw source /
R-only access respectively). Same PCA+kNN metric run on this panel: **mean
housekeeping/network separation ratio = 0.455** — real but substantially weaker than the
network genes' own separation, consistently across every single dataset (never above 1.0).
If the network-gene separation were purely technical, housekeeping genes should separate
about as strongly; they don't (about half). Argues for a real biology-specific excess
signal, not a pervasive technical confound.

**ComBat correction + revalidation** (the direct test): ComBat-corrected the network-gene
panel (dataset as batch covariate) across the same 19 datasets, then rescored using a
POOLED (not per-sample) quantile reference — necessary because BoBa-T's standard
`norm=0.3` per-sample quantile-clip is invariant to any monotonic per-gene correction
applied beforehand and would otherwise silently erase ComBat's effect entirely. Guarded
explicitly against the numerical failure mode that broke an earlier, related
"global-reference-norm" prototype (R² blew up to -1.8e14 when a gene's corrected variance
collapsed near zero) by flagging and excluding any gene with post-correction actual-value
std < 1e-3 from the mean, rather than letting it silently dominate. See
`combat_vs_baseline_comparison.csv` for the full before/after table.

**Result: ComBat correction makes validation dramatically worse, not better — a clear,
decisive negative result, not an inconclusive one.** Across the 17 scoreable datasets, mean
R² collapsed from a healthy baseline of **0.223 to -0.065** (worse than predicting the
mean), and **0/17 datasets improved**. Some collapses were severe (`organoid_celltag`:
0.308 → **-2.764**; `rpr2_allograft`: 0.135 → **-0.509**); even the mildest cases got worse
(`allograft_TKO-luc`: 0.550 → 0.525). Zero genes hit the numerical-degeneracy guard (the
failure mode that broke the earlier global-reference-norm prototype) — this is a real,
consistent effect, not an artifact of a few broken genes dominating the mean. **Mean AUC
stayed roughly flat** (0.751 → 0.755) — diagnostic: AUC depends only on prediction rank
order, R² on absolute deviation from actual, so a correction that shifts predictions'
absolute calibration without much scrambling their relative ranking produces exactly this
signature (R² craters, AUC barely moves).

**Conclusion**: this reinforces, rather than resolves differently from, the earlier
global-reference-norm failure (`BoBa-T_hyperparameters.md` §4) — BoBa-T's rules were fit
expecting each dataset to be rescaled against *its own* quantiles (`node_normalization=0.3`
applied per-sample), not a shared/pooled reference frame. Forcing external data onto a
common calibration breaks that assumption regardless of how the shared calibration is
derived (a logistic squash to GEMM's own scale there, ComBat + a pooled quantile reference
here). Combined with §4's finding that network genes separate ~2x more than housekeeping
genes (real biology, not a pervasive technical confound) and the positive
separation-vs-R² correlation, the full weight of evidence argues against "batch effects are
why extrapolation is failing" — and now additionally shows that actually attempting batch
correction, not just diagnosing for it, makes things worse. This is consistent with this
project's independent, earlier finding (via leaf-conditional agreement on organoid_shGFP)
that poor cross-context validation reflects genuine biological rewiring, not a fixable
technical artifact.

Outputs: `combat_corrected/adata_<name>_combat_globalnorm.csv` (per-dataset corrected +
rescaled data), `combat_vs_baseline_comparison.csv` (full before/after table),
`6667/validation/external_validation_combat/<name>/` (per-dataset validation output).
