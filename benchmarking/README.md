# GRN Benchmark: boba-T vs. CellOracle (and other GRN-inference methods)

This folder benchmarks boba-T (BooleaBayes) against CellOracle and, in principle, any other GRN-inference method (SCENIC, GENIE3, WGCNA, ...). It grew out of a design discussion about how to compare boba-T's fitted Boolean rules to CellOracle's per-target regressions; that discussion's conclusions are folded into the sections below so the reasoning survives alongside the code.

Everything method-specific normalises to one **canonical edge table** (`source, target, weight, score, sign` — see [`grn_benchmark/edges.py`](grn_benchmark/edges.py)), so a new method is just a loader function registered in [`grn_benchmark/loaders.py`](grn_benchmark/loaders.py)'s `METHOD_LOADERS`, and every comparison below runs on it unchanged.

There are three datasets in play, and it matters which one a given quickstart uses:

- **Run `6667`** — your actual SCLC/AA network (53 genes, 228 DIRECT-NET candidate edges), already fit and validated in `network-inference-DIRECT-NET/6667/`. No independent ground truth exists for these genes, so comparisons on this data are boba-T vs. CellOracle *directly*, or vs. the DIRECT-NET candidate network as a stand-in reference.
- **The mouse ground-truth benchmark** — Tabula Muris scRNA + Cusanovich scATAC + ChIP-Atlas ground truth, the data CellOracle's own paper used for Fig. S2, already downloaded (see [`data/README.md`](data/README.md)). This is the one with a real, independent gold standard, so it's the one that can produce an actual Fig.-S2-style AUROC/EPR figure — but boba-T has never been run on it, and doing so needs a real decision about where its candidate network comes from, since there's no paired multiome ATAC+RNA here the way DIRECT-NET expects. That decision, and the full pipeline to get there, is in [Running boba-T on the mouse ground-truth benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark).
- **BEELINE's synthetic GSD dataset** (bundled at `data/beeline/GSD/`) — a third, independent dataset, unrelated to CellOracle or your SCLC data. It's synthetic single-cell data simulated from a *literally known* Boolean network (via BoolODE, the BEELINE paper's simulator), so its ground truth is exact rather than ChIP-Atlas-inferred. Right now it's only used to sanity-check the harness itself (`run_beeline_selftest`, with synthetic `perfect`/`random` baselines, no real method) — it isn't wired to CellOracle or boba-T. It's flagged here because, in principle, it's arguably a *better*-suited ground truth for boba-T than the mouse benchmark: it's Boolean-model-generated data being scored against a Boolean network, which is boba-T's own modeling paradigm, not CellOracle's linear-regression one. Nobody has run boba-T on it; that would be new work, parallel to (not a prerequisite for) the mouse-benchmark plan below — full plan in [Running boba-T on BEELINE's synthetic GSD dataset](#running-boba-t-on-beelines-synthetic-gsd-dataset).

## The comparison points

### 1. Network structure vs. a reference — **implemented**

Do two methods (or a method and a reference) agree on which edges/hubs exist? `metrics.structure_metrics` restricts both networks to their shared node universe, then reports edge Jaccard / precision / recall / F1, degree-profile Spearman correlation, and sign concordance (activator vs. repressor) on edges present in both. `metrics.pairwise_jaccard` gives the same edge-overlap number method-vs-method with no reference required.

Note on scope: this is edge/degree agreement, not the network-science centrality comparison (betweenness, eigenvector centrality, scale-free-ness) that CellOracle's `Links.get_network_score()` / `plot_degree_distributions()` produce internally. Those are available if a deeper topology comparison is wanted later, but the harness doesn't need to reimplement them — CellOracle computes them on its own `Links` object.

**Roadmap:**

1. **Done** — boba-T (`6667`) vs. the DIRECT-NET candidate network it was fit on.
2. **Done** — boba-T (`6667`) vs. CellOracle (`6667`), both scored against that same candidate network as the reference, using the coefficient matrix comparison 3 already fit (below).
3. **Done** — boba-T (`6667`) vs. CellOracle (`6667`) vs. three real, independent ASCL1 ChIP-seq ground truths (two human, one RPR2-mouse — not the candidate network), scored separately. The mouse one (25 shared nodes) is the most statistically meaningful of the three; the two human ones are underpowered (4–5 shared nodes) — see [Sourcing a real SCLC ground truth](#sourcing-a-real-sclc-ground-truth-done-with-a-correction).
4. **Not done** — boba-T (mouse tissue) vs. CellOracle (mouse tissue) vs. ChIP-Atlas. Blocked on rerunning boba-T on the mouse ground-truth benchmark; see [that section](#running-boba-t-on-the-mouse-ground-truth-benchmark).

Steps 1–2, run end to end just now:

```python
from grn_benchmark.loaders import load_bobat, load_bobat_topology
from grn_benchmark.edges import matrix_to_edges
from grn_benchmark.metrics import run_structure_comparison
import pandas as pd

bobat_edges = load_bobat(run="6667")   # -> 6667/rules/signed_strengths.csv (boba-T's fitted, pruned network)
candidate_net = load_bobat_topology(
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
coef = pd.read_csv("../network-inference-DIRECT-NET/6667/rules/celloracle_coef_matrix.csv", index_col=0)
celloracle_edges = matrix_to_edges(coef, regulators_on_columns=False)

print(run_structure_comparison({"boba-T": bobat_edges, "CellOracle": celloracle_edges}, candidate_net))
```

Output lands at `benchmarking_out/comparison1_structure_6667.csv`, but read those numbers with the same caution as the roadmap above already flags: the DIRECT-NET candidate network is not a ground truth, so this only measures agreement with the candidate set each method started from, not correctness. See [Sourcing a real SCLC ground truth](#sourcing-a-real-sclc-ground-truth-done-with-a-correction) below for the fix — an actual independent reference (ChIP-seq from an SCLC mouse model) would make this comparison mean something.

`celloracle_coef_matrix.csv` above comes from comparison 3's fitting script, [`comparison3_fit_celloracle_6667.py`](comparison3_fit_celloracle_6667.py) — see that comparison for how it was produced.

**On the mouse ground-truth benchmark**, once boba-T has been rerun per tissue (see the dedicated section below), compare it to CellOracle's already-released network for the same tissue, scored against ChIP-Atlas as the reference for both:

```python
from grn_benchmark.loaders import load_bobat
from grn_benchmark.co_reproduction import load_co_link, load_chip_atlas_gt
from grn_benchmark.metrics import run_structure_comparison

sample, tissue = "Heart-10X_P7_4", "Heart"
bobat_edges = load_bobat(run=f"tm_{sample}")   # your new per-tissue boba-T run (see below)
co_edges = load_co_link(f"data/celloracle/inference_results/{sample}/celloracle_cluster_mouseAtacBaseGRN")
gt = load_chip_atlas_gt(tissue)
print(run_structure_comparison({"boba-T": bobat_edges, "CellOracle": co_edges}, gt))
```

`celloracle_cluster_mouseAtacBaseGRN` is the released CellOracle variant to compare against, specifically because it uses the *same* mouse-scATAC-atlas base GRN that the boba-T run below is built on — the other released variants (`celloracle_cluster_promoterBaseGRN`, `celloracle_cluster_scrambledPromoterBaseGRN`, `DCOL`, `SCENIC_10kb`) use different priors and aren't a same-base-GRN comparison.

### 2. Edge-weight recovery, BEELINE-style (AUROC / EPR) — **implemented**

CellOracle's Supplementary Fig. S2 treats each method's edge list as a binary classifier of "is this TF→target edge real" against a ground-truth network, following the [BEELINE](https://github.com/Murali-group/Beeline) paradigm (Pratapa et al., *Nat Methods* 2020):

- **AUROC** — rank every candidate TF→target pair by the method's score, label by ground-truth membership, score with ROC-AUC. 0.5 = random.
- **EPR (Early Precision Ratio)** — precision within the top-*k* predicted edges (*k* = number of ground-truth edges), divided by the random-baseline precision (*k* / number of candidate pairs). EPR > 1 = better than random; it rewards getting the *strongest* edges right rather than the whole ranking.

`metrics.beeline_metrics` / `run_beeline_comparison` implement this generically (candidate universe = ground-truth gene set × itself, self-loops excluded by default — the BEELINE convention). `co_reproduction.py` is a **faithful re-implementation of CellOracle's own scoring** (from their released `GRN_benchmarking.py`), used to validate the harness: it reproduces their Fig. S2 AUROC/EPR to 3 decimals on their released `inference_results` scored against ChIP-Atlas ground truth, before boba-T's own edges are ever dropped in as a method. (Note: this benchmark corresponds to CellOracle's *Supplementary* Fig. S2, not the main-text Fig. 7 — the two were conflated earlier in this project; every reference in this repo has been corrected to say Fig. S2.)

**Prerequisite**: a gold-standard edge set independent of both methods. Without one, AUROC/EPR are undefined — there's nothing to score edges against.

**Roadmap:**

1. **Done** — harness self-test on BEELINE's synthetic GSD ground truth (`run_beeline_selftest`): a `perfect` baseline (the ground truth itself) scores AUROC 1.0, a `random` baseline scores AUROC ≈ 0.5 / EPR ≈ 1. Sanity-checks the metric code; no real method involved.
2. **Done** — CellOracle's own released `inference_results` reproduced against real ChIP-Atlas ground truth (`run_co_reproduction_all`), matching their published Fig. S2 numbers to 3 decimals. Validates the scoring against real biology; still no boba-T.
3. **Done, sanity-check only** — boba-T (`6667`) ranked against the DIRECT-NET candidate network itself, since these SCLC genes have no independent ground truth (`data/celloracle/chip_atlas_gt/` is mouse-tissue, unrelated genes). See below.
4. **Not done** — rerun boba-T on the mouse ground-truth benchmark with the atlas base GRN, so it gets a real row/point in an actual Fig.-S2-style AUROC/EPR figure. This is the one that answers "how does boba-T's edge recovery compare to CellOracle/SCENIC/DCOL against real ChIP-Atlas ground truth" — full plan in [Running boba-T on the mouse ground-truth benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark).

Step 3, the `6667` fallback:

```python
from grn_benchmark.loaders import load_bobat, load_bobat_topology
from grn_benchmark.metrics import run_beeline_comparison

bobat_edges = load_bobat(run="6667")
candidate_net = load_bobat_topology(
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
nodes = set(candidate_net.source) | set(candidate_net.target)   # 53 genes
print(run_beeline_comparison({"boba-T": bobat_edges}, candidate_net, nodes))
```

Treat that one as a sanity check on ranking quality, not a Fig-S2-equivalent claim about recovering *true* biology — the "ground truth" there is the same LASSO network boba-T was handed, so it can only show whether rule-fitting agrees with the candidate edges, not whether those edges are correct.

**Fairness caveat carried over from the original design discussion**: CellOracle's Fig-S2-style benchmark scores *base GRN + regression pruning* together, so an apples-to-apples run should feed both methods the **same base GRN** rather than letting CellOracle use its own scATAC/promoter base while boba-T uses something else. Doing that turns the comparison into "given the same candidate edges, whose pruning/weighting ranks the true edges higher" — the interpretable, confound-free version of this benchmark. This is exactly the principle the mouse-benchmark section below applies.

### 3. Predicted vs. actual expression — **implemented and run, on `6667`**

This is the more direct accuracy question: given a method's fitted rules/coefficients, how well do they predict a TF's actual expression on held-out cells? This mirrors boba-T's own validation exactly, so the target shape is fixed by [`bobaT/tl.py:640`](../../BoBa-T/bobaT/tl.py) (`get_sklearn_metrics`) in the boba-T repo:

- boba-T's `fit_validation()` writes one `<gene>_validation.csv` per TF under `accuracy_plots/`, columns `CellID, predicted, actual` (`predicted` = P(TF ON | regulator states) from the fitted Boolean rule; `actual` = normalized expression), scored on **held-out** samples, never the training cells.
- `get_sklearn_metrics(VAL_DIR)` then computes per-gene accuracy, balanced accuracy, F1, precision, recall, ROC-AUC, explained variance, max error, R², log-loss — thresholding continuous values at 0.5 for the classification metrics.

To put CellOracle on the same footing: after `oracle.fit_GRN_for_simulation(...)`, `oracle.coef_matrix` (rows = regulators, columns = targets) is the analog of boba-T's fitted rule. Apply the **train-fitted** coefficient matrix to a genuinely held-out expression matrix and write the same `<gene>_validation.csv` shape, so `get_sklearn_metrics` runs on CellOracle's output unmodified.

**Roadmap:**

1. **Done** (pre-existing) — boba-T's own `6667` validation, at `6667/validation/in_sample_validation/accuracy_plots/`.
2. **Done** — CellOracle fit and scored on the identical `6667` train/test split. Two scripts, one per conda env:
   - [`comparison3_fit_celloracle_6667.py`](comparison3_fit_celloracle_6667.py) *(run in `celloracle_env`)* — fits CellOracle on `6667/data_split/train_t0combined.csv` using the same 228-edge DIRECT-NET network as a custom base GRN, and the same already-normalized `[0,1]`-scale expression values boba-T used (no re-normalization, so both methods fit on identical numbers). Applies the train-fitted `oracle.coef_matrix` to the held-out `test_t0combined.csv` and writes matching `<gene>_validation.csv` files to `6667/validation/celloracle_validation/accuracy_plots/`.
3. **Done** — GENIE3 (Huynh-Thu et al. 2010) added as a third method, same treatment: [`comparison_genie3_fit_6667.py`](comparison_genie3_fit_6667.py) *(either env — plain sklearn)* fits a `RandomForestRegressor` per target gene on that gene's candidate regulators from the *same* 228-edge network (restricting candidate regulators per target is why this is a direct sklearn implementation of GENIE3's algorithm rather than the `arboreto` package, which only supports one global regulator list for every target, not a per-target restriction), applies it to the held-out test split the same way, writes to `6667/validation/genie3_validation/accuracy_plots/`.
   - [`comparison3_score_celloracle_vs_bobat_6667.py`](comparison3_score_celloracle_vs_bobat_6667.py) *(run in `bobaT_env`)* — now scores *all three* methods' validation directories through boba-T's own `get_sklearn_metrics` and joins them into one long-format table, `benchmarking_out/comparison3_all_methods_vs_bobat_6667.csv`. Written to extend cleanly: add a `"MethodName": path` entry to its `VALIDATION_DIRS` dict once a new method's fitting script has written `<gene>_validation.csv` files in the same shape.
4. **Not attempted** — full SCENIC and WGCNA. SCENIC's network-inference step is itself GENIE3 (or GRNBoost2, the same algorithm) — already covered above — but SCENIC's regulon-*pruning* step (RcisTarget, motif-enrichment-based) needs species-specific cisTarget ranking databases (multi-GB downloads for human) not present in either conda env; running that step wasn't attempted. WGCNA builds an undirected soft-thresholded co-expression network (R package), which doesn't naturally produce a predictive model to apply to held-out cells the way Ridge/RF do — it would need a different comparison design (e.g. module-membership-based prediction) to fit this comparison's shape at all, so it wasn't attempted here; it may still be usable for comparison 1 (structure) directly.
5. **Not done** — the same comparison on the mouse ground-truth data. Needs an explicit train/test split there first (CellOracle's own released `inference_results` are fit on *all* cells for that sample, so they can't be reused for a held-out comparison) — a smaller variant of the per-tissue pipeline in the mouse-benchmark section, once that exists.

**Real result**, 42 genes with surviving regulators in all three networks (of 53 total in the `6667` network):

| metric | boba-T | CellOracle | GENIE3 |
|---|---|---|---|
| R² | 0.832 | 0.780 | **0.899** |
| ROC-AUC | 0.968 | 0.965 | **0.978** |
| F1 | 0.913 | 0.909 | **0.941** |

GENIE3's random forest comes out ahead of both boba-T and CellOracle on every metric here — plausible on its own terms: an RF is a much more flexible, nonlinear predictor than either boba-T's Boolean rules or CellOracle's linear ridge, so on a pure held-out-prediction-accuracy question it has room to fit patterns the other two structurally can't. That's a real result, not an artifact — the held-out split, base GRN, and shared-node rule are identical to the boba-T/CellOracle comparison — but it says more about raw predictive flexibility than about which method recovers the "right" regulatory structure (that's comparison 1's job, not this one's); a highly flexible model winning a pure prediction contest while still getting the causal structure wrong is a very ordinary failure mode, not ruled out by this number. `alpha=10` for CellOracle and `n_estimators=1000` for GENIE3 were both left at defaults, not tuned — a fairer three-way race would sweep both. Full per-gene numbers: `benchmarking_out/comparison3_all_methods_vs_bobat_6667.csv`.

Two crash bugs had to be worked around before the CellOracle fitting script above would even run (GENIE3's plain-sklearn script had none of these) — neither touches the ridge fit, the predictions, or the R²/AUC/F1 numbers reported above; both are CellOracle plumbing unrelated to the actual modeling math, and both would raise an exception (no output at all) if left unfixed rather than silently changing a result. Noting them in case they recur on other data: `Oracle.import_anndata_as_normalized_count` reads `adata.obsm[embedding_name]` even though `embedding_name` defaults to `None` in its signature — pass a placeholder 2D array if there's no real embedding (only used by CellOracle's own 2D-plotting features, never read by the regression). It also calls an internal QC step that reads `adata.layers["raw_count"]`, but the line that would set that layer from `adata.X` is commented out in installed `celloracle==0.20.0`'s own source (`oracle_core.py`) — set `adata.layers["raw_count"] = adata.X.copy()` yourself before the call (that QC step only feeds a print-only warning used later in `simulate_shift`, comparison 4; it's not consumed by `fit_GRN_for_simulation`), or `simulate_shift` will later hit a missing `self.high_var_genes` attribute.

### 4. In-silico perturbation — **stubbed**

`metrics.run_perturbation_comparison` is a placeholder for comparing predicted vs. observed responses to TF perturbation: CellOracle's `oracle.simulate_shift(...)` expression-shift vectors vs. boba-T's attractor-landscape/perturbation output, scored by direction agreement (cosine/sign) and, if ground-truth KO/OE data exists, DE correlation and rank agreement on "most impactful TF".

**Roadmap** (both steps on `6667` specifically — there's no KO/OE ground truth on the mouse tissues either, so the mouse benchmark doesn't unlock anything new here):

1. **Not done** — collapse boba-T's `6667/perturbations/<attractor_id>/results.csv` (52 attractor-state directories, each with one row per gene's knockdown/activation destabilization score for random walks *started from that attractor*) into a single per-gene or per-cluster effect vector, comparable in shape to CellOracle's per-cluster shift output. Probably via `bb.utils.get_perturbation_dict`, already used in the `6667` run script for this exact kind of aggregation — but the aggregation itself hasn't been written.
2. **Not done** — CellOracle's `simulate_shift` on the same `6667` inputs, reusing the `oracle` object comparison 3's fitting script already builds:

```python
# same knockdown boba-T's own walk_to_basin step used (RORA and RORB were combined
# into one node for this network, per main_all_data_remove_selfloops_6667's header comment).
oracle.simulate_shift(perturb_condition={"RORA_RORB": 0}, n_propagation=3)
shift = oracle.adata.layers["simulated_count"] - oracle.adata.layers["imputed_count"]
```

Step 1 is the bigger lift and the real blocker; step 2 is a few lines once step 1 defines what shape to compare it to.

## Sourcing a real SCLC ground truth (done, with a correction)

Comparison 1's `6667`-vs-candidate-network quickstart above only measures self-consistency with a shared starting point, not correctness. This section documents sourcing (and then actually scoring against) a ground truth independent of both methods.

**Correction, since resolved**: two datasets were identified — GSE69394 (Borromeo et al. 2016, *Cell Reports*, [PMID 27452466](https://pubmed.ncbi.nlm.nih.gov/27452466/)) and GSE150999 (Pozo et al. 2021, *iScience*) — with GSE69394's Table S9 initially expected to be the RPR2-mouse match, since that paper does include ASCL1 ChIP-seq in an actual RPR2 mouse tumor (`GSM1700644`, *Trp53;Rb1;Rbl2* triple-knockout). Once Table S9 was opened, that turned out to be wrong: **Table S9 is human data** — ASCL1 target genes correlated across 81 human primary SCLC tumors and 38 human cell lines, with ChIP-seq confirmation *in human ASCL1-high cell lines* (its own "Notes" sheet says so explicitly). The real RPR2-mouse ChIP-seq data is **Table S8** of the same paper — since obtained and scored below.

What was used, all three ASCL1-only (single source TF — none of the tables cleanly attributes targets to NKX2-1 or PROX1 individually; see below):

- **`data/sclc_chipseq_gt/borromeo2016_ascl1_human_chip.csv`** — Table S9's "ASCL1 genes union set" sheet, filtered to `Bound by Ascl1 (2 out of 3 cell lines) == "yes"`: 620 `ASCL1 → target` edges, unsigned (binding evidence only, no direction). Human.
- **`data/sclc_chipseq_gt/borromeo2016_ascl1_mouse_chip.csv`** — Table S8's `mSCLC_fpkm_RNA-seq` sheet, filtered to `ASCL1 bound in mSCLC != "No"` (that column holds peak IDs when bound, the literal string `"No"` when not): 3,992 `ASCL1 → target` edges, unsigned. **This is the actual RPR2-mouse ground truth.** Gene symbols are mouse case convention (`Ephb2`) in the source table; upper-cased in the saved CSV to match the human-convention symbols `6667`'s network uses, which is what ortholog symbol-matching across species conventionally does.
- **`data/sclc_chipseq_gt/pozo2021_ascl1_direct.csv`** — Table S6's "105 ASCL1 Direct repressed" + "190 ASCL1 Direct activated" sheets (the two sheets behind Fig. 5C's Venn, ChIP binding *and* siASCL1-knockdown differential expression in NCI-H2107): 295 signed `ASCL1 → target` edges (−1 repressed, +1 activated). Human. Table S6's other sheets (`132 3TF shared...`, `145 ASCL1+NKX2-1...`, `163 NKX2-1...`, etc.) were **not** used for NKX2-1/PROX1 edges: those genes were only knocked down *in combination with ASCL1* (the "3TF" condition), never individually, so an expression change there can't be cleanly attributed to NKX2-1 or PROX1 alone rather than to the co-knocked-down ASCL1. Table S4 (ChIP peaks + HiChIP gene assignment, no functional filter, broader) was left unused since S6 already gave a usable, more rigorous list.

Scored with [`comparison1_structure_6667_vs_chipseq_gt.py`](comparison1_structure_6667_vs_chipseq_gt.py) — reuses `load_bobat`/the already-fit `celloracle_coef_matrix.csv` from comparison 3, `run_structure_comparison`, and a new `loaders.load_sclc_chipseq_gt`, against each ground truth **separately**, three output files:

| ground truth | species | n_shared_nodes | method | n_pred | n_overlap | precision | recall | f1 | sign_concordance |
|---|---|---|---|---|---|---|---|---|---|
| Borromeo human (620 edges) | human | 5 | boba-T | 4 | 1 | 0.250 | 0.20 | 0.222 | n/a (unsigned GT) |
| Borromeo human (620 edges) | human | 5 | CellOracle | 3 | 1 | 0.333 | 0.20 | 0.250 | n/a (unsigned GT) |
| **Borromeo mouse, RPR2 (3,992 edges)** | **mouse** | **25** | **boba-T** | **61** | **7** | **0.115** | **0.28** | **0.163** | n/a (unsigned GT) |
| **Borromeo mouse, RPR2 (3,992 edges)** | **mouse** | **25** | **CellOracle** | **55** | **7** | **0.127** | **0.28** | **0.175** | n/a (unsigned GT) |
| Pozo (295 edges) | human | 4 | boba-T | 4 | 2 | 0.500 | 0.50 | 0.500 | 0.0 |
| Pozo (295 edges) | human | 4 | CellOracle | 3 | 2 | 0.500 | 0.50 | 0.500 | 1.0 |

Full tables: `benchmarking_out/comparison1_structure_6667_vs_{borromeo2016_ascl1_human,borromeo2016_ascl1_mouse,pozo2021_ascl1_direct}.csv`.

**Read the two human rows as a proof that the pipeline works, not a real accuracy comparison** — 4–5 shared nodes and 1–2 overlapping edges has no meaningful confidence interval either direction. The one qualitative signal worth noting without over-reading it: on the Pozo ground truth, boba-T got the *existence* of both overlapping edges right but the *sign* wrong on one of them (`sign_concordance` 0.0), while CellOracle got both signs right (1.0) — interesting, but n=2, treat it as a lead, not a conclusion.

**The RPR2-mouse row is the one actually worth reading** — 25 shared nodes and 7 overlapping edges is a real, if still modest, sample. Both methods land close together (F1 0.163 boba-T vs. 0.175 CellOracle, identical n_overlap of 7, identical recall of 0.28) — CellOracle predicts fewer total edges among these 25 nodes (55 vs. boba-T's 61) for the same overlap, giving it a slightly better precision/F1, but this is a small enough gap and small enough N that "CellOracle edges out boba-T on RPR2-mouse ASCL1 targets" would be overstating it. What it does support: both methods recover a real, non-random fraction (28%) of actual ASCL1 ChIP-seq targets from the RPR2 mouse model, restricted to genes both methods could call.

Not yet tried: feeding any of these ground truths into `run_beeline_comparison` (comparison 2) instead of `run_structure_comparison` — all three are real independent references, so AUROC/EPR would be as legitimate here as the structure numbers above (and the mouse one, with 25 shared nodes, has enough of a candidate universe to make AUROC less noisy than on the two human ones).

## Running boba-T on the mouse ground-truth benchmark

This is the path to an actual Fig.-S2-style figure — real AUROC/EPR numbers against an independent ChIP-Atlas gold standard, with boba-T plotted alongside CellOracle/SCENIC/DCOL instead of against itself. It needs three things that don't exist yet: a candidate network for boba-T on data with no paired multiome, boba-T's expression/cluster inputs converted from the benchmark's preprocessed format, and a per-tissue boba-T run. None of this has been run — everything below is a verified-API plan, not a tested pipeline; treat it as a checklist to work through one sample at a time, not a script to fire off across all 13 at once.

### The candidate-network decision

boba-T's SCLC runs get their candidate network from DIRECT-NET, which needs paired single-cell multiome (ATAC + RNA from the same cells) to link peaks to genes. The mouse benchmark doesn't have that — Tabula Muris (scRNA) and the Cusanovich atlas (scATAC) are unpaired, matched only by tissue, which is also how CellOracle itself used them. DIRECT-NET simply cannot run here; boba-T needs a different source for its candidate edges.

The resolution: CellOracle ships a base GRN already built from that exact Cusanovich atlas — `celloracle.data.load_mouse_scATAC_atlas_base_GRN()` (mm9, TF motif scan over ~92k peaks from the atlas, verified against the installed package). Handing this to boba-T as its candidate network means both methods start from the same TF-binding prior, which is exactly the "same base GRN" fairness principle used everywhere else in this document — and it's a real ATAC-derived prior, not a workaround. The comparable released CellOracle variant, `celloracle_cluster_mouseAtacBaseGRN`, uses the same base GRN, which is why that's the variant to compare against in the structure quickstart above rather than the promoter-based or scrambled variants.

```python
# Run once in celloracle_env: cache the mouse scATAC-atlas base GRN as boba-T's candidate network too.
import celloracle as co

base = co.data.load_mouse_scATAC_atlas_base_GRN()   # 91,976 peaks x 1093 TFs, mm9, same atlas as data/cusanovich_atac/
base.to_parquet("/Users/xpz5km/Documents/GitHub/multiome-analysis/benchmarking/data/celloracle/mouse_scATAC_atlas_base_GRN.parquet")
```

This base GRN is genome-wide (peaks x 1093 TFs), not restricted to any one tissue or gene set, so it needs restricting to a tractable node universe per sample before it can be a boba-T candidate network — see the next subsection for why and how.

### Preprocessing: from the benchmark's format to boba-T's

`grn_benchmark/preprocess.py` already reproduces CellOracle's own scRNA pipeline for all 13 Tabula Muris channels (see [`data/README.md`](data/README.md)), writing `log_data.mtx` (genes x cells, log1p), `all_genes.csv`, `var_genes.csv` (the ~3000-gene HVG set CellOracle itself models on), and `meta_data.csv` (has a Louvain `cluster` column — the GRN unit) to `data/preprocessed/<channel>/`. boba-T needs its inputs in a different shape: a 2-column candidate-network CSV, a CellID-rows/gene-columns expression CSV, and a CellID/class cluster CSV — matching exactly what `main_all_data_remove_selfloops_6667 copy.py` reads for run `6667`.

**A scale decision that has no single right answer**: run `6667`'s network has 53 genes. Restricting the atlas base GRN to the ~3000 HVGs CellOracle itself uses would make boba-T's candidate network roughly 60x larger than anything it's been run on before — every target gene could have dozens to hundreds of candidate TF regulators once restricted only to `var_genes`. `bb.tl.get_rules` fits each target's rule independently (closer to CellOracle's own per-gene regression cost than to BooleaBayes' attractor search, which does scale with 2^n and is *not* needed for this comparison — validation and rule-fitting are), so it's plausibly tractable, but this hasn't been tested. Recommended first move: run the smallest tissue by ground-truth size (Heart, 12 ChIP-Atlas TFs, whose CellOracle results are already unpacked locally per the structure quickstart above) with the full HVG set, time it, and only then decide whether to subset further (e.g. intersect with a smaller top-N-by-dispersion set, or restrict to genes within a couple of hops of the ChIP-Atlas gene set) before committing to all 13 samples.

```python
# Build one sample's boba-T inputs from the benchmark's preprocessed files + the cached base GRN.
import os
import scipy.io
import pandas as pd
from grn_benchmark.loaders import load_celloracle_base_grn

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
PREPROC_DIR = f"{REPO}/benchmarking/data/preprocessed"
BASE_GRN_PATH = f"{REPO}/benchmarking/data/celloracle/mouse_scATAC_atlas_base_GRN.parquet"

def build_bobat_inputs(sample: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    all_genes = pd.read_csv(f"{PREPROC_DIR}/{sample}/all_genes.csv")["x"].tolist()
    var_genes = pd.read_csv(f"{PREPROC_DIR}/{sample}/var_genes.csv")["x"].tolist()   # start here; subset further if too slow
    meta = pd.read_csv(f"{PREPROC_DIR}/{sample}/meta_data.csv", index_col=0)

    # 1. candidate network: base GRN restricted to this sample's modeled genes on both ends.
    edges = load_celloracle_base_grn(BASE_GRN_PATH)
    edges = edges[edges.source.isin(var_genes) & edges.target.isin(var_genes)]
    edges[["source", "target"]].drop_duplicates().to_csv(
        f"{out_dir}/{sample}_candidate_net.csv", header=False, index=False
    )
    node_genes = sorted(set(edges.source) | set(edges.target))

    # 2. expression: log_data.mtx is genes x cells -- transpose to boba-T's CellID-rows convention,
    #    restricted to genes that actually made it into the candidate network.
    X = scipy.io.mmread(f"{PREPROC_DIR}/{sample}/log_data.mtx").T.tocsr()
    expr = pd.DataFrame.sparse.from_spmatrix(X, index=meta.index, columns=all_genes)
    expr = expr.loc[:, expr.columns.isin(node_genes)]
    expr.index.name = "CellID"
    expr.to_csv(f"{out_dir}/{sample}_expr.csv")

    # 3. clusters: reuse the benchmark's own Louvain clusters as boba-T's phenotype/GRN-unit column,
    #    so both methods use the same clustering, not just the same base GRN.
    clusters = meta[["cluster"]].rename(columns={"cluster": "class"})
    clusters.index.name = "CellID"
    clusters.to_csv(f"{out_dir}/{sample}_clusters.csv")

    return node_genes
```

### Running boba-T per tissue

This mirrors `main_all_data_remove_selfloops_6667 copy.py` almost line for line, with `fit_rules=True` (there's no pre-fit network for this data) and paths pointing at the files just built. Output lands under `network-inference-DIRECT-NET/tm_<sample>/`, following the same `<brcd>/rules/`, `<brcd>/data_split/`, `<brcd>/validation/` layout run `6667` uses, so `load_bobat(run=f"tm_{sample}")` and the rest of `grn_benchmark`'s loaders work against it completely unmodified.

```python
# Run in bobaT_env (this part doesn't need CellOracle).
import os
import bobaT as bb

REPO = "/Users/xpz5km/Documents/GitHub/multiome-analysis"
DN = f"{REPO}/network-inference-DIRECT-NET"
INPUT_DIR = f"{REPO}/benchmarking/data/bobat_mouse_inputs"   # from build_bobat_inputs() above

def run_bobat_on_sample(sample: str):
    brcd = f"tm_{sample}"
    os.makedirs(f"{DN}/{brcd}", exist_ok=True)

    graph, vertex_dict = bb.load.load_network(
        f"{INPUT_DIR}/{sample}_candidate_net.csv",
        remove_sinks=False, remove_selfloops=True, remove_sources=False,
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    data_t0 = bb.load.load_data(
        f"{INPUT_DIR}/{sample}_expr.csv", nodes,
        norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )
    clusters = bb.utils.get_clusters(
        data_t0, cellID_table=f"{INPUT_DIR}/{sample}_clusters.csv", cluster_header_list=["class"],
    )

    os.makedirs(f"{DN}/{brcd}/data_split", exist_ok=True)
    (data_train_t0, data_test_t0, _, _, clusters_train, clusters_test) = bb.utils.split_train_test(
        data_t0, None, clusters, f"{DN}/{brcd}/data_split", suffix=sample,
    )

    rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
        data=data_train_t0, vertex_dict=vertex_dict, plot=False, threshold=0,
    )
    os.makedirs(f"{DN}/{brcd}/rules", exist_ok=True)
    bb.tl.save_rules(rules, regulators_dict, fname=f"{DN}/{brcd}/rules/rules_{brcd}.txt")
    signed_strengths.to_csv(f"{DN}/{brcd}/rules/signed_strengths.csv")

    VAL_DIR = f"{DN}/{brcd}/validation"
    os.makedirs(VAL_DIR, exist_ok=True)
    bb.tl.fit_validation(
        data_test_t0, data_test_t1=None, nodes=nodes, regulators_dict=regulators_dict, rules=rules,
        save=True, save_dir=VAL_DIR, plot=False, save_df=True, fname=sample,
    )
    return nodes   # needed as genes_used for the BEELINE/S2 scoring below

# Start with one sample -- see the scale-decision note above before looping over all 13.
nodes = run_bobat_on_sample("Heart-10X_P7_4")
```

### Scoring boba-T into a Fig.-S2-style figure

Once at least one sample has boba-T rules, `co_reproduction.co_reproduce_metrics` — already validated to reproduce CellOracle's own published numbers to 3 decimals — scores boba-T's edges exactly the same way, with no new scoring code needed:

```python
from grn_benchmark.co_reproduction import load_chip_atlas_gt, load_all_tfs, co_reproduce_metrics, run_co_reproduction
from grn_benchmark.loaders import load_bobat

def score_bobat(sample: str, tissue: str, node_genes: list) -> dict:
    bobat_edges = load_bobat(run=f"tm_{sample}")
    gt = load_chip_atlas_gt(tissue)
    all_tfs = load_all_tfs()
    return co_reproduce_metrics(bobat_edges, gt, node_genes, all_tfs)

# Combine with the released methods already scored by run_co_reproduction() for this sample.
sample, tissue = "Heart-10X_P7_4", "Heart"
scores = run_co_reproduction(sample).reset_index().rename(columns={"index": "method"})
scores.loc[len(scores)] = {"method": "boba-T", **score_bobat(sample, tissue, nodes)}
print(scores[["method", "auroc", "epr"]])
```

Looped across samples, this builds exactly the table `run_co_reproduction_all` already produces for the released methods, with boba-T added as one more row per sample — which is what a Fig.-S2-style boxplot needs:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from grn_benchmark.config import CFG

SAMPLES = {  # sample -> tissue; extend as more boba-T runs finish
    "Heart-10X_P7_4": "Heart",
}

all_scores = []
for sample, tissue in SAMPLES.items():
    node_genes = ...  # nodes returned by run_bobat_on_sample(sample), or reload from the saved network
    df = run_co_reproduction(sample).reset_index().rename(columns={"index": "method"})
    df.loc[len(df)] = {"method": "boba-T", **score_bobat(sample, tissue, node_genes)}
    df["sample"], df["tissue"] = sample, tissue
    all_scores.append(df)
all_scores = pd.concat(all_scores, ignore_index=True)
all_scores.to_csv(f"{CFG.out_dir}/s2_style_scores.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=all_scores, x="method", y="auroc", ax=axes[0])
axes[0].set_title("AUROC (Fig. S2a-style)")
sns.boxplot(data=all_scores, x="method", y="epr", ax=axes[1])
axes[1].set_title("EPR (Fig. S2b-style)")
for ax in axes:
    ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{CFG.out_dir}/s2_style_boxplots.pdf")
```

With only Heart run so far this is one point per method, not yet a boxplot — the figure only becomes meaningful once several tissues/samples have a boba-T run, the same way the original Fig. S2 pools multiple replicates per tissue.

## Running boba-T on BEELINE's synthetic GSD dataset

Nobody has done this yet; it's new work, independent of the mouse-benchmark plan above, not a prerequisite for it. The appeal: GSD's ground truth (`data/beeline/GSD/GroundTruthNetwork.csv`, 10 genes, 86 signed edges — the gonadal-sex-determination model from the BEELINE paper) is a *literal* Boolean network, generated by BoolODE, with single-cell data simulated directly from it. That's a closer match to boba-T's own modeling assumptions than either the mouse benchmark (ChIP-Atlas — inferred, not exact) or CellOracle's linear-regression paradigm, and there's no "which base GRN" decision to make the way there was for the mouse benchmark: the ground-truth network itself is the obvious candidate network to hand boba-T, since it's exactly what generated the data.

What's missing, concretely:

1. **The simulated expression matrix isn't bundled.** Only `GroundTruthNetwork.csv` is tracked in this repo (per [`data/README.md`](data/README.md)'s "small, tracked ground truth" policy) — the actual BoolODE-simulated single-cell data (`ExpressionData.csv`, and probably a `PseudoTime.csv`) lives in BEELINE's data release on Zenodo, not the GitHub repo: [`BEELINE-data.zip`](https://zenodo.org/doi/10.5281/zenodo.3378975) (262MB, DOI 10.5281/zenodo.3378975). Per BEELINE's documented input layout, this should extract to a `GSD/ExpressionData.csv` (genes x cells) alongside a matching ground-truth/reference-network file — worth confirming against the actual zip contents once downloaded, since that wasn't verified here. Alternative if the zip's layout doesn't match: regenerate it directly from the ground-truth model with [BoolODE](https://github.com/Murali-group/BoolODE) itself, which takes a Boolean model + rules and simulates single-cell trajectories — more setup, but guarantees the exact data/network correspondence.
2. **Convert to boba-T's input shape.** Same conversion this repo already does for the mouse benchmark (see [Preprocessing](#preprocessing-from-the-benchmarks-format-to-boba-ts)): `ExpressionData.csv` (likely genes x cells, BEELINE convention) needs transposing to boba-T's CellID-rows convention, and `GroundTruthNetwork.csv`'s `Gene1,Gene2,Type` needs reshaping to a plain 2-column `(source,target)` CSV for `bb.load.load_network`. GSD has no explicit cluster/phenotype labels the way the mouse benchmark's Louvain clusters or `6667`'s NE/non-NE classes do — check whether `PseudoTime.csv` (if present) should become boba-T's `class` column, or whether a single global class works for a 10-gene network this small.
3. **Run boba-T, same shape as every other run in this document**: `bb.load.load_network` on the 2-column ground-truth edges, `bb.load.load_data` + `bb.utils.split_train_test` on the converted expression matrix, `bb.tl.get_rules` + `bb.tl.fit_validation`, writing to e.g. `network-inference-DIRECT-NET/gsd/` so `load_bobat(run="gsd")` works unmodified — this part is a smaller version of the [mouse per-tissue run](#running-boba-t-per-tissue) above (10 genes vs. hundreds, so the scale concerns flagged there don't apply here).
4. **Score it against the real thing.** Since `GroundTruthNetwork.csv` is exact (not ChIP-Atlas-inferred), this is the one dataset in this document where comparison 2's AUROC/EPR (`run_beeline_comparison`) is scoring against unambiguous ground truth rather than a proxy — and comparison 3 (predicted vs. actual) is unusually well-posed too, since BoolODE's simulated states are themselves generated by discrete Boolean updates, the same object boba-T's fitted rules are trying to recover. Both `run_beeline_comparison({"boba-T": load_bobat(run="gsd")}, gt, nodes)` and a `get_sklearn_metrics`-based predicted-vs-actual run (comparison 3's pattern, but scored against boba-T's own validation output — no CellOracle fit needed unless CellOracle is also run on GSD for the same comparison) slot in with no new scoring code.

## Repo layout

```
benchmarking/
├── cell-oracle-benchmark.py                        # thin CLI: `from grn_benchmark.runners import main`
├── preprocess_benchmark_data.py                     # CLI: reproduce CellOracle's scRNA preprocessing
├── comparison3_fit_celloracle_6667.py               # comparison 3, fitting step (celloracle_env)
├── comparison3_score_celloracle_vs_bobat_6667.py    # comparison 3, scoring step (bobaT_env)
├── comparison1_structure_6667_vs_chipseq_gt.py      # comparison 1, real ChIP-seq ground truth (either env)
├── setup_celloracle_env.sh                          # builds the celloracle_env conda env (Apple Silicon)
├── grn_benchmark/                                   # the package (split from one file 2026-07-17)
│   ├── config.py                 # paths + BenchmarkConfig (CFG)
│   ├── edges.py                  # canonical edge schema + graph helpers
│   ├── loaders.py                # per-method + ground-truth loaders, METHOD_LOADERS registry
│   ├── metrics.py                # comparisons 1, 2, 4: structure / BEELINE / perturbation (stub)
│   ├── co_reproduction.py        # reproduces CellOracle's own Fig-S2 AUROC/EPR scoring (comparison 2)
│   ├── preprocess.py             # CellOracle-standard scRNA preprocessing recipe
│   └── runners.py                # run_benchmarks / self-test / main orchestration
├── benchmarking_out/                                # metrics tables + plots land here (gitignored contents)
│   ├── comparison1_structure_6667.csv
│   ├── comparison1_structure_6667_vs_borromeo2016_ascl1_human.csv
│   ├── comparison1_structure_6667_vs_borromeo2016_ascl1_mouse.csv
│   ├── comparison1_structure_6667_vs_pozo2021_ascl1_direct.csv
│   └── comparison3_celloracle_vs_bobat_6667.csv
└── data/                                            # inputs + ground truth; see data/README.md for provenance
    └── sclc_chipseq_gt/                              # real SCLC ChIP-seq ground truth (see below); .xlsx gitignored, derived .csv tracked
```

Comparison 3's own scoring output also lands next to boba-T's rules: `network-inference-DIRECT-NET/6667/rules/celloracle_coef_matrix.csv` and `network-inference-DIRECT-NET/6667/validation/celloracle_validation/accuracy_plots/*.csv`.

## Quickstart

Run everything in the `celloracle_env` conda env (Python 3.10; built by `setup_celloracle_env.sh` — see gotchas below). The core benchmark itself only needs pandas/numpy/networkx/scikit-learn, so it also runs fine in `bobaT_env`.

```bash
cd benchmarking
/opt/anaconda3/envs/celloracle_env/bin/python cell-oracle-benchmark.py
```

This runs `grn_benchmark.runners.main()`, which currently does two things:

1. **Self-test** (`run_beeline_selftest`) — validates the harness on the bundled BEELINE GSD ground truth with two synthetic baselines (`perfect` = the ground truth itself → AUROC 1.0; `random` → AUROC ≈ 0.5, EPR ≈ 1), so the metrics are demonstrably sane before any real method is scored.
2. **CellOracle Fig-S2 reproduction** (`run_co_reproduction_all`, only if `data/celloracle/inference_results/` is present) — scores CellOracle's own released method outputs against ChIP-Atlas ground truth across all 13 Tabula Muris samples, confirming the AUROC/EPR implementation matches the paper to 3 decimals.

For interactive use:

```python
from grn_benchmark import run_beeline_selftest, run_co_reproduction_all, run_methods_vs_reference

run_beeline_selftest()                 # sanity-check the metrics
run_co_reproduction_all()              # reproduce CellOracle's own Fig-S2 numbers
run_methods_vs_reference("beeline")    # load every registered method, score vs. a reference
```

`run_methods_vs_reference` calls `loaders.load_all_methods()`, which loads every method in `METHOD_LOADERS` and **skips (with a printed message) any whose loader still raises `NotImplementedError`** — currently CellOracle (the generic `Links`-object loader; `6667`'s CellOracle network is instead loaded directly from `celloracle_coef_matrix.csv` via `matrix_to_edges`, see comparison 1), SCENIC, and WGCNA. `load_bobat` works out of the box against `network-inference-DIRECT-NET/<run>/rules/signed_strengths.csv` (`CFG.bobat_run`, default `"6667"`).

## Data

Large inputs (Tabula Muris scRNA, Cusanovich scATAC, CellOracle's released `inference_results`) are gitignored; only the small BEELINE and ChIP-Atlas ground truth are tracked. Provenance, exact channel/sample naming, and re-download commands are in [`data/README.md`](data/README.md).

## Status

| Comparison | Status |
|---|---|
| 1. Network structure vs. reference | Implemented; run on `6667` vs. the DIRECT-NET candidate net *and* vs. three real ASCL1 ChIP-seq ground truths (2 human, 1 RPR2-mouse — mouse one has real N=25, humans are small-N); mouse-tissue-benchmark version (Fig-S2-style) still needs the boba-T mouse run |
| 2. Edge-weight recovery (BEELINE AUROC/EPR) | Implemented + validated against CellOracle's own Fig-S2 numbers; real Fig-S2-style figure with boba-T needs the boba-T mouse run |
| 3. Predicted vs. actual TF expression | Implemented and run on `6667` — real result: boba-T R² 0.832 vs. CellOracle R² 0.780 (42 shared genes); mouse-benchmark version needs a held-out split there |
| 4. In-silico perturbation | Stubbed; boba-T's per-attractor output needs an aggregation step before this is buildable |

The one prerequisite shared by everything still marked "needs the boba-T mouse run": [Running boba-T on the mouse ground-truth benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark) above.

## Environment notes

CellOracle 0.20.0 does not `pip install` cleanly on Apple Silicon out of the box (velocyto needs OpenMP, gimmemotifs is pinned to a version whose C sources fail under modern clang, etc.) — `setup_celloracle_env.sh` has the full workaround and is idempotent-ish to re-run. Base GRNs from `co.data.load_human_promoter_base_GRN()` / `load_mouse_scATAC_atlas_base_GRN()` download to `~/celloracle_data/`.
