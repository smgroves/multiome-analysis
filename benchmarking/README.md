# GRN Benchmark: boba-T vs. CellOracle (and other GRN-inference methods)

This folder benchmarks boba-T (BooleaBayes) against CellOracle and, in principle, any other GRN-inference method (SCENIC, GENIE3, WGCNA, ...). It grew out of a design discussion about how to compare boba-T's fitted Boolean rules to CellOracle's per-target regressions; that discussion's conclusions are folded into the sections below so the reasoning survives alongside the code.

Everything method-specific normalises to one **canonical edge table** (`source, target, weight, score, sign` — see [`grn_benchmark/edges.py`](grn_benchmark/edges.py)), so a new method is just a loader function registered in [`grn_benchmark/loaders.py`](grn_benchmark/loaders.py)'s `METHOD_LOADERS`, and every comparison below runs on it unchanged.

There are three datasets in play, and it matters which one a given quickstart uses:

- **Run `6667`** — your actual SCLC/AA network (53 genes, 228 DIRECT-NET candidate edges), already fit and validated in `network-inference-DIRECT-NET/6667/`. No independent ground truth exists for these genes, so comparisons on this data are boba-T vs. CellOracle *directly*, or vs. the DIRECT-NET candidate network as a stand-in reference.
- **The mouse ground-truth benchmark** — Tabula Muris scRNA + Cusanovich scATAC + ChIP-Atlas ground truth, the data CellOracle's own paper used for Fig. S2, already downloaded (see [`data/README.md`](data/README.md)). This is the one with a real, independent gold standard, so it's the one that can produce an actual Fig.-S2-style AUROC/EPR figure — but boba-T has never been run on it, and doing so needs a real decision about where its candidate network comes from, since there's no paired multiome ATAC+RNA here the way DIRECT-NET expects. That decision, and the full pipeline to get there, is in [Running boba-T on the mouse ground-truth benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark).
- **BEELINE's synthetic GSD dataset** (bundled at `data/beeline/GSD/`) — a third, independent dataset, unrelated to CellOracle or your SCLC data. It's synthetic single-cell data simulated from a *literally known* Boolean network (via BoolODE, the BEELINE paper's simulator), so its ground truth is exact rather than ChIP-Atlas-inferred. Right now it's only used to sanity-check the harness itself (`run_beeline_selftest`, with synthetic `perfect`/`random` baselines, no real method) — it isn't wired to CellOracle or boba-T. It's flagged here because, in principle, it's arguably a *better*-suited ground truth for boba-T than the mouse benchmark: it's Boolean-model-generated data being scored against a Boolean network, which is boba-T's own modeling paradigm, not CellOracle's linear-regression one. Nobody has run boba-T on it; that would be new work, parallel to (not a prerequisite for) the mouse-benchmark plan below — full plan in [Running boba-T on BEELINE's synthetic GSD dataset](#running-boba-t-on-beelines-synthetic-gsd-dataset).

## What boba-T and CellOracle are actually trying to do — and why that changes how to read every ChIP-seq-scored number below

boba-T (BooleaBayes) and CellOracle are not trying to solve the same problem, and comparison 1 (network structure vs. a ChIP-seq/ChIP-Atlas reference) and the Fig.-S2-style AUROC/EPR reproduction both score them as if they were.

boba-T's whole design starts from a deliberately small, curated candidate network (here, DIRECT-NET+LASSO's 53-gene, 228-edge panel for `6667`) — genes chosen because they're believed to drive a specific, coarse-grained phenotypic transition (NE ↔ non-NE state switching, in this project), then fits Boolean-ish rules whose job is to reproduce and predict *that* dynamical behavior — attractor states, basins, perturbation responses — not to comprehensively enumerate every real TF→target binding event in the genome. Success for boba-T looks like comparison 3 (predicted vs. actual expression, on its own curated genes): does the fitted rule set correctly predict expression state for the genes it was built to model.

CellOracle (and, as run here, SCENIC/GENIE3 "from scratch") is explicitly a broad, genome-scale GRN-inference method — its own validation paradigm, the Fig. S2 benchmark this harness reproduces, literally *is* "how many ChIP-Atlas/ChIP-seq-confirmed edges did you recover, across as much of the genome as your candidate set allows." ChIP-seq edge recovery isn't just *a* metric CellOracle happens to do reasonably on — it's close to the metric CellOracle was built and evaluated against in its own paper.

So scoring both against a ChIP-seq ground truth's full edge list structurally favors whichever method is trying to maximize genome-wide edge recovery — which is CellOracle/SCENIC by design, not boba-T. The small shared-node counts for boba-T seen throughout this document (5 on the human GTs, 25 on the RPR2-mouse GT, vs. CellOracle/SCENIC's 41–597 on the same TKO/`6667` cells) aren't evidence boba-T "only recovers a sliver" of the real regulatory network — boba-T was never trying to model most of those genes at all; its candidate network was deliberately restricted to the curated phenotype-driving panel before any fitting happened. Scoring a 53-gene, phenotype-focused dynamical model's recall against a 3,992-edge genome-wide ChIP-seq list and reading a low number as "worse" is close to a category error.

This doesn't make comparison 1 or the Fig.-S2 reproduction useless — "of the edges each method commits to, how many are independently real" is still a meaningful precision/recall question, and it's the reason boba-T's DIRECT-NET-restricted numbers (e.g. F1 0.163–0.500 on the mouse RPR2 GT) hold up reasonably well against CellOracle's ATAC-restricted variants scored the same way — both are working from a similarly curated candidate set there, so the comparison is closer to fair. It just means: whenever a from-scratch/genome-scale CellOracle or SCENIC run is shown recovering more absolute ChIP-seq edges, more of the ground truth, or a higher raw n_overlap than boba-T on the *same* ground truth, that's expected by construction — those methods were pointed at genome-wide recovery and boba-T was not — and it says nothing about whether boba-T's actual job (predicting the phenotype-relevant dynamics of its own curated genes) is being done well or poorly. The one comparison in this document that actually tests boba-T on its own terms is [comparison 3, predicted vs. actual expression](#3-predicted-vs-actual-expression--implemented-and-run-on-6667) — read that, not the ChIP-seq structure numbers, as the "is boba-T doing its job" answer.

## The comparison points

### 1. Network structure vs. a reference — **implemented**

Do two methods (or a method and a reference) agree on which edges/hubs exist? `metrics.structure_metrics` restricts both networks to their shared node universe, then reports edge Jaccard / precision / recall / F1, degree-profile Spearman correlation, and sign concordance (activator vs. repressor) on edges present in both. `metrics.pairwise_jaccard` gives the same edge-overlap number method-vs-method with no reference required.

Note on scope: this is edge/degree agreement, not the network-science centrality comparison (betweenness, eigenvector centrality, scale-free-ness) that CellOracle's `Links.get_network_score()` / `plot_degree_distributions()` produce internally. Those are available if a deeper topology comparison is wanted later, but the harness doesn't need to reimplement them — CellOracle computes them on its own `Links` object.

**Read this comparison with [the caveat above](#what-boba-t-and-celloracle-are-actually-trying-to-do--and-why-that-changes-how-to-read-every-chip-seq-scored-number-below) in mind whenever the reference is a ChIP-seq/ChIP-Atlas ground truth**: it tells you "of the edges each method commits to, how many are independently real," which is fair when both methods start from a similarly curated candidate set, but structurally favors whichever method is trying to maximize genome-wide edge recovery when they don't.

**Roadmap:**

1. **Done** — boba-T (`6667`) vs. the DIRECT-NET candidate network it was fit on.
2. **Done** — boba-T (`6667`) vs. CellOracle (`6667`), both scored against that same candidate network as the reference, using the coefficient matrix comparison 3 already fit (below).
3. **Done** — `6667` networks at two scales vs. three real, independent ASCL1 ChIP-seq ground truths (two human, one RPR2-mouse), scored separately. DIRECT-NET-restricted (boba-T, CellOracle, GENIE3), all on the same 53-gene candidate set, recover real edges — the mouse ground truth (25 shared nodes) is the most statistically meaningful of the three. A first "from scratch" attempt (CellOracle/SCENIC with no DIRECT-NET edge-list restriction, but still on boba-T's 53-gene export) recovered *zero* true edges on all three — caught in review as still gene-universe-restricted, not truly independent. Re-run at real genome scale (2,999 HVGs from the actual Box multiome data behind `6667`, no boba-T export or network involved at all): both recover real edges on all three ground truths, ASCL1 emerges as a genuine hub in both (110 and 118 outgoing edges respectively), and SCENIC's motif pruning gives it meaningfully better precision than CellOracle's ridge-only fit — see [Sourcing a real SCLC ground truth](#sourcing-a-real-sclc-ground-truth-done-with-a-correction).
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

**This entire comparison inherits its yardstick directly from CellOracle's own paper** — see [the caveat above](#what-boba-t-and-celloracle-are-actually-trying-to-do--and-why-that-changes-how-to-read-every-chip-seq-scored-number-below) before treating a future boba-T-vs-CellOracle Fig.-S2-style number as a verdict: AUROC/EPR against a genome-wide ChIP-Atlas ground truth is close to the exact metric CellOracle was designed and evaluated to do well on, not a neutral third-party test either method was aiming for equally.

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

**Original result**, 42 genes with surviving regulators in all three networks (of 53 total in the `6667` network):

| metric | boba-T | CellOracle | GENIE3 |
|---|---|---|---|
| R² | 0.832 | 0.780 | **0.899** |
| ROC-AUC | 0.968 | 0.965 | **0.978** |
| F1 | 0.913 | 0.909 | **0.941** |

**Correction — this table understated CellOracle specifically, and has been superseded by
the fair re-score below.** Found while re-verifying a similar CellOracle comparison built for
the HSC combinatorial-logic work: `comparison3_fit_celloracle_6667.py` applies the
train-fitted `coef_matrix` to `test_t0combined.csv` — the **raw**, un-imputed test values —
but CellOracle's own `fit_GRN_for_simulation` actually fits its Ridge models on
`oracle.adata.layers["imputed_count"]` (the KNN-smoothed output of `knn_imputation()`), not on
the raw values it was given. Evaluating on a different representation than the model was
trained for understates its held-out accuracy. (A second, separate bug — a missing Ridge
intercept — was also found and fixed in the HSC work, but does *not* affect this specific
script: `to01()`'s min-max rescaling here happens to be invariant to a missing additive
constant, so that part of the original 0.780 figure was always safe.)

**Fair re-score** (`redo_comparison3_6667_fair.py`): rather than picking whichever
representation happens to favor one method, all three methods are refit/re-evaluated against
a single, identical target — the real `imputed_count` from one shared `Oracle` object built on
train+test combined. boba-T's already-fitted rule and CellOracle's coefficients (now with the
intercept included) are simply re-evaluated against it; GENIE3 is refit fresh on the same
imputed training data, since it had never previously been evaluated against this
representation either. Restricted to the identical 42-gene "shared candidates in all three
methods" set the original comparison used — confirmed by getting exactly 42 genes again,
independently, not by construction:

| metric | boba-T | CellOracle | GENIE3 |
|---|---|---|---|
| R² | 0.858 | **0.864** | **0.963** |
| ROC-AUC | 0.975 | 0.973 | **0.992** |
| F1 | 0.918 | 0.922 | **0.974** |

**boba-T and CellOracle are now essentially tied on R² (0.858 vs. 0.864 — CellOracle a hair
ahead) — the opposite of the originally reported 0.832 vs. 0.780 "boba-T wins" margin.**
AUC/F1 stay close to their original values for both (the representation mismatch mattered far
more for R², which is sensitive to exact calibrated values, than for the 0.5-thresholded
classification metrics). GENIE3 remains the clear overall winner and its lead over both other
methods actually *widens* under fair scoring (R² 0.899→0.963) — the same "predictive
flexibility, not structural correctness" caveat from below still applies to it. `alpha=10` for
CellOracle and `n_estimators=1000` for GENIE3 were both left at defaults, not tuned — a fairer
three-way race would still sweep both.

**This tie is arguably the single most flattering result for boba-T in this whole document, and it's worth being precise about *why*, rather than just noting the R² numbers match.** It is not that boba-T uses dramatically fewer regulators per gene than a *DIRECT-NET-restricted* CellOracle — checked directly, not assumed: boba-T's fitted rules use a mean of 4.26 regulators/gene (range 1–8) across these same 42 genes, vs. CellOracle's ridge model drawing on a mean of 5.43 candidate regulators/gene (range 2–8) from the identical DIRECT-NET network — comparable, not "far fewer." The real distinction *at this restricted scale* is **model class**, not parameter count: boba-T's rule is a constrained, interpretable Boolean/threshold-style function, purpose-built to reproduce discrete phenotype-state dynamics (see [the section above on what boba-T and CellOracle are actually trying to do](#what-boba-t-and-celloracle-are-actually-trying-to-do--and-why-that-changes-how-to-read-every-chip-seq-scored-number-below)), while CellOracle's is a dense, unconstrained continuous ridge regression with one free coefficient per candidate regulator. Matching CellOracle's accuracy with that more constrained functional form — on a task CellOracle's model class has no particular structural disadvantage at — is a genuinely meaningful result: the coarser, dynamically-interpretable Boolean rule isn't leaving real predictive accuracy on the table relative to an unconstrained linear fit *when both are given the same small, curated candidate set*. It doesn't extend to GENIE3's win, whose random-forest regressors are a strictly more flexible model class than either — that gap (R² 0.963) is nonlinear/flexible vs. constrained, not evidence against boba-T specifically.

**But that "comparable regulator count" is itself an artifact of restricting CellOracle to DIRECT-NET's network — not how CellOracle would actually be run.** In practice, CellOracle builds its own base GRN from a dataset's real ATAC peaks and lets the ridge fit find structure across everything that base GRN allows, the way [`comparison_tko_fit_celloracle_atac.py`](comparison_tko_fit_celloracle_atac.py) already does for TKO's genome-scale HVG panel above. Since TKO_final_arc's scRNA-seq is the same underlying cells behind `6667` ([see the TKO section](#tko_final_arc-a-real-dataset-specific-atac-informed-base-grn-not-a-promoter-scan)), that same real ATAC-informed base GRN can be applied directly to `6667`'s own 53-gene train/test split instead of DIRECT-NET's candidate set — [`redo_comparison3_6667_fair_with_realatac.py`](redo_comparison3_6667_fair_with_realatac.py) does exactly this, reusing the identical fair-scoring `imputed_count` target from above. Restricting the real ATAC base GRN to `6667`'s 53 genes (47/53 TFs, 52/53 targets matched — only `RORA_RORB` missing, expected) gives a mean of **30.15 candidate regulators/gene** (range 10–46) *before* any sparsity-inducing selection — about 7× denser than DIRECT-NET's LASSO-pruned 5.43/gene, and about 7× denser than boba-T's own fitted 4.26/gene:

| method | n_genes | regulators/gene (mean) | R² | AUC | F1 |
|---|---|---|---|---|---|
| boba-T | 42 | 4.26 | 0.858 | 0.975 | 0.918 |
| CellOracle (DIRECT-NET base) | 42 | 5.43 | 0.864 | 0.973 | 0.922 |
| **CellOracle (real ATAC base, not DIRECT-NET-restricted)** | **41** | **30.15** | **0.957** | **0.993** | **0.965** |
| GENIE3 (DIRECT-NET-restricted) | 42 | 5.43 | 0.963 | 0.992 | 0.974 |

**This is the real "how would CellOracle actually be used" comparison, and it resolves the earlier tie: given its own realistic, ATAC-derived candidate breadth instead of DIRECT-NET's curated 53-gene starting point, CellOracle's R² jumps from 0.864 to 0.957 — nearly closing the entire gap to GENIE3's flexible random-forest fit (0.963), using the *same* ridge-regression model class as the DIRECT-NET-restricted run, just ~7× more candidate regulators per gene.** That's an important disentangling of the two things that were previously confounded in "GENIE3 wins": model flexibility (ridge vs. random forest, same small candidate set: 0.864→0.963) and candidate breadth (ridge, small vs. large candidate set: 0.864→0.957) turn out to independently explain most of the same-sized gap — breadth alone gets CellOracle nearly all the way to GENIE3's number, without changing model class at all. Read together with the parsimony paragraph above: boba-T's Boolean rule ties a *comparably-restricted* CellOracle, but a CellOracle actually run the way it's designed to be run — with real ATAC breadth, not an artificially small curated candidate set — clearly outpredicts boba-T's 53-gene panel, at the cost of roughly 7× more regulators per gene and a model with no coarse-grained phenotypic-dynamics interpretation the way boba-T's rules have. Neither framing is "wrong" — they answer different questions ("does boba-T's constrained rule lose accuracy on a fixed candidate set" vs. "does boba-T's whole curated-53-gene approach lose raw predictive accuracy to CellOracle used at realistic scope"), and this document now has the numbers for both.

**Limitation, stated plainly**: boba-T's small, curated candidate network means it structurally cannot compete with CellOracle run at realistic ATAC-informed breadth on raw predictive accuracy (R² 0.858 vs. 0.957) — it was never given access to that candidate-regulator breadth in the first place, so any comparison on this axis alone will favor genome/ATAC-scale methods by construction.

**The advantage that same limitation buys**: that small network is exactly what makes boba-T's fitted rules a phenotypic-dynamics model, not just a predictor. Each gene's rule is a compact, human-readable function over a handful of regulators — small enough to support attractor/basin analysis and in-silico perturbation simulation of coarse-grained phenotypic state transitions (e.g. NE↔non-NE), the actual scientific question this project cares about. A 30-regulator-per-gene ridge fit has no comparable notion of a discrete cell state, an attractor, or a basin to perturb — it predicts expression well, but doesn't give you a dynamical model of phenotype switching to interrogate. The two methods aren't competing on the same axis: CellOracle-at-scale wins on genome-wide predictive accuracy, boba-T's small network is what makes phenotypic-dynamics interpretability possible at all.

Full re-scored per-gene numbers:
`benchmarking_out/comparison3_all_methods_vs_bobat_6667_fair_imputed_target.csv` (3-method,
DIRECT-NET-restricted) and `benchmarking_out/comparison3_all_methods_vs_bobat_6667_fair_imputed_target_with_realatac.csv`
(adds the real-ATAC CellOracle row above); original per-gene numbers (superseded, kept for
reference): `benchmarking_out/comparison3_all_methods_vs_bobat_6667.csv`.

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

Scored with [`comparison1_structure_6667_vs_chipseq_gt.py`](comparison1_structure_6667_vs_chipseq_gt.py) — reuses `load_bobat` plus five methods' fitted networks, `run_structure_comparison`, and a new `loaders.load_sclc_chipseq_gt`, against each ground truth **separately**, three output files, five methods each. Two treatments are compared side by side:

- **Restricted to the DIRECT-NET candidate network** (the fairness principle used throughout this document): boba-T, `CellOracle (DIRECT-NET base)` ([`comparison3_fit_celloracle_6667.py`](comparison3_fit_celloracle_6667.py)), `GENIE3 (DIRECT-NET-restricted)` ([`comparison_genie3_fit_6667.py`](comparison_genie3_fit_6667.py)).
- **Run from the beginning, attempt 1 — no DIRECT-NET candidate-edge restriction, but still on boba-T's 53-gene export (superseded, see below)**: `CellOracle (from scratch)` ([`comparison_celloracle_fromscratch_fit_6667.py`](comparison_celloracle_fromscratch_fit_6667.py), using CellOracle's own human promoter base GRN) and `SCENIC (from scratch)` ([`comparison_scenic_fit_6667.py`](comparison_scenic_fit_6667.py), full GRNBoost2 + RcisTarget motif pruning — a dedicated `scenic_env` conda env was needed; pySCENIC 0.12.1's dependency chain doesn't run on a modern numpy/pandas/dask stack, see that script's docstring for the exact version pins that fixed each import-time crash). SCENIC's `--min_genes` was set to 1 (its default of 20 assumes thousands of candidate targets, not 53). **This turned out to still be gene-universe-restricted, not truly independent — see the correction and the real genome-scale rerun further down.**

| ground truth | species | n_shared_nodes | method | n_pred | n_overlap | precision | recall | f1 | sign_concordance |
|---|---|---|---|---|---|---|---|---|---|
| Borromeo human (620 edges) | human | 5 | boba-T | 4 | 1 | 0.250 | 0.20 | 0.222 | n/a (unsigned GT) |
| Borromeo human (620 edges) | human | 5 | CellOracle (DIRECT-NET base) | 3 | 1 | 0.333 | 0.20 | 0.250 | n/a (unsigned GT) |
| Borromeo human (620 edges) | human | 5 | GENIE3 (DIRECT-NET-restricted) | 3 | 1 | 0.333 | 0.20 | 0.250 | n/a (unsigned GT) |
| Borromeo human (620 edges) | human | 5 | **CellOracle (from scratch)** | 4 | **0** | 0.000 | 0.00 | 0.000 | n/a |
| Borromeo human (620 edges) | human | 4 | **SCENIC (from scratch)** | 1 | **0** | 0.000 | 0.00 | 0.000 | n/a |
| **Borromeo mouse, RPR2 (3,992 edges)** | **mouse** | **25** | **boba-T** | **61** | **7** | **0.115** | **0.28** | **0.163** | n/a (unsigned GT) |
| **Borromeo mouse, RPR2 (3,992 edges)** | **mouse** | **25** | **CellOracle (DIRECT-NET base)** | **55** | **7** | **0.127** | **0.28** | **0.175** | n/a (unsigned GT) |
| **Borromeo mouse, RPR2 (3,992 edges)** | **mouse** | **25** | **GENIE3 (DIRECT-NET-restricted)** | **55** | **7** | **0.127** | **0.28** | **0.175** | n/a (unsigned GT) |
| Borromeo mouse, RPR2 (3,992 edges) | mouse | 25 | **CellOracle (from scratch)** | 109 | **0** | 0.000 | 0.00 | 0.000 | n/a |
| Borromeo mouse, RPR2 (3,992 edges) | mouse | 25 | **SCENIC (from scratch)** | 99 | **0** | 0.000 | 0.00 | 0.000 | n/a |
| Pozo (295 edges) | human | 4 | boba-T | 4 | 2 | 0.500 | 0.50 | 0.500 | 0.0 |
| Pozo (295 edges) | human | 4 | CellOracle (DIRECT-NET base) | 4 | 2 | 0.500 | 0.50 | 0.500 | 1.0 |
| Pozo (295 edges) | human | 4 | GENIE3 (DIRECT-NET-restricted) | 4 | 2 | 0.500 | 0.50 | 0.500 | 0.5 |
| Pozo (295 edges) | human | 4 | **CellOracle (from scratch)** | 1 | **0** | 0.000 | 0.00 | 0.000 | n/a |
| Pozo (295 edges) | human | 4 | **SCENIC (from scratch)** | 3 | **0** | 0.000 | 0.00 | 0.000 | n/a |

Full tables: `benchmarking_out/comparison1_structure_6667_vs_{borromeo2016_ascl1_human,borromeo2016_ascl1_mouse,pozo2021_ascl1_direct}.csv`.

**Read the two human ground truths as a proof that the pipeline works, not a real accuracy comparison** — 4–5 shared nodes and 1–2 overlapping edges has no meaningful confidence interval either direction. The one qualitative signal worth noting without over-reading it: on the Pozo ground truth, boba-T got the *existence* of both overlapping edges right but the *sign* wrong on one of them (`sign_concordance` 0.0), CellOracle (DIRECT-NET base) got both signs right (1.0), and GENIE3 landed in between (0.5) — its importances are unsigned by construction, so its "sign" here is a post-hoc correlation-sign convention, not a modeled direction.

**The RPR2-mouse row is the one actually worth reading** — 25 shared nodes and 7 overlapping edges is a real, if still modest, sample, for the DIRECT-NET-restricted methods. boba-T, CellOracle (DIRECT-NET base), and GENIE3 (DIRECT-NET-restricted) all land close together (F1 0.163–0.175, identical n_overlap of 7, identical recall of 0.28). CellOracle and GENIE3 have identical numbers here not by coincidence of scoring but because their edge sets are **exactly identical** — verified directly, not just same counts. The reason: both exclude self-loops (13 of the 228 candidate edges are `gene → itself`), CellOracle internally as part of its own fitting code, GENIE3 by explicit choice in `comparison_genie3_fit_6667.py`; boba-T's BooleaBayes rules keep self-loops (real auto-regulation), which is most of why its predicted-edge count (61) is higher than the other two's (55) among these nodes. Beyond that shared exclusion, CellOracle's ridge and GENIE3's random forest — two structurally different fitting methods — independently zeroed out the exact same 13 candidate edges once self-loops are set aside too.

**The "from scratch" result is the most informative single finding in this section: both CellOracle (from scratch) and SCENIC (from scratch) recover *zero* true edges, on all three ground truths, despite predicting plenty of edges overall** (up to 109 among the RPR2-mouse ground truth's 25 shared nodes — more than any DIRECT-NET-restricted method). This was checked directly rather than taken at face value:

- **CellOracle (from scratch)**: `ASCL1` is a valid candidate regulator in the restricted human promoter base GRN (confirmed present in `keep_tfs`), but its fitted ridge coefficient is **exactly zero for every one of the 52 target genes** — the promoter-motif-scan signal for ASCL1 apparently isn't strong or clean enough, competing against other candidate TFs in the same ridge fit, to survive regularization for *any* target.
- **SCENIC (from scratch)**: `ASCL1` never made it into the final 36 regulons at all. But it's not absent from the pipeline — the raw GRNBoost2 step (before motif pruning) ranks `ASCL1 → PROX1` as its single strongest edge (importance 246.5, out of 52 candidate targets) — and `PROX1` is a real, ChIP+knockdown-confirmed ASCL1 target per Pozo's Table S6. That real signal got dropped at the RcisTarget motif-enrichment (`ctx`) step: enrichment testing asks whether a candidate gene set is non-randomly enriched for a motif *against a genome-wide background*, which has essentially no statistical power when the candidate set is drawn from a 53-gene universe instead of the thousands of genes SCENIC is designed around — the same "intended to use around 1000-3000 genes" scale mismatch flagged for CellOracle's own from-scratch run above, and the same mechanism discussed for [why GENIE3 underperforms in the real Fig. S2 benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark) relative to its restricted showing in comparison 3: a real regulatory signal in the raw co-expression numbers, thrown out for lack of statistical power at small scale, not necessarily because the underlying signal was wrong.

The takeaway for `6667` specifically: on a 53-gene curated network, letting CellOracle or SCENIC discover their own candidate structure — the way they'd actually be used in a real project — currently recovers none of the independently-confirmed ASCL1 ChIP-seq edges checked here, while every method that was handed the DIRECT-NET-restricted candidate set (including boba-T) recovers a real, non-random fraction. That's informative about what the DIRECT-NET restriction is doing for this comparison — supplying exactly the kind of ATAC-informed prior that a 53-gene network is too small for generic motif-scan/co-expression methods to rediscover on their own — not necessarily a statement about which method is "better" in the genome-scale regime either is actually designed for.

**Important caveat on "from scratch," caught in review**: both "from scratch" runs above removed the DIRECT-NET *candidate-edge-list* restriction, but both still ran on `train_t0combined.csv`/`test_t0combined.csv` — boba-T-specific exports already subset to its network's 53 genes. So neither run was actually given an unrestricted *gene universe*; both still only had 53 genes to work with, which plausibly explains the RcisTarget/ridge statistical-power problem described above rather than ruling it out.

### The real from-scratch run: genome-wide, no boba-T involvement at all

A genuinely independent run needs the real underlying expression data, not a boba-T export, and shouldn't be scored against boba-T's network either (that reintroduces the 53-node restriction from the other direction). Both fixed: the real source data was located at `adata_02_filtered.h5ad` in the Box multiome dataset behind `6667` (`data/converting_seurat_data.R` points to it) — 8,908 cells (the same cells as `6667`'s train+test split) × 16,719 genes, with real raw UMI counts (a `raw_counts` layer; earlier attempts had to fake this layer because only pre-normalized data was available). Gene symbols are mouse-case in the source (`Ascl1`, allograft = mouse-grown tumor); upper-cased to match this benchmark's human-convention naming, same as the RPR2-mouse ChIP-seq ground truth already uses.

[`preprocess_sclc_full_6667.py`](preprocess_sclc_full_6667.py) reproduces CellOracle's own preprocessing recipe (the same one already used for the mouse Tabula Muris benchmark) to select ~3,000 HVGs, with **no manual inclusion of ASCL1 or any network gene** — whether they survive the cut is itself part of what's being checked. Result: 18 of `6667`'s 53 genes made the HVG cut (including ASCL1 itself), 34 are present in the data but not selected as HVGs, and `RORA_RORB` (the combined pseudo-node) isn't a real gene so isn't present at all.

[`comparison_celloracle_fullscale_fit_6667.py`](comparison_celloracle_fullscale_fit_6667.py) fits CellOracle on this HVG-scale data using its actual tutorial-standard entry point, `import_anndata_as_raw_count` (real raw counts this time, not a faked layer), with its own human promoter base GRN restricted to the 2,999 HVGs (125 candidate TFs matched). Scored directly against the three ChIP-seq ground truths — **no boba-T network involved in this comparison at all**, so the node universe is genuinely this run's own (HVGs ∩ each ground truth's genes), not capped at 53:

| ground truth | n_shared_nodes | n_pred_edges | n_overlap | precision | recall | f1 | sign_concordance |
|---|---|---|---|---|---|---|---|
| Borromeo human (620 edges) | 90 | 162 | 2 | 0.0123 | 0.0222 | 0.0159 | n/a (unsigned GT) |
| Borromeo mouse, RPR2 (3,992 edges) | 564 | 8,846 | 29 | 0.0033 | 0.0514 | 0.0062 | n/a (unsigned GT) |
| Pozo (295 edges) | 49 | 32 | 2 | 0.0625 | 0.0408 | 0.0494 | **1.0** |

**This resolves the open question from the 53-gene run**: at real scale, CellOracle-from-scratch is *not* stuck at zero. ASCL1 itself has 110 nonzero outgoing edges in the fitted `coef_matrix` (vs. exactly zero in the 53-gene run) — genuinely regulatory-shaped, not degenerate — and directly overlaps real confirmed targets: 29 of the RPR2-mouse ground truth's genes, plus `HIVEP3`, `THBS1`, `TBC1D1` among others; 2 apiece on the two human ground truths, including `HEPACAM2` shared between the mouse and Pozo lists. Where CellOracle *does* find a true edge, it gets the direction right (`sign_concordance` 1.0 on the two Pozo overlaps). What doesn't hold up at this scale is precision: with 8,846 predicted edges among 564 shared nodes, only 29 are confirmed true — 0.33% precision, because the total candidate-pair space grows enormously faster than the (fixed) number of real ASCL1 targets. That's the same mechanism discussed above for GENIE3's real-world Fig. S2 numbers: genome-scale unrestricted inference recovers real signal, but at much lower precision than a curated, ATAC-informed candidate set gets for free.

SCENIC's genome-scale run ([`comparison_scenic_fullscale_fit_6667.py`](comparison_scenic_fullscale_fit_6667.py)) used the same 2,999 HVGs and same raw counts, with its TF candidates restricted to the 209 HVGs that are also in the standard human TF reference list (`allTFs_hg38.txt` — a different, larger TF universe than CellOracle's 125 promoter-base-GRN TF columns, since SCENIC doesn't tie candidacy to any base-GRN prior). GRNBoost2 found 272,315 raw candidate edges across all TFs (up from 2,600 at 53-gene scale); RcisTarget motif pruning (the default `--min_genes 20` this time, not the `1` override the 53-gene run needed) cut that to 42 final regulons, 2,489 edges. **ASCL1 gets a real regulon this time** — 3 significantly enriched motifs, 118 target genes, including `RBPJ`, `ZBTB20`, and `LMX1B`, all three of which are also `6667` network nodes.

| ground truth | n_shared_nodes | n_pred_edges | n_overlap | precision | recall | f1 | sign_concordance |
|---|---|---|---|---|---|---|---|
| Borromeo human (620 edges) | 44 | 16 | **7** | **0.4375** | 0.1591 | **0.2333** | n/a (unsigned GT) |
| Borromeo mouse, RPR2 (3,992 edges) | 315 | 275 | **36** | 0.1309 | 0.1143 | **0.1220** | n/a (unsigned GT) |
| Pozo (295 edges) | 30 | 3 | 1 | 0.3333 | 0.0333 | 0.0606 | n/a (unsigned GT) |

**SCENIC's genome-scale precision is much higher than CellOracle's genome-scale precision on every ground truth** (0.44 vs. 0.01 human, 0.13 vs. 0.003 mouse, 0.33 vs. 0.06 Pozo), despite CellOracle predicting far more total edges. The likely reason: RcisTarget's motif-enrichment step is a real second filter beyond co-expression — it throws away GRNBoost2 edges that aren't independently supported by sequence motif enrichment, shrinking the final edge set but concentrating it on plausible direct-binding relationships; CellOracle's ridge regression only applies L2 shrinkage to the same base-GRN candidates, with no comparable sequence-based check, so it keeps far more (weaker) edges and dilutes precision. Both, though, land nowhere near boba-T/CellOracle/GENIE3's DIRECT-NET-restricted numbers from earlier (F1 0.16–0.5 there, on a node universe capped at 53) — genome-scale inference is a harder, lower-precision regime by construction, for both methods, regardless of which one handles it better.

**Overall verdict on "run from the beginning" for `6667`, now actually checked at real scale**: neither CellOracle nor SCENIC is fundamentally incapable of recovering ASCL1's real regulatory targets — the earlier zero-overlap result was a genuine artifact of the 53-gene universe, not a property of either method. At real scale, both recover real signal on all three independent ChIP-seq ground truths, ASCL1 emerges as a real regulatory hub in both, and SCENIC's motif-pruning step gives it a real precision advantage over CellOracle's ridge-only approach in this specific from-scratch setting. What still hasn't been tested: whether boba-T, given the same real genome-scale data (rather than its own 53-gene DIRECT-NET export), would show the same pattern — that's a different, not-yet-attempted experiment, since boba-T's candidate network is fundamentally tied to DIRECT-NET's peak-to-gene output rather than an HVG selection.

Not yet tried: feeding any of these ground truths into `run_beeline_comparison` (comparison 2) instead of `run_structure_comparison` — all three are real independent references, so AUROC/EPR would be as legitimate here as the structure numbers above (and the mouse one, with 25 shared nodes, has enough of a candidate universe to make AUROC less noisy than on the two human ones).

### TKO_final_arc: a real, dataset-specific ATAC-informed base GRN (not a promoter scan)

Every CellOracle base GRN used above — the DIRECT-NET-restricted one and the "from scratch" human-promoter-scan one — is either tied to a candidate network boba-T also uses, or built from a generic genome-wide promoter motif scan, not this dataset's own chromatin accessibility. `TKO_final_arc.rds` (a real mouse RPR2 — *Trp53;Rb1;Rbl2* triple-knockout, i.e. genotype-matched to the Borromeo mouse ground truth — multiome object, with its own `peaks` ChromatinAssay) makes a genuinely ATAC-informed base GRN possible: build it from this dataset's own peaks, not a reference-genome promoter scan.

[`extract_tko_atac.R`](../network-inference-DIRECT-NET/extract_tko_atac.R) pulls the `peaks` assay (mm10, 158,859 peaks) and assigns each peak to its nearest gene within 10kb (Signac's `ClosestFeature` — the simpler documented alternative to Cicero co-accessibility, not attempted here): 115,027 peaks within 10kb of a gene, 19,455 unique genes. [`comparison_tko_atac_base_grn.py`](comparison_tko_atac_base_grn.py) motif-scans those peaks against the real mm10 genome (`celloracle.motif_analysis.TFinfo`, gimmemotifs, `fpr=0.02`) → a 115,027-peak × 1,093-TF × 19,455-target-gene base GRN, CellOracle's own convention, genuinely ATAC-derived this time. [`preprocess_tko_rna.py`](preprocess_tko_rna.py) applies the same CellOracle-paper HVG recipe used for `6667`'s genome-scale run (`filter_genes` → `normalize_per_cell` → `filter_genes_dispersion` cell_ranger top 3,000 → `log1p`) to TKO's own RNA: 8,779 cells × 32,285 genes → 3,000 HVGs, ASCL1 included.

**Two real problems hit and fixed while building this, worth recording**:
- The gimmemotifs scan takes about an hour and was twice lost mid-run when this session's background shell was killed by an unrelated Claude Code restart — the process was reparented to `launchd` (`nohup` + backgrounding without a controlling shell) so a third restart can't repeat this, and a `tfi.to_hdf5(...)`/`co.motif_analysis.load_TFinfo(...)` checkpoint was added right after the scan completes (before the slower `to_dataframe()` aggregation step) so a crash there doesn't force a full rescan again. (`to_hdf5` also silently requires the filename to literally end in `.celloracle.tfinfo` — a `ValueError` that cost one more full rescan before it was caught.)
- SCENIC's `ctx` step (RcisTarget) failed almost completely on the first attempt — a 202-byte, essentially empty output — because this repo's usual upper-cased gene-symbol convention (`ASCL1`, matching the ChIP-seq ground truths and used freely elsewhere, including on CellOracle's side of this exact run) doesn't match the mm10 rankings database's real mouse-case symbols (`Ascl1`, `0610005C13Rik`); RcisTarget does exact string matching, so every module failed the "genes map to the rankings" check. Fixed by reconstructing an exact upper-case → original-case mapping from TKO's pre-upper-cased gene list (replicating anndata's `var_names_make_unique()` suffixing so the 3,000 HVG names line up exactly) and rerunning GRNBoost2 + `ctx` with proper mouse-case symbols throughout; edges are only upper-cased again at the final scoring step, to match the ground truths' convention.

[`comparison_tko_fit_celloracle_atac.py`](comparison_tko_fit_celloracle_atac.py): base GRN restricted to the 3,000-HVG set → 133 candidate TFs, 2,159 target genes, 280,276 nonzero peak-TF entries; ridge fit (`GRN_unit="whole"`, `alpha=10`) → `coef_matrix` 3,000×3,000, 117,686 nonzero entries. [`comparison_tko_fit_scenic.py`](comparison_tko_fit_scenic.py): same TKO expression, freshly downloaded mouse cisTarget resources (`allTFs_mm.txt`, the mm10 10kbp-up/down rankings feather, `motifs-v10nr_clust-nr.mgi-*.tbl`) → GRNBoost2 (295,731 raw edges) → `ctx` → 61 regulons, 6,643 edges; ASCL1 present as a source.

Scored against the same three independent ChIP-seq ground truths ([`comparison_tko_atac_vs_chipseq_gt.py`](comparison_tko_atac_vs_chipseq_gt.py)), alongside boba-T's **existing, already-fit `6667` network** (`load_bobat`, no refitting/rerunning here) — this **is** a same-dataset comparison, and `6667` **is** boba-T's own TKO-ATAC-derived fit, not a stand-in for one: `6667`'s underlying scRNA-seq is the same TKO_final_arc cells used for the CellOracle/SCENIC fits above, and DIRECT-NET's own peak-to-gene + LASSO candidate-network construction for `6667` was run directly on TKO_final_arc's own ATAC data — just outside this project's own scripts, so it wasn't visible here until pointed out. The real difference between the three rows below is candidate-network **scope**, not dataset or ATAC source: boba-T's DIRECT-NET+LASSO candidate network is a curated 53-gene panel derived from TKO's ATAC peaks, while CellOracle/SCENIC here ran genome-scale on the same cells' ~3,000-HVG set, using an independently-built (Signac `ClosestFeature` + motif-scan) ATAC-informed candidate structure over the same underlying peaks, just not pruned down by DIRECT-NET+LASSO:

| ground truth | method | n_shared_nodes | n_pred_edges | n_overlap | precision | recall | f1 |
|---|---|---|---|---|---|---|---|
| Borromeo human (620 edges) | CellOracle (real ATAC, TKO) | 123 | 680 | 50 | 0.074 | 0.407 | 0.125 |
| Borromeo human (620 edges) | SCENIC (from scratch, TKO) | 102 | 49 | 40 | 0.816 | 0.392 | 0.530 |
| Borromeo human (620 edges) | boba-T (`6667`, TKO-ATAC-derived, DIRECT-NET-restricted) | 5 | 4 | 1 | 0.250 | 0.20 | 0.222 |
| **Borromeo mouse, RPR2 (3,992 edges)** | **CellOracle (real ATAC, TKO)** | **597** | **19,598** | **241** | **0.012** | **0.404** | **0.024** |
| **Borromeo mouse, RPR2 (3,992 edges)** | **SCENIC (from scratch, TKO)** | **455** | **1,389** | **161** | **0.116** | **0.354** | **0.175** |
| Borromeo mouse, RPR2 (3,992 edges) | boba-T (`6667`, TKO-ATAC-derived, DIRECT-NET-restricted) | 25 | 61 | 7 | 0.115 | 0.28 | 0.163 |
| Pozo (295 edges) | CellOracle (real ATAC, TKO) | 48 | 129 | 26 | 0.202 | 0.542 | 0.294 (sign_concordance 0.5) |
| Pozo (295 edges) | SCENIC (from scratch, TKO) | 41 | 34 | 12 | 0.353 | 0.293 | 0.320 |
| Pozo (295 edges) | boba-T (`6667`, TKO-ATAC-derived, DIRECT-NET-restricted) | 4 | 4 | 2 | 0.500 | 0.50 | 0.500 (sign_concordance 0.0) |

boba-T's numbers here are identical to its rows in the very first table in this section (as they should be — same network, same ground truths, no refit happened). Its F1 (0.163–0.500) sits between CellOracle-real-ATAC's and SCENIC's on the mouse RPR2 ground truth. Read carefully, though: the much smaller shared-node counts (5–25 vs. 41–597) reflect boba-T's DIRECT-NET+LASSO candidate network being a curated 53-gene panel, not a different or smaller dataset — all three methods are evaluated on the same underlying TKO_final_arc cells. boba-T's higher apparent precision/F1 on the mouse RPR2 ground truth (0.115/0.163 vs. CellOracle-real-ATAC's 0.012/0.024) is consistent with the same pattern already established earlier in this document (DIRECT-NET-restricted methods concentrating signal on a small curated candidate set vs. genome-scale methods diluting precision over a much larger candidate-pair space) — a real, comparable finding here, not an artifact of comparing different data.

**The RPR2-mouse row is the most directly relevant of the three here** — TKO's own genotype (*Trp53;Rb1;Rbl2* triple-knockout) is an exact match to the Borromeo RPR2-mouse tumor's genotype, not just an ortholog-matched human comparison. And the pattern is the same one already found for `6667`'s genome-scale from-scratch run, now with a genuinely dataset-specific ATAC base GRN instead of a generic promoter scan: **CellOracle has consistently higher recall** (0.35–0.54 vs. SCENIC's 0.29–0.39 across all three ground truths) **but far lower precision** (0.01–0.20 vs. SCENIC's 0.12–0.82), and SCENIC wins on F1 for the two ground truths with the most shared nodes (human, mouse RPR2) while CellOracle edges it out narrowly on Pozo. Swapping a real, dataset-specific chromatin-accessibility prior in for CellOracle's generic promoter scan did not change this qualitative result: CellOracle's ridge regression still has no analogue of RcisTarget's independent motif-enrichment filter, so real ATAC-derived candidate peaks still get regularized down to a large, comparatively low-precision edge set rather than a small, high-precision one. The real-ATAC prior appears to matter for *which* genes get access to the candidate-regulator pool (2,159 target genes here, vs. the human-promoter-scan run's coverage) more than it does for the precision/recall trade-off itself, which seems to be a property of the two methods' downstream filtering, not of the base GRN's source.

## Running boba-T on the mouse ground-truth benchmark

This is the path to an actual Fig.-S2-style figure — real AUROC/EPR numbers against an independent ChIP-Atlas gold standard, with boba-T plotted alongside CellOracle/SCENIC/DCOL instead of against itself. It needs three things that don't exist yet: a candidate network for boba-T on data with no paired multiome, boba-T's expression/cluster inputs converted from the benchmark's preprocessed format, and a per-tissue boba-T run. None of this has been run — everything below is a verified-API plan, not a tested pipeline; treat it as a checklist to work through one sample at a time, not a script to fire off across all 13 at once.

### The candidate-network decision

boba-T's SCLC runs get their candidate network from DIRECT-NET, which needs paired single-cell multiome (ATAC + RNA from the same cells) to link peaks to genes. The mouse benchmark doesn't have that — Tabula Muris (scRNA) and the Cusanovich atlas (scATAC) are unpaired, matched only by tissue, which is also how CellOracle itself used them. DIRECT-NET simply cannot run here; boba-T needs a different source for its candidate edges.

The resolution: CellOracle ships a base GRN already built from that exact Cusanovich atlas — `celloracle.data.load_mouse_scATAC_atlas_base_GRN()` (mm9, TF motif scan over ~92k peaks from the atlas, verified against the installed package). Handing this to boba-T as its candidate network means both methods start from the same TF-binding prior, which is exactly the "same base GRN" fairness principle used everywhere else in this document — and it's a real ATAC-derived prior, not a workaround. The comparable released CellOracle variant, `celloracle_cluster_mouseAtacBaseGRN`, uses the same base GRN, which is why that's the variant to compare against in the structure quickstart above rather than the promoter-based or scrambled variants.

**Why this restriction matters, not just for fairness but for whether a method finds anything at all**: CellOracle's own Fig. S2 benchmark shows GENIE3 doing much worse than CellOracle's ATAC/promoter-restricted variants at recovering true ChIP-Atlas edges — and the `6667`/from-scratch experiment above ([Sourcing a real SCLC ground truth](#sourcing-a-real-sclc-ground-truth-done-with-a-correction)) reproduces the same pattern directly: CellOracle and SCENIC, run on `6667` with no ATAC/DIRECT-NET restriction at all, recovered *zero* true ASCL1 ChIP-seq edges across three independent ground truths, while every DIRECT-NET-restricted method (including boba-T) recovered a real, non-random fraction. The mechanism in both cases is the same: without a curated, ATAC-informed candidate set, these methods have to rediscover regulatory structure from co-expression and/or generic sequence motifs alone, across a much larger and noisier space of candidate gene pairs — and that's a much harder problem than pruning/weighting a set of edges someone already told you were plausible. This is why handing boba-T the same mouse-scATAC-atlas restriction CellOracle's own best-performing Fig.-S2 variant uses, rather than leaving it to find structure unaided, is the right choice for a benchmark that's supposed to isolate rule-fitting quality rather than re-litigating whether ATAC-informed priors help (they clearly do, on both datasets checked here).

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

## Combinatorial regulatory logic: the HSC ground truth

Comparisons 1-4 all score a single scalar per (regulator, target) pair — an edge weight, a
coefficient, a correlation. They can't tell apart "TF1 always activates GeneX" from "TF1
activates GeneX only when TF2 is absent," because the *object* being scored (one number per
edge) has no room to express that. boba-T's fitted rule is a full pseudo-Boolean function of
all of a gene's regulators at once (a $2^n$-entry probability table, one entry per regulator
state combination), so it's the one method here whose output is even the right *shape* to
represent AND/OR/NOT logic between TFs — the kind of combinatorial regulation Buchler et al.
(2003, [PNAS](https://doi.org/10.1073/pnas.0930314100)) and Balaji et al. (2006,
[JMB](https://doi.org/10.1016/j.jmb.2006.04.029)) argue is pervasive in real regulatory
networks. This section is a comparison built specifically to make that difference visible,
in two tracks: a synthetic dataset with an exact, literal Boolean ground truth (so "did the
method get the logic right" has an unambiguous answer), and a real multiome dataset for the
same biological system (so the same question can be asked of real data).

### Track 1: synthetic data from a literal Boolean model (Krumsiek et al. 2011 HSC network)

[BEELINE](https://github.com/Murali-group/BEELINE)/[BoolODE](https://github.com/Murali-group/BoolODE)
ship the Krumsiek et al. (2011) 11-gene hematopoietic stem cell differentiation model as a
literal Boolean network (`data/BoolODE/data/HSC.txt`) — the classic GATA1/PU.1 toggle switch,
with genuinely nested AND/OR/NOT rules, e.g. `Gata2 = Gata2 and (not(Pu1 or (Gata1 and
Fog1)))`. BoolODE simulates single-cell expression trajectories directly from these rules
(Hill-equation ODEs with the Boolean function as the production term), so the "ground truth"
here isn't inferred or literature-curated after the fact — it's the literal generative model.

```bash
# Simulate 2000 cells from the Boolean model (do_parallel: False -- True hangs on macOS,
# spawn/fork multiprocessing incompatibility with this 2019-era codebase).
cd benchmarking/data/BoolODE
python boolode.py --config config-files/hsc-config.yaml    # -> output-HSC/HSC/ExpressionData.csv

# Parse HSC.txt into boba-T's candidate network + per-gene ground-truth truth tables.
cd ../..
python hsc_ground_truth.py           # -> data/hsc_ground_truth/{candidate_network.csv, truth_table_*.csv}
python prepare_hsc_bobat_input.py    # -> data/hsc_ground_truth/{expr_bobat.csv, clusters_bobat.csv}

# Fit boba-T and score its fitted rule against the literal ground-truth truth table.
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_fit_bobat.py
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_truth_table_scoring.py
```

**Caveat on "regulator-set recovery" — this is not yet a discovery result.** boba-T's fitted
`regulators_dict` matches the ground-truth regulator set for all 11 genes (e.g. GATA2's fitted
parents are exactly `{GATA2, GATA1, FOG1, PU1}`), but that's guaranteed by construction, not
evidence boba-T found the right parents among a larger pool: `hsc_ground_truth.py`'s
candidate network is built directly from HSC.txt's own regulator lists (each gene's candidate
edges = exactly its true regulators, nothing else), and `comparison_hsc_fit_bobat.py` calls
`get_rules(..., threshold=0)`, which disables `bobaT/tl.py`'s own irrelevant-regulator-pruning
step (a regulator is only dropped if its relevance is `< threshold`, and relevance is never
negative). So boba-T was handed the answer as its candidate set and its own pruning mechanism
was turned off — this step currently shows boba-T *can fit* over a given true parent set
without being told to drop any of it, not that it can *discover* the parent set from a larger
one. A real structure-recovery test (not yet run) needs a genuinely larger candidate pool per
gene (e.g. all 10 other genes, or explicit decoys) and `threshold > 0` so pruning is active.

**Result 2 — the new metric: truth-table agreement.** Regulator-set recovery says boba-T found
the right *inputs*; it says nothing about whether the fitted *function* of those inputs is
correct. `comparison_hsc_truth_table_scoring.py` decodes boba-T's fitted rule (`gene|regulators|
p_0,...,p_(2^n-1)`, using the exact bit convention from `bobaT/tl.py:parent_heatmap` — regulator
order in the stored list is MSB-first over the leaf index) and compares it, entry-by-entry
against the literal ground-truth truth table for the same gene (rebuilt in boba-T's own fitted
regulator order, so both sides share one bit convention before comparing):

| gene | regulators | truth-table accuracy | truth-table AUC | (for reference) predictive R² |
|---|---|---|---|---|
| CEBPA | CEBPA,GATA1,FOG1,SCL | 0.688 | 0.846 | 0.980 |
| GATA1 | GATA1,PU1,FLI1,GATA2 | 0.562 | 0.714 | 0.933 |
| GATA2 | GATA2,GATA1,FOG1,PU1 | 0.688 | 0.821 | 0.932 |
| PU1   | PU1,GATA1,CEBPA,GATA2 | 0.562 | 0.641 | 0.900 |
| GFI1  | CEBPA,EGRNAB | 0.750 | 1.000 | 0.909 |
| EGRNAB| CJUN,GFI1,PU1 | 0.875 | 1.000 | 0.539 |
| CJUN  | GFI1,PU1 | 1.000 | 1.000 | 0.575 |
| SCL   | PU1,GATA1 | 1.000 | 1.000 | 0.334 |
| FOG1  | GATA1 | 1.000 | 1.000 | 0.236 |
| EKLF  | GATA1,FLI1 | 0.750 | 0.667 | 0.055 |
| FLI1  | GATA1,EKLF | 0.750 | 0.667 | 0.076 |

(mean truth-table accuracy 0.784, mean AUC 0.850; predictive R² from `comparison_hsc_fit_bobat.py`'s
`get_sklearn_metrics` output, same fit)

**The interesting result is the *disagreement* between the last two columns, in both
directions — and both directions have a verified (not guessed) mechanism, checked directly
against the validation CSVs and the simulated expression matrix:**

- **High R², mediocre truth-table accuracy (GATA1, GATA2, PU1, CEBPA — all 4-regulator
  rules).** For GATA1's 4 regulators there are 16 logically possible parent-state combinations,
  but only **8 of them are ever visited** by any of the 2000 simulated cells (leaves
  3,5,6,7,12,13,14,15 have zero occupancy, checked directly against `expr_bobat.csv`) — a
  committed differentiation trajectory doesn't explore the full combinatorial space, and one
  leaf (`GATA1=0,PU1=1,FLI1=0,GATA2=0`) alone accounts for 59% of all cells. R² is computed only
  on real observed cells, so a rule that nails the 2-3 dominant, well-supported leaves gets a
  great R² (0.93) regardless of what it fits for the 8 never-visited hypothetical states —
  which is exactly where most of its truth-table disagreement (7/16 wrong) comes from.
- **Low R², good truth-table accuracy (EKLF, FLI1, SCL, FOG1 — 1-2 regulator rules).** Checked
  two candidate explanations directly against the data rather than guessing:
  - *Not* a normalization artifact. One hypothesis: per-gene min-max normalization could
    "inflate" a narrow real range into an artificially large apparent [0,1] spread, making a
    genuinely-flat gene look falsely variable. Checked FLI1's raw (pre-normalization) BoolODE
    output directly: it already shows a real, large-scale split before any rescaling — roughly
    75% of cells sit near 0, but a genuine ~15-25% (not 1-2 outliers) have substantially
    elevated raw values (1.5-3.2 vs. <0.05 for the majority). The variance is real pre-rescaling
    too, so this isn't the mechanism.
  - The real mechanism: **within-leaf variance**. Holding FLI1's two Boolean parents fixed at
    `GATA1=1,EKLF=0` (n=254 cells, the "should be ON" leaf), FLI1's actual value still has
    std=0.253 — nearly half the entire possible [0,1] range, *within a single fixed parent
    state*. No rule (however fit) can explain that with one number per leaf. Contrast with
    GATA1's own best-populated leaf (n=1183): std=0.009, essentially deterministic given its 4
    parents. So GATA1's high R² reflects real near-determinism from its parents; FLI1's low R²
    reflects real residual variance that its 2 named parents don't explain — plausibly because
    FLI1 sits in a mutual-inhibition loop with EKLF (`Fli1=Gata1 and not(Eklf)`,
    `Eklf=Gata1 and not(Fli1)`), so a cell's *position along the commitment trajectory* carries
    information a static 2-parent snapshot can't see. FLI1's leaves are already
    well-populated (unlike GATA1's sparse ones), so this isn't fixable with more data — it needs
    more regulators, or pseudo-time, not more cells.

Predictive fit and combinatorial-logic fidelity are answering genuinely different questions —
R²/AUC alone (comparison 3's metric) would have missed both of these mechanisms — which is the
real justification for this being its own comparison rather than reading it off the existing
ones. It also means truth-table accuracy is noisier than it looks for genes with several
never-visited leaves: those entries are effectively unscored noise relative to what the
simulated trajectory actually supports, not a real test of learned biology.

CellOracle has since been fit on this same candidate network + simulated data too (see the
"Track 1 vs. Track 2, side by side" R² table further down this document, in the Track 2
section — mean R² 0.709, actually beating boba-T's 0.618 on the identical true-network
setup). **Still not yet done for Track 1:** GENIE3 (full SCENIC is inapplicable since
RcisTarget's motif step needs a real genome, not synthetic gene symbols); reconstructing
either method's *implied* truth table (predict every regulator-state combination, threshold
it) to make the comparison concrete: a method that fits one scalar coefficient per regulator
independently structurally cannot represent an AND/OR/NOT gate, and the reconstructed truth table should
show that directly (e.g. failing specifically on the states where the true rule is non-linear
in its inputs, not uniformly).

### Track 3: ChEA as a real, independent candidate network

Track 1's candidate network is the ground-truth edges themselves — useful for testing whether
boba-T can *fit* a correct combinatorial rule when handed the right parents, but not a test of
whether it can *find* them. Track 3 fixes that: it hands boba-T only the 11 gene **names** (no
connection information from HSC.txt at all) and builds the candidate network from
[ChEA](https://maayanlab.cloud/Enrichr/) instead — a real, independently-compiled database of
ChIP-seq-confirmed TF-target relationships. The resulting candidate network is real,
denser than the truth, and partly wrong/incomplete relative to this specific 11-gene toy model
— genuinely testing structure recovery for the first time in this comparison.

```bash
# Download ChEA_2022 (Enrichr gene-set library format) and restrict to edges where
# both the source TF and target gene are one of the 11 HSC-model genes (upper-cased,
# mapped to the model's own naming: ZFPM1->FOG1, KLF1->EKLF, SPI1->PU1, JUN->CJUN,
# TAL1->SCL, EGR1->EGRNAB (disclosed proxy, see Track 2) -- the only genes we have
# simulated expression for, so the only genes any candidate regulator could be scored on.
python hsc_chea_candidate_network.py     # -> data/hsc_ground_truth/candidate_network_chea.csv (57 edges)
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_fit_bobat_chea.py
```

`comparison_hsc_fit_bobat_chea.py` uses `threshold=0`, matching the real 6667 network's own
convention (`main_all_data_remove_selfloops_6667.py`: `node_threshold = 0  # don't remove any
parents`) rather than bobaT's package default of 0.1 — this project never actually uses bobaT's
own irrelevant-regulator pruning in its real runs, so Track 3 stays consistent with that. This
means any regulator-set recovery here has to come from the ChEA candidate network itself, not
from bobaT's own relevance filtering (which this project doesn't otherwise rely on).

**Result: regulator-set recovery is now a real, non-trivial number (mean F1 = 0.414, precision
0.360, recall 0.621)** — a meaningful drop from Track 1's tautological 1.0, and it decomposes
cleanly into one unavoidable limitation and one absent (by design) fitting behavior:

| gene | true regulators | in ChEA candidates | kept (threshold=0, no pruning) |
|---|---|---|---|
| CJUN | GFI1, PU1 | **0/2** | 0/0 |
| CEBPA | CEBPA,FOG1,GATA1,SCL | 2/4 | 2/2 |
| EGRNAB | CJUN,GFI1,PU1 | 1/3 | 1/1 |
| EKLF | FLI1,GATA1 | 1/2 | 1/1 |
| FLI1 | EKLF,GATA1 | 1/2 | 1/1 |
| FOG1 | GATA1 | 1/1 | 1/1 |
| GATA1 | FLI1,GATA1,GATA2,PU1 | 2/4 | 2/2 |
| GATA2 | FOG1,GATA1,GATA2,PU1 | 3/4 | 3/3 |
| GFI1 | CEBPA,EGRNAB | 2/2 | 2/2 |
| PU1 | CEBPA,GATA1,GATA2,PU1 | 3/4 | 3/3 |
| SCL | GATA1,PU1 | 2/2 | 2/2 |

1. **Candidate-coverage ceiling** (can't be fixed by better fitting, at any threshold): CJUN's
   true parents `{GFI1, PU1}` are never ChEA candidates at all — GFI1 has zero ChIP-seq entries
   in ChEA as a source TF, and `SPI1→JUN` (PU1→CJUN) isn't in ChEA either. Recovery here is
   capped at 0 regardless of how boba-T fits. Most other genes are only partially covered (e.g.
   GATA1: all 4 true regulators exist biologically, but ChEA only offers 2 of them as
   candidates) — a real database has real gaps relative to a specific curated toy model.
2. **Given `threshold=0`, boba-T keeps every offered candidate** — with no internal pruning,
   *every* true-and-offered regulator is retained (recall = exactly "fraction of true
   regulators ChEA offered": e.g. GATA2/PU1 now recover 3/4 instead of the 1/4 and 2/4 an
   earlier `threshold=0.1` pass got, since that pruning step was occasionally dropping correct
   candidates for no benefit — it wasn't removing any *wrong* ones either). Precision is
   correspondingly lower (0.360) since every ChEA false positive rides along uncorrected — the
   whole precision/recall trade here is entirely a property of ChEA's own edge list, not of
   boba-T's fitting, since fitting does no filtering at all at this threshold.

Predictive fit stays strong across the board (R² 0.47-0.98, mean well above Track 1's — more
candidate regulators generally only helps a flexible fuzzy-rule fit, even when several of them
are structurally wrong) — reinforcing the point that predictive accuracy and structural/logic
correctness are different axes, now from the opposite direction: a method can predict
expression well from a noisy, partly-incorrect candidate scaffold while still recovering the
wrong combinatorial structure.

**Entry-by-entry truth-table scoring, via empirical marginalization**
(`comparison_hsc_chea_truth_table_scoring.py`): since Track 3's fitted regulator sets differ in
size/identity from ground truth, a direct leaf-by-leaf comparison isn't defined the way it was
for Track 1. Fix: evaluate the fitted rule on every one of the 2000 simulated cells (using the
fitted regulators' real continuous values, bobaT's own fuzzy-weighting scheme), then group
cells by their *true* regulators' binarized state and average the fitted prediction within each
true leaf — an empirical marginalization over whatever extra/wrong regulators got carried
along, weighted by their actual joint distribution in the data.

**Headline numbers (mean accuracy 0.832, AUC 0.985) are real but not trustworthy on their own —
flag this every time this number is cited.** CJUN is the clearest case: its fitted rule has
**zero regulator overlap with ground truth** (never saw GFI1 or PU1 at all) yet scores perfect
truth-table accuracy and AUC. In this deterministic synthetic system, every gene is ultimately a
function of a handful of shared upstream drivers, so a rule built on the *wrong* parents can
still separate true-ON from true-OFF cells by riding a correlated proxy, not by using the real
causal information. This means marginalized truth-table accuracy, on its own, is not a reliable
signal of genuine logic recovery here — it has to be read together with the regulator-set F1
(0.414) above, which is the more honest measure of whether boba-T found the *right* combinatorial
inputs, not just something correlated with them. Several genes also have very small leaf sample
sizes backing part of the score (`min_leaf_n` as low as 1 for CEBPA/PU1) — those cells'
contribution to the accuracy/AUC is closer to anecdotal than statistically robust.

### Track 2: a real multiome dataset for the same system

The Krumsiek model's TFs (GATA1, GATA2, SPI1/PU1, FLI1, CEBPA, KLF1/EKLF, GFI1, ZFPM1/FOG1,
TAL1/SCL, JUN/cJun — 10 of 11; only the EGR1/NAB2 protein complex EGRNAB has no clean
single-gene proxy) and the cell populations that sit at this branch point (HSC, myeloid/
erythroid progenitors) are exactly what the NeurIPS 2021 Open Problems Multimodal Single-Cell
Integration competition dataset covers: real paired 10x Multiome (RNA+ATAC) from human bone
marrow, 12 donors, 4 sites (GEO [GSE194122](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)).

```bash
mkdir -p benchmarking/data/hsc_multiome && cd benchmarking/data/hsc_multiome
curl -O https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
gunzip -k GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
```

Confirmed (backed-mode `.h5ad` inspection): 69,249 cells x 129,921 features (`var[
'feature_types']`: 116,490 ATAC + 13,431 GEX); `obs['cell_type']` includes the exact target
populations — `HSC` (1,072 cells), `MK/E prog` (884), `G/M prog` (1,203) — plus downstream
`Erythroblast`/`Proerythroblast`/`Normoblast` and `CD14+`/`CD16+ Mono` populations that the
GATA1/PU.1 branch point actually resolves into; all 10 mappable model genes present as real
`GEX` features.

### Preprocessing real single-cell data for boba-T (a real bug, caught and fixed)

```bash
/opt/anaconda3/envs/bobaT_env/bin/python preprocess_hsc_multiome.py
```

Subsets to the 11 branch-point cell types (HSC, MK/E prog, G/M prog, plus the downstream
fates the branch resolves into on both arms — Erythroblast/Proerythroblast/Normoblast and
CD14+/CD16+ Mono/cDC2/pDC/ID2-hi myeloid prog; excludes all lymphoid T/B/NK, outside this
model's scope) → 27,050 real cells, and extracts the 10 mappable genes (EGRNAB dropped, no
real single-gene proxy).

**First attempt produced an implausible result — R²=1.0 on several genes — that turned out
to be a real, disclosable bug, not a fluke.** Traced it directly (never trust a suspiciously
perfect metric): `bb.load.load_data(..., norm=0.3)` applies its own quantile-clip
normalization (`lq,uq = quantile(0.3),quantile(0.7)`; `(data-lq)/(uq-lq)`). GATA1 etc. are
~92% *exact zeros* in this real data — extreme dropout, nothing like BoolODE's smooth
synthetic values — so both the 30th and 70th percentile land on 0, making `uq-lq=0`. Every
zero cell becomes `0/0→NaN→filled to 0`; every nonzero cell becomes `v/0→+inf→clipped to 1`.
Real continuous signal collapsed into a hard binary encoding *before boba-T ever saw it* — an
interaction between real single-cell dropout and a normalization step that was harmless on
synthetic data but pathological here. Caught by noticing `fit_validation`'s "actual" column
had only 2 distinct values where >2000 were expected — the same "verify against raw values,
don't trust a transform's name or a metric's face value" lesson as the organoid/mets_compiled
correction earlier in this project.

**Fix** (matches this project's own established real-single-cell convention, independently
re-derived for the same reason in `preprocess_organoid.py`'s docstring): normalize raw counts
by GSE194122's own precomputed genome-wide size factor (`obs['GEX_size_factors']`) → log1p →
MAGIC impute (same `magic-impute` package/`solver='approximate'` used throughout this
project) — no additional min-max step; let `load_data`'s own `norm=0.3` do the final [0,1]
rescaling on the now-smooth values, exactly like every other real-data script here. A second,
smaller issue: the first MAGIC attempt built the imputation graph from only the 10 panel
genes — too thin (`RuntimeWarning: zero distance between 43770 pairs of samples`, i.e. many
genuinely-different real cells look identical in just 10 dimensions since most are
simultaneously all-zero there). Fixed by computing ~2000 dispersion-based HVGs across the
full 13,431-gene real GEX matrix, unioned with the 10 panel genes, and running MAGIC on that
broader ~2006-gene matrix before keeping just the panel genes' imputed columns — the warning
disappeared and every panel gene's post-MAGIC distinct-value count became exactly 27,050
(fully continuous, confirming the fix).

### Two real, independent candidate networks

Two different real, non-cheating candidate networks were built for the same real cells and
scored against each other — this is where Track 2 earns its keep over Track 3, since it's now
testing a genuinely different modality (real ATAC + real motifs, not literature ChIP-seq):

1. **ChEA, reused from Track 3, restricted to the 10 real-data genes** (`candidate_network_
   chea_real.csv`, 44 edges) — same source/method as Track 3, EGRNAB dropped.
2. **Real ATAC + real motifs** (`build_hsc_atac_base_grn.py`, `candidate_network_atac_real.csv`,
   22 edges) — the thing repeatedly flagged earlier in this project as "the actual right
   answer to does CellOracle look at ATAC data," built here for the first time:
   - Real TSS coordinates for the 10 genes fetched from the Ensembl REST API (GRCh38, matching
     GSE194122's build).
   - Real ATAC peaks within 10kb of each TSS, pulled directly from GSE194122's own peak
     coordinates (3–7 peaks per gene, 52 total) — their real DNA sequence fetched from the
     UCSC REST API (targeted per-region fetch; no full-genome download needed, since
     `genomepy` has no genome installed in this environment).
   - `gimmemotifs`'s bundled 1,796-motif default vertebrate database, restricted to each of
     the 10 TFs' *direct*-binding motifs (via each `Motif.factors['direct']` annotation — the
     gene-symbol association lives there, not in the opaque `.id`). 9 of 10 TFs have real
     direct motifs; **ZFPM1 (FOG1) has zero** — the *third* independent line of evidence
     agreeing on this (also zero ChEA source entries in Track 3, and a known non-DNA-binding
     GATA1 cofactor biologically), so FOG1 can only ever be a target in any real base-network
     method here, never a source.
   - `celloracle.motif_analysis.scan_dna_for_motifs` with a properly FPR-calibrated
     `gimmemotifs.scanner.Scanner` (`fpr=0.02`, CellOracle's own tutorial default) — **a hand-
     rolled raw PWM-score cutoff was tried first and rejected**: `score >= 0.9 * max_score`
     with no background/FPR calibration returned an implausibly dense, near-complete graph (77
     of ~90 possible edges) — a bare score fraction ignoring the motif's minimum score is not
     a rigorous motif call. `Scanner.set_background(genome=...)` needs a locally installed
     genome; `set_background(fname=<fasta>)` doesn't, so 500 *other* real ATAC peaks sampled
     from this same dataset (excluding the 52 target peaks) were used as a real,
     dataset-matched null distribution. Result: 26 raw hits across 19 of 52 peaks — sparse and
     plausible, unlike the uncalibrated attempt.
   - Two more real gotchas hit and fixed while wiring this up: (a) several motifs are shared
     "direct" binders across TFs (e.g. one GATA motif lists GATA1, GATA2, *and* TAL1 as direct
     factors), so the raw per-TF motif list has duplicate IDs — deduped before scanning, or
     gimmemotifs' internal threshold table (which dedupes) ends up shorter than the motif list
     and crashes; (b) `celloracle.motif_analysis.process_bed_file.peak_M1` subtracts 1 from
     every result's start coordinate (1-based→0-based BED convention) before returning it —
     matched that offset when mapping scan results back to genes, or every lookup silently
     misses (this produced a real "26 hits, 0 edges" result on the way to the fix).

### Fitting all three methods on real cells, both candidate networks

```bash
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_bobat.py       # ChEA network
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_bobat_atac.py  # ATAC network
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_genie3.py       # ChEA network
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_genie3_atac.py  # ATAC network
/opt/anaconda3/envs/celloracle_env/bin/python comparison_hsc_multiome_fit_celloracle.py  # ATAC network only
```

CellOracle needed two more small, disclosed fixes on real MAGIC-imputed input specifically:
clipping a small fraction of values (0.3–23% per gene) that were slightly negative — a known
property of MAGIC's diffusion smoothing near zero, not real negative expression — and setting
`adata.layers["raw_count"]` manually (a CellOracle 0.20.0 gotcha already hit and documented
elsewhere in this project: the line that would set it is commented out in `oracle_core.py`).

| method | network | mean predictive R² | regulator-set F1 (vs. 10-gene ground truth) |
|---|---|---|---|
| boba-T | ChEA (44 edges) | 0.858 | 0.454 (precision 0.438, recall 0.650) |
| GENIE3 | ChEA (44 edges) | 0.760 | *(same candidates as boba-T, no pruning by either)* |
| boba-T | real ATAC+motif (22 edges) | 0.680 | 0.190 (precision 0.233, recall 0.250) |
| GENIE3 | real ATAC+motif (22 edges) | 0.649 (8/10 genes*) | *(same as boba-T's ATAC F1)* |
| CellOracle (ridge, α=10) | real ATAC+motif (22 edges) | — | 0.157 (precision 0.142, recall 0.225) |

*GENIE3 explicitly drops self-loops before fitting; CEBPA and CJUN had only a self-loop as a
candidate in the sparser ATAC network, so GENIE3 skips them (0 candidates) while boba-T's own
fallback (`if no regulators survive, use the gene itself`) keeps a self-loop-only fit for both.

**Real findings, not just numbers to report:**
- **GFI1 is the hardest gene for every method that touches it** (boba-T R²=0.36–0.22, GENIE3
  R²=0.47–0.34 across the two networks) — a real, cross-method, cross-network signal that
  GFI1 is genuinely harder to predict in this real dataset, not a boba-T-specific weakness.
- **The real ATAC+motif network is a much stricter, harder-to-match evidence source than
  ChEA's broad ChIP-seq literature compilation for this specific curated toy model** —
  structure recovery drops sharply on every method that used it (boba-T F1 0.454→0.190,
  same candidates for GENIE3), and predictive R² drops too (0.858→0.680 for boba-T) since the
  ATAC network is both sparser (22 vs 44 edges) and picks a partly different set of candidates
  (10kb-TSS-proximal, motif-confirmed) than ChEA's literature-wide compilation.
- **CellOracle's ridge (α=10) is even more conservative than boba-T on the identical ATAC
  candidate scaffold** (F1 0.157 vs 0.190) — consistent with the regularization behavior
  already observed on `6667`'s real data (comparison 1), not a new phenomenon specific to this
  system.
- On the ChEA network, **boba-T's real-data R² (0.858) beats GENIE3's (0.760)** — the opposite
  direction from the earlier synthetic-`6667` comparison (comparison 3), where GENIE3's
  random-forest flexibility won. Flagged as a real, not-yet-explained direction flip rather
  than smoothed over — worth investigating if this comparison is revisited.

**Not yet done at this point:** GENIE3/CellOracle haven't both been run on *both* the ChEA and
ATAC+motif networks (CellOracle only on ATAC, to keep that pass scoped). Superseded by the
comparison below for the specific "does DIRECT-NET+boba-T vs. default CellOracle" question.

### DIRECT-NET + boba-T vs. CellOracle's own real-multiome tutorial pipeline

The ChEA and hand-rolled ATAC+motif networks above are useful independent tests, but they
aren't *this project's own* real pipeline, nor CellOracle's own documented one. This section
runs the two methods' actual real-world recipes on the same real cells and compares both R²
and (an adaptation of) truth-table logic recovery — the specific comparison this section was
built to answer.

**boba-T's side: DIRECT-NET.** DIRECT-NET (Zhang lab, [Sci. Adv. 2022](https://www.science.org/doi/10.1126/sciadv.abl7393))
is a gradient-boosting (xgboost) method that regresses each gene's real expression against
nearby real ATAC peaks' real accessibility to discover CREs, then motif-scans the CREs it
finds for real TF binding — this project's own real 6667 network uses exactly this method
(`network-inference-DIRECT-NET/`), but it had never been run on GSE194122 before (no driver
script survived in this repo from the original run; DIRECT-NET is GitHub-only, `Run_DIRECT_
NET()`, and depends on Cicero itself internally).

```bash
# Real dependency chain installed fresh (chromVAR, motifmatchr, JASPAR2020/2016,
# BSgenome.Hsapiens.UCSC.hg38, monocle3, cicero, xgboost, DIRECT-NET) -- see setup notes below.
/opt/anaconda3/envs/celloracle_env/bin/python export_for_direct_net.py   # h5ad -> CSVs for R
Rscript run_direct_net.R              # builds Seurat obj, runs Run_DIRECT_NET, focus_markers=our 10 genes
Rscript run_direct_net_tf_links.R     # motif-scans the real CREs DIRECT-NET found (JASPAR2016 + hg38)
# -> data/hsc_multiome/candidate_network_directnet_real.csv (45 edges)
/opt/anaconda3/envs/bobaT_env/bin/python comparison_hsc_multiome_fit_bobat_directnet.py
```

Three real, disclosed installation/compatibility issues hit and fixed along the way (none by
editing the installed packages themselves):
- `BPCells` (a `monocle3` dependency) needs `libhdf5` to compile; not discoverable on this
  system — installed via Homebrew (`brew install hdf5 pkg-config`).
- `Run_DIRECT_NET`'s own `reduction.name` parameter is never actually forwarded to its
  internal `Aggregate_data()` call (a real bug in DIRECT-NET's own source, confirmed by
  reading it directly) — it always looks for a reduction literally named `wnn.umap`.
  Workaround: name our real GEX PCA embedding (from the h5ad's own precomputed
  `obsm['GEX_X_pca']`) `wnn.umap` — Seurat doesn't enforce that a reduction's name matches
  its algorithm.
- `DIRECTNET::isSparseMatrix` does `class(x) %in% c("dgCMatrix","dgTMatrix")`; for a plain
  dense matrix `class(x)` returns `c("matrix","array")` (length 2 since R 4.0), so `%in%`
  returns a length-2 vector and `if()` on it throws under R's stricter checking. Confirmed
  upstream bug — patched at the call site via `assignInNamespace` in our own script (this
  project's established pattern for fixing a dependency without touching its installed
  source, e.g. the `np.trapz` shim used elsewhere).
- Seurat 5's default `Assay5` class has no `@counts` slot (DIRECT-NET's own code expects the
  classic V3/V4 `Assay` API) — fixed with Seurat's own backward-compatibility escape hatch,
  `options(Seurat.object.assay.version = "v3")`.

Restricted to our 10 genes as `focus_markers` (DIRECT-NET's own `±250kb` promoter/enhancer
search window, real ATAC peaks, real GEX): 506 real CRE-gene links across 9/10 genes (SPI1
had fewer than 2 usable enhancer peaks after DIRECT-NET's own promoter/distal split), then a
real JASPAR2016 motif scan restricted to our 10 TFs' direct-binding motifs against those real
CREs (hg38, via `BSgenome.Hsapiens.UCSC.hg38`) → **45 real candidate edges**
(`candidate_network_directnet_real.csv`).

**CellOracle's side: Cicero + TSS + motif scan** (CellOracle's own documented real-multiome
recipe, not the generic promoter base GRN):

```bash
Rscript run_cicero.R                          # real co-accessibility on the same 584 real peaks
python build_hsc_cicero_base_grn.py           # link each gene's real promoter peak to Cicero-coaccessible
                                                # peaks (coaccess >= 0.2, Cicero/CellOracle's own convention),
                                                # then the SAME FPR-calibrated motif scan as the ATAC track
/opt/anaconda3/envs/celloracle_env/bin/python comparison_hsc_multiome_fit_celloracle_cicero.py
```

One Cicero-specific gotcha: `run_cicero()`'s internal distance-parameter estimation expects
`genome_coords` as exactly 2 columns (chr, length); passing 3 (chr, start, end) shifts its
positional column access and throws `wrong sign in 'by' argument`.

Real result: 35,000 real peak-pair co-accessibility scores; 4/10 genes' promoter peaks had
real Cicero-linked distal peaks above the 0.2 threshold (GATA2/SPI1/FLI1 had none — their
promoter peaks show no real co-accessibility partner in this window) → **40 real candidate
edges** (`candidate_network_cicero_real.csv`) after the same motif scan.

**Results — R² (held out the same way for both) and structure recovery:**

| method | network | mean R² (held-out test cells) | regulator-set F1 |
|---|---|---|---|
| boba-T | DIRECT-NET (45 edges) | 0.855 | 0.292 |
| boba-T | Cicero (40 edges) | 0.765 | 0.291 |
| CellOracle (ridge, α=10) | Cicero (40 edges) | **0.514** | 0.239 |

**Correction, caught by re-verifying before trusting a strikingly one-sided result** (see
`verify_celloracle_fit.py`): an earlier version of this table reported CellOracle's R² as
0.125, computed by reconstructing predictions as `coef_matrix · expression` and comparing
directly to raw expression. Two real bugs in that reconstruction, not in CellOracle itself:
(1) CellOracle's own `_getCoefMatrix` fits `sklearn.Ridge` with the library default
`fit_intercept=True`, but only ever saves `.coef_` into `coef_matrix` — `.intercept_` is
silently discarded by CellOracle's own code and never appears anywhere I could read it back
from, so my manual reconstruction was missing an additive constant for every gene; (2) the
Ridge model is actually fit on `oracle.adata.layers["imputed_count"]` (the KNN-smoothed
output of `oracle.knn_imputation()`), not the raw input expression I was evaluating against.
Fixed by refitting Ridge myself (same candidate TFs, same alpha, same held-out split) and
capturing both `.coef_` and `.intercept_`, evaluated against the real `imputed_count` the
model was actually trained on. Corrected mean R² is 0.514, not 0.125 — still behind boba-T's
0.765 on the identical network, but a real, much narrower gap than first reported. (Checked
whether this same bug affects the project's *earlier*, already-established comparison-3
result on the real 6667 SCLC data: it doesn't — that script applies a min-max rescale to both
actual and predicted before scoring, and min-max rescaling is invariant to a missing additive
constant, so CellOracle's R²=0.780 there is unaffected. This bug is specific to the ad-hoc
verification scripts written for this HSC comparison.)

**Real findings:**
- **boba-T beats CellOracle on R² on the identical real candidate network** (0.765 vs. 0.514)
  — a real, if now much narrower, fitting-method gap, not a network-choice artifact (both
  used the same 40-edge Cicero network).
- **DIRECT-NET and Cicero are two genuinely different real methods that landed on
  strikingly similar answers**: boba-T's structure-recovery F1 is 0.292 vs. 0.291 — a
  near-exact match — despite one method being gradient-boosting-based CRE discovery and the
  other co-accessibility-based. GFI1 is again among the hardest genes on both.

**Truth-table comparison, adapted to real data.** Unlike Tracks 1/3's synthetic data, there's
no literal simulator output to serve as ground truth for real cells. The meaningful analog
built here (`comparison_hsc_multiome_truth_table_scoring.py`): group real cells by their
*true* regulators' (from `hsc_ground_truth.py`) binarized real expression state, then check
whether each fitted model's prediction — averaged within that group — lands on the correct
side of the literal HSC.txt rule's 0/1 label for that state. AUROC (not raw accuracy) is used
since it's scale-invariant across boba-T's naturally-[0,1]-bounded prediction and CellOracle's
unbounded raw linear combination. Genes whose true rule depends on EGRNAB (e.g. GFI1) are
skipped — there's no real EGRNAB data to evaluate that rule against, and marginalizing over an
unmodeled regulator's state isn't the same as testing recovery of the modeled ones.

| method | network | mean truth-table AUC (8-9 genes; GFI1 skipped, needs EGRNAB) |
|---|---|---|
| boba-T | DIRECT-NET | 0.811 |
| boba-T | Cicero | 0.812 |
| CellOracle (ridge) | Cicero | 0.802 |

(Re-verified after the R² correction above, using the same correctly-refit, intercept-included
Ridge models: AUROC is invariant to a missing additive constant — adding the same constant to
every prediction doesn't change rank order — so this table was actually unaffected by that bug
the whole time; re-running it with the corrected models reproduced the same numbers to within
floating-point noise.)

**All three are close (0.80–0.81) here.** boba-T's real-data R² lead over CellOracle (0.765 vs.
0.514, corrected) is real but narrower than truth-table AUC alone would suggest — CellOracle's
ridge coefficients rank the true combinatorial states in roughly the right *direction* almost
as well as boba-T does, even where its absolute predicted expression *levels* are less
accurate.

**Track 1 vs. Track 2, side by side.** Both tracks score against the exact same object —
`hsc_ground_truth.py`'s literal truth tables — so the per-gene AUCs are directly comparable
even though Track 1 fits on synthetic data with the true regulator set handed to it, and
Track 2 fits on real cells with a real, independently-discovered candidate network. (Merged
directly from the already-computed `comparison_hsc_truth_table_bobat.csv` and
`comparison_hsc_multiome_truth_table_*.csv` outputs above — nothing in either track was rerun
to build this table.)

| gene | Track 1: boba-T (synthetic, true network) | Track 2: boba-T + DIRECT-NET (real) | Track 2: boba-T + Cicero (real) | Track 2: CellOracle + Cicero (real) |
|---|---|---|---|---|
| GATA2 | 0.821 | 0.667 | 0.500 | 0.533 |
| GATA1 | 0.714 | 0.833 | 0.738 | 0.714 |
| FOG1  | 1.000 | 1.000 | 1.000 | 1.000 |
| EKLF  | 0.667 | 1.000 | 1.000 | 1.000 |
| FLI1  | 0.667 | 0.333 | 0.333 | — (0 surviving candidates) |
| SCL   | 1.000 | 1.000 | 1.000 | 1.000 |
| CEBPA | 0.846 | 0.944 | 0.889 | 0.722 |
| PU1   | 0.641 | 0.852 | 0.852 | 0.444 |
| CJUN  | 1.000 | 0.667 | 1.000 | 1.000 |
| EGRNAB| 1.000 | — (no real EGRNAB data) | — | — |
| GFI1  | 1.000 | — (true rule needs EGRNAB) | — | — |
| **mean (8 genes present in all 4 columns)** | **0.836** | **0.870** | **0.872** | **0.802** |

**No advantage at all for the synthetic, cheat-network Track 1 setup over real data — if
anything, real data with a *discovered* network scores slightly higher.** Track 1 handed
boba-T the exact true regulators directly as candidates (a best-case, non-discovery scenario,
per the caveat above); Track 2 had to discover its own candidate network from real ATAC/RNA
from scratch, and both real-data boba-T runs (0.870, 0.872) *edge out* Track 1's synthetic,
cheat-network score (0.836) — CellOracle's real-data run (0.802) is the only one that trails
it, and only slightly. The two tracks disagree gene-by-gene in informative ways rather than
Track 2 simply trailing Track 1 everywhere:
- **CJUN**: Track 1 gets AUC=1.0 (trivial — GFI1 and PU1 were handed to it directly as the
  true candidates); Track 2's DIRECT-NET run drops to 0.667 because DIRECT-NET's real,
  independently-discovered candidates for CJUN didn't include the literal true pair — but
  Track 2's *Cicero* run recovers the full 1.0 anyway, on a different real candidate set.
- **EKLF and PU1**: both real-data runs (DIRECT-NET and Cicero) *exceed* Track 1's synthetic
  score (1.000/0.852 vs. Track 1's 0.667/0.641) — real cells' actual regulator-state
  distribution apparently makes these two genes' true logic easier to separate than the
  synthetic BoolODE trajectory did, not harder.
- **FLI1** is the one gene that's hard for literally everyone across both tracks and every
  network (0.667 synthetic, 0.333 on both real networks) — the same gene flagged earlier
  (Track 1's mechanism section) as sitting in a real mutual-inhibition loop with EKLF, whose
  static combinatorial snapshot can't see trajectory/hysteresis information. That this
  specific gene is *also* the hardest on two independently-built real candidate networks is
  a second, independent piece of evidence for that explanation, not just a synthetic-data
  quirk.

**Same comparison, for predictive R² instead of truth-table AUC.** This table also required
running CellOracle on Track 1's synthetic data for the first time (it had never been fit
there before — see the correction below), so unlike the tables above it isn't purely a
re-merge of pre-existing outputs; the boba-T columns are.

| gene | Track 1: boba-T (synthetic, true network) | Track 1: CellOracle (synthetic, true network) | Track 2: boba-T + DIRECT-NET (real) | Track 2: boba-T + Cicero (real) | Track 2: CellOracle + Cicero (real) |
|---|---|---|---|---|---|
| GATA2 | 0.932 | 0.315 | 0.936 | 0.022 | 0.000 |
| GATA1 | 0.933 | 0.885 | 0.887 | 0.989 | 0.974 |
| FOG1  | 0.236 | 0.927 | 0.704 | 0.771 | 0.811 |
| EKLF  | 0.055 | 0.776 | 0.845 | 0.982 | 0.982 |
| FLI1  | 0.076 | 0.795 | 0.950 | 0.926 | — (0 surviving candidates) |
| SCL   | 0.334 | 0.928 | 0.861 | 0.959 | 0.847 |
| CEBPA | 0.980 | 0.225 | 0.934 | 0.934 | 0.348 |
| PU1   | 0.900 | 0.874 | 0.952 | 0.947 | 0.238 |
| CJUN  | 0.575 | 0.739 | 0.941 | 0.934 | 0.357 |
| EGRNAB| 0.539 | 0.842 | — (no real EGRNAB data) | — | — |
| GFI1  | 0.909 | 0.906 | 0.544 | 0.184 | 0.068 |
| **mean (8 genes present in all 5 columns)** | **0.618** | **0.709** | **0.882** | **0.817** | **0.570** |

**Correction before drawing any conclusions here** (caught by double-checking a strikingly
one-sided result rather than trusting it — see `verify_celloracle_fit.py`): the CellOracle
columns above were originally computed by reconstructing predictions as
`coef_matrix · expression` and comparing to raw expression directly. This has two real bugs,
not in CellOracle itself but in that reconstruction: (1) CellOracle's `_getCoefMatrix` fits
`sklearn.Ridge` with the library default `fit_intercept=True`, but only saves `.coef_` — the
intercept is silently discarded by CellOracle's own code and never appears in `coef_matrix`,
so the reconstruction was missing an additive constant per gene; (2) the Ridge model is
actually fit on `oracle.adata.layers["imputed_count"]` (KNN-smoothed), not the raw expression
being evaluated against. The original (wrong) numbers had CellOracle catastrophically failing
on synthetic data (mean R² = **-0.413**, several genes strongly negative — e.g. PU1 = -4.12)
and badly on real data (mean R² = **0.125**) — both *far* worse than boba-T in both regimes,
which was the "great result" this correction was prompted by double-checking. Fixed by
refitting Ridge myself (same candidate TFs, same alpha, same held-out cells) and capturing
both `.coef_` and `.intercept_`, evaluated against the real `imputed_count` the model was
actually trained on. **Confirmed this bug is specific to this session's ad-hoc scripts, not
a problem with the project's earlier, already-established 6667 comparison-3 result**: that
script rescales both actual and predicted to `[0,1]` before scoring, and min-max rescaling is
invariant to a missing additive constant, so CellOracle's R²=0.780 there was never affected.
Truth-table AUC (the table above) was also unaffected throughout — confirmed by re-running it
with the corrected, intercept-included models and getting the same numbers back — since
AUROC doesn't change under a constant shift applied to every prediction.

**The corrected picture is genuinely mixed, not one-directional.** CellOracle actually *beats*
boba-T on Track 1's synthetic data (0.709 vs. 0.618) — the reverse of what the (wrong) first
pass showed — while boba-T still beats CellOracle on Track 2's real data (0.817 vs. 0.570),
though by a real but much narrower margin than the uncorrected -0.413/0.125 numbers implied.
Both real-data boba-T runs (0.882, 0.817) still clearly beat Track 1's synthetic boba-T score
(0.618) — that comparison was never affected by this bug, since it doesn't involve
CellOracle — for the same reason already established in Track 1's own mechanism section:
EKLF/FLI1/SCL/FOG1/GFI1/CJUN had low synthetic R² specifically because BoolODE's continuous
trajectory leaves real within-leaf variance a 1-2-regulator boba-T rule can't explain; real
cells don't carry that same simulation artifact. GATA2 is the one gene where *every* method on
*every* real network does badly (boba-T 0.022, CellOracle 0.000) — its real promoter peak
apparently offers very little real regulatory signal in this dataset regardless of method,
already flagged from the structure side (Track 2's ChEA/ATAC/Cicero sections) as its
candidate set collapsing to just FLI1 across three independently-built real networks.

## Alternative edge-weight summaries: is collapsing the fitted rule to a sum the right call?

Every comparison above that scores boba-T's *structure* (comparison 1, the HSC tracks) reads its edge weights straight from `signed_strengths.csv`/`strengths.csv`. Those aren't a direct measure of anything — they're a **summary** of boba-T's actual fitted object, which is a full pseudo-Boolean truth table (one probability per combination of a target's regulators), collapsed down to one number per regulator by *summing* that regulator's ON-vs-OFF effect across every combination of its co-regulators (`bobaT/tl.py:52`, `detect_irrelevant_regulator`'s `tot_dif`/`signed_tot_dif`). Two things about a plain sum are worth questioning: it isn't comparable across targets with different regulator counts (summing over `2**(n-1)` contexts inflates edges into heavily-co-regulated targets, regardless of whether the effect is actually stronger), and it can hide a regulator whose effect is genuinely concentrated in a few contexts (canalizing/conditional regulation) or that flips sign across contexts (summed toward zero even if individually large). This section tests that concern directly, first on the SCLC data, then — with a much stronger ground truth — on HSC Track 2.

**Eight summaries**, all derived from the *same* already-fitted rule (no refitting needed — `rules_<run>.txt` already stores the full truth table), computed by [`comparison_edge_weight_summaries_6667.py`](comparison_edge_weight_summaries_6667.py):

| summary | what it computes |
|---|---|
| `sum_abs` / `sum_signed` | boba-T's current `strengths.csv`/`signed_strengths.csv`: sum of \|Δ\| / Δ across every context |
| `mean_abs` / `mean_signed` | the same sum, divided by the number of contexts — comparable across targets with different regulator counts, unlike the raw sum |
| `max_abs` / `max_signed` | the single largest \|Δ\| across contexts (and its sign) — boba-T computes this internally to decide pruning, then discards it; captures a regulator whose effect is concentrated in one or few contexts even if its average effect is small |
| `dataw_abs` / `dataw_signed` | mean of \|Δ\| / Δ, weighted by how often each context's combination of co-regulators actually occurs in the real training data (recomputes boba-T's own per-cell soft leaf-membership weighting, computed during fitting and normally discarded) — contexts that never really happen in the data stop contributing; contexts that dominate the real cell population dominate the score |

### First pass: SCLC (`6667`), scored against the ChIP-seq ground truths

Scoring each summary against the same three ChIP-seq ground truths from [Sourcing a real SCLC ground truth](#sourcing-a-real-sclc-ground-truth-done-with-a-correction) exposed a methodology gap before it produced a real answer: comparison 2's usual ground-truth-gene-universe AUROC came back numerically identical (to 6 decimal places) across all 8 summaries. Not a bug — with only ~13 ASCL1-sourced edges in a 380K+-candidate universe, AUROC is dominated by "any nonzero score beats the sea of always-zero negatives," which is true for every monotonic-positive summary regardless of how those 13 edges rank *relative to each other*. Fixed by scoring a second way, restricted to boba-T's own 52 possible ASCL1-edges (its real candidate set) instead of the ground truth's full gene universe:

| ground truth (n true positives / 52) | best summary | current method (`sum_abs`) |
|---|---|---|
| RPR2-mouse (n=24) | `dataw_signed`: AUROC 0.528 | AUROC 0.524 — close |
| Pozo human (n=3) | **`sum_abs`: AUROC 0.762, EPR 5.78** — current method wins clearly | — |
| Borromeo human (n=4) | all ~equally weak (AUROC ≈ 0.48–0.50) | — |

Inconclusive on its own: the current sum-based method wasn't losing, but n=3–24 is too small to trust either way.

### Second pass: HSC Track 2 — a real, direct test against literal ground truth

HSC Track 2 (see above) is a far better test bed: a *literal* Boolean ground truth (not ChIP-seq) for every fitted gene, across four independently-built real candidate networks (ChEA, ATAC+motif, DIRECT-NET, Cicero), giving 39 true-regulator observations instead of 3–24. Run with [`comparison_edge_weight_summaries_hsc_track2.py`](comparison_edge_weight_summaries_hsc_track2.py), which reuses `comparison_edge_weight_summaries_6667.py`'s summary functions directly against the four Track 2 rules files.

**Structure recovery** (pooled AUROC per network — does the summary rank true regulators above false candidates?):

| summary | ChEA | ATAC | DIRECT-NET | Cicero |
|---|---|---|---|---|
| `sum_abs` (current) | **0.304** (worst) | 0.691 | 0.671 | 0.601 |
| `sum_signed` (current) | 0.382 | 0.779 | **0.746** (tied best) | 0.571 |
| `mean_abs` | 0.533 | 0.750 | 0.671 | 0.619 |
| `mean_signed` | 0.583 | 0.750 | **0.746** (tied best) | 0.625 |
| `max_abs` | 0.467 | **0.838** (best) | 0.580 | 0.640 |
| `dataw_abs` | 0.592 | 0.603 | 0.637 | 0.616 |
| `dataw_signed` | 0.636 | 0.706 | 0.714 | **0.634** (best) |

No summary wins on every network. The current method is the outright *worst* on ChEA (0.30–0.38, barely better than random) but ties for best on DIRECT-NET (0.746). `max_abs` dominates on ATAC (0.838) but is near-worst on ChEA. `dataw_signed` never wins outright but is the most *consistent* (0.64–0.71 everywhere) — a reasonable property for a default if no single network is going to be "the" one used going forward.

**The direct hypothesis test — canalization degree vs. summary ranking.** Using the literal Krumsiek AND/OR/NOT formulas, every true regulator's *ground-truth* "canalization degree" (how concentrated its real influence is across contexts — 0 = matters almost everywhere, up to `true_max_abs − true_mean_abs`'s ceiling = matters in only a handful of contexts) was computed directly from the Boolean rule, then correlated (Spearman) against how much more `max_abs`/`dataw_abs` ranks each true regulator relative to `sum_abs`/`mean_abs`:

```
Spearman(canalization_degree, rank_drop_sum_vs_max)    = -0.109  (n=39)
Spearman(canalization_degree, rank_drop_mean_vs_dataw) = -0.069  (n=39)
```

**Both are essentially null, and if anything point the opposite direction from the hypothesis.** On real, literal ground truth, truly canalized/context-dependent regulators are *not* systematically rescued by max- or data-weighted summaries relative to the current sum-based ones — the specific failure mode this whole investigation was checking for doesn't show up. (One genuine, if incidental, finding along the way: in a literal 0/1 Boolean truth table, every true regulator's `max_abs` is exactly 1 and `min_abs` is exactly 0 — any AND/OR/NOT regulator has some context where it fully determines the output and some context where it doesn't matter at all — so a naive `max − min` "context range" carries zero information here; `true_max_abs − true_mean_abs` is the graded quantity that actually varies, 0.125–0.875 across the 39 true regulators.)

**No alternative summary decisively and consistently beats the current sum-based one across networks, and the specific mechanism motivating this investigation (averaging washing out canalized regulators) isn't supported by the one test built specifically to check it.** That said, `dataw_signed`/`dataw_abs` were the most *consistent* performer of the eight (never the worst on any of the four real Track 2 networks) — on that basis, the decision was made to adopt them as the new default anyway, even without a decisive statistical win, since "never worst" is a reasonable property to prefer when no summary is a clear overall winner.

### Switching the default: `strengths.csv`/`signed_strengths.csv` now mean `dataw_abs`/`dataw_signed`

[`regenerate_strengths_as_dataw.py`](regenerate_strengths_as_dataw.py) overwrites `rules/strengths.csv` and `rules/signed_strengths.csv` **in place**, for every existing boba-T run (`6667`, HSC Tracks 1/2/3 — `hsc`, `hsc_chea`, `hsc_multiome{,_atac,_directnet,_cicero}`), from the already-fitted `rules_<run>.txt` — **no refitting**, just a different collapse of the same fitted truth table. The sum-based originals are preserved alongside as `strengths_sum_abs.csv.bak` / `signed_strengths_sum_signed.csv.bak` in each run's `rules/` directory, so this is fully reversible.

One thing to know rather than be surprised by: genes with exactly one total regulator (including boba-T's self-loop fallback when no real regulator survives — 11 of `6667`'s 53 genes) are **unchanged** by this switch — with only one regulator there's exactly one context, so sum/mean/max/data-weighted all reduce to the same single number by construction. Only genes with 2+ regulators actually get a different value (214 of `6667`'s 226 edges, for example). The script verifies this directly against the pre-existing files (not just assumed) before writing, and raises rather than silently overwriting if a self-loop-only gene's value ever doesn't match.

```bash
python3 regenerate_strengths_as_dataw.py
```

Every loader in this benchmark (`load_bobat`, `load_bobat_topology`, everything built on `matrix_to_edges`) reads `signed_strengths.csv` by path and has no opinion on what summary produced its numbers, so nothing downstream needed to change — re-running any comparison in this document that touches boba-T's edges now transparently uses `dataw_signed`/`dataw_abs`.

## Repo layout

```
benchmarking/
├── cell-oracle-benchmark.py                        # thin CLI: `from grn_benchmark.runners import main`
├── preprocess_benchmark_data.py                     # CLI: reproduce CellOracle's scRNA preprocessing
├── comparison3_fit_celloracle_6667.py               # comparison 3, CellOracle fitting step, DIRECT-NET-restricted (celloracle_env)
├── comparison_genie3_fit_6667.py                    # comparison 3, GENIE3 fitting step, DIRECT-NET-restricted (either env)
├── comparison_celloracle_fromscratch_fit_6667.py    # comparison 1, CellOracle from scratch (53-gene, superseded), own promoter base GRN (celloracle_env)
├── comparison_scenic_fit_6667.py                    # comparison 1, SCENIC from scratch (53-gene, superseded), GRNBoost2+RcisTarget (scenic_env)
├── preprocess_sclc_full_6667.py                     # comparison 1, preprocess the REAL genome-wide data behind 6667 (celloracle_env)
├── comparison_celloracle_fullscale_fit_6667.py      # comparison 1, CellOracle from scratch, real genome-wide/HVG scale (celloracle_env)
├── comparison_scenic_fullscale_fit_6667.py          # comparison 1, SCENIC from scratch, real genome-wide/HVG scale (scenic_env)
├── comparison3_score_celloracle_vs_bobat_6667.py    # comparison 3, scoring step, all methods (bobaT_env)
├── verify_celloracle_6667.py                        # comparison 3: checks CellOracle's real vs. raw-test representation mismatch (celloracle_env)
├── verify_bobat_vs_imputed_6667.py                  # comparison 3: fairness check -- does boba-T's own R2 also rise against the same imputed target? (bobaT_env)
├── redo_comparison3_6667_fair.py                    # comparison 3: full fair re-score, all 3 methods vs. one shared imputed_count target (celloracle_env)
├── comparison1_structure_6667_vs_chipseq_gt.py      # comparison 1, real ChIP-seq ground truth, all methods (either env)
├── hsc_ground_truth.py                              # HSC combinatorial-logic: parse HSC.txt -> candidate network + per-gene truth tables
├── prepare_hsc_bobat_input.py                       # HSC: BoolODE's ExpressionData.csv -> boba-T input format
├── comparison_hsc_fit_bobat.py                      # HSC: fit boba-T on the BoolODE-simulated data (bobaT_env)
├── comparison_hsc_truth_table_scoring.py            # HSC: score boba-T's fitted rule against the literal ground-truth truth table
├── hsc_chea_candidate_network.py                    # HSC Track 3: build a real candidate network from ChEA (TF names only, no true connections)
├── comparison_hsc_fit_bobat_chea.py                 # HSC Track 3: fit boba-T on the ChEA-derived candidate network (bobaT_env)
├── comparison_hsc_chea_truth_table_scoring.py       # HSC Track 3: marginalized truth-table scoring (fitted regulator set != ground truth's)
├── preprocess_hsc_multiome.py                       # HSC Track 2: real GSE194122 subset -> boba-T input (normalize+log1p+MAGIC; bobaT_env)
├── build_hsc_atac_base_grn.py                       # HSC Track 2: real ATAC peaks + real motifs -> candidate network (celloracle_env)
├── comparison_hsc_multiome_fit_bobat.py             # HSC Track 2: fit boba-T, ChEA network, real data (bobaT_env)
├── comparison_hsc_multiome_fit_bobat_atac.py        # HSC Track 2: fit boba-T, real ATAC network, real data (bobaT_env)
├── comparison_hsc_multiome_fit_genie3.py            # HSC Track 2: fit GENIE3, ChEA network, real data (bobaT_env)
├── comparison_hsc_multiome_fit_genie3_atac.py       # HSC Track 2: fit GENIE3, real ATAC network, real data (bobaT_env)
├── comparison_hsc_multiome_fit_celloracle.py        # HSC Track 2: fit CellOracle, real ATAC network, real data (celloracle_env)
├── export_for_direct_net.py                         # HSC Track 2: h5ad -> CSVs for building a Seurat object in R (celloracle_env)
├── run_direct_net.R                                 # HSC Track 2: build Seurat obj + run DIRECT-NET, focus_markers=10 genes
├── run_direct_net_tf_links.R                        # HSC Track 2: motif-scan DIRECT-NET's real CREs (JASPAR2016 + hg38)
├── run_cicero.R                                     # HSC Track 2: real Cicero co-accessibility on the same real peaks
├── build_hsc_cicero_base_grn.py                     # HSC Track 2: Cicero-linked peaks -> FPR-calibrated motif scan (celloracle_env)
├── comparison_hsc_multiome_fit_bobat_directnet.py   # HSC Track 2: fit boba-T, real DIRECT-NET network (bobaT_env)
├── comparison_hsc_multiome_fit_bobat_cicero.py      # HSC Track 2: fit boba-T, real Cicero network (bobaT_env)
├── comparison_hsc_multiome_fit_celloracle_cicero.py # HSC Track 2: fit CellOracle, real Cicero network (celloracle_env)
├── comparison_hsc_multiome_truth_table_scoring.py   # HSC Track 2: real-data truth-table/AUC comparison, DIRECT-NET+boba-T vs Cicero+CellOracle
├── comparison_hsc_fit_celloracle.py                 # HSC Track 1: fit CellOracle on synthetic data, same true-network candidates boba-T got (celloracle_env)
├── score_celloracle_hsc_truthtable.py               # HSC Track 1: CellOracle truth-table/AUC scoring
├── verify_celloracle_fit.py                         # HSC Track 1+2: corrected R2 (intercept + imputed_count fix) -- re-verification after catching a bug in the ad-hoc scripts above
├── verify_celloracle_truthtable.py                  # HSC Track 1+2: re-verifies truth-table AUC with the corrected models (confirms it was unaffected)
├── comparison_edge_weight_summaries_6667.py         # alt. edge-weight summaries (sum/mean/max/data-weighted) vs. SCLC ChIP-seq GTs
├── comparison_edge_weight_summaries_hsc_track2.py   # same summaries, scored on HSC Track 2's 4 real networks + literal Boolean ground truth
├── regenerate_strengths_as_dataw.py                 # switches every run's strengths.csv/signed_strengths.csv default to dataw_abs/dataw_signed
├── setup_celloracle_env.sh                          # builds the celloracle_env conda env (Apple Silicon)
├── setup_scenic_env.sh                              # builds the scenic_env conda env
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
│   ├── comparison1_structure_fullscale_vs_borromeo2016_ascl1_human.csv       # CellOracle, genome-scale
│   ├── comparison1_structure_fullscale_vs_borromeo2016_ascl1_mouse.csv       # CellOracle, genome-scale
│   ├── comparison1_structure_fullscale_vs_pozo2021_ascl1_direct.csv          # CellOracle, genome-scale
│   ├── comparison1_structure_fullscale_vs_borromeo2016_ascl1_human_scenic.csv # SCENIC, genome-scale
│   ├── comparison1_structure_fullscale_vs_borromeo2016_ascl1_mouse_scenic.csv # SCENIC, genome-scale
│   ├── comparison1_structure_fullscale_vs_pozo2021_ascl1_direct_scenic.csv    # SCENIC, genome-scale
│   └── comparison3_all_methods_vs_bobat_6667.csv
└── data/                                            # inputs + ground truth; see data/README.md for provenance
    ├── sclc_chipseq_gt/                              # real SCLC ChIP-seq ground truth (see below); .xlsx gitignored, derived .csv tracked
    ├── scenic/                                       # cisTarget databases + SCENIC intermediates (gitignored; ~410MB, re-downloadable)
    └── sclc_full/                                    # genome-wide 6667 expression, preprocessed from Box (gitignored; ~206MB, regenerate via preprocess_sclc_full_6667.py)
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
| 1. Network structure vs. reference | Implemented; run on `6667` at two scales — DIRECT-NET-restricted (boba-T, CellOracle, GENIE3: recover real edges) and genuinely genome-scale, no boba-T involved at all (CellOracle, SCENIC via the real Box multiome data: also recover real edges, ASCL1 emerges as a real hub in both, SCENIC's motif pruning gives it better precision) — vs. three real ASCL1 ChIP-seq ground truths; mouse-tissue-benchmark version (Fig-S2-style) still needs the boba-T mouse run |
| 2. Edge-weight recovery (BEELINE AUROC/EPR) | Implemented + validated against CellOracle's own Fig-S2 numbers; real Fig-S2-style figure with boba-T needs the boba-T mouse run |
| 3. Predicted vs. actual TF expression | Implemented and run on `6667` for boba-T, CellOracle, and GENIE3, then fully re-scored against a single fair, consistent target (42 shared genes both times) after finding CellOracle's original R² was computed on the wrong data representation — **corrected result: R² 0.858 (boba-T) / 0.864 (CellOracle, now essentially tied) / 0.963 (GENIE3, still the clear winner)**, vs. the original, now-superseded 0.832/0.780/0.899; full SCENIC/WGCNA not attempted (see roadmap); mouse-benchmark version needs a held-out split there |
| 4. In-silico perturbation | Stubbed; boba-T's per-attractor output needs an aggregation step before this is buildable |
| 5. Combinatorial logic (HSC ground truth) | Track 1 (synthetic, true candidate network): regulator-set "recovery" is by construction not a real discovery test (see caveat), mean truth-table accuracy 0.784 (AUC 0.850); boba-T R²=0.618 vs. CellOracle R²=0.709 (CellOracle wins here — see the R² correction below); GENIE3 not yet run. Track 3 (synthetic, ChEA-derived candidate network, `threshold=0`): regulator-set recovery F1=0.414; marginalized truth-table accuracy 0.832/AUC 0.985 but flagged as not trustworthy alone. Track 2 (real GSE194122 data, 27,050 cells): ChEA/hand-rolled-ATAC networks — boba-T F1 0.454 (ChEA) / 0.190 (ATAC), GENIE3 R² 0.760 (ChEA) / 0.649 (ATAC), CellOracle F1 0.157 (ATAC); GFI1 hardest gene for every method/network. **DIRECT-NET+boba-T vs. CellOracle's own real-multiome tutorial pipeline (Cicero+motif scan)**: boba-T R²=0.855(DIRECT-NET)/0.765(Cicero) vs. CellOracle R²=0.514 (corrected — see `verify_celloracle_fit.py`, an earlier pass had this wrong at 0.125 due to a missing-intercept bug in the verification, not in CellOracle) on the *identical* Cicero network — boba-T still wins on real data, narrower margin than first computed; truth-table AUC close for all three (0.80–0.81) throughout, unaffected by that bug. **Net picture: CellOracle beats boba-T on R² for synthetic data, boba-T beats CellOracle on R² for real data — genuinely mixed, not a one-sided result in either direction** |

The one prerequisite shared by everything still marked "needs the boba-T mouse run": [Running boba-T on the mouse ground-truth benchmark](#running-boba-t-on-the-mouse-ground-truth-benchmark) above.

## Environment notes

CellOracle 0.20.0 does not `pip install` cleanly on Apple Silicon out of the box (velocyto needs OpenMP, gimmemotifs is pinned to a version whose C sources fail under modern clang, etc.) — `setup_celloracle_env.sh` has the full workaround and is idempotent-ish to re-run. Base GRNs from `co.data.load_human_promoter_base_GRN()` / `load_mouse_scATAC_atlas_base_GRN()` download to `~/celloracle_data/`.

pySCENIC 0.12.1 similarly needs its own env (`setup_scenic_env.sh`) — its 2022-era dependency chain (arboreto, dask, numpy) conflicts with both `celloracle_env` and `bobaT_env`'s pins. The cisTarget motif databases it needs (`data/scenic/`, ~410MB) are downloaded fresh from `resources.aertslab.org` and gitignored; see `comparison_scenic_fit_6667.py` for the exact URLs.

**R environment (DIRECT-NET + Cicero, added for the "DIRECT-NET+boba-T vs. CellOracle real-multiome pipeline" comparison above)**: this project's global R 4.6.1 library now also has `DIRECTNET` (`remotes::install_github("zhanglhbioinfor/DIRECT-NET")`), `cicero` (`remotes::install_github("cole-trapnell-lab/cicero-release", ref="monocle3")`), `monocle3` (`remotes::install_github("cole-trapnell-lab/monocle3")` — needs `BPCells`, which needs `libhdf5`; installed via `brew install hdf5 pkg-config` first), `xgboost`, and `chromVAR`/`motifmatchr`/`JASPAR2020`/`JASPAR2016`/`BSgenome.Hsapiens.UCSC.hg38` (all via `BiocManager::install`, Bioconductor 3.23). `getJasparMotifs()` (used by DIRECT-NET's own TF-linking step) hardcodes JASPAR2016 specifically, not JASPAR2020 — both are needed. Two real, disclosed compatibility workarounds needed at *call time* (not by editing any installed package): `options(Seurat.object.assay.version = "v3")` before building any Seurat object DIRECT-NET will touch (its code expects the classic `Assay` class's `@counts` slot, not Seurat 5's `Assay5`/`@layers`), and `assignInNamespace("isSparseMatrix", ..., ns="DIRECTNET")` patching a real upstream bug (`class(x) %in% c(...)` returns a length-2 vector for a plain dense matrix under R≥4.0, tripping R's stricter `if()` check) — see `run_direct_net.R` for both, applied only in that script's own session, not persisted to the package.
