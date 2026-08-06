# BoBa-T hyperparameters: a decision guide

Reference for choosing BoBa-T's fitting hyperparameters for a given dataset. Written
from the investigation into why shGFP organoid external validation (run `6667`) wasn't
as good as expected for a null/control condition — see
`/Users/xpz5km/.claude/plans/it-is-looking-like-elegant-patterson.md` for the full
narrative and `../comparisons/6667_vs_6668_norm_sweep/` /
`../comparisons/domain_shift_diagnostic_and_organoid_walks/FINDINGS.md` for the actual results referenced
below. Covers what's implemented today and what's designed-but-deferred; each deferred
item is written as a full spec so the reasoning and the eventual code stay in sync.

*(All scripts named below live under `network-inference-DIRECT-NET/claude_analysis/`,
organized by phase: `01_normalization_sweep/`, `02_hyperparameter_experiments/`,
`03_domain_shift_diagnostics/`, `04_rorb_perturbation_validation/`, `05_preprocessing/`.
This file itself now lives in that same `claude_analysis/` folder.)*

## 1. `node_normalization` (`norm=` in `bb.load.load_data`)

**What it does**: for a float in `(0,1)`, takes the gene's `norm`-th and `(1-norm)`-th
quantiles as a window, linearly rescales inside it, and hard-clips everything outside to
exactly 0 or 1 (`load.py:164-170`). At the repo-wide default of `0.3`, the window is the
middle 40% of each gene's distribution — the outer 60% of cells get clipped to exact
0/1.

**Why it matters for rule quality**: `get_rules` (`tl.py:244`) fits each truth-table leaf
as a heat-weighted continuous average, then adds a pseudo-observation
(`tl.py:324-333`, comment in the source itself): every leaf gets a weight-`(1-max(heat))`
sample at probability 0.5. A leaf only escapes this pull-to-0.5 if **at least one cell is
a confident (near-1 heat) match** — not based on how many cells nominally belong to it.
Whether any cell can be a confident match depends on `norm`: a **larger** `node_normalization`
widens the clip window, so more cells land at exact 0/1, so more leaves get a confident
match, so less of the pull-to-0.5 survives. A **smaller** value keeps more cells graded;
since `heat` is a product across regulators, several graded values multiply toward the
middle for nearly every leaf, so more of the network gets pulled toward 0.5 regardless of
true sample count.

**Tested result (run `6668`, `norm=0.4` vs. `6667`'s `0.3`)**: raising it did **not**
improve external validation — mean R² was flat-to-worse across all 5 external validation
sets, including the target diagnostic (`organoid_shGFP` mean R² 0.0232 -> 0.0040). At the
per-gene level the effect is a wash: some genes improved substantially (`NFIB` +0.19,
`TCF7L1` +0.16, `JUN` +0.15), others got much worse (`PKNOX2` -0.28, `RORA_RORB` -0.22,
`BACH2` -0.19). **Takeaway: this parameter alone is not the lever that fixes null-condition
underperformance** — it reshuffles which genes validate well rather than raising the
floor. See §6 for the more targeted fix this result motivates.

**How to choose it for a new dataset**: there's no dataset-size-independent "right"
value — it should track how bimodal/switch-like the genes in the network actually are
(§5). A quick empirical check: after loading with a candidate `norm`, look at the
fraction of exactly-0/exactly-1 entries in the binarized data; going from ~60% (at 0.3)
to ~80% (at 0.4) is a large swing in how much of the fit leans on the graded-cell
mechanism above. Sweep 2-3 values the way `6667`/`6668` did, and check external
validation, not just in-sample AUC (in-sample validation reflects the same clipping
choice that generated the training labels and doesn't guard against this in either
direction).

## 2. `node_threshold` (`threshold=` in `get_rules`)

**What it does**: prunes a regulator whose max ON-vs-OFF leaf-probability swing
(`max_dif`, `detect_irrelevant_regulator`, `tl.py:52`) falls below the threshold.
Currently `0` everywhere in this repo (no pruning). This is independent of §1's
mechanism — it decides which regulators survive the fit at all, not how confidently
their combined effect is estimated once they're in.

**Bias/variance tradeoff**: raising it removes regulators whose apparent effect might be
noise (reducing overfitting / truth-table sparsity, since each additional regulator
doubles the truth table), at the risk of dropping a real but modest regulatory
relationship. With 53 nodes and up to ~5-6 regulators/gene in this network, most
per-gene truth tables are already reasonably sparse relative to ~7100 training cells, so
raising this from 0 is a lower-priority lever than §1 unless a specific gene's fit looks
like it's driven by a spurious regulator.

## 3. Regulators-per-gene cap (Phase 2 — implemented and tested, `6669`)

**Why it matters**: truth-table size is `2^n` for `n` regulators. Per-leaf data gets
exponentially sparser as `n` grows, which is exactly what feeds the §1 mechanism (fewer
cells per leaf -> more leaves depend on a single confident match). Real GRNs also tend
toward modest per-gene in-degree — low-`K` Kauffman networks show more ordered, robust
dynamics than high-`K` ones — so a cap is biologically motivated, not just a
sparsity patch.

**Sizing heuristic**: choose a cap `k` such that `2^k` stays small relative to the
smaller of {training cell count, minority-class cell count for that gene}. This
project's own numbers as a worked example: ~7100 training cells, `2^5 = 32` leaves ->
~220 cells/leaf on average if evenly distributed (never is — real distributions are
skewed toward a few common leaf combinations), `2^7 = 128` leaves -> ~55 cells/leaf. A
gene with 6+ candidate regulators in a <10k-cell dataset is a candidate for capping to
4-5; the same cap is too aggressive for a 100k-cell dataset. Compute this cap
per-dataset, not as a fixed constant across projects.

**Implementation**: no BoBa-T source change needed — a pre-filter on the network edge
list (`build_capped_network.py`), dropping the lowest-fitted-relevance regulator(s) per
over-cap gene (ranked by 6667's own `strengths.csv`) before `bb.load.load_network`, fully
on the multiome-analysis side.

**Tested result (`6669`, cap=6, applied to the 10 genes that had 7-8 regulators)**:
essentially neutral. In-sample mean R² 0.855 (identical to 6667's 0.855 — no fit quality
lost from removing up to 2 regulators per gene). External validation nearly unchanged
too: `organoid_shGFP` 0.016 (vs. 6667's 0.023), `organoid_shRORB2` 0.162 (vs. 0.162,
essentially identical), `mets_compiled` 0.013 (vs. 0.019) — all within noise of baseline,
neither a clear win nor a loss. **Interpretation**: consistent with §6-9's finding that the
worst external-validation shortfalls are driven by genuine cross-context rewiring, not by
overfitting to sparse high-order truth tables — a capacity-reduction fix like this
wouldn't be expected to move that kind of shortfall, and it didn't. Low-risk to adopt as a
simplification (fewer parameters, same performance) but not a fix for §6-9's core finding.

## 4. Normalization scheme choice, and the "artificial spread" problem (Phase 2 — prototyped and tested, `6670`)

`bb.load.load_data`'s `norm=` also accepts `"gmm"` (2-component Gaussian-mixture soft
assignment) and `"minmax"` (exact min/max rescale) as alternatives to the quantile-clip
scheme in §1. **All three force every gene's own distribution to span the full `[0,1]`
range**, regardless of whether that gene actually varies that much in the data. For a
gene with genuinely low biological variance, this manufactures spread that isn't real
signal — the fitted rule then risks being confident about a distinction the data doesn't
actually support, and validation against external data (where the same gene may have
even less real variance) inherits that overconfidence.

**Design principle for a fix**: the best scheme restricts output to `[0,1]` *without*
requiring every gene's own min and max to hit the boundary. Concretely: reference each
gene's rescaling to a **global** spread statistic (e.g. the median or a fixed percentile
of the per-gene IQRs across the whole network) rather than that gene's own min/max, and
pass the result through a fixed-scale squashing function (e.g. a logistic centered at the
gene's median with a shared slope). A genuinely low-variance gene then stays compressed
near the middle of `[0,1]` instead of being stretched to occupy the same range as a truly
bimodal gene.

**Prototyped (`build_global_reference_norm.py`, `6670`)**: logistic squash referenced to
the median of all 53 genes' raw standard deviations (shared scale), slope `K` swept
empirically (`K=2.0`, an initial guess, collapsed the whole network to ~1% confidently
-binarized cells on average — devastating given §1's `1-max(heat)` mechanism; `K=0.2` kept
the network-wide mean extreme-fraction (0.47) comparable to `node_normalization=0.3`'s
fixed 0.60 while preserving real per-gene differentiation, e.g. `BACH2` 0.04 vs `ZBTB20`
0.96 exact-0/1 fraction, vs. both pinned to exactly 0.60 under the old scheme).

**Tested result**: in-sample mean R² 0.80 (`6670`) vs. 6667's 0.855 — a modest, plausible
cost from an untuned prototype scheme. **Scoring external data against it surfaced a
real, important failure mode**: applying the training data's *own* fixed reference
(median + scale) to a new dataset — the mechanistically "correct" way to use this scheme
across datasets, and the direct answer to "how would you compensate for the
diversity<->R² relationship" — caused several genes' *actual* values to saturate to an
essentially constant 0 or 1 in `organoid_shGFP` (e.g. `EPCAM`: actual std collapsed to
8.5e-8), because organoid's raw baseline for that gene sits systematically offset from
GEMM's median, not just differently spread. With near-zero true variance, R² (dividing by
it) becomes numerically degenerate — one run produced a mean R² of roughly -1.8e14, not a
real signal. **This scheme correctly targets the right mechanism (§6's diversity finding)
but needs more care before it's usable for cross-dataset scoring**: candidates for a
follow-up prototype — robust/winsorized scaling instead of a raw logistic, or excluding
genes with near-zero true variance from R² aggregation and reporting a bounded metric
instead.

## 5. Bimodality / true-spread screening (Phase 2/3)

Before trusting a gene's fitted rule as "this gene is switch-like, regulated the way the
Boolean model assumes," check whether it actually is:
- A **dip-test** statistic (Hartigan's dip test) on the gene's raw (pre-normalization)
  expression distribution — low p-value supports genuine bimodality; a high p-value
  means the "on/off" framing is imposed by normalization rather than data-driven.
- The `"gmm"` normalization scheme's own component separation (distance between the two
  fitted Gaussian means, relative to their pooled variance) as a byproduct diagnostic,
  since it's already fitting exactly this structure.
- A gene's **true spread** (raw expression IQR or a coefficient-of-variation, computed
  pre-normalization) vs. its **post-normalization spread** — a large gap between the two
  is the direct symptom of §4's artificial-spread problem for that specific gene, and is
  a good per-gene confidence flag for any rule the network reports (a rule fit on
  artificially-spread data should be trusted less than one fit on a gene that was
  already naturally bimodal).

Not yet implemented as a standing diagnostic run across the network — would slot in as a
pre-fit QC step, flagging genes for extra scrutiny rather than blocking the fit.

## 6. Domain-shift interpretation

Low external-validation R² isn't automatically a rule-fitting problem — see
`../comparisons/domain_shift_diagnostic_and_organoid_walks/FINDINGS.md` for the full analysis (9 sections,
summarized here). Checked across all 33 of 6667's scored external samples (allografts,
human tumors, organoid variants, mets_compiled), not just organoid:

- A continuous per-cell stress score does **not** robustly explain the worst-validating
  genes (correlations are weak and inconsistent in sign) (§1-2).
- Distance from the training distribution — UMAP-space distance, kNN distance in raw
  expression space, distance to the 8 known GEMM attractors — is at best a weak-to-moderate
  predictor of a sample's mean R² (r=-0.09 to -0.49) (§5). A sample can have a
  perfectly ordinary, GEMM-like average expression profile and still validate badly.
- What actually predicts mean R² well: **how much of a sample's own variance lies along
  GEMM's specific fitted identity axes** (r=0.65, the best predictor found) — not how much
  variance it has overall (r=0.58), and not how far its average state sits from training
  data (§4, §6-7).
- The two worst-validating samples (`mets_compiled`, `organoid_shGFP`) fail for different
  specific reasons: `mets_compiled`'s worst subpopulation (cluster 2) is a
  dissociation-stress technical artifact (§3), while `organoid_shGFP` has real,
  substantial lineage heterogeneity (§8) that simply doesn't align to GEMM's specific
  identity-axis combination.
- **Decisive test (§9)**: hard-binarizing cells into their exact combinatorial regulator
  state and comparing to GEMM's fitted rule value for that *exact* state (removing any
  population-mixture confound) shows organoid_shGFP's disagreement is genuine rewiring,
  not a compositional artifact — the same TF combination really does imply a different
  outcome in organoid vs. GEMM, consistent with real context-dependent TF activity
  (cofactor/chromatin/signaling differences between in vivo tissue and culture), not a
  BoBa-T fitting defect.

**Read**: the shortfall isn't spread evenly across a dataset — it's concentrated in
specific subpopulations or driven by specific rewired relationships, not a uniform
degradation. Before concluding a hyperparameter change should compensate for a validation
set's low aggregate score, check whether that low score reflects a few concentrated
subpopulations or specific rewired edges (§10's checklist) rather than a general BoBa-T
fitting weakness.

## 7. Pseudo-observation weighting: opt-in alternative, default unchanged (Phase 2/3, BoBa-T source)

Spec for a `weighting=` parameter on `get_rules`/`get_rules_scvelo`:
- **Default** (unset, or `weighting="max_heat"`): today's exact behavior —
  `1 - max(heat)` weight at probability 0.5 (`tl.py:324-333`). No prior script or result
  changes unless it explicitly opts in.
- **New option** `weighting="aggregate"`: a Beta-style pseudocount tied to the leaf's
  *total* weighted evidence (`sum(heat[:, leaf])`, or an effective-sample-size version of
  it) rather than its single best-matching cell, with a tunable pseudocount-strength
  constant. This is the more principled version of the exact lever §1's `6668` test
  exercised indirectly through `norm` — where `norm` only changes how many cells *can*
  become confident matches, this changes the rule itself to require aggregate evidence
  rather than one lucky cell.
- **Why deferred, now confirmed**: `6668`'s result (§1) showed the norm-based lever isn't a
  clean fix on its own; §9's leaf-conditional test now confirms why a pseudo-observation
  change wouldn't fix the organoid/mets_compiled shortfall either — the problem isn't weak
  aggregate evidence for an otherwise-correctly-specified leaf, it's that the leaf's *true*
  value genuinely differs by context (real rewiring). No amount of evidence-weighting
  changes what a leaf's fitted value should be when the training and target contexts
  disagree about it. This fix would still be worth having for its original purpose
  (reducing single-cell-driven noise within one context), just not as an extrapolation fix.

## 8. Canalizing-function structure: opt-in mode, not default (Phase 2/3, BoBa-T source)

Spec for an opt-in fitting mode biased toward canalizing Boolean functions (one dominant
regulator can force the gene's output regardless of the rest), grounded in real GRNs'
enrichment for this structure. Explicitly additive: default fitting stays the current
general-truth-table `get_rules` behavior; this would be a separate mode a user opts into
per-gene or per-network, not a replacement. Not yet implemented — deferred alongside §7.

## 9. RORA_RORB gene-name labeling bug — fixed

`get_sklearn_metrics` (`bobaT/tl.py`) used to derive a gene name from each
`accuracy_plots/*.csv` filename by splitting on the first underscore, truncating the
combined `RORA_RORB` node to `"RORA"` in every summary-stats table. Fixed directly in
BoBa-T source (strips the known `_validation.csv` suffix instead of splitting on `_`) —
display-only, doesn't change any fitted value or metric. `6667`'s and `6668`'s
`summary_stats.csv` (in-sample and all 5 external validation sets) have been
regenerated under the fix.

## 10. Pre-extrapolation checklist: will this rule set actually transfer to a new dataset?

Before trusting a fitted rule set's predictions on a genuinely new dataset (a new
cell-culture system, a new species, a new experimental condition), run these checks in
order — cheapest/most diagnostic first. All are generalizations of scripts already built
and run for organoid_shGFP/mets_compiled in `../comparisons/domain_shift_diagnostic_and_organoid_walks/`
(reusable, just point them at the new dataset).

1. **Diversity ratio** (`diagnose_sample_diversity.py`, §6/§7's baseline check): compute
   the new dataset's mean per-gene raw standard deviation relative to the training data's
   own. A ratio well below 1 (e.g. <0.3-0.4, per the samples that validated worst) is an
   early warning that the dataset may not vary enough for R²-style validation to be
   meaningful at all, independent of whether the rules are "right."
2. **Identity-axis variance fraction** (`diagnose_variance_composition.py`, §7 — **the
   single best predictor found**, r=0.65 across 33 samples): fit PCA on the *training*
   data, project the new dataset onto that fixed basis, and compute what fraction of the
   new dataset's *own* variance lands on the training data's top identity PCs. Low
   fraction (<0.3, per the worst-validating samples) is a real warning sign — but **do
   not stop here**: §8-9 showed a dataset can score low on this check while still having
   real, relevant biological heterogeneity that simply isn't aligned to the training data's
   specific axis combination, which this metric cannot distinguish from genuine unfitness.
3. **The decisive check: leaf-conditional agreement** (`diagnose_leaf_conditional_agreement.py`,
   §9). For a handful of the network's genes (start with ones with 4-6 regulators — few
   enough regulators that leaves stay well-populated), hard-binarize (>0.5) the new
   dataset's cells into their exact combinatorial regulator state and compare each
   well-populated leaf's mean *actual* target value to the training rule's fitted value
   for that *exact* leaf. Run the same check on the training data's own held-out test set
   first as a positive control (should show small residuals, ~0.03-0.06 per the GEMM
   check) — this confirms the test itself is working before interpreting the new
   dataset's result. **This is the only check in this list that directly measures whether
   the fitted logic itself transfers**, rather than a proxy (variance overlap, distance)
   that can be fooled by population composition (§9's core finding: marginal correlation
   and even PCA-axis overlap can look bad or good somewhat independently of whether the
   actual conditional relationships hold).
4. **If check 3 shows large, systematic leaf-conditional residuals**: don't try to fix
   this with a `node_normalization`/`node_threshold` hyperparameter sweep (§1's `6668`
   result already showed the norm knob doesn't move this kind of shortfall) — it means
   specific regulatory relationships are genuinely different in the new context, most
   plausibly because TF *activity* (not just measured expression level) depends on
   cofactor/chromatin/signaling context that differs between the training and target
   systems. The rules may still be directionally/qualitatively useful, but shouldn't be
   trusted for genes/leaves that fail this check, and any perturbation predictions
   involving those genes should be flagged as training-context-specific rather than
   general claims.

## 11. Robust-core subnetwork: identifying which edges transfer, not just whether the whole network does (implemented, `diagnose_leaf_conditional_agreement_full.py`)

§9's leaf-conditional test (originally run on 5 genes against `organoid_shGFP` alone) is
directly generalizable into a systematic map of which *specific* parts of the network are
context-robust vs. context-specific, rather than a single pass/fail per external sample.
Relevant to BoBa-T's own claim that Boolean structure lets it predict dynamics/attractors
in unseen conditions: that claim is likely true for some edges and not others, and this
gives a way to say which.

**Design**:
1. Run `diagnose_leaf_conditional_agreement.py`'s core check (already built) across
   *every* network gene (not just the 5 tested) and across *every* scored external sample
   (not just `organoid_shGFP`) — both the well-validating ones (`allograft_TKO-luc`,
   `human_RU1311`) and the poorly-validating ones (`organoid_shGFP`, `mets_compiled`,
   `human_RU1215`), to see whether the same specific edges are the ones that fail
   everywhere, or whether different samples rewire different parts of the network.
2. For each gene, compute a transferability score: the cell-count-weighted fraction of its
   well-populated leaves (across all tested samples) with small residual (<0.15, per §9's
   threshold) against the training rule value.
3. Genes/edges with high transferability across most/all tested contexts form a **robust
   core** — the part of the network where a rule fit on GEMM is likely to actually hold in
   a new, previously-unseen context, and where claims about predicting unseen-cluster
   dynamics or attractor structure are best supported. Genes/edges that fail
   context-specifically (fail in some samples, hold in others) point to *what kind* of
   context shift matters for that specific regulatory relationship — worth cross-referencing
   against sample metadata (culture vs. tissue, tumor subtype, technical depth) the way §3
   and §8 did for specific cases.
4. **Done, at full scale (53 genes x 33 external samples)**: see
   `../comparisons/domain_shift_diagnostic_and_organoid_walks/FINDINGS.md` §10. Headline result: per-sample mean
   transferability correlates with that sample's mean R² at **r=0.96** — the strongest
   predictor found across this entire investigation, as expected since it directly
   measures rule-holds-in-this-sample with the composition confound removed by
   construction. **Caveat that matters for reading the per-gene ranking**: the network's
   11 self-loop-only "source" nodes score a trivial 1.0 (a self-loop predicts a gene
   largely from its own level, which "transfers" anywhere) and must be excluded before
   ranking genes — same reason the existing pipeline already excludes them from averaged
   ROC plots. Among real (>=2-regulator) genes, even the single most-transferable one
   (`TEAD1`) only reaches 0.63 across the *full*, deliberately heterogeneous 33-sample
   population (mouse allografts + human tumors + mouse organoid culture together) — no
   edge is universally robust across that much biological diversity, which is a realistic
   ceiling, not a bug. Least-transferable genes (`NFIX`, `JUNB`, `KMT2A`, `TCF7L2`, `FOS`,
   `LMX1B`, `MEIS2`, `EHF`, `JUN`, `ASCL1`) substantially overlap with the genes flagged
   throughout this investigation as worst-validating/most-rewired — a good consistency
   check. The full per-gene x per-sample matrix supports re-aggregating over any sample
   subset relevant to a specific claim (e.g. in-vivo-only for a more permissive "robust
   core for in-vivo extrapolation" ranking).
