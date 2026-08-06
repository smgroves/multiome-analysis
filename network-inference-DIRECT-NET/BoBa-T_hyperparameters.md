# BoBa-T hyperparameters: a decision guide

Reference for choosing BoBa-T's fitting hyperparameters for a given dataset. Written
from the investigation into why shGFP organoid external validation (run `6667`) wasn't
as good as expected for a null/control condition — see
`/Users/xpz5km/.claude/plans/it-is-looking-like-elegant-patterson.md` for the full
narrative and `comparisons/6667_vs_6668_norm_sweep/` /
`comparisons/domain_shift_diagnostic/FINDINGS.md` for the actual results referenced
below. Covers what's implemented today and what's designed-but-deferred; each deferred
item is written as a full spec so the reasoning and the eventual code stay in sync.

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

## 3. Regulators-per-gene cap (Phase 2/3 — not yet implemented)

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

**Implementation note (not yet built)**: this doesn't require a BoBa-T source change —
it's a pre-filter on the network edge list (drop the lowest-strength incoming edges per
gene, e.g. by DIRECT-NET/LASSO edge weight) applied before `bb.load.load_network`, fully
on the multiome-analysis side.

## 4. Normalization scheme choice, and the "artificial spread" problem (Phase 2/3)

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
bimodal gene. Not yet prototyped — a natural Phase 2/3 companion to §5's screening, since
you'd want the bimodality/spread check in place first to know which genes this would
actually change the fit for.

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
`comparisons/domain_shift_diagnostic/FINDINGS.md` for the full analysis. Summary: a
continuous per-cell stress score does **not** robustly explain organoid/mets_compiled's
worst-validating genes (correlations are weak and inconsistent in sign). But grouping by
discrete cell-type/cluster identity finds a much sharper signal — mets_compiled's cluster
`2` (1221 cells, not a small/noisy group) is the single highest-residual cluster for 7 of
its 10 worst-validating genes. **Read**: the shortfall isn't spread evenly across a
dataset — it's concentrated in specific subpopulations that are either genuinely
out-of-distribution relative to the GEMM training data, or represent a real biological
state the current regulator set doesn't capture. Before concluding a hyperparameter
change should compensate for a validation set's low aggregate score, check whether that
low score is actually a few concentrated subpopulations dragging down an otherwise-solid
fit.

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
- **Why deferred**: `6668`'s result (§1) showed the norm-based lever isn't a clean fix on
  its own, which argues for waiting to see whether §6's domain-shift read changes the
  interpretation before investing in a BoBa-T source change here — if the shortfall is
  concentrated in specific out-of-distribution subpopulations rather than spread by weak
  aggregate evidence generally, this fix would help less than it might first appear to.

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
