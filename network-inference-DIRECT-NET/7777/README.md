# Barcode 7777: aggregate-evidence pseudocount prior

## Status: paused, not adopted

**Bottom line: once a real setup bug was fixed, this idea showed no meaningful benefit on
the datasets that matter most (Ireland et al. 2025), and the investigation was paused at
the user's call rather than continued.** The early "43/43 samples improved" / "46/53 genes
improved in-sample" headline numbers reported during development were real computations,
but turned out to be almost entirely an artifact of a network-loading mismatch, not a
genuine property of the prior. See "What actually happened" below for the full trail —
kept in detail because the failure mode (and how it was caught) is worth remembering for
any future rule-fitting experiment on this or another network.

## What this is

An attempt to refit the 6667 network's rules on **GEMM's own training data** (identical
train/test split and network as 6667), changing exactly one thing: the pseudo-observation
prior used to smooth under-evidenced combinatorial leaves during rule fitting.

**No external validation data was used to fit these rules.** Only `6667/data_split/
train_t0combined.csv` (the same GEMM training cells 6667 was fit on) was used.

### Provenance note

This barcode number was originally requested for a different idea ("refit rules with a
transfer-regularized objective — penalize disagreement with held-out external data during
fitting"). That idea was substituted, mid-development, for the prior-change idea below
(prompted by a separate discussion about `BoBa-T_hyperparameters.md` §7's stubbed-but-never-
executed "aggregate" pseudocount alternative). The transfer-regularized version was never
built. This substitution wasn't flagged clearly at the time it happened — noted here so the
history is traceable.

## The method

BoBa-T's rule fitting (`bobaT.tl.get_rules`) estimates each leaf of a gene's combinatorial
truth table as a heat-weighted average of that gene's real training-cell values, then adds
a pseudo-observation to smooth leaves with little support: **every leaf gets a
pseudo-observation of weight `1 - max(heat[:, leaf])` at probability 0.5.**

This only looks at the single best-matching cell for that leaf. A leaf with 500 cells each
at heat=0.4 (no single confident match, but substantial collective support) gets pulled
toward 0.5 exactly as hard as a leaf with zero supporting cells at all — aggregate evidence
across many moderately-supporting cells is ignored.

**The change**: replace the pseudo-observation weight with a Beta-style pseudocount tied to
the leaf's *total* weighted evidence instead of its single best cell:

```
total_heat = sum(heat[:, leaf])          # across all training cells
pseudo_weight = C / (C + total_heat)     # C = 1.0
```

`C=1.0` means a leaf needs roughly one fully-confident-cell's worth of aggregate evidence
before the prior's pull meaningfully fades. Well-populated leaves (many confident-equivalent
cells) get a near-negligible pull toward 0.5; genuinely under-observed leaves still get
pulled hard, same as before.

Now implemented as an actual opt-in option in the installed `bobaT` package itself
(`bobaT.tl.get_rules(..., pseudocount_mode="aggregate", pseudocount_c=1.0)`), default
`pseudocount_mode="max_heat"` reproduces the original behavior exactly, byte-for-byte.
(`../claude_analysis/08_alternative_prior_7777/fit_rules_aggregate_prior.py` predates that
option and reimplements the same math standalone — kept for its diagnostic/comparison code,
but the package option is now the reference implementation.)

## What actually happened (the full trail)

### 1. Initial result looked dramatic

First fit (`fit_rules_aggregate_prior.py` v1) reported: mean |diff| in rule values across
genes only 0.0022 (tiny), yet 46/53 genes improved in-sample against GEMM's held-out test
set (mean R² 0.855→0.905) and 43/43 external-validation samples improved (mean R²
0.202→0.375, more than doubled). This was independently re-audited (fitting code traced
line-by-line against `bobaT.tl.get_rules`, file timestamps checked, a per-cell spot check
confirmed the "actual" ground truth was byte-identical between runs and only "predicted"
values differed) and looked solid.

### 2. A simple question exposed the real story

Asked directly: *"how much do the rules actually change?"* Checking regulator sets (not
just rule values) between 6667 and 7777 showed 37/53 genes had a *different* regulator list
— far more than "rules barely moved." Splitting that down further: 24/53 were just
reordered (behaviorally inert — the rule array is re-permuted to match, so predictions for
a given regulator combination are unchanged), but **13/53 genuinely gained a new
regulator: themselves.** All 13 (`BACH1, BACH2, EHF, FOXO3, GRHL2, HES1, NFIA, PBX1, REST,
RUNX1, SIX1, TCF7L2, THRB`) picked up a self-loop. The four largest reported in-sample
gains — `BACH2, PBX1, THRB, EHF` — were all four among these 13. Not a coincidence: a gene
predicting itself from its own already-known state is a trivially easy task (this project
already knew self-loops inflate R²/AUC, from 6667's own 11 pre-existing self-loop genes).

A follow-up test (`test_selfloop_contribution.py`: marginalize the self-term back out of
these 13 genes' rules, holding the rest of the fit fixed, rescore in-sample) was decisive:
**for all 13/13 genes, removing the self-term didn't just erase the gain — it left the fit
worse than 6667's original rule** (134%–966% of the reported gain was the self-term alone).

### 3. Root cause: not the prior, a network-loading mismatch

The next question — *"why would 6667 have pruned the self-loop but 7777 not?"* — turned out
to have a much simpler answer than "the new prior changes what survives pruning." Checking
`bobaT.load.load_network`'s `remove_selfloops` flag directly: 6667's original fitting script
(`main_all_data_remove_selfloops_6667.py`) calls it with `remove_selfloops=True`, which
strips self-edges (`source==target`) from the graph *before* fitting ever starts — the gene
never has itself as a regulator candidate at all. The first version of
`fit_rules_aggregate_prior.py` called it with `remove_selfloops=False` by mistake (a plain
setup error, unrelated to the pseudocount idea being tested) — so these 13 genes got a
self-candidate in 7777 that 6667 never had access to in the first place. There was never an
iterative "pruning battle" the new prior won and the old one lost; it was two differently
configured networks being compared as if they were the same one.

### 4. Corrected refit: clean regulator structure, negligible effect

Fixed `fit_rules_aggregate_prior.py` to use `remove_selfloops=True` (matching 6667 exactly)
and refit. Result: **zero genes now have a genuinely different regulator set.** All 53
genes have the identical regulator set to 6667; 22 also have identical order (mean |diff|
in rule values 0.0025, same tiny scale as before); the other 31 are reordered only
(behaviorally inert, as established above). This is the correct, confound-free comparison —
same causal structure throughout, only fitted probability values shift, and only slightly.

**Partial re-validation against the Ireland et al. 2025 external samples** (flagged as the
most important test set; checked smallest-first for fast feedback, 5 of 11 samples
completed before the investigation was paused):

| sample | n cells | R² mean (6667→7777) | R² median (6667→7777) | AUC (6667→7777) |
|---|---|---|---|---|
| organoid_celltag_RPM | 1191 | 0.266 → 0.264 | 0.319 → 0.318 | 0.767 → 0.767 |
| tbo_allograft_5khvg_RPM_CTpreCre | 1599 | 0.207 → 0.203 | 0.217 → 0.214 | 0.740 → 0.739 |
| organoid_celltag_WT | 1767 | 0.315 → 0.314 | 0.385 → 0.386 | 0.790 → 0.790 |
| tbo_allograft_5khvg_RPM_CTpostCre | 2836 | 0.086 → 0.078 | 0.076 → 0.080 | 0.691 → 0.691 |
| organoid_celltag_RPMA | 6131 | 0.229 → 0.226 | 0.240 → 0.240 | 0.753 → 0.754 |

AUC is flat within noise (differences of 0.0001–0.001) and R² is flat to marginally
*worse*, not improved, on every sample checked. **This is the honest, confound-free
picture: once the self-loop artifact is removed, the aggregate-evidence prior does not show
a real benefit on the external data that matters most.**

## Current state / what's not done

Paused here at the user's call once the Ireland results came back unpromising:
- The remaining 6 (larger) Ireland samples were not scored.
- The full 43-sample external-validation re-run (`main_validate_7777_all_samples.py`) and
  the in-sample re-check (`check_in_sample_7777_vs_6667.py`) were **not** re-run against the
  corrected rules — the old output files (`7777_vs_6667_full_validation.csv`,
  `in_sample_6667_vs_7777.csv`, and the two per-sample dumbbell plots) reflect the **old,
  self-loop-confounded** `rules_7777.txt`, not the corrected one now on disk. Treat those
  files and plots as historical/superseded, not current results.
- `7777/rules/rules_7777.txt` on disk **is** the corrected (`remove_selfloops=True`) fit.

## Recommendation

Do not adopt this prior as a default or route it into other benchmarks (e.g. HSC vs.
CellOracle) based on this investigation — the corrected, confound-free result did not show
a real improvement on the most important external data. The `pseudocount_mode="aggregate"`
option added to `bobaT.tl.get_rules` is harmless to keep (default behavior is unchanged),
but there's no evidence yet that it's worth using. If anyone picks this up again: rerun the
Ireland comparison to completion first (cheap, and it's the test that mattered), and always
check a network for self-edge candidates before trusting any pruning-sensitive rule-fitting
change on it.

## How this relates to the other two barcodes

**7778 and 7779 do not include this prior change.** Both are built on 6667's *original*
rules — they test data-side corrections (rescaling/marginalizing scale-mismatched genes in
new external samples), independent of this fitting-side experiment. As of this writing 7778
and 7779 are still running (separate, ongoing work — see task #20).
