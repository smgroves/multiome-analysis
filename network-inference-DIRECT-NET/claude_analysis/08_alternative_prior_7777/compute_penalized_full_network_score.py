"""Post-processing: compute an honest "penalized full-network" score for 7778/7779
alongside the restricted, survivors-only score already in
7778_vs_6667_full_validation.csv / 7779_vs_6667_full_validation.csv.

Why this exists: both 7778 and 7779 exclude some genes from scoring per sample (7778 via
marginalization -- a collapsed gene is dropped as both regulator and target; 7779 via the
target-exclusion fix -- a collapsed gene's pole-shifted value is only used as a regulator
input, never as its own ground truth). Comparing their mean R2/AUC over that SURVIVING
subset against 6667's baseline restricted to the SAME subset is internally consistent, but
it silently narrows the question from "did this method improve generalization" to "did it
improve on the genes that already had cooperative data" -- a method that just declines to
score its hardest genes would look artificially better by this measure alone, without
actually being more useful. Real user feedback caught this directly.

Fix 1: for every one of the 53 network genes NOT scored in a given sample, credit it with a
NEUTRAL score (r2=0, the "predict the mean" baseline; roc_auc_score=0.5, chance) instead of
omitting it, then average over all 53 genes every time -- same denominator as 6667's
baseline (which always scores all 53), so an "improvement" that only survives by excluding
hard genes shows up honestly as a penalty, not a hidden gain.

Fix 2 (also caught by real user feedback, a second real bug/limitation, not the same one as
Fix 1): a single near-zero-real-variance gene can make R2's MEAN across genes wildly
unrepresentative even when that gene's underlying data is completely honest -- e.g.
allograft_mt3 under 7779: THRB's actual values are genuinely real (1469 distinct values,
std=0.013 -- not manufactured, not collapsed to a constant) but its predicted values deviate
enough that R2 = -388 for that one gene alone, dragging the sample's MEAN r2 from a
reasonable 0.11 down to -7.2, even though every other gene (52/53) is essentially unaffected
and the MEDIAN r2 (0.114) matches baseline almost exactly. This isn't a bug in the rescaling
-- it's R2's well-known mathematical instability when the true response has very small but
nonzero variance (small SS_tot in the denominator amplifies any residual). Using MEDIAN
instead of MEAN as the primary per-sample R2 summary (computed AFTER Fix 1's neutral-credit
padding, so both fixes combine into one robust statistic) fixes this without needing to
hand-pick a "how small is too small" variance threshold. AUC doesn't have this problem
(bounded [0,1], and sklearn's roc_auc_score already returns NaN -- correctly excluded from
the mean by pandas' default skipna -- when the ground truth is genuinely single-class), so
mean is kept for AUC.

Run in bobaT_env_py3.13 (no bobaT calls needed, just reads already-written CSVs):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/compute_penalized_full_network_score.py
"""

import glob
import os

import numpy as np
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
RULES_PATH = f"{DIR_PREFIX}/6667/rules/rules_6667.txt"
NEUTRAL_R2 = 0.0
NEUTRAL_AUC = 0.5


def load_all_nodes():
    nodes = []
    with open(RULES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def penalized_score(brcd):
    all_nodes = set(load_all_nodes())
    val_dirs = glob.glob(f"{DIR_PREFIX}/{brcd}/validation/external_validation/*/summary_stats_fixed.csv")
    rows = []
    for f in val_dirs:
        name = os.path.basename(os.path.dirname(f))
        stats = pd.read_csv(f)
        scored_genes = set(stats["gene"])
        excluded = all_nodes - scored_genes
        # Restricted (survivors-only) mean/median -- what the earlier comparison reported
        # (mean), plus median (Fix 2) for robustness to a single degenerate-variance gene.
        restricted_r2 = stats["r2"].mean()
        restricted_r2_median = stats["r2"].median()
        restricted_auc = stats["roc_auc_score"].mean()
        # Penalized (full-53-gene, neutral-credit-for-excluded) mean AND median (Fix 1 + 2
        # combined: pad with neutral credit for excluded genes, THEN take the median of all
        # 53 values so neither an excluded gene NOR a single degenerate-variance survivor
        # can dominate the summary).
        r2_padded = np.concatenate([stats["r2"].values, np.full(len(excluded), NEUTRAL_R2)])
        auc_padded = np.concatenate([stats["roc_auc_score"].dropna().values, np.full(len(excluded), NEUTRAL_AUC)])
        penalized_r2 = r2_padded.mean()
        penalized_r2_median = np.median(r2_padded)
        penalized_auc = auc_padded.mean()
        rows.append({
            "name": name, "n_scored": len(scored_genes), "n_excluded": len(excluded),
            "restricted_r2": restricted_r2, "restricted_r2_median": restricted_r2_median,
            "penalized_r2": penalized_r2, "penalized_r2_median": penalized_r2_median,
            "restricted_auc": restricted_auc, "penalized_auc": penalized_auc,
        })
    return pd.DataFrame(rows)


def gather_baseline_full(name):
    """6667's baseline over its full scored gene set (always ~53, no exclusion)."""
    if name.startswith("allograft_"):
        c = f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"
    elif name.startswith("human_"):
        c = f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv"
    elif name == "organoid_combined":
        c = f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv"
    else:
        c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
        if not os.path.exists(c):
            c = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"
    if not os.path.exists(c):
        return None, None, None
    df = pd.read_csv(c)
    return df["r2"].mean(), df["r2"].median(), df["roc_auc_score"].mean()


def main():
    for brcd in ["7778", "7779"]:
        df = penalized_score(brcd)
        if df.empty:
            print(f"{brcd}: no results yet")
            continue
        df["baseline_r2_full"], df["baseline_r2_full_median"], df["baseline_auc_full"] = zip(*df["name"].map(gather_baseline_full))
        out_path = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777/{brcd}_penalized_vs_baseline.csv"
        df.to_csv(out_path, index=False)

        valid = df.dropna(subset=["baseline_r2_full"])
        pd.set_option("display.width", 160)
        print(f"\n=== {brcd} ({len(df)} samples scored so far) ===")
        print(df[["name", "n_scored", "n_excluded", "restricted_r2_median", "penalized_r2_median", "restricted_auc", "penalized_auc"]].to_string(index=False))
        if len(valid):
            print(f"\n--- Headline (MEDIAN-based, robust to single-gene R2 blowups) ---")
            print(f"Mean baseline (full 53-gene) median-R2: {valid['baseline_r2_full_median'].mean():.4f}")
            print(f"Mean {brcd} penalized (full 53-gene, neutral-credit) median-R2: {valid['penalized_r2_median'].mean():.4f}")
            print(f"Datasets improved on PENALIZED MEDIAN R2 vs baseline: {(valid['penalized_r2_median'] > valid['baseline_r2_full_median']).sum()}/{len(valid)}")
            print(f"\n--- For reference (MEAN-based, vulnerable to single-gene R2 blowups -- see module docstring Fix 2) ---")
            print(f"Mean baseline (full 53-gene) R2: {valid['baseline_r2_full'].mean():.4f}")
            print(f"Mean {brcd} penalized (full 53-gene, neutral-credit) R2: {valid['penalized_r2'].mean():.4f}")
            print(f"Mean {brcd} restricted (survivors-only) R2: {valid['restricted_r2'].mean():.4f}")
            print(f"\n--- AUC (mean, NaN-safe by construction) ---")
            print(f"Mean baseline (full 53-gene) AUC: {valid['baseline_auc_full'].mean():.4f}")
            print(f"Mean {brcd} penalized (full 53-gene, neutral-credit) AUC: {valid['penalized_auc'].mean():.4f}")
            print(f"Mean {brcd} restricted (survivors-only) AUC: {valid['restricted_auc'].mean():.4f}")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
