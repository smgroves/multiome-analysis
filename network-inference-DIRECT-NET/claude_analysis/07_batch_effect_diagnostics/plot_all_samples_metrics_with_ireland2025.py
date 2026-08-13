"""Remake claude_analysis/03_domain_shift_diagnostics/plot_all_samples_metrics_by_category.py's
R2/AUC (and F1) jitter-by-category plots with the 5 new Ireland et al. 2025 samples added as
a 6th category. Imports that script's exact plotting logic (`plot_metric`) and constants
(self-loop gene list, colors) rather than reimplementing them, so the two are guaranteed to
render identically apart from the added category.

Sample selection, following the SAME "exclude the combined aggregate, include its condition
splits" convention already used for organoid (`organoid_combined` excluded there since it's
literally organoid_shGFP+shRORB1+shRORB2 merged, not an independent sample): uses the 11
Ireland condition-level splits, not the 5 whole-dataset combined versions, since each whole
dataset (e.g. `cgrp_k5`) is exactly its own condition splits merged (e.g. `cgrp_k5_K5` +
`cgrp_k5_CGRP`) and would double-count the same underlying cells' signal if both were shown.

No scrambled-data null exists yet for the Ireland samples (would need
run_scrambled_null_validation.py rerun for them, out of scope here) -- their category simply
has no null diamond, which plot_metric already handles gracefully per-category.

Run in bobaT_env_py3.13 (same env as the original script):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/07_batch_effect_diagnostics/plot_all_samples_metrics_with_ireland2025.py
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_domain_shift_diagnostics"))
import plot_all_samples_metrics_by_category as base

DIR_PREFIX = base.DIR_PREFIX
OUT_DIR = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/comparisons/ireland2025_external_validation"

IRELAND_SAMPLES = [
    "cgrp_k5_K5", "cgrp_k5_CGRP",
    "organoid_celltag_RPM", "organoid_celltag_RPMA", "organoid_celltag_WT",
    "tbo_allograft_5khvg_RPM_CTpostCre", "tbo_allograft_5khvg_RPM_CTpreCre",
    "rpr2_allograft_RPM", "rpr2_allograft_RPR2",
    "celltag_fate_dpt_RPM", "celltag_fate_dpt_RPMA",
]

CATEGORY_ORDER = base.CATEGORY_ORDER + ["ireland_2025"]
CATEGORY_COLORS = {**base.CATEGORY_COLORS, "ireland_2025": "tab:brown"}


def ireland_summary_stats_path(name):
    p = f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats_fixed.csv"
    return p if os.path.exists(p) else f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"


def load_combined_metrics_table(exclude_selfloops):
    df = base.load_metrics_table(exclude_selfloops=exclude_selfloops)

    rows = []
    for name in IRELAND_SAMPLES:
        stats = pd.read_csv(ireland_summary_stats_path(name))
        if exclude_selfloops:
            stats = stats[~stats["gene"].isin(base.SELF_LOOP_GENES)]
        rows.append({
            "name": name, "category": "ireland_2025",
            "mean_r2": stats["r2"].mean(), "mean_f1": stats["f1"].mean(), "mean_auc": stats["roc_auc_score"].mean(),
        })
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def run_comparison(exclude_selfloops):
    tag = "excl_selfloops" if exclude_selfloops else "incl_selfloops"
    gene_desc = "42 non-self-loop network genes" if exclude_selfloops else "all 53 network genes (11 self-loop)"
    print(f"\n=== {tag} ({gene_desc}) ===")

    df = load_combined_metrics_table(exclude_selfloops=exclude_selfloops)
    out_csv = f"{OUT_DIR}/all_samples_metrics_by_category_with_ireland2025_{tag}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(df.groupby("category")[["mean_r2", "mean_f1", "mean_auc"]].agg(["count", "median"]))

    null_df = base.load_aggregate_null() if not exclude_selfloops else base.load_pergene_null(exclude_selfloops=True)

    # Patch base module's globals so plot_metric (imported, unmodified) picks up the new
    # category/color -- plot_metric reads these as module-level names, not parameters.
    base.CATEGORY_ORDER = CATEGORY_ORDER
    base.CATEGORY_COLORS = CATEGORY_COLORS
    base.OUT_DIR = OUT_DIR

    suffix = "" if not exclude_selfloops else "_excl_selfloops"
    base.plot_metric(df, null_df, "mean_r2", f"Mean R² (per-sample, averaged across {gene_desc})", None, f"all_samples_r2_by_category_with_ireland2025{suffix}")
    base.plot_metric(df, null_df, "mean_auc", f"Mean ROC AUC (per-sample, averaged across {gene_desc})", 0.5, f"all_samples_auc_by_category_with_ireland2025{suffix}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_comparison(exclude_selfloops=False)
    run_comparison(exclude_selfloops=True)


if __name__ == "__main__":
    main()
