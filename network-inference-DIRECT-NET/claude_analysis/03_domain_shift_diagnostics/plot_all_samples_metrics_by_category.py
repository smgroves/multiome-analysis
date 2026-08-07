"""Grouped jitter-by-sample-type summary for all three per-gene validation metrics BoBa-T's
summary_stats.csv carries: R^2, F1, and ROC AUC. One point per sample, one row per sample
type (allograft / TKO / human_tumor / organoid / mets_compiled), each sample's own per-gene
metric averaged across all 53 network genes.

SELF-LOOPS ARE INCLUDED, NOT DROPPED: 11 of the 53 network genes are their own regulator
(TFDP1, NFYC, CREB1, TCF4, ZEB1, ESR1, STAT1, RBPJ, JUND, NR6A1, SOX9 -- confirmed via
regulators_dict from rules_6667.txt), and summary_stats.csv scores all 53, including these
11, matching the production validation scripts (main_external_validation_organoid_mets.py
etc., which all load the network with remove_selfloops=False). This matters because
self-loop genes trivially inflate a gene's own score (its scrambled/permuted value still
partially predicts itself through the self-referential regulator link -- see
run_scrambled_null_validation.py's self-loop diagnostic: null AUC=1.0/R²=0.94 for the 11
self-loop genes vs. AUC=0.51/R²=-0.33 for the other 42). The real vs. null comparison here
is still fair (both include the same 11 genes), but see the companion "excluding
self-loops" plots for the more conservative 42-gene-only version.

Reuses all_samples_corr_vs_r2.csv (comparisons/domain_shift_diagnostic_low_R2_samples/,
built by compare_correlation_shift_across_samples.py) for the name/category/mean_r2 columns
and sample filtering (organoid_combined excluded as an aggregate; samples with no
summary_stats.csv on disk dropped) -- then re-derives the same summary_stats.csv path per
sample to also pull mean F1 and mean ROC AUC, so all three metrics are guaranteed to cover
the identical 32-sample set.

Run in bobaT_env_py3.13 (no bobaT import needed, just pandas/matplotlib):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/plot_all_samples_metrics_by_category.py
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
IN_CSV = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples/all_samples_corr_vs_r2.csv"
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples"
# Scrambled-data null (see run_scrambled_null_validation.py): each network gene's values
# independently permuted across cells, then re-scored with the same fitted rules. Pools
# every scrambled_null_*.csv on disk -- for allograft/human_tumor/organoid this spans
# multiple distinct real samples per category (not just repeated permutations of one),
# so the null reflects both scrambling noise and real inter-sample variation.
NULL_CSV_GLOB = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples/scrambled_null_*.csv"
NULL_CSV_COMBINED = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_low_R2_samples/scrambled_null_all.csv"

CATEGORY_ORDER = ["allograft", "TKO", "human_tumor", "organoid", "mets_compiled"]
CATEGORY_COLORS = {
    "allograft": "tab:blue", "TKO": "tab:purple", "human_tumor": "tab:orange",
    "organoid": "tab:green", "mets_compiled": "tab:red",
}
LABEL_SAMPLES = {"organoid_shGFP", "mets_compiled"}
RNG_SEED = 0
JITTER = 0.16

# Genes that are their own regulator in rules_6667.txt (confirmed via regulators_dict) --
# see run_scrambled_null_validation.py's self-loop diagnostic for why this matters.
SELF_LOOP_GENES = {"TFDP1", "NFYC", "CREB1", "TCF4", "ZEB1", "ESR1", "STAT1", "RBPJ", "JUND", "NR6A1", "SOX9"}

# organoid_shGFP_D_Sample* / allograft_mt5 etc. -> the sample's category, mirroring
# run_scrambled_null_validation.py's CATEGORY_OF (needed to tag the pergene null files).
NULL_SAMPLE_CATEGORY = {
    "allograft_mt5": "allograft", "allograft_mt2": "allograft", "allograft_1L": "allograft",
    "allograft_TKO-luc": "TKO",
    "human_RU1293": "human_tumor", "human_RU1215": "human_tumor", "human_RU1311": "human_tumor",
    "organoid_shGFP_D_Sample4": "organoid", "organoid_shGFP_D_Sample1": "organoid", "organoid_shGFP_D_Sample6": "organoid",
    "mets_compiled": "mets_compiled",
}

METRICS = [
    # (summary_stats.csv column, output column name, axis label, chance/reference line)
    ("r2", "mean_r2", "Mean R² (per-sample, averaged across all 53 network genes)", None),
    ("f1", "mean_f1", "Mean F1 (per-sample, averaged across all 53 network genes)", None),
    ("roc_auc_score", "mean_auc", "Mean ROC AUC (per-sample, averaged across all 53 network genes)", 0.5),
]


def summary_stats_path(name):
    if name.startswith("allograft_"):
        return f"{DIR_PREFIX}/6667/validation/allografts/{name[len('allograft_'):]}/summary_stats.csv"
    if name.startswith("human_"):
        return f"{DIR_PREFIX}/6667/validation/human_tumor_MSK/{name[len('human_'):]}/summary_stats.csv"
    if name == "organoid_combined":
        return f"{DIR_PREFIX}/6667/validation/external_validation/organoid/summary_stats.csv"
    return f"{DIR_PREFIX}/6667/validation/external_validation/{name}/summary_stats.csv"


def load_metrics_table(exclude_selfloops=False):
    df = pd.read_csv(IN_CSV)
    df = df[df["name"] != "organoid_combined"]
    df = df.dropna(subset=["mean_r2"]).reset_index(drop=True)

    r2s, f1s, aucs = [], [], []
    for name in df["name"]:
        stats = pd.read_csv(summary_stats_path(name), index_col=0)
        if exclude_selfloops:
            stats = stats[~stats["gene"].isin(SELF_LOOP_GENES)]
        r2s.append(stats["r2"].mean())
        f1s.append(stats["f1"].mean())
        aucs.append(stats["roc_auc_score"].mean())
    df["mean_r2"] = r2s
    df["mean_f1"] = f1s
    df["mean_auc"] = aucs
    return df[["name", "category", "mean_r2", "mean_f1", "mean_auc"]]


def load_aggregate_null():
    """Every scrambled_null_<sample>[_suffix].csv on disk (aggregate mean_r2/f1/auc per
    iteration, all 53 genes) -- excludes the per-gene detail files, which have a different
    schema and are handled separately by load_pergene_null()."""
    files = [f for f in glob.glob(NULL_CSV_GLOB) if "pergene" not in f and not f.endswith("_all.csv")]
    if not files:
        return pd.DataFrame(columns=["sample", "category", "iteration", "mean_r2", "mean_f1", "mean_auc"])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(NULL_CSV_COMBINED, index=False)
    return df


def load_pergene_null(exclude_selfloops):
    """Combine every scrambled_null_pergene_<sample>_iter<i>.csv (one row per gene, from
    run_scrambled_null_validation.py --save-pergene) into one mean_r2/f1/auc per
    sample-iteration, optionally excluding the 11 self-loop genes first."""
    files = glob.glob(f"{OUT_DIR}/scrambled_null_pergene_*.csv")
    rows = []
    for f in files:
        base = os.path.basename(f).removeprefix("scrambled_null_pergene_").removesuffix(".csv")
        sample, _, iter_tag = base.rpartition("_iter")
        if sample not in NULL_SAMPLE_CATEGORY:
            continue
        stats = pd.read_csv(f)
        if exclude_selfloops:
            stats = stats[~stats["gene"].isin(SELF_LOOP_GENES)]
        rows.append({
            "sample": sample, "category": NULL_SAMPLE_CATEGORY[sample], "iteration": iter_tag,
            "mean_r2": stats["r2"].mean(), "mean_f1": stats["f1"].mean(), "mean_auc": stats["roc_auc_score"].mean(),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["sample", "category", "iteration", "mean_r2", "mean_f1", "mean_auc"])


def plot_metric(df, null_df, col, xlabel, ref_line, out_name):
    rng = np.random.default_rng(RNG_SEED)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    if ref_line is not None:
        ax.axvline(ref_line, color="0.6", linewidth=1.2, linestyle="--", zorder=1)
        ax.text(ref_line, len(CATEGORY_ORDER) - 0.55, "chance level", color="0.5",
                fontsize=8, ha="center", va="top", style="italic")

    for row_i, cat in enumerate(CATEGORY_ORDER):
        sub = df[df["category"] == cat]
        color = CATEGORY_COLORS[cat]
        y = row_i + rng.uniform(-JITTER, JITTER, size=len(sub))
        ax.scatter(sub[col], y, s=42, color=color, edgecolor="black",
                   linewidth=0.5, alpha=0.85, zorder=3)

        median = sub[col].median()
        ax.plot([median, median], [row_i - 0.28, row_i + 0.28], color=color,
                linewidth=2.5, zorder=4, solid_capstyle="round")
        ax.text(median, row_i + 0.34, f"median={median:.2f}", color=color,
                fontsize=8, ha="center", va="bottom", fontweight="bold")

        for pos, (_, r) in zip(y, sub.iterrows()):
            if r["name"] in LABEL_SAMPLES:
                ax.annotate(r["name"], (r[col], pos),
                            xytext=(8, 0), textcoords="offset points",
                            fontsize=8, va="center", color="black",
                            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))

        null_sub = null_df[null_df["category"] == cat]
        if len(null_sub) > 1:
            n = len(null_sub)
            null_mean = null_sub[col].mean()
            sem = null_sub[col].std(ddof=1) / np.sqrt(n)
            ci95 = sp_stats.t.ppf(0.975, df=n - 1) * sem
            ax.plot([null_mean - ci95, null_mean + ci95], [row_i - 0.42, row_i - 0.42], color="0.35",
                    linewidth=2.0, zorder=4, solid_capstyle="round")
            ax.scatter([null_mean], [row_i - 0.42], marker="D", s=30, color="0.35",
                       edgecolor="white", linewidth=0.5, zorder=5)
        elif len(null_sub):
            ax.scatter(null_sub[col], [row_i - 0.42] * len(null_sub), marker="D", s=30,
                       color="0.35", edgecolor="white", linewidth=0.5, zorder=5)

    # single legend entry for the null marker (same for every row, so only add once)
    ax.plot([], [], color="0.35", marker="D", markersize=5, linewidth=2.0,
             label="scrambled-data null (mean ± 95% CI per category -- see note)")
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    ax.set_yticks(range(len(CATEGORY_ORDER)))
    ax.set_yticklabels([f"{cat} (n={(df['category'] == cat).sum()})" for cat in CATEGORY_ORDER], fontsize=11)
    ax.set_ylim(-0.6, len(CATEGORY_ORDER) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=11)
    lo_candidates = [df[col].min()]
    if ref_line is not None:
        lo_candidates.append(ref_line)
    if len(null_df):
        lo_candidates.append(null_df[col].min())
    lo = min(lo_candidates) - 0.03
    hi = df[col].max() * 1.08
    ax.set_xlim(lo, hi)
    ax.grid(axis="x", color="0.85", linewidth=0.8, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    gene_scope = xlabel.split("averaged across ")[-1].rstrip(")")
    ax.set_title(
        f"External-validation {xlabel.split(' (')[0]} across all scored samples, by sample type\n"
        "GEMM run 6667's fitted rules, scored against each sample's own held-out data\n"
        f"({gene_scope})",
        fontsize=12,
    )
    if len(null_df):
        n_range = f"{null_df.groupby('category').size().min()}-{null_df.groupby('category').size().max()} iters"
        samp_range = f"{null_df.groupby('category')['sample'].nunique().min()}-{null_df.groupby('category')['sample'].nunique().max()} samples"
        null_desc = f"{n_range}/category, {samp_range}/category"
    else:
        null_desc = "not yet run"
    note = (
        "Each dot = one sample's own metric (organoid_combined excluded -- it's an aggregate of the 3 organoid rows below, not an\n"
        "independent sample). organoid_shGFP and mets_compiled are labeled directly since they're the focus of this investigation.\n"
        f"Grey diamond/bar = scrambled-data null, mean ± 95% CI ({null_desc}) -- most real samples score above it."
    )
    fig.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7.5, color="0.3")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ["png", "pdf"]:
        fig.savefig(f"{OUT_DIR}/{out_name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_DIR}/{out_name}.{{png,pdf}}")


def run_comparison(exclude_selfloops):
    tag = "excl_selfloops" if exclude_selfloops else "incl_selfloops"
    gene_desc = "42 non-self-loop network genes" if exclude_selfloops else "all 53 network genes (11 self-loop)"
    print(f"\n=== {tag} ({gene_desc}) ===")

    df = load_metrics_table(exclude_selfloops=exclude_selfloops)
    df.to_csv(f"{OUT_DIR}/all_samples_metrics_by_category_{tag}.csv", index=False)
    print(f"Wrote {OUT_DIR}/all_samples_metrics_by_category_{tag}.csv")
    print(df.groupby("category")[["mean_r2", "mean_f1", "mean_auc"]].agg(["count", "median"]))

    null_df = load_aggregate_null() if not exclude_selfloops else load_pergene_null(exclude_selfloops=True)
    if len(null_df):
        print(null_df.groupby("category")[["mean_r2", "mean_f1", "mean_auc"]].agg(["count", "mean"]))
    else:
        print("(no null data yet)")

    suffix = "" if not exclude_selfloops else "_excl_selfloops"
    plot_metric(df, null_df, "mean_r2", f"Mean R² (per-sample, averaged across {gene_desc})", None, f"all_samples_r2_by_category{suffix}")
    plot_metric(df, null_df, "mean_f1", f"Mean F1 (per-sample, averaged across {gene_desc})", None, f"all_samples_f1_by_category{suffix}")
    plot_metric(df, null_df, "mean_auc", f"Mean ROC AUC (per-sample, averaged across {gene_desc})", 0.5, f"all_samples_auc_by_category{suffix}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    run_comparison(exclude_selfloops=False)
    run_comparison(exclude_selfloops=True)


if __name__ == "__main__":
    main()
