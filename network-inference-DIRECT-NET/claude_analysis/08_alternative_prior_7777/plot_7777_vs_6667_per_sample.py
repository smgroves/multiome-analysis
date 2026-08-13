"""Dumbbell (before/after) plot of every sample's R2 and AUC, 6667 baseline vs 7777
(aggregate-evidence prior). One dot per condition per sample, connected by a line;
sorted by baseline R2 so the "does 7777 move everything the same direction" story reads
top-to-bottom at a glance. Colored by category (allograft / human_tumor / organoid /
mets_compiled / ireland_2025).

Run in bobaT_env_py3.13 (matplotlib only, no bobaT needed):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/plot_7777_vs_6667_per_sample.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"

CATEGORY_COLORS = {
    "allograft": "#2a78d6", "human_tumor": "#eb6834", "organoid": "#1baf7a",
    "mets_compiled": "#e34948", "ireland_2025": "#8B5E3C",
}


def categorize(name):
    if name.startswith("allograft_"):
        return "allograft"
    if name.startswith("human_"):
        return "human_tumor"
    if name.startswith("organoid"):
        return "organoid"
    if name == "mets_compiled":
        return "mets_compiled"
    return "ireland_2025"


def plot_metric(df, base_col, new_col, ylabel_stub, out_name, ref_line=None):
    df = df.dropna(subset=[base_col, new_col]).copy()
    df["category"] = df["name"].map(categorize)
    df = df.sort_values(base_col).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.22)))
    y = range(len(df))

    for i, row in df.iterrows():
        color = CATEGORY_COLORS[row["category"]]
        ax.plot([row[base_col], row[new_col]], [i, i], color=color, linewidth=1.2, alpha=0.6, zorder=2)
        ax.scatter(row[base_col], i, color=color, s=22, marker="o", facecolor="none", linewidth=1.3, zorder=3)
        ax.scatter(row[new_col], i, color=color, s=26, marker="o", zorder=4)

    if ref_line is not None:
        ax.axvline(ref_line, color="0.7", linewidth=1, linestyle=":", zorder=1)

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["name"], fontsize=6)
    ax.set_xlabel(ylabel_stub)
    ax.set_title(f"{ylabel_stub}: 6667 baseline (hollow) → 7777 (filled), per sample\nsorted by baseline, colored by category")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.9", linewidth=0.7, zorder=0)

    handles = [plt.Line2D([0], [0], marker="o", color=c, linestyle="", markersize=6, label=cat)
               for cat, c in CATEGORY_COLORS.items() if cat in df["category"].unique()]
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=True)

    n_improved = (df[new_col] > df[base_col]).sum()
    fig.text(0.5, 0.005, f"{n_improved}/{len(df)} samples improved", ha="center", fontsize=8, color="0.3")

    plt.tight_layout(rect=(0, 0.02, 1, 1))
    for ext in ["png", "pdf"]:
        plt.savefig(f"{OUT_DIR}/{out_name}.{ext}", dpi=150)
    plt.close()
    print(f"Wrote {OUT_DIR}/{out_name}.{{png,pdf}} ({n_improved}/{len(df)} improved)")


def main():
    df = pd.read_csv(f"{OUT_DIR}/7777_vs_6667_full_validation.csv")
    plot_metric(df, "baseline_r2", "r2_7777", "R²", "7777_vs_6667_r2_per_sample")
    plot_metric(df, "baseline_auc", "auc_7777", "ROC AUC", "7777_vs_6667_auc_per_sample", ref_line=0.5)


if __name__ == "__main__":
    main()
