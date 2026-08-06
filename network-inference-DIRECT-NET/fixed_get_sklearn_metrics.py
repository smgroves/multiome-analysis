"""Corrected replacement for `bobaT.tl.get_sklearn_metrics`'s gene-name parsing.

Real bug in the installed bobaT package (bobaT/tl.py:655), present since before this
session -- confirmed to also affect the original 6667/validation/in_sample_validation:
    gene = f.split("/")[-1].split("_")[0]
This takes only the first underscore-delimited token of the filename, so any node whose
name itself contains an underscore -- in this network, only `RORA_RORB` -- gets mislabeled
as a nonexistent gene "RORA" in every summary_stats.csv this project has ever produced. It
doesn't crash or collide with a real node (there's no separate "RORA" node), so it's easy
to miss: RORA_RORB has always been silently inaccessible under its real name.

Not editing the bobaT package (out of scope for this repo) -- this reimplements the exact
same metric computation with correct gene-name resolution, matched against the actual
node list instead of naively splitting on "_".
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, explained_variance_score, max_error, r2_score, log_loss,
)


def resolve_gene_name(filename: str, nodes: list) -> str:
    """Match a `<gene>_validation.csv` filename against the real node list, longest match
    first, so multi-underscore names (RORA_RORB) resolve correctly instead of truncating
    at the first underscore."""
    base = os.path.basename(filename)
    for node in sorted(nodes, key=len, reverse=True):
        if base.startswith(f"{node}_"):
            return node
    return base.split("_")[0]  # fallback: same behavior as the buggy original


def get_sklearn_metrics_fixed(val_dir: str, nodes: list) -> pd.DataFrame:
    files = glob.glob(f"{val_dir}/accuracy_plots/*_validation.csv")
    rows = []
    for f in files:
        gene = resolve_gene_name(f, nodes)
        val_df = pd.read_csv(f, header=0, index_col=0)
        val_df["actual_binary"] = (val_df["actual"] > 0.5).astype(int)
        val_df["predicted_binary"] = (val_df["predicted"] > 0.5).astype(int)

        try:
            roc_auc = roc_auc_score(val_df["actual_binary"], val_df["predicted"])
        except ValueError:
            roc_auc = np.nan
        try:
            ll = log_loss(val_df["actual_binary"], val_df["predicted"])
        except ValueError:
            ll = np.nan

        rows.append({
            "gene": gene,
            "accuracy": accuracy_score(val_df["actual_binary"], val_df["predicted_binary"]),
            "balanced_accuracy_score": balanced_accuracy_score(val_df["actual_binary"], val_df["predicted_binary"]),
            "f1": f1_score(val_df["actual_binary"], val_df["predicted_binary"]),
            "roc_auc_score": roc_auc,
            "precision": precision_score(val_df["actual_binary"], val_df["predicted_binary"]),
            "recall": recall_score(val_df["actual_binary"], val_df["predicted_binary"]),
            "explained_variance": explained_variance_score(val_df["actual"], val_df["predicted"]),
            "max_error": max_error(val_df["actual"], val_df["predicted"]),
            "r2": r2_score(val_df["actual"], val_df["predicted"]),
            "log-loss": ll,
        })
    return pd.DataFrame(rows)
