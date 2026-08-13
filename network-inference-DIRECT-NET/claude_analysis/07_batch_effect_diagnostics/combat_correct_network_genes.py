"""ComBat-correct the network-gene panel across the 19 in-scope datasets (GEMM training
data, organoid_combined, the 5 Ireland et al. 2025 datasets, 12 allografts), then rescale
to [0,1] for BoBa-T scoring -- deliberately NOT using bobaT's standard per-sample
`norm=0.3` quantile-clip for this, since that recomputes each gene's low/high quantile from
THAT sample's own data alone. Any correction applied before a per-sample-quantile rescale
is a monotonic (in fact linear, for ComBat) transform per gene per batch, and monotonic
transforms are exactly invariant to a subsequent per-sample-quantile rescale -- so scoring
ComBat-corrected data with the standard norm=0.3 pipeline would silently discard the
correction entirely. Instead: after ComBat, compute ONE shared low/high quantile per gene
from the pooled (all-dataset) corrected distribution, and apply that same pair to every
dataset -- this preserves whatever cross-dataset alignment ComBat established.

Known risk, guarded against explicitly (the earlier "global reference norm" prototype hit
this and produced numerically degenerate R2=-1.8e14, see BoBa-T_hyperparameters.md sec 4 /
comparisons/domain_shift_diagnostic_low_R2_samples/SUMMARY.md sec 1): if a gene's true
variance within some dataset collapses under a shared external reference (e.g. because that
gene's real baseline was already offset, not just differently-scaled, and ComBat/rescaling
pins it to a near-constant value), naive R2 explodes. This script does NOT filter that here
-- it's checked and reported explicitly downstream by
combat_run_validation_and_compare.py's per-gene actual-variance flag, so a degenerate case
is visible and excluded from any mean, not silently averaged in.

Run in celloracle_env (has scanpy for sc.pp.combat; bobaT_env/bobaT_env_py3.13 don't):
    /opt/anaconda3/envs/celloracle_env/bin/python claude_analysis/07_batch_effect_diagnostics/combat_correct_network_genes.py
"""

import os
import pickle

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/07_batch_effect_diagnostics/combat_corrected"
NODES_PATH = f"{DIR_PREFIX}/6667/rules/rules_6667.txt"

ALLOGRAFTS = ["1L", "2L", "2LR", "3L", "5B", "TKO-luc", "mt2", "mt3", "mt4", "mt4Rf", "mt5", "mt6"]
IRELAND_2025_SAMPLES = ["cgrp_k5", "organoid_celltag", "tbo_allograft_5khvg", "rpr2_allograft", "celltag_fate_dpt"]
QUANTILE_LO, QUANTILE_HI = 0.3, 0.7  # matches the 6667 network's own fitted node_normalization=0.3


def load_nodes():
    nodes = []
    with open(NODES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def gather_paths():
    paths = {}
    for a in ALLOGRAFTS:
        paths[f"allograft_{a}"] = f"{DIR_PREFIX}/data/allografts/adata_{a}_allografts_v3_RORA_RORB_ave.csv"
    paths["organoid_combined"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_v3_RORA_RORB_ave.csv"
    for name in IRELAND_2025_SAMPLES:
        paths[name] = f"{DIR_PREFIX}/data/{name}/adata_{name}_v3_RORA_RORB_ave.csv"
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nodes = load_nodes()

    # GEMM training data: same source + same train-split restriction used everywhere else
    # in this diagnostic series (test_batch_effects_all_samples.py etc).
    gemm_full = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm_full.columns = [c.upper() for c in gemm_full.columns]
    with open(f"{DIR_PREFIX}/6667/data_split/test_train_indicescombined.p", "rb") as f:
        split_indices = pickle.load(f)
    train_cellids = set(split_indices["train_cellID"])
    gemm = gemm_full.loc[gemm_full.index.isin(train_cellids), nodes]

    pooled = {"GEMM_train": gemm}
    for name, path in gather_paths().items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        df = pd.read_csv(path, index_col=0)
        df.columns = [c.upper() for c in df.columns]
        pooled[name] = df[nodes]

    print(f"Pooling {len(pooled)} datasets: {list(pooled.keys())}")
    frames = []
    for name, df in pooled.items():
        frames.append(df.assign(_dataset=name))
    combined = pd.concat(frames)
    labels = combined.pop("_dataset")

    adata = ad.AnnData(X=combined.values.astype(np.float64))
    adata.obs["dataset"] = labels.values
    adata.obs["dataset"] = adata.obs["dataset"].astype("category")
    adata.var_names = combined.columns

    print("Running ComBat...")
    corrected = sc.pp.combat(adata, key="dataset", inplace=False)
    corrected_df = pd.DataFrame(corrected, index=combined.index, columns=combined.columns)
    corrected_df["_dataset"] = labels.values

    # Shared [0,1] rescale from the POOLED corrected distribution (not per-sample) -- see
    # module docstring for why per-sample norm=0.3 would erase the correction entirely.
    lq = corrected_df.drop(columns="_dataset").quantile(QUANTILE_LO)
    uq = corrected_df.drop(columns="_dataset").quantile(QUANTILE_HI)
    print(f"Pooled quantile range width per gene (uq-lq): min={float((uq - lq).min()):.4f}, "
          f"max={float((uq - lq).max()):.4f}, median={float((uq - lq).median()):.4f}")

    for name, sub in corrected_df.groupby("_dataset"):
        sub = sub.drop(columns="_dataset")
        scaled = ((sub - lq) / (uq - lq)).clip(0, 1)
        out_path = f"{OUT_DIR}/adata_{name}_combat_globalnorm.csv"
        scaled.index.name = "CellID"
        scaled.to_csv(out_path)
        print(f"Wrote {out_path}: {scaled.shape}")

    lq.to_frame("lq").join(uq.to_frame("uq")).to_csv(f"{OUT_DIR}/pooled_quantile_reference.csv")
    print(f"Wrote {OUT_DIR}/pooled_quantile_reference.csv")


if __name__ == "__main__":
    main()
