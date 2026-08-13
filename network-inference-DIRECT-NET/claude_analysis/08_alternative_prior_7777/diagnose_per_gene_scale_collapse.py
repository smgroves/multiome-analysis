"""Which of the 53 network genes show ASCL1-style raw-scale collapse across external
datasets -- i.e. a dataset where that gene's typical raw (pre-normalization) level sits far
outside GEMM's own natural variation, or where the gene's within-dataset spread collapses
to near-zero? This is the diagnostic the ASCL1 finding motivated: per-sample rank
normalization (node_normalization=0.3) assumes every dataset's own top/bottom 30% is
biologically meaningful, which fails specifically for genes whose absolute dynamic range
shifts or collapses in some datasets (confirmed directly for ASCL1: raw mean ranges from
0.0002 in organoid_celltag to 1.46 in allograft_1L, a >7000x swing). Checking all 53 genes
here, not just ASCL1, across every scored external sample (33 original + 5 Ireland 2025
whole datasets + 11 Ireland condition splits = 44 samples).

Two metrics per (gene, dataset), both referenced to GEMM's OWN raw distribution for that
gene (not a shared/pooled scale -- this is diagnostic only, not a normalization scheme):
  - mean_shift_in_gemm_sds = (dataset_mean - gemm_mean) / gemm_std -- how many GEMM standard
    deviations away this dataset's typical level sits. Large |value| = mean has shifted far
    outside GEMM's own natural variation.
  - std_ratio = dataset_std / gemm_std -- near 0 = this dataset's within-sample spread for
    this gene has collapsed relative to GEMM's own variation (the specific failure mode that
    broke the earlier global-reference-norm prototype: near-zero true variance -> R2
    degenerate under any shared-scale rescaling).

Run in bobaT_env_py3.13 (pandas/numpy only, no bobaT needed):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/08_alternative_prior_7777/diagnose_per_gene_scale_collapse.py
"""

import os

import numpy as np
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
OUT_DIR = f"{DIR_PREFIX}/claude_analysis/08_alternative_prior_7777"
RULES_PATH = f"{DIR_PREFIX}/6667/rules/rules_6667.txt"

ALLOGRAFTS = ["1L", "2L", "2LR", "3L", "5B", "TKO-luc", "mt2", "mt3", "mt4", "mt4Rf", "mt5", "mt6"]
HUMAN_TUMORS = [
    "PleuralEffusion", "RU1065", "RU1066", "RU1080", "RU1108", "RU1124", "RU1144", "RU1145",
    "RU1152", "RU1181", "RU1195", "RU1215", "RU1229", "RU1231", "RU1293", "RU1311", "RU426", "RU779",
]
IRELAND_WHOLE = ["cgrp_k5", "organoid_celltag", "tbo_allograft_5khvg", "rpr2_allograft", "celltag_fate_dpt"]
IRELAND_SPLITS = [
    "cgrp_k5_K5", "cgrp_k5_CGRP", "organoid_celltag_RPM", "organoid_celltag_RPMA", "organoid_celltag_WT",
    "tbo_allograft_5khvg_RPM_CTpostCre", "tbo_allograft_5khvg_RPM_CTpreCre",
    "rpr2_allograft_RPM", "rpr2_allograft_RPR2", "celltag_fate_dpt_RPM", "celltag_fate_dpt_RPMA",
]
IRELAND_DIR = {
    "cgrp_k5": "cgrp_k5", "cgrp_k5_K5": "cgrp_k5", "cgrp_k5_CGRP": "cgrp_k5",
    "organoid_celltag": "organoid_celltag", "organoid_celltag_RPM": "organoid_celltag",
    "organoid_celltag_RPMA": "organoid_celltag", "organoid_celltag_WT": "organoid_celltag",
    "tbo_allograft_5khvg": "tbo_allograft_5khvg", "tbo_allograft_5khvg_RPM_CTpostCre": "tbo_allograft_5khvg",
    "tbo_allograft_5khvg_RPM_CTpreCre": "tbo_allograft_5khvg",
    "rpr2_allograft": "rpr2_allograft", "rpr2_allograft_RPM": "rpr2_allograft", "rpr2_allograft_RPR2": "rpr2_allograft",
    "celltag_fate_dpt": "celltag_fate_dpt", "celltag_fate_dpt_RPM": "celltag_fate_dpt", "celltag_fate_dpt_RPMA": "celltag_fate_dpt",
}


def load_nodes():
    nodes = []
    with open(RULES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def gather_paths():
    paths = {}
    for a in ALLOGRAFTS:
        paths[f"allograft_{a}"] = f"{DIR_PREFIX}/data/allografts/adata_{a}_allografts_v3_RORA_RORB_ave.csv"
    for h in HUMAN_TUMORS:
        paths[f"human_{h}"] = f"{DIR_PREFIX}/data/human_tumor_MSK/adata_{h}_v3_RORA_RORB_ave.csv"
    paths["organoid_shGFP"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB1"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB1_v3_RORA_RORB_ave.csv"
    paths["organoid_shRORB2"] = f"{DIR_PREFIX}/data/organoid/adata_organoid_shRORB2_v3_RORA_RORB_ave.csv"
    paths["mets_compiled"] = f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv"
    for name in IRELAND_WHOLE + IRELAND_SPLITS:
        d = IRELAND_DIR[name]
        paths[name] = f"{DIR_PREFIX}/data/{d}/adata_{name}_v3_RORA_RORB_ave.csv"
    return paths


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    nodes = load_nodes()
    nodes_cols = [n for n in nodes if n != "RORA_RORB"] + ["RORA_RORB"]  # RORA_RORB is a literal column already

    gemm = pd.read_csv(f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv", index_col=0)
    gemm.columns = [c.upper() for c in gemm.columns]
    gemm_mean = gemm[nodes_cols].mean()
    gemm_std = gemm[nodes_cols].std()

    paths = gather_paths()
    rows = []
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"MISSING: {name} -> {path}")
            continue
        df = pd.read_csv(path, index_col=0)
        df.columns = [c.upper() for c in df.columns]
        d_mean = df[nodes_cols].mean()
        d_std = df[nodes_cols].std()
        for gene in nodes_cols:
            rows.append({
                "gene": gene, "dataset": name,
                "mean_shift_in_gemm_sds": (d_mean[gene] - gemm_mean[gene]) / gemm_std[gene],
                "std_ratio": d_std[gene] / gemm_std[gene],
                "dataset_mean": d_mean[gene], "dataset_std": d_std[gene],
            })

    result = pd.DataFrame(rows)
    result.to_csv(f"{OUT_DIR}/per_gene_scale_diagnostic_all_datasets.csv", index=False)
    print(f"Wrote {len(result)} (gene, dataset) rows across {result['dataset'].nunique()} datasets, {result['gene'].nunique()} genes")

    per_gene = result.groupby("gene").agg(
        worst_abs_mean_shift=("mean_shift_in_gemm_sds", lambda x: x.abs().max()),
        worst_std_ratio=("std_ratio", "min"),
        n_datasets_shift_gt3sd=("mean_shift_in_gemm_sds", lambda x: (x.abs() > 3).sum()),
        n_datasets_std_collapse=("std_ratio", lambda x: (x < 0.1).sum()),
    ).sort_values("worst_abs_mean_shift", ascending=False)
    per_gene.to_csv(f"{OUT_DIR}/per_gene_scale_diagnostic_summary.csv")

    pd.set_option("display.width", 160)
    print("\n=== Genes with the worst cross-dataset MEAN shift (relative to GEMM's own SD) ===")
    print(per_gene.head(15).to_string())

    print("\n=== Genes with the worst within-dataset VARIANCE collapse (std_ratio near 0 in at least one dataset) ===")
    print(per_gene.sort_values("worst_std_ratio").head(15).to_string())

    # Which specific (gene, dataset) pairs are the most extreme -- concretely, which
    # datasets would be at risk under a shared-scale rescaling, and for which genes.
    extreme = result[(result["mean_shift_in_gemm_sds"].abs() > 3) | (result["std_ratio"] < 0.1)]
    extreme = extreme.sort_values("mean_shift_in_gemm_sds", key=lambda x: x.abs(), ascending=False)
    extreme.to_csv(f"{OUT_DIR}/per_gene_scale_diagnostic_extreme_pairs.csv", index=False)
    print(f"\n{len(extreme)} (gene, dataset) pairs cross either threshold (|shift|>3 GEMM-SDs, or std_ratio<0.1)")
    print(f"{extreme['gene'].nunique()} distinct genes affected, {extreme['dataset'].nunique()} distinct datasets affected")
    print("\n=== Top 20 most extreme (gene, dataset) pairs ===")
    print(extreme.head(20)[["gene", "dataset", "mean_shift_in_gemm_sds", "std_ratio"]].to_string(index=False))

    print(f"\nWrote {OUT_DIR}/per_gene_scale_diagnostic_{{all_datasets,summary,extreme_pairs}}.csv")


if __name__ == "__main__":
    main()
