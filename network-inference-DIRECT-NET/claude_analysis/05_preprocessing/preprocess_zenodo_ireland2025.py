"""Preprocess the 5 Ireland et al. 2025 Zenodo h5ad datasets (record 15857303) for external
validation of the 6667 network, matching the established convention used for
organoid/mets_compiled (see preprocess_organoid.py): QC filter -> normalize by total UMI
count -> log1p -> restrict to network genes -> MAGIC impute -> add RORA_RORB.

Gene panel is pulled directly from the 6667 network's own 53 nodes (rules_6667.txt), not
the broader Direct_net.csv/FigR_DORC_TF.csv/extra_genes union used by the older
preprocess_adata.py/preprocess_organoid.py convention -- simpler, and it's what's actually
required for validation (per user feedback while planning this).

Inspection findings (inspect_ireland2025_h5ad.py), true of all 5 files:
- Same scanpy-derived obs schema across every file: QC columns are `n_genes_by_counts`
  (~nFeature_RNA), `total_counts` (~nCount_RNA), `pct_counts_mito` (~percent.mt, already on
  a 0-100 percent scale).
- `layers["counts"]` is reliably genuine raw integer counts in every one of the 5 files
  (verified: whole-number values in every file) -- `X` is NOT reliable, it's raw in some
  files and already log-transformed in others (e.g. cgrp_k5, organoid_celltag), so `counts`
  is used here, never `X`, consistent with this project's "verify a layer is really raw by
  its values, not its name" rule (two prior real bugs were caught exactly this way).
- All 5 files matched 54/54 of the 6667 network's genes (53 nodes, RORA/RORB checked
  separately since RORA_RORB isn't a literal gene symbol) -- zero missing genes in any of
  them, so the marginalize-vs-zero-fill comparison built in
  claude_analysis/06_marginal_rule_validation/ has no live case to run on this batch.
- Each file has a real genotype/condition column analogous to organoid's shGFP/shRORB1/
  shRORB2 experimental split (not just a Leiden cluster label) -- used here for a
  per-condition cut of each dataset's export, same convention as organoid.

Run in bobaT_env (has `magic`; celloracle_env has anndata but not `magic`):
    /opt/anaconda3/envs/bobaT_env/bin/python claude_analysis/05_preprocessing/preprocess_zenodo_ireland2025.py
"""

import os

import anndata as ad
import magic
import numpy as np
import pandas as pd

BOX = "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/Ireland_2025_Basal_Cell"
RULES_PATH = "6667/rules/rules_6667.txt"
OUT_ROOT = "data"

DATASETS = [
    {
        "name": "cgrp_k5",
        "path": f"{BOX}/021825_RPM_CGRPvK5_adata3.h5ad",
        "condition_col": "Cre",  # K5 vs CGRP
    },
    {
        "name": "organoid_celltag",
        "path": f"{BOX}/030525_RPM_RPMA_WT_Organoids_forCellTagwStates.h5ad",
        "condition_col": "Genotype",  # WT / RPM / RPMA
    },
    {
        "name": "tbo_allograft_5khvg",
        "path": f"{BOX}/042725_RPM_TBOAllos_OriginalandAllo3_adata3_5kHVG_subsamplebycluster.h5ad",
        "condition_col": "GenoCT",  # RPM_CTpostCre / RPM_CTpreCre (Original vs Allo3)
    },
    {
        "name": "rpr2_allograft",
        "path": f"{BOX}/050925_RPM_TBOAllo_OriginalandAllo3_RPR2_adata2.h5ad",
        "condition_col": "Genotype",  # RPM vs RPR2
    },
    {
        "name": "celltag_fate_dpt",
        "path": f"{BOX}/050125_RPM_RPMA_TBOAllo_CellTagAnalysis_New_1.2_fate_FAprojection_DPT_final.h5ad",
        "condition_col": "Genotype",  # RPM vs RPMA
    },
]


def load_network_nodes():
    nodes = []
    with open(RULES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def preprocess_one(name, path, condition_col):
    out_dir = os.path.join(OUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {name} ===")
    adata = ad.read_h5ad(path)
    print(f"Loaded: {adata.shape}")

    keep = (adata.obs["pct_counts_mito"] < 20) & (adata.obs["n_genes_by_counts"] > 200)
    print(f"QC filter (pct_counts_mito<20, n_genes_by_counts>200): {keep.sum()}/{len(keep)} cells kept")
    adata = adata[keep.values].copy()

    nodes = load_network_nodes()
    nodes_check = [n for n in nodes if n != "RORA_RORB"] + ["RORA", "RORB"]
    var_upper = {v.upper(): v for v in adata.var_names}
    overlap = sorted(set(nodes_check) & set(var_upper))
    missing = sorted(set(nodes_check) - set(var_upper))
    print(f"{len(overlap)}/{len(nodes_check)} network genes matched. Missing: {missing if missing else 'NONE'}")

    dataset_genes = [var_upper[g] for g in overlap]
    raw = adata[:, dataset_genes].layers["counts"]
    raw = raw.toarray() if hasattr(raw, "toarray") else raw
    raw = pd.DataFrame(raw, index=adata.obs_names, columns=[g.upper() for g in dataset_genes])

    size_factor = adata.obs["total_counts"] / adata.obs["total_counts"].median()
    normalized = raw.div(size_factor.values, axis=0)
    log_normalized = np.log1p(normalized)
    ref_gene = "ASCL1" if "ASCL1" in log_normalized.columns else log_normalized.columns[0]
    print(f"Normalized + log1p. Value range sample ({ref_gene}): "
          f"{log_normalized[ref_gene].min():.3f} to {log_normalized[ref_gene].max():.3f}, "
          f"{log_normalized[ref_gene].nunique()} distinct values")

    magic_operator = magic.MAGIC(solver="approximate")
    imputed = magic_operator.fit_transform(log_normalized)
    print(f"MAGIC done: {imputed.shape}")

    if "RORA" in imputed.columns and "RORB" in imputed.columns:
        imputed["RORA_RORB"] = imputed[["RORA", "RORB"]].mean(axis=1)

    imputed.index.name = "CellID"
    imputed.to_csv(f"{out_dir}/adata_{name}_v3_RORA_RORB_ave.csv")
    adata.obs.to_csv(f"{out_dir}/{name}_clusters.csv")
    print(f"Wrote {out_dir}/adata_{name}_v3_RORA_RORB_ave.csv: {imputed.shape}")

    # Per-condition cut, same convention as organoid's shGFP/shRORB1/shRORB2 split.
    if condition_col is not None and condition_col in adata.obs.columns:
        for cond_value, cond_cells in adata.obs.groupby(condition_col, observed=True).groups.items():
            cond_cells = [c for c in cond_cells if c in imputed.index]
            if not cond_cells:
                continue
            safe_cond = str(cond_value).replace("/", "-").replace(" ", "_")
            sub = imputed.loc[cond_cells]
            sub.to_csv(f"{out_dir}/adata_{name}_{safe_cond}_v3_RORA_RORB_ave.csv")
            print(f"  condition '{cond_value}' ({condition_col}): {len(cond_cells)} cells -> "
                  f"adata_{name}_{safe_cond}_v3_RORA_RORB_ave.csv")

    return {"name": name, "n_cells_kept": int(keep.sum()), "n_genes_matched": len(overlap), "missing_genes": missing}


def main():
    summary = []
    for ds in DATASETS:
        summary.append(preprocess_one(ds["name"], ds["path"], ds["condition_col"]))
    print("\n=== Summary ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
