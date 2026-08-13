"""Inspect one Ireland et al. 2025 Zenodo h5ad file: obs columns, layers, var_names case,
which raw-counts-candidate layer actually looks raw (whole numbers, not already
log/normalized), and which of the 6667 network's 53 genes are present. Read-only,
no output files written except the printed report (redirect to a log file per dataset).

Run in bobaT_env (has anndata):
    /opt/anaconda3/envs/bobaT_env/bin/python inspect_ireland2025_h5ad.py <path_to_h5ad>
"""

import sys

import anndata as ad
import numpy as np

RULES_PATH = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET/6667/rules/rules_6667.txt"


def load_network_nodes():
    nodes = []
    with open(RULES_PATH) as f:
        for line in f:
            nodes.append(line.split("|")[0].strip().upper())
    return nodes


def layer_summary(mat, name):
    arr = mat.toarray() if hasattr(mat, "toarray") else np.asarray(mat)
    sample = arr[:2000] if arr.shape[0] > 2000 else arr
    is_whole = np.allclose(sample, np.round(sample))
    print(f"  layer '{name}': dtype={arr.dtype}, min={sample.min():.4f}, max={sample.max():.4f}, "
          f"n_distinct(sample)={len(np.unique(sample))}, all_whole_numbers={is_whole}")


def main(path):
    print(f"\n{'='*80}\n{path}\n{'='*80}")
    adata = ad.read_h5ad(path)
    print(f"shape: {adata.shape}")

    print("\n--- obs.columns ---")
    for c in adata.obs.columns:
        print(f"  {c}  (dtype={adata.obs[c].dtype})")

    print("\n--- candidate QC columns (name contains mt/count/feature/pct/percent) ---")
    for c in adata.obs.columns:
        cl = c.lower()
        if any(k in cl for k in ["mt", "count", "feature", "pct", "percent"]):
            print(f"  {c}: min={adata.obs[c].min()}, max={adata.obs[c].max()}, "
                  f"n_unique={adata.obs[c].nunique()}, dtype={adata.obs[c].dtype}")

    print("\n--- candidate condition/grouping columns (categorical/object, 2-20 unique values) ---")
    for c in adata.obs.columns:
        if adata.obs[c].dtype.name in ("category", "object", "bool") or (
            adata.obs[c].dtype.kind in "iu" and adata.obs[c].nunique() <= 20
        ):
            nu = adata.obs[c].nunique()
            if 1 < nu <= 20:
                print(f"  {c} ({nu} unique): {adata.obs[c].value_counts().to_dict()}")

    print("\n--- layers ---")
    print(f"  X: dtype={adata.X.dtype}, shape={adata.X.shape}")
    layer_summary(adata.X, "X")
    for lname in adata.layers.keys():
        layer_summary(adata.layers[lname], lname)

    print("\n--- X vs layers identity check (is X byte-identical to any layer, i.e. mislabeled?) ---")
    x_sample = adata.X[:500]
    x_sample = x_sample.toarray() if hasattr(x_sample, "toarray") else np.asarray(x_sample)
    for lname in adata.layers.keys():
        l_sample = adata.layers[lname][:500]
        l_sample = l_sample.toarray() if hasattr(l_sample, "toarray") else np.asarray(l_sample)
        same = np.allclose(x_sample, l_sample)
        print(f"  X == layer '{lname}'? {same}")

    print(f"\n--- var_names sample (case check) ---")
    print(f"  first 10: {list(adata.var_names[:10])}")

    print("\n--- network gene match (53 nodes from rules_6667.txt) ---")
    nodes = load_network_nodes()
    # RORA_RORB isn't a literal gene symbol -- check RORA/RORB separately for match purposes
    nodes_check = [n for n in nodes if n != "RORA_RORB"] + ["RORA", "RORB"]
    var_upper = {v.upper(): v for v in adata.var_names}
    matched = sorted(set(nodes_check) & set(var_upper))
    missing = sorted(set(nodes_check) - set(var_upper))
    print(f"  {len(matched)}/{len(nodes_check)} matched")
    print(f"  MISSING: {missing}")


if __name__ == "__main__":
    main(sys.argv[1])
