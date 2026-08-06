"""Phase 2, item 3 (BoBa-T_hyperparameters.md sec 4): prototype the "global-reference"
normalization scheme. The existing quantile-clip/gmm/minmax schemes all force every
gene's own distribution to span [0,1], regardless of how much that gene actually varies --
manufacturing spread for genuinely low-variance genes (screen_bimodality_spread.py found
4/53 genes, e.g. BACH2/ZEB1/PBX1/JUNB, where this risk is concrete: not clearly bimodal by
Hartigan's dip test AND genuinely low raw spread relative to the rest of the network).

This scheme instead: (1) centers each gene at its own median (preserves each gene's
natural on/off balance point), then (2) squashes through a logistic referenced to a
SHARED, network-wide scale constant (the median of all 53 genes' raw standard deviations)
rather than that gene's own min/max/IQR. A gene with much less real spread than the
network's typical gene stays compressed near 0.5 instead of being stretched to the same
[0,1] range as a strongly bimodal gene; a gene with much more spread saturates towards 0/1
more readily. No BoBa-T source change -- this is a multiome-side preprocessing step;
output feeds into bb.load.load_data with norm=None (pass-through, already scaled).

Run in bobaT_env_py3.13:
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python build_global_reference_norm.py
"""

import os

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
DATA_PATH = "data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
OUT_PATH = "data/adata_imputed_combined_v3_RORA_RORB_ave_globalnorm.csv"
# Logistic slope: how many "shared scale" units map to a full 0->1 transition. Smaller K
# means genes saturate to 0/1 more easily (closer to today's per-gene schemes); larger K
# means more of the network stays graded/intermediate. Swept K in {0.2..2.0}: K=2.0 (an
# initial guess) collapsed the whole network to ~1% confidently-binarized cells on average
# (vs. ~60% under node_normalization=0.3) -- given the 1-max(heat) pull-to-0.5 mechanism in
# get_rules (BoBa-T_hyperparameters.md sec 1), that would devastate the fit. K=0.2 keeps
# the network-wide mean extreme-fraction (0.47) roughly comparable to norm=0.3's fixed 0.60
# while still preserving real per-gene differentiation (e.g. BACH2 0.04 vs ZBTB20 0.96,
# vs. both pinned to exactly 0.60 under the old scheme).
K = 0.2


def main():
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    raw = bb.load.load_data(
        f"{DIR_PREFIX}/{DATA_PATH}", nodes,
        norm=None, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0,
    )

    gene_medians = raw.median()
    gene_stds = raw.std()
    shared_scale = gene_stds.median()
    print(f"Shared scale (median of {len(gene_stds)} genes' raw std): {shared_scale:.4f}")
    print(f"Per-gene raw std range: {gene_stds.min():.4f} ({gene_stds.idxmin()}) to {gene_stds.max():.4f} ({gene_stds.idxmax()})")

    centered = raw - gene_medians
    scaled = 1.0 / (1.0 + np.exp(-centered / (K * shared_scale)))

    print("\nFraction of exact-ish extreme values (>0.95 or <0.05) per gene, low vs high raw_std:")
    frac_extreme = ((scaled > 0.95) | (scaled < 0.05)).mean()
    compare = pd.DataFrame({"raw_std": gene_stds, "frac_extreme_globalnorm": frac_extreme}).sort_values("raw_std")
    print(compare.head(5))
    print("...")
    print(compare.tail(5))

    # bb.load.load_data's final returned orientation (rows=samples, cols=genes) is
    # identical to this source CSV's own on-disk orientation when loaded with
    # transpose=True (confirmed directly: the file's header row is "CellID,<gene>,<gene>...").
    # `scaled` (built from that same `raw` output) is therefore already in the right
    # on-disk shape -- write as-is, no re-transpose.
    scaled.index.name = "CellID"
    scaled.to_csv(f"{DIR_PREFIX}/{OUT_PATH}")
    print(f"\nWrote {DIR_PREFIX}/{OUT_PATH}: {scaled.shape} (samples x genes, matching source CSV orientation)")

    # Save the fitted reference (gene medians + shared scale) so external validation data
    # can be scored under the SAME reference (GEMM's), not a freshly-recomputed one from
    # the external sample itself -- recomputing per-sample would silently reintroduce the
    # artificial-spread problem this scheme is meant to avoid (see
    # apply_global_reference_norm_external.py).
    params = gene_medians.to_frame("gene_median")
    params["shared_scale"] = shared_scale
    params["K"] = K
    params.to_csv(f"{DIR_PREFIX}/data/global_reference_norm_params_gemm.csv")
    print(f"Wrote GEMM reference params to {DIR_PREFIX}/data/global_reference_norm_params_gemm.csv")


if __name__ == "__main__":
    main()
