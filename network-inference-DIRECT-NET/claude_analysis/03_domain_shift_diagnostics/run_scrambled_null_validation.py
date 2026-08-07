"""Scrambled-data null control for one representative sample per category: permute each
network gene's values independently across cells (breaking real regulator-target joint
structure while preserving each gene's own marginal distribution), re-run BoBa-T's own
fit_validation() with the SAME fitted rules (run 6667) against this scrambled input, and
score it with the same get_sklearn_metrics() used everywhere else in this investigation.
Repeated N times (default 5) per sample to get a small empirical null distribution to
compare the real R^2/F1/AUC numbers against.

fit_validation's runtime scales with cell count (measured ~0.08s/cell for all 53 genes
together), NOT with permutation -- each of the 5 representative samples here (one per
category: allograft, TKO, human_tumor, organoid, mets_compiled) takes several minutes per
iteration, so this is meant to be run per-sample in parallel background processes, not
serially for all 5 at once.

Run in bobaT_env_py3.13, one sample at a time (e.g. in separate background shells):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/03_domain_shift_diagnostics/run_scrambled_null_validation.py organoid_shGFP --iters 5
"""

import argparse
import os
import shutil

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
OUT_DIR = f"{DIR_PREFIX}/comparisons/domain_shift_diagnostic_and_organoid_walks"

# One representative sample per category (chosen close to that category's own median
# mean_r2 in all_samples_corr_vs_r2.csv, except organoid/TKO/mets_compiled which are the
# specific samples this investigation is about).
SAMPLES = {
    "allograft_mt5": f"{DIR_PREFIX}/data/allografts/adata_mt5_allografts_v3_RORA_RORB_ave.csv",
    "allograft_mt2": f"{DIR_PREFIX}/data/allografts/adata_mt2_allografts_v3_RORA_RORB_ave.csv",
    "allograft_1L": f"{DIR_PREFIX}/data/allografts/adata_1L_allografts_v3_RORA_RORB_ave.csv",
    "allograft_TKO-luc": f"{DIR_PREFIX}/data/allografts/adata_TKO-luc_allografts_v3_RORA_RORB_ave.csv",
    "human_RU1293": f"{DIR_PREFIX}/data/human_tumor_MSK/adata_RU1293_v3_RORA_RORB_ave.csv",
    "human_RU1215": f"{DIR_PREFIX}/data/human_tumor_MSK/adata_RU1215_v3_RORA_RORB_ave.csv",
    "human_RU1311": f"{DIR_PREFIX}/data/human_tumor_MSK/adata_RU1311_v3_RORA_RORB_ave.csv",
    # Multiple organoid_shGFP lines (see plot_organoid_shgfp_by_line.py) instead of the
    # full pooled organoid_shGFP sample, to keep runtime down while still varying which
    # real cells get scrambled within the category.
    "organoid_shGFP_D_Sample4": f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_D_Sample4_raw_subset.csv",
    "organoid_shGFP_D_Sample1": f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_D_Sample1_raw_subset.csv",
    "organoid_shGFP_D_Sample6": f"{DIR_PREFIX}/data/organoid/adata_organoid_shGFP_D_Sample6_raw_subset.csv",
    "mets_compiled": f"{DIR_PREFIX}/data/mets_compiled/adata_mets_compiled_v3_RORA_RORB_ave.csv",
}
CATEGORY_OF = {
    "allograft_mt5": "allograft", "allograft_mt2": "allograft", "allograft_1L": "allograft",
    "allograft_TKO-luc": "TKO",
    "human_RU1293": "human_tumor", "human_RU1215": "human_tumor", "human_RU1311": "human_tumor",
    "organoid_shGFP_D_Sample4": "organoid", "organoid_shGFP_D_Sample1": "organoid", "organoid_shGFP_D_Sample6": "organoid",
    "mets_compiled": "mets_compiled",
}


def permute_raw(df, nodes, rng):
    """Permute each network gene's column independently across cells. Matched
    case-insensitively -- mouse allograft data uses Title Case gene symbols (e.g. Ascl1)
    while `nodes` are all-caps human symbols (ASCL1); a literal `gene in df.columns` check
    silently matches almost nothing for those samples and leaves them unscrambled."""
    df = df.copy()
    col_by_upper = {c.upper(): c for c in df.columns}
    matched, unmatched = 0, []
    for gene in nodes:
        col = col_by_upper.get(gene.upper())
        if col is not None:
            df[col] = rng.permutation(df[col].values)
            matched += 1
        else:
            unmatched.append(gene)
    if unmatched:
        print(f"  WARNING: {len(unmatched)}/{len(nodes)} nodes had no matching raw column: {unmatched}")
    assert matched >= len(nodes) - 2, f"only matched {matched}/{len(nodes)} nodes -- check column naming"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sample", choices=list(SAMPLES.keys()))
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out-suffix", default="", help="appended to the output CSV name, so a follow-up run for the same sample doesn't overwrite the first")
    ap.add_argument("--save-pergene", action="store_true", help="also save per-gene r2/f1/auc for each iteration, needed to split by self-loop status later")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=False, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    rules, regulators_dict = bb.load.load_rules(fname=f"{DIR_PREFIX}/6667/rules/rules_6667.txt")

    tag = args.sample.replace("/", "_")
    pid_tag = f"{tag}_pid{os.getpid()}"  # unique per process -- two concurrent runs for the
    # same sample (e.g. a base run + a follow-up --out-suffix run) would otherwise share
    # /tmp paths keyed only by sample+iteration and delete each other's temp files mid-run.
    raw = pd.read_csv(SAMPLES[args.sample], index_col=0)
    print(f"[{args.sample}] {len(raw)} cells, running {args.iters} scrambled iterations...")

    rows = []
    for i in range(args.iters):
        rng = np.random.default_rng(args.seed_offset + i)
        permuted = permute_raw(raw, nodes, rng)
        tmp_path = f"/tmp/scrambled_{pid_tag}_{i}.csv"
        permuted.to_csv(tmp_path)

        data = bb.load.load_data(tmp_path, nodes, norm=0.3, delimiter=",", log1p=False, transpose=True, sample_order=False, fillna=0)
        val_dir = f"/tmp/scrambled_val_{pid_tag}_{i}"
        os.makedirs(val_dir, exist_ok=True)
        _, _, _, area_all = bb.tl.fit_validation(
            data, nodes, regulators_dict, rules, data_test_t1=None,
            save=True, save_dir=val_dir, plot=False, show_plots=False, save_df=True, fname="scrambled",
        )
        bb.tl.save_auc_by_gene(area_all, nodes, val_dir)
        stats = bb.tl.get_sklearn_metrics(val_dir, plot_cm=False, show=False)
        if args.save_pergene:
            pergene_path = f"{OUT_DIR}/scrambled_null_pergene_{tag}_iter{args.seed_offset + i}.csv"
            stats[["gene", "r2", "f1", "roc_auc_score"]].to_csv(pergene_path, index=False)
        row = {
            "sample": args.sample, "category": CATEGORY_OF[args.sample], "iteration": i,
            "mean_r2": stats["r2"].mean(), "mean_f1": stats["f1"].mean(), "mean_auc": stats["roc_auc_score"].mean(),
        }
        rows.append(row)
        print(f"[{args.sample}] iter {i}: r2={row['mean_r2']:.3f} f1={row['mean_f1']:.3f} auc={row['mean_auc']:.3f}")

        os.remove(tmp_path)
        shutil.rmtree(val_dir, ignore_errors=True)

    out_df = pd.DataFrame(rows)
    out_path = f"{OUT_DIR}/scrambled_null_{tag}{args.out_suffix}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
