"""Comparison 1, "run from the beginning" step: full pySCENIC (GRNBoost2 + RcisTarget
motif pruning), on ALL of 6667's cells, with NO restriction to the DIRECT-NET candidate
network -- unlike comparison_genie3_fit_6667.py, which deliberately restricted GENIE3 to
DIRECT-NET's 228 candidate edges for a same-base-GRN comparison, this is SCENIC run the
way it's actually used in practice: given the expression data and nothing else (plus a
generic human TF list and RcisTarget's genome-wide motif databases), let it find its own
structure.

Requires a dedicated conda env -- pySCENIC 0.12.1 (2022) and its arboreto/dask dependency
chain do not run on a modern numpy/pandas/scikit-learn/dask stack. Build it with:
    bash benchmarking/setup_scenic_env.sh
(see that script for what each version pin fixes.)

Data, all downloaded fresh from Aertslab's public cisTarget resources, not redistributed
in this repo -- see data/README.md-style provenance note in benchmarking/README.md:
    data/scenic/allTFs_hg38.txt                                            (~12KB)
    data/scenic/motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl                 (~99MB)
    data/scenic/hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather (~311MB)

Run (bash, in scenic_env; the two pyscenic CLI steps, then a small df2regulons step):

    cd benchmarking/data/scenic
    ENV=/opt/anaconda3/envs/scenic_env/bin

    $ENV/pyscenic grn expr_all_cells_6667.csv tfs_6667.txt \
        -o adjacencies_6667.csv --method grnboost2 --num_workers 4 --seed 0

    $ENV/pyscenic ctx adjacencies_6667.csv \
        hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
        --annotations_fname motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
        --expression_mtx_fname expr_all_cells_6667.csv \
        --min_genes 1 --num_workers 4 --mode custom_multiprocessing \
        -o regulons_6667.csv

    $ENV/python comparison_scenic_fit_6667.py   # this file: df2regulons -> canonical edges

`--min_genes 1` overrides pySCENIC's default of 20: with only 53 genes and ~50 TF
candidates in this network, the default would filter out essentially every regulon (a
setting meant for genome-wide runs with thousands of candidate targets per TF).
"""

import pandas as pd
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons

SCENIC_DIR = "data/scenic"
OUT_PATH = "../network-inference-DIRECT-NET/6667/rules/scenic_regulon_edges.csv"


def main():
    motifs = load_motifs(f"{SCENIC_DIR}/regulons_6667.csv")
    regulons = df2regulons(motifs)
    print(f"{len(motifs)} enriched motifs -> {len(regulons)} regulons (TFs with >=1 significant motif)")

    rows = [
        {"source": reg.transcription_factor, "target": gene, "weight": weight}
        for reg in regulons
        for gene, weight in reg.gene2weight.items()
    ]
    edges = pd.DataFrame(rows).drop_duplicates(subset=["source", "target"])
    edges.to_csv(OUT_PATH, index=False)
    print(f"{len(edges)} TF->target edges -> {OUT_PATH}")


if __name__ == "__main__":
    main()
