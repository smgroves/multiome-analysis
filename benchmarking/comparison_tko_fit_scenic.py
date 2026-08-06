"""Comparison 1's SCENIC-on-TKO counterpart: mirrors comparison_scenic_fullscale_fit_6667.py
but for TKO_final_arc's mouse data (mm10 cisTarget databases, benchmarking/data/scenic_mouse/).

Upstream steps already run (scenic_env):
    cd benchmarking/data/scenic_mouse
    ENV=/opt/anaconda3/envs/scenic_env/bin
    $ENV/pyscenic grn expr_tko.csv tfs_tko.txt -o adjacencies_tko.csv --method grnboost2 --num_workers 6 --seed 0
    $ENV/pyscenic ctx adjacencies_tko.csv \
        mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
        --annotations_fname motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl \
        --expression_mtx_fname expr_tko.csv --num_workers 6 --mode custom_multiprocessing \
        -o regulons_tko.csv

expr_tko.csv/tfs_tko.txt use proper mouse-case gene symbols (Ascl1, not ASCL1) to match the
mm10 rankings database's case-sensitive gene symbols -- ALL genes failed to map on a first
attempt that reused this repo's upper-cased convention. See
benchmarking/comparison_tko_fit_celloracle_atac.py for that same upper-casing on the
CellOracle side (which doesn't need case-sensitive matching against an external database).

Run in scenic_env:
    /opt/anaconda3/envs/scenic_env/bin/python comparison_tko_fit_scenic.py
"""

import pandas as pd
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons

SCENIC_DIR = "data/scenic_mouse"
OUT_PATH = "../network-inference-DIRECT-NET/data/tko_atac/scenic_tko_regulon_edges.csv"


def main():
    motifs = load_motifs(f"{SCENIC_DIR}/regulons_tko.csv")
    regulons = df2regulons(motifs)
    print(f"{len(motifs)} enriched motifs -> {len(regulons)} regulons (TFs with >=1 significant motif)")

    rows = [
        {"source": reg.transcription_factor, "target": gene, "weight": weight}
        for reg in regulons
        for gene, weight in reg.gene2weight.items()
    ]
    edges = pd.DataFrame(rows).drop_duplicates(subset=["source", "target"])
    edges.to_csv(OUT_PATH, index=False)
    print(f"{len(edges)} TF->target edges across {edges.source.nunique()} TFs -> {OUT_PATH}")
    print("Ascl1 present as source:", "Ascl1" in edges.source.values)


if __name__ == "__main__":
    main()
