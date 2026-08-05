"""Comparison 1, the CORRECTED "run from the beginning" step, SCENIC half: full pySCENIC
(GRNBoost2 + RcisTarget) on the real genome-wide SCLC/AA data behind 6667
(preprocess_sclc_full_6667.py's output -- 2999 HVGs, 8908 cells), not the 53-gene boba-T
export the earlier from-scratch attempt (comparison_scenic_fit_6667.py) mistakenly used.

Uses the same scenic_env as comparison_scenic_fit_6667.py and the same cisTarget databases
(data/scenic/) -- no new downloads needed, just a different, larger input.

Run (bash, in scenic_env):

    cd benchmarking/data/scenic
    ENV=/opt/anaconda3/envs/scenic_env/bin

    # TF candidates: the HVGs that are also in the standard human TF reference list (not
    # CellOracle's promoter-base-GRN TF columns -- SCENIC's own TF universe is broader).
    python -c "
import pandas as pd
raw = pd.read_csv('../sclc_full/raw_counts_hvg.csv', index_col=0)
tfs = set(open('allTFs_hg38.txt').read().split())
matched = [g for g in raw.columns if g in tfs]
open('tfs_6667_fullscale.txt', 'w').write('\n'.join(sorted(matched)) + '\n')
raw.index.name = 'CellID'
raw.to_csv('expr_fullscale_6667.csv')
"

    $ENV/pyscenic grn expr_fullscale_6667.csv tfs_6667_fullscale.txt \
        -o adjacencies_fullscale_6667.csv --method grnboost2 --num_workers 6 --seed 0

    $ENV/pyscenic ctx adjacencies_fullscale_6667.csv \
        hg38_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather \
        --annotations_fname motifs-v10nr_clust-nr.hgnc-m0.001-o0.0.tbl \
        --expression_mtx_fname expr_fullscale_6667.csv \
        --num_workers 6 --mode custom_multiprocessing \
        -o regulons_fullscale_6667.csv

    $ENV/python comparison_scenic_fullscale_fit_6667.py   # this file: df2regulons -> canonical edges

Note: `--min_genes` is left at pySCENIC's default (20) here, unlike the 53-gene run
(comparison_scenic_fit_6667.py), which had to override it to 1 -- at real HVG scale there
are enough candidate targets per TF for the default to make sense.
"""

import pandas as pd
from pyscenic.utils import load_motifs
from pyscenic.transform import df2regulons

SCENIC_DIR = "data/scenic"
OUT_PATH = "../network-inference-DIRECT-NET/6667/rules/scenic_fullscale_regulon_edges.csv"


def main():
    motifs = load_motifs(f"{SCENIC_DIR}/regulons_fullscale_6667.csv")
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
    print("ASCL1 present as source:", "ASCL1" in edges.source.values)


if __name__ == "__main__":
    main()
