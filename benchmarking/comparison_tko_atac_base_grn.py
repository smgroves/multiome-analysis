"""Build a REAL, ATAC-informed CellOracle base GRN from TKO_final_arc.rds's actual peaks
assay (mm10, 158,859 peaks) -- not a generic genome-wide promoter motif scan. This is the
direct answer to "does the CellOracle/SCENIC-from-scratch comparison use real ATAC data":
this base GRN is built from real, dataset-specific chromatin accessibility.

Pipeline:
    1. extract_tko_atac.R (Signac) -- peak genomic ranges -> nearest-gene assignment
       within 10kb (Signac's ClosestFeature; the simpler alternative to Cicero
       co-accessibility that CellOracle's own tutorial documents), raw RNA counts, QC
       metadata. Writes data/tko_atac/{peak_to_gene.csv,raw_counts.mtx,genes.csv,
       cells.csv,tko_clusters.csv}.
    2. This script (celloracle_env) -- motif-scan those peaks against the real mm10
       genome (`celloracle.motif_analysis.TFinfo`, gimmemotifs under the hood) ->
       TF_info_matrix (peak x TF, same shape as the human_promoter_base_GRN used
       elsewhere in this repo, but now genuinely ATAC-derived for this dataset).

Run in celloracle_env:
    /opt/anaconda3/envs/celloracle_env/bin/python comparison_tko_atac_base_grn.py
"""

import celloracle as co
import pandas as pd

import os

DATA_DIR = "../network-inference-DIRECT-NET/data/tko_atac"
OUT_PATH = f"{DATA_DIR}/tko_atac_base_grn.parquet"
SCAN_CHECKPOINT = f"{DATA_DIR}/tko_atac_tfi_scanned.celloracle.tfinfo"


def main():
    if os.path.exists(SCAN_CHECKPOINT):
        print(f"[scan] found checkpoint at {SCAN_CHECKPOINT}, skipping rescan")
        tfi = co.motif_analysis.load_TFinfo(SCAN_CHECKPOINT)
    else:
        peak_gene = pd.read_csv(f"{DATA_DIR}/peak_to_gene.csv")
        peak_gene = peak_gene.rename(columns={"gene_short_name": "gene_short_name"})[["peak_id", "gene_short_name"]]
        peak_gene = peak_gene.dropna().reset_index(drop=True)  # check_peak_format misaligns on a non-contiguous index
        print(f"[load] {len(peak_gene)} peak-gene pairs, {peak_gene.gene_short_name.nunique()} unique genes")

        peak_gene = co.motif_analysis.check_peak_format(peak_gene, ref_genome="mm10")
        print(f"[check] {len(peak_gene)} peaks passed format check")

        tfi = co.motif_analysis.TFinfo(peak_data_frame=peak_gene, ref_genome="mm10")
        print("[scan] starting motif scan (gimmemotifs, this is the slow step)...")
        tfi.scan(fpr=0.02, n_cpus=-1, verbose=True)
        print("[scan] done")
        tfi.to_hdf5(file_path=SCAN_CHECKPOINT)
        print(f"[scan] checkpoint saved -> {SCAN_CHECKPOINT}")

    tfi.reset_filtering()
    tfi.filter_motifs_by_score(threshold=10)
    tfi.make_TFinfo_dataframe_and_dictionary()

    base_grn = tfi.to_dataframe()
    base_grn.to_parquet(OUT_PATH)
    tf_cols = [c for c in base_grn.columns if c not in ("peak_id", "gene_short_name")]
    print(f"[write] base GRN: {base_grn.shape[0]} peaks x {len(tf_cols)} TFs, "
          f"{base_grn['gene_short_name'].nunique()} target genes -> {OUT_PATH}")


if __name__ == "__main__":
    main()
