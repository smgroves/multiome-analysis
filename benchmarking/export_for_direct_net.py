"""Export what's needed to build a Seurat object for DIRECT-NET in R (avoids needing an
h5ad-reading R package -- this project's own precedent for cross-language handoffs is via
intermediate files, e.g. preprocess_mets_compiled.R/preprocess_mets_compiled_magic.py).

Real ATAC peaks within +-250kb of each of our 10 genes' real TSS -- DIRECT-NET's OWN
hardcoded window (Run_DIRECT_NET: `p2 <- Starts[i]-250000 : Starts[i]+250000`), not the
+-10kb window used for the earlier hand-rolled motif-scan comparison -- so this is a
genuinely different, wider real cis-regulatory search than what was already built.

Also exports the h5ad's own real, already-computed genome-wide GEX PCA embedding
(obsm['GEX_X_pca']) to use as DIRECT-NET's cell_coord for KNN-based cell aggregation,
rather than recomputing a PCA from just 10 genes in R.

Run in celloracle_env (has anndata):
    /opt/anaconda3/envs/celloracle_env/bin/python export_for_direct_net.py
"""
import numpy as np
import pandas as pd
import anndata as ad

H5AD_PATH = "data/hsc_multiome/multiome_BMMC_processed.h5ad"
OUT_DIR = "data/hsc_multiome/direct_net"

CELL_TYPES = [
    "HSC", "MK/E prog", "G/M prog",
    "Erythroblast", "Proerythroblast", "Normoblast",
    "CD14+ Mono", "CD16+ Mono", "cDC2", "pDC", "ID2-hi myeloid prog",
]
# Real gene symbols this time (DIRECT-NET matches focus_markers against genome.info$genes,
# both kept as real symbols until the final candidate-network conversion step).
PANEL_GENES = ["GATA1", "GATA2", "SPI1", "FLI1", "CEBPA", "KLF1", "GFI1", "ZFPM1", "TAL1", "JUN"]
TSS = {
    "GATA1": ("X", 48783122), "GATA2": ("3", 128493208), "SPI1": ("11", 47409369),
    "FLI1": ("11", 128686535), "CEBPA": ("19", 33302534), "KLF1": ("19", 12887201),
    "GFI1": ("1", 92486925), "ZFPM1": ("16", 88453208), "TAL1": ("1", 47232225),
    "JUN": ("1", 58784048),
}
WINDOW = 250000


def main():
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading full h5ad...")
    a = ad.read_h5ad(H5AD_PATH)
    mask = a.obs["cell_type"].isin(CELL_TYPES).to_numpy()
    sub = a[mask]
    print(f"{mask.sum()} cells")

    rna = sub[:, PANEL_GENES].layers["counts"]
    rna = rna.toarray() if hasattr(rna, "toarray") else np.asarray(rna)
    rna_df = pd.DataFrame(rna, index=sub.obs_names, columns=PANEL_GENES)
    rna_df.to_csv(f"{OUT_DIR}/rna_counts.csv")
    print(f"RNA: {rna_df.shape}")

    atac_mask = (a.var["feature_types"] == "ATAC").to_numpy()
    atac_names = a.var_names[atac_mask]
    peaks = []
    for name in atac_names:
        chrom, s, e = name.split("-")
        peaks.append((chrom.replace("chr", ""), int(s), int(e), name))
    peaks_df = pd.DataFrame(peaks, columns=["chrom", "start", "end", "peak_id"])

    keep_peaks = set()
    for gene, (chrom, tss) in TSS.items():
        hits = peaks_df[(peaks_df.chrom == chrom) & (peaks_df.start < tss + WINDOW) & (peaks_df.end > tss - WINDOW)]
        keep_peaks |= set(hits.peak_id)
        print(f"  {gene}: {len(hits)} peaks within {WINDOW}bp")
    keep_peaks = sorted(keep_peaks)
    print(f"{len(keep_peaks)} unique peaks total (union across all 10 genes' +-250kb windows)")

    atac = sub[:, keep_peaks].layers["counts"]
    atac = atac.toarray() if hasattr(atac, "toarray") else np.asarray(atac)
    atac_df = pd.DataFrame(atac, index=sub.obs_names, columns=keep_peaks)
    atac_df.to_csv(f"{OUT_DIR}/atac_counts.csv")
    print(f"ATAC: {atac_df.shape}")

    peaks_df[peaks_df.peak_id.isin(keep_peaks)].to_csv(f"{OUT_DIR}/peaks_info.csv", index=False)

    pca = pd.DataFrame(sub.obsm["GEX_X_pca"], index=sub.obs_names)
    pca.columns = [f"PC_{i+1}" for i in range(pca.shape[1])]
    pca.to_csv(f"{OUT_DIR}/gex_pca.csv")
    print(f"PCA: {pca.shape}")

    meta = pd.DataFrame({"cell_type": sub.obs["cell_type"].to_numpy()}, index=sub.obs_names)
    meta.to_csv(f"{OUT_DIR}/cell_meta.csv")

    genome_info = pd.DataFrame({
        "genes": list(TSS.keys()),
        "Chrom": [f"chr{c}" for c, _ in TSS.values()],
        "Starts": [t for _, t in TSS.values()],
        "Ends": [t for _, t in TSS.values()],
    })
    genome_info.to_csv(f"{OUT_DIR}/genome_info.csv", index=False)
    print(f"\nAll exports -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
