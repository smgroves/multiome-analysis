"""Extract the HSC branch-point cell types + the 10 mappable Krumsiek-model genes from the
real GSE194122 BMMC multiome dataset (see benchmarking/README.md, Track 2), and build
boba-T's CellID x gene input format.

CORRECTED (caught by inspecting bb.load.load_data's actual "actual" column in
fit_validation output -- only 2 distinct values where 2287 were expected): an earlier
version of this script min-max normalized the already-log1p X directly, then handed it to
bb.load.load_data(..., norm=0.3). That's a double-normalization bug specific to real,
dropout-heavy scRNA data: GATA1 etc. are ~92% exact zeros here, so load_data's own
norm=0.3 quantile-clip (`lq,uq = quantile(0.3),quantile(0.7)`; `(data-lq)/(uq-lq)`) hits
`uq-lq=0` (both percentiles land on the shared zero mode), collapsing EVERY nonzero cell to
`v/0 -> +inf -> clipped to 1` and every zero cell to `0/0 -> NaN -> filled to 0` --
i.e. real continuous signal collapsed to a hard binary encoding before boba-T ever saw it.
This never showed up in Tracks 1/3 because BoolODE's synthetic data has no real dropout.

Fix, matching this project's own established real-single-cell-data convention (see
network-inference-DIRECT-NET/preprocess_organoid.py's docstring for the same lesson learned
independently there): normalize raw counts by a genome-wide size factor -> log1p -> MAGIC
impute (same package/solver used throughout this project) -- do NOT additionally min-max
normalize here; let bb.load.load_data's own norm=0.3 do the final [0,1] rescaling on the
now-smooth, dropout-free imputed values, exactly as every other real-data script in this
project does.

Cell types: HSC + MK/E prog + G/M prog (the branch point itself) plus the downstream fates
it resolves into on both arms -- erythroid (Erythroblast, Proerythroblast, Normoblast) and
myeloid (CD14+ Mono, CD16+ Mono, cDC2, pDC, ID2-hi myeloid prog). Excludes lymphoid
(T/B/NK) populations entirely -- outside the Krumsiek model's scope.

EGRNAB is dropped here (unlike Tracks 1/3): it has no clean single-gene real expression
proxy (EGR1/NAB2 protein complex in the model), so there's no real column to extract for it.

Run in bobaT_env (has both anndata and magic; celloracle_env has anndata but not magic):
    /opt/anaconda3/envs/bobaT_env/bin/python preprocess_hsc_multiome.py
"""
import magic
import numpy as np
import pandas as pd
import anndata as ad

H5AD_PATH = "data/hsc_multiome/multiome_BMMC_processed.h5ad"
OUT_DIR = "data/hsc_multiome"

CELL_TYPES = [
    "HSC", "MK/E prog", "G/M prog",
    "Erythroblast", "Proerythroblast", "Normoblast",
    "CD14+ Mono", "CD16+ Mono", "cDC2", "pDC", "ID2-hi myeloid prog",
]

# Real GEX symbol -> HSC-model gene name (same mapping as Track 3's ChEA build).
GENE_MAP = {
    "GATA1": "GATA1", "GATA2": "GATA2", "SPI1": "PU1", "FLI1": "FLI1",
    "CEBPA": "CEBPA", "KLF1": "EKLF", "GFI1": "GFI1", "ZFPM1": "FOG1",
    "TAL1": "SCL", "JUN": "CJUN",
}


N_HVG = 2000


def main():
    print("Loading full h5ad (69249 cells x 129921 features)...")
    a = ad.read_h5ad(H5AD_PATH)

    mask = a.obs["cell_type"].isin(CELL_TYPES).to_numpy()
    print(f"{mask.sum()} cells across {len(CELL_TYPES)} branch-point cell types")

    gex_mask = (a.var["feature_types"] == "GEX").to_numpy()
    sub = a[mask, gex_mask]  # ALL 13431 GEX genes, not just the 10 -- MAGIC needs a real
                             # neighbor graph; 10 genes alone is too thin (many all-zero
                             # cells become spurious exact-distance-0 "duplicates").
    raw = sub.layers["counts"]
    raw = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
    raw = pd.DataFrame(raw, index=sub.obs_names, columns=sub.var_names)

    size_factor = sub.obs["GEX_size_factors"].to_numpy()
    normalized = raw.div(size_factor, axis=0)
    log_normalized = np.log1p(normalized)

    # Simple dispersion-based HVG selection (var/mean on the log-normalized matrix; no
    # scanpy in bobaT_env, and this doesn't need scanpy's exact binning method) -- top 2000,
    # force-union with the 10 HSC-panel genes so they're always in the graph regardless of
    # whether they're independently highly variable in this specific cell subset.
    gene_mean = log_normalized.mean(axis=0)
    gene_var = log_normalized.var(axis=0)
    dispersion = gene_var / gene_mean.replace(0, np.nan)
    hvgs = dispersion.dropna().sort_values(ascending=False).head(N_HVG).index.tolist()
    panel_genes_real = list(GENE_MAP.keys())
    keep_genes = sorted(set(hvgs) | set(panel_genes_real))
    print(f"{len(hvgs)} HVGs + {len(panel_genes_real)} panel genes -> {len(keep_genes)} genes for MAGIC's graph")
    log_normalized_hvg = log_normalized[keep_genes]

    print("Panel genes' post normalize+log1p distinct-value counts (sanity check vs. the earlier bug):")
    print(log_normalized_hvg[panel_genes_real].nunique())

    magic_operator = magic.MAGIC(solver="approximate")
    imputed = magic_operator.fit_transform(log_normalized_hvg)
    print(f"\nMAGIC done: {imputed.shape}")
    imputed_panel = imputed[panel_genes_real].rename(columns=GENE_MAP)
    print("Post-MAGIC distinct-value counts (should be ~n_cells, confirming real imputed continuity):")
    print(imputed_panel.nunique())

    imputed_panel.index.name = "CellID"
    imputed_panel.to_csv(f"{OUT_DIR}/expr_bobat_real.csv")
    print(f"\n{imputed_panel.shape[0]} cells x {imputed_panel.shape[1]} genes -> {OUT_DIR}/expr_bobat_real.csv")

    clusters = pd.DataFrame({"class": sub.obs["cell_type"].to_numpy()}, index=sub.obs_names)
    clusters.index.name = "CellID"
    clusters.to_csv(f"{OUT_DIR}/clusters_bobat_real.csv")
    print(clusters["class"].value_counts())


if __name__ == "__main__":
    main()
