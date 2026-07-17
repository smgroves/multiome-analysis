"""Preprocess raw scRNA-seq the same way the CellOracle benchmark did.

The CellOracle Fig-7 benchmark handed every GRN method (CellOracle, GENIE3, SCENIC,
WGCNA, DCOL) the SAME per-sample input, produced by CellOracle's standard scRNA
pipeline. Paper Methods (Kamimoto et al. 2023), verbatim:

    "zero-count genes are first filtered out by UMI count using
     scanpy.pp.filter_genes(min_counts=1). After normalization by total UMI count per
     cell using sc.pp.normalize_per_cell(key_n_counts='n_counts_all'), highly variable
     genes are detected by scanpy.pp.filter_genes_dispersion(n_top_genes=2000~3000)."

This module reproduces that and writes the four files the benchmark's runners consume
(see original_code/do_celloracle.py in janursa/CO_evaluation):

    log_data.mtx    genes x cells, log1p(normalized) counts over ALL kept genes
    all_genes.csv   column "x" = gene symbols (rows of log_data.mtx)
    var_genes.csv   column "x" = highly variable genes (the modelling feature set)
    meta_data.csv   indexed by cell barcode; has a "cluster" column (GRN unit) + tissue

boba-T consumes the same log_data / var_genes, so the comparison is like-for-like.

NOTE on choices the paper leaves open (flagged so they're easy to change):
  - n_top_genes default 3000 (paper says "2000~3000").
  - clusters (the GRN unit) via Louvain on the HVG/scaled/PCA space; resolution 1.0.
    The benchmark aggregates edges across clusters (max), so resolution has modest
    effect, but it is exposed as a parameter.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

from .config import BENCH_DATA

TABULA_MURIS_DIR = os.path.join(BENCH_DATA, "tabula_muris", "droplet")
TM_ANNOTATIONS = os.path.join(BENCH_DATA, "tabula_muris", "annotations_droplet.csv")
PREPROCESSED_DIR = os.path.join(BENCH_DATA, "preprocessed")

# The 13 Tabula Muris droplet channels matching the CellOracle benchmark samples.
# (Heart was renamed from TM's "Heart_and_Aorta-10X_P7_4" on download.)
BENCHMARK_CHANNELS = [
    "Heart-10X_P7_4",
    "Kidney-10X_P4_5", "Kidney-10X_P4_6", "Kidney-10X_P7_5",
    "Liver-10X_P4_2", "Liver-10X_P7_0", "Liver-10X_P7_1",
    "Lung-10X_P7_8", "Lung-10X_P7_9", "Lung-10X_P8_12", "Lung-10X_P8_13",
    "Spleen-10X_P4_7", "Spleen-10X_P7_6",
]


def load_10x_channel(channel_dir: str) -> sc.AnnData:
    """Load a Tabula Muris 10x triplet (barcodes/genes/matrix) as cells x genes."""
    adata = sc.read_mtx(os.path.join(channel_dir, "matrix.mtx")).T  # -> cells x genes
    genes = pd.read_csv(os.path.join(channel_dir, "genes.tsv"), sep="\t", header=None)
    barcodes = pd.read_csv(os.path.join(channel_dir, "barcodes.tsv"), header=None)[0].values
    # genes.tsv is [ensembl_id, symbol]; use the symbol to match ChIP-Atlas / base GRN.
    symbols = genes[1].values if genes.shape[1] > 1 else genes[0].values
    adata.var_names = [str(s) for s in symbols]
    adata.var_names_make_unique()
    adata.obs_names = [str(b) for b in barcodes]
    return adata


def annotated_cells(channel: str, annotations_path: str = TM_ANNOTATIONS) -> pd.Series:
    """Map QC'd barcode -> cell_ontology_class for a channel, from Tabula Muris annotations.

    The raw 10x matrices include all droplet barcodes; the benchmark used only the
    annotated (QC-passed) cells. Annotation 'cell' ids are '<channel_id>_<barcode>'
    with no '-1' suffix (channel_id e.g. '10X_P8_12'); channel folders are
    '<Tissue>-<channel_id>' with barcodes like 'AAAC...-1'.
    """
    channel_id = channel.split("-", 1)[1]              # "Lung-10X_P8_12" -> "10X_P8_12"
    ann = pd.read_csv(annotations_path, usecols=["cell", "cell_ontology_class"])
    ann = ann[ann["cell"].str.startswith(channel_id + "_")]
    barcodes = ann["cell"].str[len(channel_id) + 1:]   # strip channel prefix
    return pd.Series(ann["cell_ontology_class"].values, index=barcodes.values)


def preprocess_channel(
    channel: str,
    tabula_muris_dir: str = TABULA_MURIS_DIR,
    out_root: str = PREPROCESSED_DIR,
    n_top_genes: int = 3000,
    min_counts: int = 1,
    louvain_resolution: float = 1.0,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    seed: int = 123,
) -> str:
    """Preprocess one channel and write the four benchmark input files. Returns out dir."""
    channel_dir = os.path.join(tabula_muris_dir, channel)
    tissue = channel.split("-")[0]
    out_dir = os.path.join(out_root, channel)
    os.makedirs(out_dir, exist_ok=True)

    adata = load_10x_channel(channel_dir)
    adata.obs["tissue"] = tissue

    # 0. keep only Tabula Muris QC'd / annotated cells (raw mtx has all droplets)
    cell_types = annotated_cells(channel)
    bc_no_suffix = adata.obs_names.str.replace(r"-\d+$", "", regex=True)
    keep = np.asarray(bc_no_suffix.isin(cell_types.index))
    adata = adata[keep].copy()
    adata.obs["cell_ontology_class"] = cell_types.reindex(bc_no_suffix[keep]).values
    print(f"[preprocess] {channel}: {int(keep.sum())} annotated cells "
          f"(of {len(keep)} droplets)")

    # 1. drop zero-count genes
    sc.pp.filter_genes(adata, min_counts=min_counts)

    # 2. normalize by total UMI per cell (stores raw totals in obs['n_counts_all'])
    sc.pp.normalize_per_cell(adata, key_n_counts="n_counts_all")

    # 3. highly variable genes on the normalized (non-log) counts
    filt = sc.pp.filter_genes_dispersion(
        adata.X, flavor="cell_ranger", n_top_genes=min(n_top_genes, adata.n_vars), log=False
    )
    var_genes = adata.var_names[filt.gene_subset].tolist()

    # 4. log1p -> this is log_data, over ALL kept genes
    sc.pp.log1p(adata)

    # 5. cluster (the GRN unit): HVG subset -> scale -> PCA -> neighbours -> Louvain
    clus = adata[:, var_genes].copy()
    sc.pp.scale(clus, max_value=10)
    n_comps = min(n_pcs, clus.n_obs - 1, clus.n_vars - 1)
    sc.tl.pca(clus, n_comps=n_comps, random_state=seed)
    sc.pp.neighbors(clus, n_neighbors=min(n_neighbors, clus.n_obs - 1),
                    n_pcs=n_comps, random_state=seed)
    sc.tl.louvain(clus, resolution=louvain_resolution, random_state=seed)
    adata.obs["cluster"] = "cl" + clus.obs["louvain"].astype(str).values

    # --- write the four files the benchmark methods consume ---
    # log_data.mtx is genes x cells (do_celloracle reads it and transposes).
    X = adata.X.T  # genes x cells
    scipy.io.mmwrite(os.path.join(out_dir, "log_data.mtx"),
                     sp.csr_matrix(X) if not sp.issparse(X) else X)
    pd.DataFrame({"x": adata.var_names.tolist()}).to_csv(
        os.path.join(out_dir, "all_genes.csv"), index=False)
    pd.DataFrame({"x": var_genes}).to_csv(
        os.path.join(out_dir, "var_genes.csv"), index=False)
    adata.obs.to_csv(os.path.join(out_dir, "meta_data.csv"))

    print(f"[preprocess] {channel}: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{len(var_genes)} HVGs, {adata.obs['cluster'].nunique()} clusters -> {out_dir}")
    return out_dir


def preprocess_all_channels(channels=BENCHMARK_CHANNELS, **kwargs) -> list[str]:
    """Preprocess every benchmark channel. Extra kwargs pass through to preprocess_channel."""
    out = []
    for ch in channels:
        if not os.path.isdir(os.path.join(kwargs.get("tabula_muris_dir", TABULA_MURIS_DIR), ch)):
            print(f"[skip] {ch}: not found under {TABULA_MURIS_DIR}")
            continue
        out.append(preprocess_channel(ch, **kwargs))
    return out
