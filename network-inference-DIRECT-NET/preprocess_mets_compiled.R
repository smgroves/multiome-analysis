# Preprocess Mets_compiled.rds for external validation, matching preprocess_adata.py's
# convention for the allograft datasets (same gene panel: DIRECT-NET TF/target genes +
# FigR DORC-TF genes + the same fixed extra_genes list used for every other external
# validation set in this project).
#
# IMPORTANT, caught in review: an earlier version of this script used the RNA assay's
# "data" slot assuming it was Seurat's NormalizeData()-transformed (log-normalized) output
# -- it isn't. Checked directly: GetAssayData(obj, "RNA", "data") is byte-identical to
# "counts" (raw UMI counts) for this object; NormalizeData was never actually run/stored.
# This produced near-discrete, few-valued columns (e.g. REST had only 2 distinct values)
# that broke boba-T's [0,1] `norm=0.3` rescaling. This script now does the real pipeline
# the allografts' own preprocessing notebook used (archetype-plasticity-notebooks/
# external_network_validation/1-preprocess-tumor-datasets.ipynb): QC filter -> normalize
# by total UMI count -> log1p -> restrict to network genes -> MAGIC impute. The original
# notebook used a private lab package (`mazebox`) for its filtering/normalization step and
# `magic.MAGIC(solver='approximate')` for imputation; MAGIC itself is replicated exactly
# (same package, same solver) in preprocess_mets_compiled_magic.py, which this script's
# output feeds into -- R only exports raw counts + QC metadata here.
suppressMessages(library(Seurat))

box <- "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/from zenodo"
obj <- readRDS(file.path(box, "Mets_compiled.rds"))
cat("Loaded:", dim(obj), "cells x genes (RNA assay)\n")

direct_net <- read.csv("DIRECT-NET-FILES/Direct_net.csv", row.names = 1)
direct_net$Target_gene <- toupper(direct_net$Target_gene)
tfs <- unique(c(toupper(direct_net$TF.motif), direct_net$Target_gene))

figr <- read.csv("DIRECT-NET-FILES/FigR_DORC_TF.csv", row.names = 1)
tfs <- unique(c(tfs, toupper(figr$Motif), toupper(figr$DORC)))

extra_genes <- c('CD24', 'CD44', 'EPCAM', 'ICAM1', 'NCAM1', 'HES1', 'NFYC', 'NR6A1',
                  'RBPJ', 'RORA', 'RORB', 'SOX11', 'TFDP1')
tfs <- unique(c(tfs, extra_genes))
cat(length(tfs), "candidate genes (network + FigR + extra_genes)\n")

all_genes_upper <- toupper(rownames(obj[["RNA"]]))
overlap <- intersect(tfs, all_genes_upper)
cat(length(overlap), "of", length(tfs), "matched in Mets_compiled (mouse gene symbols)\n")
missing <- setdiff(tfs, all_genes_upper)
cat("not found:", paste(missing, collapse=", "), "\n")

gene_map <- setNames(rownames(obj[["RNA"]]), all_genes_upper)
mets_genes <- gene_map[overlap]

# Raw counts for the network gene panel -- normalization/log1p/MAGIC happens in Python
# (preprocess_mets_compiled_magic.py), using nCount_RNA (genome-wide, already computed by
# Seurat) as the size-factor basis rather than re-deriving it from only these ~93 genes.
raw_counts <- GetAssayData(obj, assay = "RNA", layer = "counts")[mets_genes, ]
raw_df <- as.data.frame(t(as.matrix(raw_counts)))
colnames(raw_df) <- toupper(colnames(raw_df))

dir.create("data/mets_compiled", showWarnings = FALSE)
write.csv(raw_df, "data/mets_compiled/raw_counts_network_genes.csv", row.names = TRUE)

qc <- obj@meta.data[, c("nCount_RNA", "nFeature_RNA", "percent.mt")]
write.csv(qc, "data/mets_compiled/qc_metadata.csv", row.names = TRUE)
write.csv(obj@meta.data, "data/mets_compiled/mets_compiled_clusters.csv", row.names = TRUE)
cat("Wrote data/mets_compiled/raw_counts_network_genes.csv:", dim(raw_df), "\n")
