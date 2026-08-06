# Extract real ATAC peak-to-gene assignments from TKO_final_arc.rds's `peaks` ChromatinAssay
# (mm10, 158,859 peaks, Signac gene annotation already attached) -- this is the real,
# dataset-specific ATAC data CellOracle's own multiome tutorial wants for base GRN
# construction, not a generic genome-wide promoter motif scan.
#
# Peak-to-gene assignment: nearest-feature (Signac's ClosestFeature), restricted to peaks
# within 10kb of a gene -- the "up/down 10kb" convention already used for the human
# cisTarget database in comparison_scenic_fit_6667.py, kept for consistency. This is the
# simpler alternative CellOracle's tutorial documents when Cicero co-accessibility isn't
# run (Cicero needs its own R package + co-accessibility computation across all peak pairs,
# a materially bigger step not attempted here).
suppressMessages({library(Seurat); library(Signac); library(GenomicRanges)})

box <- "/Users/xpz5km/Library/CloudStorage/Box-Box/_Research/SCLC_data/data_multiome/from zenodo"
obj <- readRDS(file.path(box, "TKO_final_arc.rds"))
cat("Loaded TKO_final_arc:", ncol(obj), "cells\n")

peaks_assay <- obj[["peaks"]]
gr <- granges(peaks_assay)
cat(length(gr), "peaks, genome:", unique(genome(gr)), "\n")

closest <- ClosestFeature(peaks_assay, regions = gr)
cat("ClosestFeature done:", nrow(closest), "rows\n")

peak_gene <- data.frame(
  peak_id = paste0(seqnames(gr), "_", start(gr), "_", end(gr)),
  gene_short_name = closest$gene_name,
  distance = closest$distance
)
peak_gene <- peak_gene[peak_gene$distance <= 10000, ]
cat(nrow(peak_gene), "peaks within 10kb of a gene (of", length(gr), "total)\n")
cat(length(unique(peak_gene$gene_short_name)), "unique genes with >=1 assigned peak\n")

dir.create("data/tko_atac", showWarnings = FALSE)
write.csv(peak_gene, "data/tko_atac/peak_to_gene.csv", row.names = FALSE)
cat("Wrote data/tko_atac/peak_to_gene.csv\n")

# Also export raw RNA counts + QC metadata, same shape as the fullscale 6667 pipeline
# (preprocess_sclc_full_6667.py), for the CellOracle/SCENIC fits that consume this base GRN.
raw <- GetAssayData(obj, assay = "RNA", layer = "counts")
Matrix::writeMM(raw, "data/tko_atac/raw_counts.mtx")
write.csv(data.frame(gene = rownames(raw)), "data/tko_atac/genes.csv", row.names = FALSE)
write.csv(data.frame(cell = colnames(raw)), "data/tko_atac/cells.csv", row.names = FALSE)
write.csv(obj@meta.data, "data/tko_atac/tko_clusters.csv", row.names = TRUE)
cat("Wrote raw_counts.mtx:", dim(raw), "\n")
