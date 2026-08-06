# CellOracle's own real-multiome tutorial pipeline, part 1: Cicero co-accessibility on the
# real ATAC peaks near our 10 genes (same 584-peak, +-250kb export used for DIRECT-NET),
# using the same real GEX PCA embedding for consistency.

suppressMessages({
  library(cicero)
  library(monocle3)
  library(Matrix)
})

DIR <- "data/hsc_multiome/direct_net"

atac <- read.csv(file.path(DIR, "atac_counts.csv"), row.names = 1, check.names = FALSE)
peaks_info <- read.csv(file.path(DIR, "peaks_info.csv"))
pca <- read.csv(file.path(DIR, "gex_pca.csv"), row.names = 1, check.names = FALSE)

atac_mat <- t(as(as.matrix(atac), "sparseMatrix"))  # peaks x cells
rownames(atac_mat) <- gsub("-", "_", peaks_info$peak_id)
cat(sprintf("ATAC: %d peaks x %d cells\n", nrow(atac_mat), ncol(atac_mat)))

pd <- data.frame(row.names = colnames(atac_mat))
fd <- data.frame(gene_short_name = rownames(atac_mat), row.names = rownames(atac_mat))

cds <- suppressWarnings(new_cell_data_set(atac_mat, cell_metadata = pd, gene_metadata = fd))

# Use the real GEX PCA embedding as Cicero's reduced-dimension coordinates (consistent with
# DIRECT-NET's own reduction choice) -- cicero::make_cicero_cds accepts precomputed
# reduced_coordinates directly, no need to recompute a UMAP from just 584 peaks.
pca_mat <- as.matrix(pca)
rownames(pca_mat) <- colnames(atac_mat)
umap_coords <- pca_mat[, 1:2]  # cicero's own convention expects a 2D reduced-coordinate matrix

set.seed(123)
cicero_cds <- make_cicero_cds(cds, reduced_coordinates = umap_coords)
cat("Cicero aggregated cds built.\n")

# run_cicero expects exactly 2 columns (chr, length) -- passing a 3rd "start" column shifts
# its internal positional access (x$V2 reads column 2) onto the wrong value, hitting
# "wrong sign in 'by' argument" since it then treats our all-zero start as the chr length.
genome_coords <- data.frame(
  chr = unique(peaks_info$chrom),
  length = 300000000
)  # generous chromosome-length placeholder (> any real human chromosome); only bounds the
   # internal distance-parameter search, doesn't need to be exact

conns <- run_cicero(cicero_cds, genome_coords, sample_num = 100)
cat(sprintf("\n%d real Cicero co-accessibility peak pairs computed\n", nrow(conns)))
write.csv(conns, file.path(DIR, "cicero_connections.csv"), row.names = FALSE)
cat(sprintf("-> %s\n", file.path(DIR, "cicero_connections.csv")))
print(head(conns[order(-conns$coaccess), ], 15))
