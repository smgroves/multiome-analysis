# Build a Seurat object from the exports in data/hsc_multiome/direct_net/ and run
# DIRECT-NET's real peak-gene-TF linkage pipeline, restricted to our 10 HSC-panel genes
# as focus_markers, using the same +-250kb window DIRECT-NET's own code hardcodes.
#
# Run: Rscript run_direct_net.R

suppressMessages({
  library(Seurat)
  library(Signac)
  library(DIRECTNET)
  library(Matrix)
})

# DIRECT-NET's Aggregate_data accesses object@assays$RNA@counts directly (Seurat V3/V4 API);
# Seurat 5's default Assay5 class stores data in @layers instead and has no @counts slot at
# all, which is exactly what broke on the first run. Seurat 5 keeps this global option for
# exactly this backward-compatibility case -- force classic Assay objects, not Assay5.
options(Seurat.object.assay.version = "v3")

DIR <- "data/hsc_multiome/direct_net"

rna <- read.csv(file.path(DIR, "rna_counts.csv"), row.names = 1, check.names = FALSE)
atac <- read.csv(file.path(DIR, "atac_counts.csv"), row.names = 1, check.names = FALSE)
peaks_info <- read.csv(file.path(DIR, "peaks_info.csv"))
pca <- read.csv(file.path(DIR, "gex_pca.csv"), row.names = 1, check.names = FALSE)
meta <- read.csv(file.path(DIR, "cell_meta.csv"), row.names = 1, check.names = FALSE)
genome_info <- read.csv(file.path(DIR, "genome_info.csv"))

cat(sprintf("RNA: %d cells x %d genes\n", nrow(rna), ncol(rna)))
cat(sprintf("ATAC: %d cells x %d peaks\n", nrow(atac), ncol(atac)))

# cells x genes -> genes x cells (Seurat convention), sparse
rna_mat <- t(as(as.matrix(rna), "sparseMatrix"))
atac_mat <- t(as(as.matrix(atac), "sparseMatrix"))
rownames(atac_mat) <- gsub("-", "_", peaks_info$peak_id)  # match DIRECT-NET's own convention

# old-style Assay (has @counts directly) -- DIRECT-NET's Aggregate_data accesses
# object@assays$RNA@counts / object@assays$ATAC@counts directly, a pre-Seurat-V5 API.
rna_assay <- CreateAssayObject(counts = rna_mat)
atac_assay <- CreateAssayObject(counts = atac_mat)

obj <- CreateSeuratObject(counts = rna_assay@counts, assay = "RNA", meta.data = meta)
obj[["ATAC"]] <- atac_assay
Idents(obj) <- obj$cell_type

# Aggregate_data hardcodes reduction.name=NULL in its own call to itself regardless of what
# Run_DIRECT_NET's caller passes -- a real bug in DIRECTNET's own source (checked directly:
# `Aggregate_data(object, ..., reduction.name = NULL, ...)` is hardcoded inside
# Run_DIRECT_NET's body). It then falls back to the literal, hardcoded slot name
# `object@reductions$wnn.umap`. Non-invasive workaround: name our real PCA embedding
# "wnn.umap" -- Seurat doesn't enforce that a reduction's name matches its algorithm, it's
# just a label DIRECT-NET's own hardcoded lookup needs to find.
pca_mat <- as.matrix(pca)
rownames(pca_mat) <- rownames(meta)
obj[["wnn.umap"]] <- CreateDimReducObject(embeddings = pca_mat, key = "PC_", assay = "RNA")

cat("Seurat object built:\n")
print(obj)

genome_info_dn <- data.frame(
  genes = genome_info$genes,
  Chrom = genome_info$Chrom,
  Starts = genome_info$Starts,
  Ends = genome_info$Ends
)

# DIRECTNET::isSparseMatrix does `class(x) %in% c("dgCMatrix","dgTMatrix")` -- for a plain
# dense matrix, class(x) returns c("matrix","array") (length 2, standard since R 4.0), so
# %in% returns a length-2 logical vector, and `if()` on that throws "condition has length
# > 1" under R's stricter if() checking. Confirmed real upstream bug, not an input issue --
# patched here at the call site (this project's established pattern for fixing a dependency
# without editing its installed source, e.g. the np.trapz shim used elsewhere), not by
# editing DIRECTNET's own files.
assignInNamespace("isSparseMatrix", function(x) any(class(x) %in% c("dgCMatrix", "dgTMatrix")), ns = "DIRECTNET")

cat("\nRunning DIRECT-NET...\n")
obj <- Run_DIRECT_NET(
  obj,
  peakcalling = FALSE,
  k_neigh = 50,
  atacbinary = TRUE,
  max_overlap = 0.8,
  reduction.name = "pca",
  size_factor_normalize = TRUE,
  genome.info = genome_info_dn,
  focus_markers = genome_info$genes,
  nthread = 4,
  verbose = TRUE
)

cat("\nDIRECT-NET finished. Links assay:\n")
print(obj)
saveRDS(obj, file.path(DIR, "direct_net_result.rds"))
cat(sprintf("\n-> %s\n", file.path(DIR, "direct_net_result.rds")))
