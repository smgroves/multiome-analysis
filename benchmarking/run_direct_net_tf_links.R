# Second half of DIRECT-NET's real pipeline: motif-scan the CREs it identified for our 10
# TFs' real binding motifs (matching this project's own existing DIRECT-NET-FILES/Direct_net*
# .csv schema: "regulatory regions","TF motif","motif score","Target_gene").

suppressMessages({
  library(Seurat)
  library(DIRECTNET)
  library(chromVAR)
  library(motifmatchr)
  library(JASPAR2020)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(GenomicRanges)
})

obj <- readRDS("data/hsc_multiome/direct_net/direct_net_result.rds")
dn <- Misc(obj)$direct.net
cat(sprintf("%d real CRE-gene links across %d genes (function_type: %s)\n",
            length(dn$gene), length(unique(dn$gene)), paste(table(dn$function_type), collapse=",")))

parse_peak <- function(p) {
  parts <- strsplit(p, "_")[[1]]
  n <- length(parts)
  data.frame(chrom = paste(parts[1:(n-2)], collapse = "_"),
             start = as.numeric(parts[n-1]), end = as.numeric(parts[n]))
}

all_peaks <- unique(c(dn$Peak1, dn$Peak2))
peaks_bed <- do.call(rbind, lapply(all_peaks, parse_peak))
peaks_bed$R.chrom <- peaks_bed$chrom
peaks_bed$R.start <- peaks_bed$start
peaks_bed$R.end <- peaks_bed$end
cat(sprintf("%d unique real CRE peaks to motif-scan\n", nrow(peaks_bed)))

markers <- data.frame(gene = c("GATA1","GATA2","SPI1","FLI1","CEBPA","KLF1","GFI1","ZFPM1","TAL1","JUN"),
                       group = "focus")

tf_links <- generate_peak_TF_links(
  peaks_bed_list = list(peaks_bed),
  species = "Homo sapiens",
  genome = BSgenome.Hsapiens.UCSC.hg38,
  markers = markers
)
tf_df <- tf_links[[1]]
cat(sprintf("\n%d real peak-TF hits (restricted to our 10 TFs)\n", nrow(tf_df)))
print(head(tf_df, 20))

write.csv(dn, "data/hsc_multiome/direct_net/cre_gene_links.csv", row.names = FALSE)
write.csv(tf_df, "data/hsc_multiome/direct_net/peak_tf_links.csv", row.names = FALSE)
cat("\n-> data/hsc_multiome/direct_net/{cre_gene_links,peak_tf_links}.csv\n")
