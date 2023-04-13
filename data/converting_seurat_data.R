

library(Seurat)
library(SeuratDisk)


# final seurat object from Debbie
TKO_final <- readRDS("~/Box/multiome_data/TKO_final.rds")
TKO_final_updated <- UpdateSeuratObject(TKO_final)


### THIS DOESN'T WORK
SaveH5Seurat(TKO_final_updated, filename = "~/Documents/GitHub/multiome-analysis/data/TKO_final.h5Seurat", overwrite = TRUE)
Convert("~/Documents/GitHub/multiome-analysis/data/TKO_final.h5Seurat", dest = "h5ad")

