

library(Seurat)
library(SeuratDisk)


# final seurat object from Debbie
data <- readRDS("~/Box/multiome_data/Allograft_mnn.rds")
data_updated <- UpdateSeuratObject(data)


### THIS DOESN'T WORK FOR MULTIOME DATA
SaveH5Seurat(data_updated, filename = "~/Documents/GitHub/multiome-analysis/data/allografts.h5Seurat", overwrite = TRUE)
Convert("~/Documents/GitHub/multiome-analysis/data/allografts.h5Seurat", dest = "h5ad")

