import numpy as np
import seaborn as sns
import booleabayes as bb
from bb_utils import *
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
def binarize_data_df(
    data,
    nodes,
    threshold=0.5,
):
    binaries = pd.DataFrame(columns=nodes)
    f = np.vectorize(lambda x: "0" if x < threshold else "1")
    for sample in data.index:
        idx = bb.state2idx("".join(f(data.loc[sample])))
        bin = (bb.idx2binary(idx, len(nodes)))
        att_list = [int(i) for i in bin]
        binaries = binaries.append(pd.DataFrame(att_list, index=nodes, columns=[sample]).T)
    return binaries


node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks.csv'
data_path = f'data/adata_imputed_combined_v3.csv'
t1 = False
data_t1_path = None #if no T1 (i.e. single dataset), replace with None
cellID_table = 'data/AA_clusters_splitgen.csv'
cluster_header_list = ["class"]

brcd = str(6666)
print(brcd)

# brcd =str(35468)
# random_state = str(random.Random.randint(0,99999)) #for train-test split
random_state = 1234

remove_sinks = False
remove_selfloops = True
remove_sources = False

graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                              remove_sources=remove_sources)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

embedding = "./data/umap_wnn_combined.csv"
def visualize_normalization(dir_prefix, data_path, nodes, node_normalization, embedding, transpose = True):
    data_t0 = bb.load.load_data(f'{dir_prefix}/{data_path}', nodes, norm=node_normalization,
                            delimiter=',', log1p=False, transpose=transpose,
                            sample_order=False, fillna=0)
    print("binarizing...")
    binarized_data_df = binarize_data_df(data_t0, nodes, threshold=0.5)
    umap = pd.read_csv(embedding, index_col=0, header=0)
    data_orig = pd.read_csv(f"{dir_prefix}/{data_path}", index_col=0, header=0)
    data_orig.columns = [i.upper() for i in data_orig.columns]
    clusters = bb.utils.get_clusters(data_t0, cellID_table=f"{dir_prefix}/{cellID_table}",
                                     cluster_header_list=cluster_header_list)

    print("PCA...")
    pca = PCA(n_components=2)
    binarized_data_df_new = pca.fit_transform(binarized_data_df)
    data = pd.DataFrame(binarized_data_df_new, columns=['0', '1'], index=binarized_data_df.index)
    data['color'] = clusters['class']
    pal = {"Generalist_NE": "lightgrey", "Generalist_nonNE": "grey", "Arc_1": "red", "Arc_2": "orange",
           "Arc_3": "yellow", "Arc_4": "green", "Arc_5": "blue", "Arc_6": "purple"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=200)  # 1 row, 2 columns
    sns.scatterplot(x=umap['UMAP1'], y=umap['UMAP2'], hue=clusters['class'],palette=pal, edgecolor='none', s=2, ax=axs[0])
    axs[0].set_title("UMAP of Original Data Colored by Phenotype")

    scatterplot = sns.scatterplot(data=data, x='0', y='1', hue='color', alpha=0.7, palette=pal, edgecolor='none', s=10, ax=axs[1])
    scatterplot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[1].set_title("PCA of Binarized Data Colored by Phenotype")
    plt.tight_layout()
    pdf.savefig()
    plt.clf()

    for gene in nodes:
        print(gene)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=200)  # 1 row, 2 columns
        sns.scatterplot(x=umap['UMAP1'], y=umap['UMAP2'], c=data_orig[gene], edgecolor='none', s = 2, ax = axs[0])
        cbar1 = plt.colorbar(axs[0].collections[0], orientation='vertical', ax=axs[0])
        axs[0].set_title(f"Original Expression for {gene} on UMAP")
        # Customize the colorbar label
        cbar1.set_label("Ascl1", rotation=90, labelpad=10)

        sns.scatterplot(x=umap['UMAP1'], y=umap['UMAP2'], c=data_t0[gene], edgecolor="none", s=2, ax=axs[1])
        axs[1].set_title(f"Normalized Expression for {gene} on UMAP")
        cbar2 = plt.colorbar(axs[1].collections[0], orientation='vertical', ax=axs[1])
        cbar2.set_label("Ascl1", rotation=90, labelpad=10)

        sns.scatterplot(x=umap['UMAP1'], y=umap['UMAP2'], hue=binarized_data_df[gene],palette = {0:'lightgrey',1:'black'}, edgecolor="none", s=2, ax=axs[2])
        axs[2].set_title(f"Binarized Expression for {gene} on UMAP")
        plt.tight_layout()
        pdf.savefig()
        plt.close()



norms = ['gmm','minmax',0.3, 0.2, 0.1]
for node_normalization in norms:
    print(node_normalization)
    pdf = PdfPages(f'{dir_prefix}/{brcd}/normalization_plots/norm_{node_normalization}.pdf')
    visualize_normalization(dir_prefix, data_path, nodes, node_normalization, embedding, transpose = True)
    pdf.close()
