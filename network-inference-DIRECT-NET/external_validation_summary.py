import booleabayes as bb
import os
import os.path as op
import pandas as pd
from bb_utils import *
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.gridspec as gridspec

# setup parameters for visual formatting
mpl.rcParams['figure.max_open_warning'] = 0
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1.25
cmap = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'

brcd = str(9999)
validation_fname = "validation"
VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"

subfolders = [f.path for f in os.scandir(VAL_DIR) if f.is_dir()]
print(subfolders)

stats_df = pd.DataFrame(columns=["dataset", "gene", "accuracy", "balanced_accuracy_score", "f1", "roc_auc_score",
                                 "precision", "recall", "explained_variance", "max_error", "r2", "log-loss"])
for f in subfolders:
    try:
        if f.split("-")[-1] == 'validation':
            tmp = pd.read_csv(f"{f}/summary_stats.csv")
            tmp['dataset'] = f.split("/")[-1].split("-")[0]
            stats_df = pd.concat([stats_df, tmp])
    except FileNotFoundError: pass

stats_df = stats_df.drop('Unnamed: 0', axis = 1)

graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=False, remove_selfloops=False,
                                              remove_sources=False)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

sources = []
sinks = []
for i in range(len(nodes)):
    if graph.vp.source[i] == 1: sources.append(graph.vp.name[i])
    if graph.vp.sink[i] == 1: sinks.append(graph.vp.name[i])
print("Sources: ", len(sources), sources)
print("Sinks: ",len(sinks), sinks)

stats_df['source'] = [True if x in sources else False for x in stats_df['gene']]

print(stats_df.head())

# Remove sources, which will be artificially high
stats_df = stats_df.loc[stats_df['source']== False]
#
# pdf = PdfPages(f'{VAL_DIR}/external_validation_summary.pdf')
#
# for stat in ["accuracy", "balanced_accuracy_score", "f1", "roc_auc_score",
#              "precision", "recall", "explained_variance", "max_error", "r2", "log-loss"]:
#     figure = plt.figure(figsize = (5,len(stats_df.dataset.unique())))
#     gs = gridspec.GridSpec(1, 1)
#     ax = plt.subplot(gs[0])
#     sns.boxplot(data = stats_df, y = 'dataset', x = stat, order = sorted(stats_df.dataset.unique()),
#                 ax = ax)
#     sns.stripplot(data = stats_df, y = 'dataset', x = stat, alpha = .5,
#                   order = sorted(stats_df.dataset.unique()), ax = ax, color=".2")
#     plt.tight_layout()
#     pdf.savefig()
#     plt.clf()
#
# pdf.close()
#
#
# pdf = PdfPages(f'{VAL_DIR}/external_validation_summary_by_gene.pdf')
#
# for stat in ["accuracy", "balanced_accuracy_score", "f1", "roc_auc_score",
#              "precision", "recall", "explained_variance", "max_error", "r2", "log-loss"]:
#     figure = plt.figure(figsize = (5,len(stats_df.gene.unique())))
#     gs = gridspec.GridSpec(1, 1)
#     ax = plt.subplot(gs[0])
#     sns.boxplot(data = stats_df, y = 'gene', x = stat, order = sorted(stats_df.gene.unique()),
#                 ax = ax, color = 'white')
#     sns.swarmplot(data = stats_df, y = 'gene', x = stat, alpha = .5, hue = 'dataset',
#                   order = sorted(stats_df.gene.unique()), ax = ax, )
#     plt.tight_layout()
#     pdf.savefig()
#     plt.clf()
#
#
# pdf.close()


def plot_summary_boxplots(stats_df, VAL_DIR, file_name, groupby = 'dataset', hue = None, dots = 'swarm',height = 1.0,
                          stats = ["accuracy", "balanced_accuracy_score", "f1", "roc_auc_score","precision",
                                   "recall", "explained_variance", "max_error", "r2", "log-loss"]):
    pdf = PdfPages(f'{VAL_DIR}/{file_name}')

    for stat in stats:
        print(stat)
        figure = plt.figure(figsize=(5, height*len(stats_df[groupby].unique())))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        sns.boxplot(data=stats_df, y=groupby, x=stat, order=sorted(stats_df[groupby].unique()),
                    ax=ax, color='white')
        if dots == 'swarm':
            sns.swarmplot(data=stats_df, y=groupby, x=stat, hue=hue,
                      order=sorted(stats_df[groupby].unique()), ax=ax, )
        elif dots == 'strip':
            sns.stripplot(data=stats_df, y='dataset', x=stat, alpha=.5,
                          order=sorted(stats_df.dataset.unique()), ax=ax, color=".2")
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

    pdf.close()

plot_summary_boxplots(stats_df, VAL_DIR, file_name='external_validation_summary_by_gene.pdf',groupby='gene',
                      hue = 'dataset',dots = 'swarm', height = .5, stats = ['accuracy'])