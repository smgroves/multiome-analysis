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
# mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 1.25
cmap = mpl.rcParams['axes.prop_cycle'].by_key()['color']

# Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = 'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_NO_sinks_NOCD24_expanded.csv'
brcd = str(9999)

def get_summary_stats(VAL_DIR, dir_prefix, network_path, remove_sources = True):
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
    if remove_sources:
        stats_df = stats_df.loc[stats_df['source']== False]
    return stats_df

def plot_summary_boxplots(stats_df, VAL_DIR, file_name, groupby = 'dataset', hue = None, dots = 'swarm',height = 1.0,
                          stats = ["accuracy", "balanced_accuracy_score", "f1", "roc_auc_score","precision",
                                   "recall", "explained_variance", "max_error", "r2", "log-loss"],
                          ordered_by = "alphabetical"):
    pdf = PdfPages(f'{VAL_DIR}/{file_name}')

    for stat in stats:
        print(stat)
        figure = plt.figure(figsize=(5, height*len(stats_df[groupby].unique())))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])
        if ordered_by == 'alphabetical':
            order = sorted(stats_df[groupby].unique())
        elif ordered_by == 'mean':
            order = stats_df.groupby(groupby).mean().sort_values(by = stat, ascending=False).index
        elif ordered_by == 'median':
            order = stats_df.groupby(groupby).median().sort_values(by = stat, ascending = False).index
        PROPS = {'boxprops': {'facecolor': 'none', 'edgecolor': 'black'}}
        sns.boxplot(data=stats_df, y=groupby, x=stat, order=order,
                    ax=ax, color='white', **PROPS)
        if dots == 'swarm':
            sns.swarmplot(data=stats_df, y=groupby, x=stat, hue=hue,
                      order=order, ax=ax, size = 3)
        elif dots == 'strip':
            sns.stripplot(data=stats_df, y='dataset', x=stat, alpha=.5,
                          order=order, ax=ax, color=".2")
        if stat in ["explained_variance", "r2"]:
            plt.axvline(x = 0, linestyle = '--', color = 'lightgray')
        plt.setp(ax.lines, color=".1")
        plt.xticks(size = 16)
        plt.yticks(size = 16)
        plt.xlabel(stat.capitalize(), size = 20)
        plt.ylabel(groupby.capitalize(), size = 20)
        plt.tight_layout()
        pdf.savefig()
        plt.clf()

    pdf.close()

## Plots for allograft validation datasets
validation_folder = "validation/allograft_validation"
VAL_DIR = f"{dir_prefix}/{brcd}/{validation_folder}"

stats_df = get_summary_stats(VAL_DIR, dir_prefix, network_path)
# plot_summary_boxplots(stats_df, VAL_DIR, file_name='external_validation_summary_mean.pdf',groupby='dataset',
#                       hue = None,dots = 'strip', height = 1, ordered_by='mean')

plot_summary_boxplots(stats_df, VAL_DIR, file_name='external_validation_summary_mean_by_gene.pdf',groupby='gene',
                      hue = 'dataset',dots = 'swarm', height = .5, ordered_by='mean')