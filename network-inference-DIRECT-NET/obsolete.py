##################################
## Added Nov 9, 2022
##################################
## The functions below to plot perturbations have been copied into BooleaBayes package and
# should be run instead with the following commands in main:
#---------------------------------
# bb.tl.perturbations_summary(attractor_dict,perturbations_dir, show = False, save = True, plot_by_attractor = True,
#                             save_dir = "clustered_perturb_plots", save_full = True, significance = 'both', fname = "",
#                             ncols = 5, mean_threshold = -0.3)
#
# ## gene dict and plots with threshold = -0.2
# perturb_dict, full = bb.utils.get_perturbation_dict(attractor_dict, perturbations_dir, significance = 'both', save_full=False,
#                                                     mean_threshold=-0.2)
# perturb_gene_dict = bb.utils.reverse_perturb_dictionary(perturb_dict)
# bb.plot.plot_perturb_gene_dictionary(perturb_gene_dict, full,perturbations_dir,show = False, save = True, ncols = 5, fname = "_0.2")
#---------------------------------
## original code:
import random
import numpy as np
import seaborn as sns
import booleabayes as bb
import os
import os.path as op
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from graph_tool import all as gt
from graph_tool import GraphView
from bb_utils import *
import sys
import glob

dir_prefix = "/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
brcd = "3000"

def plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = True,
                                act_kd_together = False):
    for k in attractor_dict.keys():
        print(k)
        if clustered:
            try:
                os.mkdir(f"{perturbations_dir}/clustered_perturb_plots")
            except FileExistsError:
                pass
            results = pd.DataFrame(columns = ['attr','gene','perturb','score'])
            for attr in attractor_dict[k]:
                tmp = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                tmp.columns = ["attractor_dir","cluster","gene","perturb","score"]
                for i,r in tmp.iterrows():
                    results = results.append(pd.Series([attr, r['gene'],r['perturb'],r['score']],
                                                       index = ['attr','gene','perturb','score']), ignore_index=True)
            if act_kd_together:
                plt.figure()
                my_order = sorted(np.unique(results['gene']))
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')

                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results, x = 'gene', y = 'score', hue = 'perturb', order = my_order,
                                palette = {"activate":sns.color_palette("tab10")[0], "knockdown":sns.color_palette("tab10")[1]})
                else:
                    sns.boxplot(data = results, x = 'gene', y = 'score',hue = 'perturb', order = my_order,
                                palette = {"activate":sns.color_palette("tab10")[0], "knockdown":sns.color_palette("tab10")[1]})
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Perturbation for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/clustered_perturb_plots/{k}_scores.pdf")
                    plt.close()
            else:
                results_act = results.loc[results["perturb"] == 'activate']
                plt.figure()
                # my_order = results_act.sort_values(by = 'score')['gene'].values
                my_order = results_act.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')

                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order)
                else:
                    sns.boxplot(data = results_act, x = 'gene', y = 'score', order = my_order)
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Activation for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/clustered_perturb_plots/{k}_activation_scores.pdf")
                    plt.close()

                results_kd = results.loc[results["perturb"] == 'knockdown']

                plt.figure()
                # my_order = results_act.sort_values(by = 'score')['gene'].values
                my_order = results_kd.groupby(by=["gene"]).median().sort_values(by = 'score').index.values
                plt.axhline(y = 0, linestyle = "--", color = 'lightgrey')
                if len(attractor_dict[k]) == 1:
                    sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
                else:
                    sns.boxplot(data = results_kd, x = 'gene', y = 'score', order = my_order)
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title(f"Destabilization by TF Knockdown for {k} Attractors \n {len(attractor_dict[k])} Attractors")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/clustered_perturb_plots/{k}_knockdown_scores.pdf")
                    plt.close()

        else:
            for attr in attractor_dict[k]:
                results = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
                results.columns = ["attractor_dir","cluster","gene","perturb","score"]
                #activation plot
                results_act = results.loc[results["perturb"] == 'activate']
                colormat=list(np.where(results_act['score']>0, 'g','r'))
                results_act['color'] = colormat

                plt.figure()
                my_order = results_act.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_act, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Activation")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/activation_scores.pdf")
                    plt.close()

                #knockdown plot
                results_kd = results.loc[results["perturb"] == 'knockdown']
                colormat=list(np.where(results_kd['score']>0, 'g','r'))
                results_kd['color'] = colormat

                plt.figure()
                my_order = results_kd.sort_values(by = 'score')['gene'].values
                sns.barplot(data = results_kd, x = 'gene', y = 'score', order = my_order,
                            palette = ['r','g'], hue = 'color')
                plt.xticks(rotation = 90, fontsize = 8)
                plt.xlabel("Gene")
                plt.ylabel("Stabilization Score")
                plt.title("Destabilization by TF Knockdown")
                plt.legend([],[], frameon=False)
                plt.tight_layout()
                if show:
                    plt.show()
                if save:
                    plt.savefig(f"{perturbations_dir}/{attr}/knockdown_scores.pdf")
                    plt.close()
import math
def get_ci_sig(results, group_cols = ['gene'], score_col = 'score', mean_threshold = -0.3):
    stats = results.groupby(group_cols)[score_col].agg(['mean', 'count', 'std'])
    ci95_hi = []
    ci95_lo = []
    sig = []
    mean_sig = []
    for i in stats.index:
        m, c, s = stats.loc[i]
        ci95_hi.append(m + 1.96*s/math.sqrt(c))
        ci95_lo.append(m - 1.96*s/math.sqrt(c))
        if m < mean_threshold:
            mean_sig.append("yes")
        else:
            mean_sig.append('no')
        if m + 1.96*s/math.sqrt(c) < 0:
            sig.append('yes')
        else:
            sig.append("no")

    stats['ci95_hi'] = ci95_hi
    stats['ci95_lo'] = ci95_lo
    stats['ci_sig'] = sig
    stats['mean_sig'] = mean_sig
    return stats
def get_perturbation_dict(attractor_dict, perturbations_dir, significance = 'both', save_full = False, save_dir = "clustered_perturb_plots",
                          mean_threshold = -0.3):
    perturb_dict = {}
    full_results = pd.DataFrame(columns = ['cluster','attr','gene','perturb','score'])

    for k in attractor_dict.keys():
        print(k)
        results = pd.DataFrame(columns = ['attr','gene','perturb','score'])
        for attr in attractor_dict[k]:
            tmp = pd.read_csv(f"{perturbations_dir}/{attr}/results.csv", header = None, index_col = None)
            tmp.columns = ["attractor_dir","cluster","gene","perturb","score"]
            for i,r in tmp.iterrows():
                results = results.append(pd.Series([attr, r['gene'],r['perturb'],r['score']],
                                                   index = ['attr','gene','perturb','score']), ignore_index=True)
                full_results = full_results.append(pd.Series([k, attr, r['gene'],r['perturb'],r['score']],
                                                             index = ['cluster','attr','gene','perturb','score']), ignore_index=True)
        results_act = results.loc[results["perturb"] == 'activate']
        stats_act = get_ci_sig(results_act, mean_threshold=mean_threshold)

        results_kd = results.loc[results["perturb"] == 'knockdown']
        stats_kd = get_ci_sig(results_kd, mean_threshold=mean_threshold)

        if significance == 'ci':
            #activation is significantly destabilizing = destabilizer
            act_l = []
            for i,r in stats_act.iterrows():
                if r['ci_sig'] == "yes":
                    act_l.append(i)

            #knockdown is significantly destabilizing = destabilizer
            kd_l = []
            for i,r in stats_kd.iterrows():
                if r['ci_sig'] == "yes":
                    kd_l.append(i)
        elif significance == 'mean':
            act_l = []
            for i,r in stats_act.iterrows():
                if r['mean_sig'] == "yes":
                    act_l.append(i)

            kd_l = []
            for i,r in stats_kd.iterrows():
                if r['mean_sig'] == "yes":
                    kd_l.append(i)
        elif significance == 'both':
            #activation is significantly destabilizing = destabilizer
            act_l = []
            for i,r in stats_act.iterrows():
                if r['ci_sig'] == "yes":
                    if r['mean_sig'] == "yes":
                        act_l.append(i)
                elif len(attractor_dict[k]) == 1: #ci of single attractor DNE
                    if r['mean_sig'] == "yes":
                        act_l.append(i)
            #knockdown is significantly destabilizing = destabilizer
            kd_l = []
            for i,r in stats_kd.iterrows():
                if r['ci_sig'] == "yes":
                    if r['mean_sig'] == "yes":
                        kd_l.append(i)
                elif len(attractor_dict[k]) == 1: #ci of single attractor DNE
                    if r['mean_sig'] == "yes":
                        kd_l.append(i)
        else:
            print("significance must be one of {'ci','mean', 'both'}")
        perturb_dict[k] = {"Regulators":kd_l, "Destabilizers":act_l}

    if save_full:
        try:
            os.mkdir(f"{perturbations_dir}/{save_dir}")
        except FileExistsError:
            pass
        full_results.to_csv(f"{perturbations_dir}/{save_dir}/perturbations.csv")

    return perturb_dict, full_results
def reverse_dictionary(dictionary):
    return  {v: k for k, v in dictionary.items()}
def reverse_perturb_dictionary(dictionary):
    reverse_dict = {}
    for k,v in dictionary.items():
        # v is a dictionary too
        for reg_type, genes in v.items():
            for gene in genes:
                if gene not in reverse_dict.keys():
                    reverse_dict[gene] = {"Regulators":[], "Destabilizers":[]}
                reverse_dict[gene][reg_type].append(k)
    return  reverse_dict
import json
def write_dict_of_dicts(dictionary, file):
    with open(file, 'w') as convert_file:
        for k in sorted(dictionary.keys()):
            convert_file.write(f"{k}:")
            convert_file.write(json.dumps(dictionary[k]))
            convert_file.write("\n")
def plot_perturb_gene_dictionary(p_dict, full,perturbations_dir,show = False, save = True, ncols = 5, fname = "",
                                 palette = {"activate":sns.color_palette("tab10")[0], "knockdown":sns.color_palette("tab10")[1]}):
    ncols = ncols
    nrows = int(np.ceil(len(p_dict.keys())/ncols))
    # fig = plt.Figure(figsize = (8,8))
    fig, axs = plt.subplots(ncols = ncols, nrows= nrows, figsize=(20, 30))

    for x, k in enumerate(sorted(p_dict.keys())):
        print(k)
        #for each gene, for associated clusters that are destabilized, make a df of scores to be used for plotting
        plot_df = pd.DataFrame(columns = ["cluster","attr","gene","perturb","score"])
        for cluster in p_dict[k]["Regulators"]:
            tmp = full.loc[(full['cluster']==cluster)&(full['gene']==k)&(full["perturb"]=="knockdown")]
            for i,r in tmp.iterrows():
                plot_df = plot_df.append(r, ignore_index=True)
        for cluster in p_dict[k]["Destabilizers"]:
            tmp = full.loc[(full['cluster']==cluster)&(full['gene']==k)&(full["perturb"]=="activate")]
            for i,r in tmp.iterrows():
                plot_df = plot_df.append(r, ignore_index=True)

        # fig.add_subplot(ncols, nrows,x+1)
        my_order = plot_df.groupby(by=["cluster"]).median().sort_values(by = 'score').index.values
        col = int(np.floor(x/nrows))
        row = int(x%nrows)
        sns.barplot(data= plot_df, x = "cluster",y = "score", hue = "perturb", order = my_order,
                    ax = axs[row,col], palette = palette, dodge = False)
        axs[row,col].set_title(f"{k} Perturbations")
        axs[row,col].set_xticklabels(labels = my_order,rotation = 45, fontsize = 8, ha = 'right')
    plt.tight_layout()
    if save:
        plt.savefig(f"{perturbations_dir}/destabilizing_tfs{fname}.pdf")
    if show:
        plt.show()
if True:
    ATTRACTOR_DIR = f"{dir_prefix}/{brcd}/attractors/attractors_threshold_0.5"

    attractor_dict = {}
    attr_filtered = pd.read_csv(f'{ATTRACTOR_DIR}/attractors_filtered.txt', sep = ',', header = 0, index_col = 0)
    for i,r in attr_filtered.iterrows():
        attractor_dict[i] = []

    for i,r in attr_filtered.iterrows():
        attractor_dict[i].append(bb.utils.state_bool2idx(list(r)))

    perturbations_dir = f"{dir_prefix}/{brcd}/perturbations"


    plot_destabilization_scores(attractor_dict, perturbations_dir, show = False, save = True, clustered = True,
                                act_kd_together = True)

    perturb_dict, full = get_perturbation_dict(attractor_dict, perturbations_dir, significance = 'both', save_full=False)
    perturb_gene_dict = reverse_perturb_dictionary(perturb_dict)
    write_dict_of_dicts(perturb_gene_dict, file = f"{perturbations_dir}/clustered_perturb_plots/perturbation_TF_dictionary.txt")
    plot_perturb_gene_dictionary(perturb_gene_dict, full,perturbations_dir,show = False, save = True, ncols = 5)

    full_sig = get_ci_sig(full, group_cols=['cluster','gene','perturb'])
    full_sig.to_csv(f"{perturbations_dir}/clustered_perturb_plots/perturbation_stats.csv")



    ## gene dict and plots with threshold = -0.2
    perturb_dict_2, full = get_perturbation_dict(attractor_dict, perturbations_dir, significance = 'both', save_full=False,
                                                 mean_threshold=-0.2)
    perturb_gene_dict_2 = reverse_perturb_dictionary(perturb_dict_2)
    plot_perturb_gene_dictionary(perturb_gene_dict_2, full,perturbations_dir,show = False, save = True, ncols = 5, fname = "_0.2")

#---------------------------------

