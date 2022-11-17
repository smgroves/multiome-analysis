# This code is written by SM Groves
# October 21, 2022
# The purpose of this code is to compare the edge strengths, signed_strengths, and edge_weights to the correlation matrix from Debbie
# between each pair of TFs.

import numpy as np
import seaborn as sns
import booleabayes as bb
import os
import os.path as op
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bb_utils import draw_grn

dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
brcd = str(55476)

strengths = pd.read_csv(f"{dir_prefix}/{brcd}/rules/strengths.csv", header = 0, index_col=0)
signed_strengths = pd.read_csv(f"{dir_prefix}/{brcd}/rules/signed_strengths.csv", header = 0, index_col=0)
edge_weights = pd.read_csv(f"{dir_prefix}/{brcd}/rules/edge_weights.csv", header = 0, index_col=0)

nodes = strengths.index

def plot_strength_ew(strengths, signed_strengths, edge_weights, nodes, dir_prefix = "", brcd = ""):

    if not os.path.exists(f"{dir_prefix}/{brcd}/rules/strength_plots"):
        # Create a new directory because it does not exist
        os.makedirs(f"{dir_prefix}/{brcd}/rules/strength_plots")

    for node in nodes:
        print(node)
        s = strengths.loc[node].dropna().sort_index()
        ss = signed_strengths.loc[node].dropna().sort_index()
        ew = edge_weights.loc[node].dropna().sort_index()

        plt.figure()
        plt.scatter(ew, ss)
        range = ew.max() - ew.min()
        plt.ylabel(f"Signed Strength (sum(f_{node}(TF = ON) - f_{node}(TF = OFF)))")
        plt.xlabel(f"Edge Weights (average(f_{node}(TF = ON) - f_{node}(TF = OFF)))")
        for i in ew.index:
            plt.text(x=ew[i]+.01*range, y = ss[i],s = i)
        plt.axhline(y = 0, linestyle = "--", color = 'grey')
        plt.axvline(x = 0, linestyle = "--", color = 'grey')
        plt.savefig(f"{dir_prefix}/{brcd}/rules/strength_plots/{node}_ew_ss.png")
        plt.close()

        plt.figure()
        plt.scatter(s, ss)
        range = s.max() - s.min()
        plt.ylabel(f"Signed Strength (sum(f_{node}(TF = ON) - f_{node}(TF = OFF)))")
        plt.xlabel(f"Strength (sum(|f_{node}(TF = ON) - f_{node}(TF = OFF)|))")
        for i in s.index:
            plt.text(x=s[i]+.01*range, y = ss[i],s = i)
        plt.axhline(y = 0, linestyle = "--", color = 'grey')
        plt.savefig(f"{dir_prefix}/{brcd}/rules/strength_plots/{node}_s_ss.png")
        plt.close()


gene_correlations = pd.read_csv(f"{dir_prefix}/DIRECT-NET-FILES/Dorc_TF_cor.csv", header = 0, index_col=0)
gene_correlations.index = [i.upper() for i in gene_correlations.index]
gene_correlations.columns = [i.upper() for i in gene_correlations.columns]

def plot_ew_gene_corr(edge_weights, gene_correlations, nodes, dir_prefix = "", brcd = ""):

    for node in nodes:
        if node in gene_correlations.index:
            print(node)
            ew = edge_weights.loc[node].dropna()
            gc = gene_correlations.loc[node]

            overlap = list(set(ew.index).intersection(set(gc.index)))

            ew = ew.loc[overlap].sort_index()
            gc = gc.loc[overlap].sort_index()

            plt.figure()
            plt.scatter(ew, gc)
            range = ew.max() - ew.min()
            plt.ylabel(f"Gene Correlation")
            plt.xlabel(f"Edge Weights (average(f_{node}(TF = ON) - f_{node}(TF = OFF)))")
            for i in ew.index:
                plt.text(x=ew[i]+.01*range, y = gc[i],s = i)
            plt.axhline(y = 0, linestyle = "--", color = 'grey')
            plt.axvline(x = 0, linestyle = "--", color = 'grey')
            plt.savefig(f"{dir_prefix}/{brcd}/rules/strength_plots/{node}_ew_gc.png")
            plt.close()

df = pd.DataFrame(columns = ['parent','child','ew','gc'])
for node in nodes:
    if node in gene_correlations.index:
        print(node)
        ew = edge_weights.loc[node].dropna()
        gc = gene_correlations.loc[node]

        overlap = list(set(ew.index).intersection(set(gc.index)))

        ew = ew.loc[overlap].sort_index()
        gc = gc.loc[overlap].sort_index()
        for i in ew.index:
            df = df.append(pd.Series([i,node,ew.loc[i],gc.loc[i]], index = df.columns), ignore_index=True)
fig = plt.figure()
sns.lmplot(data = df, x = 'ew', y= 'gc', hue = 'child', scatter = False, ci = None,legend = False)
           #truncate = False)
plt.ylabel(f"Gene Correlation")
plt.xlabel(f"Edge Weights")
plt.axhline(y = 0, linestyle = "--", color = 'grey')
plt.axvline(x = 0, linestyle = "--", color = 'grey')
plt.tight_layout()
plt.savefig(f"{dir_prefix}/{brcd}/rules/strength_plots/all_ew_gc.png")
plt.close()

fig = plt.figure()
sns.lmplot(data = df, x = 'ew', y= 'gc', scatter = True,legend = False)
#truncate = False)
plt.ylabel(f"Gene Correlation")
plt.xlabel(f"Edge Weights")
plt.axhline(y = 0, linestyle = "--", color = 'grey')
plt.axvline(x = 0, linestyle = "--", color = 'grey')
plt.tight_layout()
plt.savefig(f"{dir_prefix}/{brcd}/rules/strength_plots/all_ew_gc2.png")
plt.close()