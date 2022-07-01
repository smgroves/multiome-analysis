import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import graph_utils
import os
from graph_utils import *
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import seaborn as sns
def smooth_rule(rule, regulators, regulator_diffs, confidence, power=3.):
    n = len(regulators)
    if (n < 2): return rule # If there is only one regulator, there are two leaves, and we don't want to average them
    newrule = rule*0
    norm_factor = np.power(2.,n) # This is the total number of leaves
    
    
    # regulator_diffs is the total regulator relevance for each regulator. We divide them by the number of leaves
    # to get an average importance per leaf. 0.1 would mean that this regulator, on average, affects rule leaves by 0.1.
    # In contrast, 0.01 would mean that the regulator is on average weaker.
    # For smoothing, neighbors that differ by a weaker regulator are considered more similar to the leaf under question.
    weights = dict()
    
    for regulator in regulators:
        w = regulator_diffs[regulator]/norm_factor
        weights[regulator] = np.power(1.-w,power) # 0.1 -> 0.9 -> 0.9^3 = 0.73, VS 0.01 -> 0.99 -> 0.99^3 = 0.97
    print(weights)
    
    # Loop over all leaves, get their rule as True/False, and their confidence
    for leaf in range(2**n):
        binary = graph_utils.idx2binary(leaf, n)
        binary = [{'0':False,'1':True}[i] for i in binary] # This leaf means regulators are "ON,OFF,ON,ON,OFF"
        conf = confidence[leaf]
        
        # The new rule will become
        # r_new[i] = (r[i]*c[i] + (1-c[i])*SUM(r[j]*c[j]*w[j])) / (c[i] + (1-c[i])*SUM(c[j]*w[j]))
        # That is, a linear interpolation between the rule itself, and a weighted average of the rules
        # of it's neighbors, interpolated based on its own confidence, and the neighbors are weighted
        # by the product of their weight and confidence
        # This can cause PROBLEMS TODO. If c[i]=0, and c[j]=0 for all but one j*, where c[j*]>0, this 
        # rule is totally overwritten by r[j*]
        # In contrast, let
        # r_new[i] = (r[i]*c[i] + (1-c[i])*SUM(r[j]*w[j])) / (c[i] + (1-c[i])*SUM(w[j]))
        # Thus, neighbors are averaged without consideration for their confidence. Unconfident neighbors will have r[j]=0.5,
        # and that uncertainty gets averaged in to this.
        s = rule[leaf]*conf
        total_weight = conf
        
        for r, regulator in enumerate(regulators):
            neighbor = [i for i in binary]
            neighbor[r] = not neighbor[r]
            neighbor_idx = graph_utils.state_bool2idx(neighbor)
            #neighbor_weight = (1-conf)*weights[regulator]
            neighbor_weight = (1-conf)*weights[regulator]*confidence[neighbor_idx]
            s += neighbor_weight*rule[neighbor_idx]
            total_weight += neighbor_weight
        newrule[leaf] = s / total_weight
    return newrule

def reorder_binary_decision_tree(old_regulator_order, regulators):
    n = len(regulators)
    new_order = []
    
    # Map the old order into the new order
    old_index_to_new = [regulators.index(i) for i in old_regulator_order]
    
    # Loop through the leaves for the new rule
    for leaf in range(2**n):
        # Get the binary for this leaf, ordered by the new order
        binary = graph_utils.idx2binary(leaf, n)
        # Figure out what this binary would have been in the old order
        oldbinary = ''.join([binary[i] for i in old_index_to_new])
        # What leaf was that in the old order?
        oldleaf = graph_utils.state2idx(oldbinary)
        # Map that old leaf to the current, reordered leaf
        new_order.append(oldleaf)
    return new_order
        

# If A=f(B,C,D), this checks whether B being ON or OFF has an impact > threshold for any combination of C={ON/OFF} D={ON/OFF}
def detect_irrelevant_regulator(regulators, rule, threshold=0.1):
    n = len(regulators)
    max_difs = []
    tot_difs = []
    signed_tot_difs = []

    irrelevant = []
    for r, regulator in enumerate(regulators):
        print(f"...checking if {regulator} is irrelevant")
        max_dif=0
        tot_dif = 0
        signed_tot_dif = 0
        leaves = get_leaves_of_regulator(2**n, r)
        for i,j in zip(*leaves):
            dif = np.abs(rule[j] - rule[i])
            max_dif = max(dif, max_dif)
            tot_dif = tot_dif + dif
            signed_tot_dif = signed_tot_dif + rule[i] - rule[j]
        max_difs.append(max_dif)
        tot_difs.append(tot_dif)
        signed_tot_difs.append(signed_tot_dif)
        if max_dif < threshold:
            irrelevant.append(regulator)
    return dict(zip(regulators, max_difs)), dict(zip(regulators, tot_difs)),dict(zip(regulators, signed_tot_difs)) ### added signed_tot_dif to output
    #return irrelevant
    
# n = number of leaves in the rule (i.e. 2**N for a node with N regulators)
# index = index of the regulator is being looked at, when viewing the rule as a binary decision tree
# (e.g., index=0 refers to the top branch of the bdt)
def get_leaves_of_regulator(n, index):
    step_size = int(np.round(n/(2**(index+1))))
    num_steps = int(np.round(n/step_size/2))
    off_leaves = []
    on_leaves = []

    base = 0
    for step in range(num_steps):
        for i in range(step_size): off_leaves.append(base+i)
        base += step_size
        for i in range(step_size): on_leaves.append(base+i)
        base += step_size
    return off_leaves, on_leaves
def bootstrap_rules(data, vertex_dict, n = 10, plot=False, threshold=0.1, directory="rules", hlines=None):
    print("Bootstrapping...")
    #randomly sample with replacement from samples (rows)

# data=dataframe with rows=samples, cols=genes
# nodes = list of nodes in network
# vertex_dict = dictionary mapping gene name to a vertex in a graph_tool Graph()
# v_names - A dictionary mapping vertex in graph to name
# plot = boolean - make the resulting plot
# threshold = float from 0.0 to 1.0, used as threshold for removing irrelevant regulators. 0 removes nothing. 1 removes all.
def get_rules(data, vertex_dict, plot=False, threshold=0.1, directory="rules", hlines=None):
    v_names = dict()
    for vertex_name in list(vertex_dict): v_names[vertex_dict[vertex_name]] = vertex_name # invert the vertex_dict
    nodes = list(vertex_dict)
    rules = dict()
    regulators_dict = dict()
    strengths = pd.DataFrame(index=nodes,columns=nodes)
    signed_strengths = pd.DataFrame(index=nodes,columns=nodes)
    for gene in nodes:
        print(gene)
        # for each node of the network
        irrelevant = []
        n_irrelevant_new=0
        regulators = [v_names[v] for v in vertex_dict[gene].in_neighbors() if not v_names[v] in irrelevant]
        # define a set of regulators as the in_neighbors of the node
        while True: # This breaks when all regulators have been deemed irrelevant, or none have
            n_irrelevant_old = n_irrelevant_new
            regulators_dict[gene]=regulators
            n = len(regulators)


            # we have to make sure we haven't stripped all the regulators as irrelevant
            if n > 0:
                # This becomes the eventual probabilistic rule. It has 2 rows
                # that describe prob(ON) and prob(OFF). At the end these rows
                # are normalized to sum to 1, such that the rule becomes
                # prob(ON) / (prob(ON) + prob(OFF)

                prob_01 = np.zeros((2,2**n))

                # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
                heat = np.ones((data.shape[0], 2**n))

                for leaf in range(2**n):
                    if leaf%100 ==0: print(leaf)
                    binary = graph_utils.idx2binary(leaf, len(regulators))
                    binary = [{'0':False,'1':True}[i] for i in binary]
                    # binary becomes a list of lists of T and Fs to represent each column
                    for i,idx in enumerate(data.index):
                        # for each row in data column...
                        # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
                        df = data.loc[idx]
                        val = np.float(data.loc[idx,gene])
                        for col,on in enumerate(binary):
                            # for each regulator in each column in decision tree...
                            regulator = regulators[col]
                            # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                            # row of data and column of tree with a weight that = probability that that node is on in the data
                            # df(regulator) = expression value of regulator in data for that row
                            # multiply for each regulator (parent TF) in leaf
                            if on: heat[i,leaf] *= np.float(df[regulator])
                            else: heat[i,leaf] *= 1-np.float(df[regulator])
                        # the probability for that leaf becomes the value of expression (val) times that square in the heatmap
                        # this loops over the rows in the heatmap and keeps multiplying in the weight * expression value
                        prob_01[0,leaf] += val*heat[i,leaf] # Probabilitiy of being ON
                        prob_01[1,leaf] += (1-val)*heat[i,leaf]

                # We weigh each column by adding in a sample with prob=50% and
                # a weight given by 1-max(weight). So leaves where no samples
                # had high weight will end up with a high weight of 0.5. For
                # instance, if the best sample has a weight 0.1 (crappy), the
                # rule will have a sample added with weight 0.9, and 50% prob.
                max_heat = 1-np.max(heat,axis=0)
                for i in range(prob_01.shape[1]):

                    prob_01[0,i] += max_heat[i]*0.5
                    prob_01[1,i] += max_heat[i]*0.5
                    
                # The rule is normalized so that prob(ON)+prob(OFF)=1
                rules[gene]=prob_01[0,:]/np.sum(prob_01,axis=0)
                max_regulator_relevance, tot_regulator_relevance, signed_tot_regulator_relevance = detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)

                old_regulator_order = [i for i in regulators]
                regulators = sorted(regulators, key=lambda x: max_regulator_relevance[x], reverse=True)
                if max_regulator_relevance[regulators[-1]] < threshold:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(regulators, key=lambda x: tot_regulator_relevance[x], reverse=True)
                regulators_dict[gene] = regulators

#                regulators = old_regulator_order
#                irrelevant += detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)

                n_irrelevant_new=len(irrelevant)
            if len(regulators)==0 and gene not in irrelevant:
                regulators = [gene,]
                regulators_dict[gene]=[gene,]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0: break
            
        if len(regulators) > 0:

            importance_order = reorder_binary_decision_tree(old_regulator_order, regulators)
            heat = heat[:, importance_order]

            rules[gene] = rules[gene][importance_order]
            #rules[gene] = smooth_rule(rules[gene], regulators, tot_regulator_relevance, np.max(heat,axis=0))
            strengths.loc[gene] = tot_regulator_relevance
            signed_strengths.loc[gene] = signed_tot_regulator_relevance

            if plot: plot_rule(gene, rules[gene], regulators, heat, data, directory=directory, hlines=hlines)

        # else:
        #     fig = plt.figure()
        #     plt.savefig(os.path.join(directory,"%s.pdf"%gene))
        #     plt.clf()
        #     plt.close()


    return rules, regulators_dict,  strengths, signed_strengths


def get_rules_scvelo(data,data_t1, vertex_dict, plot=False, threshold=0.1, directory="rules", hlines=None):
    v_names = dict()
    for vertex_name in list(vertex_dict): v_names[vertex_dict[vertex_name]] = vertex_name  # invert the vertex_dict
    nodes = list(vertex_dict)
    rules = dict()
    regulators_dict = dict()
    strengths = pd.DataFrame(index=nodes,columns=nodes)
    signed_strengths = pd.DataFrame(index=nodes,columns=nodes)

    for gene in nodes:
        print(gene)
        # for each node of the network
        irrelevant = []
        n_irrelevant_new = 0
        regulators = [v_names[v] for v in vertex_dict[gene].in_neighbors() if not v_names[v] in irrelevant]
        # define a set of regulators as the in_neighbors of the node
        while True:  # This breaks when all regulators have been deemed irrelevant, or none have
            n_irrelevant_old = n_irrelevant_new
            regulators_dict[gene] = regulators
            n = len(regulators)

            # we have to make sure we haven't stripped all the regulators as irrelevant
            if n > 0:
                # This becomes the eventual probabilistic rule. It has 2 rows
                # that describe prob(ON) and prob(OFF). At the end these rows
                # are normalized to sum to 1, such that the rule becomes
                # prob(ON) / (prob(ON) + prob(OFF)

                prob_01 = np.zeros((2, 2 ** n))

                # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
                heat = np.ones((data.shape[0], 2 ** n))

                for leaf in range(2 ** n):
                    if leaf % 50 == 0: print(leaf)
                    binary = graph_utils.idx2binary(leaf, len(regulators))
                    binary = [{'0': False, '1': True}[i] for i in binary]
                    # binary becomes a list of lists of T and Fs to represent each column
                    for i, idx in enumerate(data.index):
                        # for each row in data column...
                        # grab that row (df) and the expression value for the current node (left side of rule plot) (val)
                        df = data.loc[idx]
                        val = np.float(data_t1.loc[idx, gene])
                        for col, on in enumerate(binary):
                            # for each regulator in each column in decision tree...
                            regulator = regulators[col]
                            # if that regulator is on in the decision tree, multiply the weight in the heatmap for that
                            # row of data and column of tree with a weight that = probability that that node is on in the data
                            # df(regulator) = expression value of regulator in data for that row
                            # multiply for each regulator (parent TF) in leaf
                            if on:
                                heat[i, leaf] *= np.float(df[regulator])
                            else:
                                heat[i, leaf] *= 1 - np.float(df[regulator])
                        # the probability for that leaf becomes the value of expression (val) times that square in the heatmap
                        # this loops over the rows in the heatmap and keeps multiplying in the weight * expression value
                        prob_01[0, leaf] += val * heat[i, leaf]  # Probabilitiy of being ON
                        prob_01[1, leaf] += (1 - val) * heat[i, leaf]

                # We weigh each column by adding in a sample with prob=50% and
                # a weight given by 1-max(weight). So leaves where no samples
                # had high weight will end up with a high weight of 0.5. For
                # instance, if the best sample has a weight 0.1 (crappy), the
                # rule will have a sample added with weight 0.9, and 50% prob.
                max_heat = 1 - np.max(heat, axis=0)
                for i in range(prob_01.shape[1]):
                    prob_01[0, i] += max_heat[i] * 0.5
                    prob_01[1, i] += max_heat[i] * 0.5

                # The rule is normalized so that prob(ON)+prob(OFF)=1
                rules[gene] = prob_01[0, :] / np.sum(prob_01, axis=0)
                max_regulator_relevance, tot_regulator_relevance, signed_tot_regulator_relevance = detect_irrelevant_regulator(regulators, rules[gene],
                                                                                               threshold=threshold)

                old_regulator_order = [i for i in regulators]
                regulators = sorted(regulators, key=lambda x: max_regulator_relevance[x], reverse=True)
                if max_regulator_relevance[regulators[-1]] < threshold:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(regulators, key=lambda x: tot_regulator_relevance[x], reverse=True)
                regulators_dict[gene] = regulators

                #                regulators = old_regulator_order
                #                irrelevant += detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)

                n_irrelevant_new = len(irrelevant)
            if len(regulators) == 0 and gene not in irrelevant:
                regulators = [gene, ]
                regulators_dict[gene] = [gene, ]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0:
                break

        if len(regulators) > 0:

            importance_order = reorder_binary_decision_tree(old_regulator_order, regulators)
            heat = heat[:, importance_order]

            rules[gene] = rules[gene][importance_order]
            # rules[gene] = smooth_rule(rules[gene], regulators, tot_regulator_relevance, np.max(heat,axis=0))

            strengths.loc[gene] = tot_regulator_relevance
            signed_strengths.loc[gene] = signed_tot_regulator_relevance

            if plot: plot_rule(gene, rules[gene], regulators, heat, data, directory=directory, hlines=hlines)

        # else:
        #     fig = plt.figure()
        #     plt.savefig(os.path.join(directory, "%s.pdf" % gene))
        #     plt.clf()
        #     plt.close()

    return rules, regulators_dict, strengths, signed_strengths


# If A=f(B,C,D), this checks whether B being ON or OFF has an impact > threshold for any combination of C={ON/OFF} D={ON/OFF}
def detect_irrelevant_regulator_bootstrap(regulators, rule):
    n = len(regulators)
    best_pvals = []
    tot_difs = []
    irrelevant = []
    for r, regulator in enumerate(regulators):
        max_dif=0
        tot_dif = 0
        pvals = []
        off_leaves, on_leaves = get_leaves_of_regulator(2**n, r)
        for i,j in zip(off_leaves, on_leaves):
            pvals.append(mannwhitneyu(rule[:,i], rule[:,j]).pvalue)
        pvals = multipletests(pvals, method='fdr_bh')[1]
        best_pvals.append(pvals.min())
        tot_difs.append(-np.log(pvals).sum())
    return dict(zip(regulators, best_pvals)), dict(zip(regulators, tot_difs))


# data=dataframe with rows=samples, cols=genes
# nodes = list of nodes in network
# vertex_dict = dictionary mapping gene name to vertex in graph
# v_names - A dictionary mapping vertex in graph to name
# plot = boolean - make the resulting plot
# 
def bootstrap_rules(data, nodes, vertex_dict, v_names, plot=True, directory="rules", n_bootstraps=10):
    rules = dict()
    regulators_dict = dict()
    for gene in nodes:
        irrelevant = []
        n_irrelevant_new=0
        regulators = [v_names[v] for v in vertex_dict[gene].in_neighbors() if not v_names[v] in irrelevant]
        while True: # This breaks when all regulators have been deemed irrelevant, or none have
            n_irrelevant_old = n_irrelevant_new
            print (gene, irrelevant, regulators)
            regulators_dict[gene]=regulators
            n = len(regulators)

            # we have to make sure we haven't stripped all the regulators as irrelevant
            if n > 0:
                current_rule = np.zeros((n_bootstraps, 2**n))
                for boot_ in range(n_bootstraps):
                    print("Bootstrap %d"%boot_)
                    
                    bootstrap_samples = data.sample(n=data.shape[0], replace=True).index
                    # This becomes the eventual probabilistic rule. It has 2 rows
                    # that describe prob(ON) and prob(OFF). At the end these rows
                    # are normalized to sum to 1, such that the rule becomes
                    # prob(ON) / (prob(ON) + prob(OFF)
                    prob_01 = np.zeros((2,2**n))


                    # This is the distribution of how much each sample reflects/constrains each leaf of the Binary Decision Diagram
                    heat = np.ones((data.shape[0], 2**n))
                    for leaf in range(2**n):
                        binary = graph_utils.idx2binary(leaf, len(regulators))
                        binary = [{'0':False,'1':True}[i] for i in binary] # This leaf means regulators are "ON,OFF,ON,ON,OFF"
                        for i,idx in enumerate(bootstrap_samples):
                            df = data.loc[idx]
                            val = np.float(data.loc[idx,gene])
                            for col,on in enumerate(binary):
                                regulator = regulators[col] 
                                if on: heat[i,leaf] *= np.float(df[regulator])
                                else: heat[i,leaf] *= 1-np.float(df[regulator])
                            prob_01[0,leaf] += val*heat[i,leaf]
                            prob_01[1,leaf] += (1-val)*heat[i,leaf]

                    # We weigh each column by adding in a sample with prob=50% and
                    # a weight given by 1-max(weight). So leaves where no samples
                    # had high weight will end up with a high weight of 0.5. For
                    # instance, if the best sample has a weight 0.1 (crappy), the
                    # rule will have a sample added with weight 0.9, and 50% prob.
                    max_heat = 1-np.max(heat,axis=0)
                    for i in range(prob_01.shape[1]):

                        prob_01[0,i] += max_heat[i]*0.5
                        prob_01[1,i] += max_heat[i]*0.5
                    
                    # The rule is normalized so that prob(ON)+prob(OFF)=1
                    current_rule[boot_,:]=prob_01[0,:]/np.sum(prob_01,axis=0)
                    
                    # END BOOSTRAP LOOP
                    
                    
                    
                    
                    
                max_regulator_relevance, tot_regulator_relevance = detect_irrelevant_regulator_bootstrap(regulators, current_rule)
                print(max_regulator_relevance)
                old_regulator_order = [i for i in regulators]
                regulators = sorted(regulators, key=lambda x: max_regulator_relevance[x], reverse=False) # Sort from most significant to least
                if max_regulator_relevance[regulators[-1]] > 0.01:
                    irrelevant.append(regulators[-1])
                    old_regulator_order.remove(regulators[-1])
                    regulators.remove(regulators[-1])
                regulators = sorted(regulators, key=lambda x: tot_regulator_relevance[x], reverse=True)
                regulators_dict[gene] = regulators

#                regulators = old_regulator_order
#                irrelevant += detect_irrelevant_regulator(regulators, rules[gene], threshold=threshold)

                n_irrelevant_new=len(irrelevant)
            if len(regulators)==0 and gene not in irrelevant:
                regulators = [gene,]
                regulators_dict[gene]=[gene,]
            elif n_irrelevant_old == n_irrelevant_new or len(regulators) == 0: break
            
        if len(regulators) > 0:

            importance_order = reorder_binary_decision_tree(old_regulator_order, regulators)
            heat = heat[:, importance_order]
            rules[gene] = current_rule[:,importance_order].mean(axis=0) # Don't just take the mean bootstrap...
            #rules[gene] = smooth_rule(rules[gene], regulators, tot_regulator_relevance, np.max(heat,axis=0))

            if plot: plot_rule(gene, rules[gene], regulators, heat, data, directory=directory)

        else:
            fig = plt.figure()
            plt.savefig(os.path.join(directory,"%s.pdf"%gene))
            plt.clf()
            plt.close()
            
            
            
            
            
                    


    return rules, regulators_dict
    
    
    

def plot_rule(gene, rule, regulators, sample_weights, data, directory="rules", hlines=None):#, hlines=[11,10,11,18]):

    n = len(regulators)

    fig = plt.figure()
    
    #                           .-.-------.
    #                  / \      | |       |
    #                 /\ /\     | |       |
    #             .-.-------.   |-|-------|
    #             | |       |   | |       |
    #             | |       |   | |       |
    #             | |       |   | |       |
    #             '-|-.-.-.-|   |-|-------|
    #               '-'-'-'-'   '-'-------'
    #
    
    gs = gridspec.GridSpec(3,2,height_ratios=[3,9,1], width_ratios=[1,8])
#    gs.update(hspace=0, wspace=0.03)
    gs.update(hspace=0, wspace=0)

    # Make the tree (plot a bunch of lines in branching pattern, starting from the bottom)
    ax = plt.subplot(gs[0,1])

    bottom_nodes = range(2**n)
    for layer in range(n):
        top_nodes = []
        for leaves in [i*2 for i in range(2**(n-layer-1))]:

            top_nodes.append((bottom_nodes[leaves] + bottom_nodes[leaves+1]) / 2.)
            
        for i in range(len(top_nodes)):
            ax.plot([bottom_nodes[2*i], top_nodes[i]],[layer,layer+1],'b--', lw=0.8)
            ax.plot([bottom_nodes[2*i+1], top_nodes[i]],[layer,layer+1],'r-', lw=0.8)
        
#        ax.annotate(regulators[n-1-layer], ((2*top_nodes[i] + bottom_nodes[2*i+1])/3., layer+1))
        progress = min(0.9,(n-layer-1)/6.) # Progress helps position the annotation along the branch - the lower in the tree, the farther along the branch the text is placed
        ax.annotate(" %s"%regulators[n-1-layer], ((1-progress)*top_nodes[i] + progress*bottom_nodes[2*i+1], layer+1-progress), fontsize=8)
        bottom_nodes = top_nodes
            
    ax.set_xlim(-0.5,2**n-0.5)
    ax.set_ylim(0,n)
    ax.set_axis_off()
    

#### OLD    
    # Label the tree regulators
#    ax = plt.subplot(gs[0,0])
#    for i,reg in enumerate(regulators):
#        ax.annotate(reg,(0,len(regulators)-i))
#    ax.set_ylim(0.5,len(regulators)+0.5)
#    ax.set_axis_off()


    # Plot the rule (horizontal bar directly under tree (now under the matrix))
    ax = plt.subplot(gs[2,1])
    pco = ax.pcolor(rule.reshape(1,rule.shape[0]), cmap="bwr", vmin=0, vmax=1)
    pco.set_edgecolor('face')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Inferred rule for %s"%gene)


    # Plot the sample weights in greyscale (big matrix)
    ax = plt.subplot(gs[1,1])
    pco = ax.pcolor(sample_weights, cmap="Greys", vmin=0, vmax=1)
    pco.set_edgecolor('face')
    if hlines is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yline = 0
        for hline in hlines[:-1]:
            yline += hline 
            plt.plot(xlim, [yline, yline], 'k--',lw=0.5)
        if n < 8:
            for xline in range(1,2**n): plt.plot([xline,xline],ylim, 'k--', lw=0.1)
        else:
            for xline in range(2,2**n,2): plt.plot([xline,xline],ylim, 'k--', lw=0.1)
#        print " XLIM ", xlim
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])


    # Plot the sample expression (vertical bar on left)
    ax = plt.subplot(gs[1,0])
    pco = ax.pcolor(data[[gene,]], cmap="bwr")
    pco.set_edgecolor('face')
    if hlines is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        yline = 0
        for hline in hlines[:-1]:
            yline += hline 
            plt.plot(xlim, [yline, yline], 'k--',lw=0.5)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("%s expression"%gene)

    
    # Save
    plt.savefig(os.path.join(directory,"%s.pdf"%gene))
    plt.cla()
    plt.clf()
    plt.close()

def save_rules(rules, regulators_dict, fname="rules.txt", delimiter="|"):
    lines = []
    for k in regulators_dict.keys():
        rule = ",".join(["%f"%i for i in rules[k]])
        regulators = ",".join(regulators_dict[k])
        lines.append("%s|%s|%s"%(k,regulators,rule))
    
    outfile = open(fname,"w")
    outfile.write("\n".join(lines))
    outfile.close()

def load_rules(fname="rules.txt", delimiter="|"):
    rules = dict()
    regulators_dict = dict()
    with open(fname,"r") as infile:
        for line in infile:
            line = line.strip().split(delimiter)
            regulators_dict[line[0]] = line[1].split(',')
            rules[line[0]] = np.asarray([float(i) for i in line[2].split(',')])
    return rules, regulators_dict

def rule_comparison(data_train, data_t1_train, data_test, data_t1_test, clusters,nodes, dir_prefix, brcd,vertex_dict, node_threshold, class_header = 'class', compare = True):
    try:
        os.mkdir(f"{dir_prefix}{brcd}/")
    except FileExistsError:
        pass
    aucs = pd.DataFrame(index=nodes, columns=sorted(list(set(clusters[class_header]))))
    for cluster in sorted(list(set(clusters[class_header]))):
        print(f"{cluster}")
        try:
            os.mkdir(f"{dir_prefix}{brcd}/{cluster}/")
        except FileExistsError:
            pass
        samples = clusters.loc[clusters['class']==cluster].index
        # save rules for each barcode and cross-validation fold
        rules, regulators_dict, strengths, signed_strengths = get_rules_scvelo(data_train.loc[list(set(samples).intersection(data_train.index))], data_t1_train.loc[list(set(samples).intersection(data_train.index))],
                                                            vertex_dict,
                                                            directory=dir_prefix + brcd + os.sep + "rules_" + brcd + '_' + str(cluster),
                                                            plot=False, threshold=node_threshold)
        save_rules(rules, regulators_dict,
                             fname=f"{dir_prefix}{brcd}/{cluster}/rules_{brcd}_{cluster}.txt")

        strengths.to_csv(f"{dir_prefix}{brcd}/{cluster}/strengths_{brcd}_{cluster}.csv")
        signed_strengths.to_csv(f"{dir_prefix}{brcd}/{cluster}/signed_strengths_{brcd}_{cluster}.csv")
        if compare:
            outfile = open(f"{dir_prefix}{brcd}/{cluster}/tprs_fprs_{brcd}_{cluster}.csv", 'w+')
            ind = [x for x in np.linspace(0, 1, 50)]
            tpr_all = pd.DataFrame(index=ind)
            fpr_all = pd.DataFrame(index=ind)
            area_all = []

            outfile.write(f",,")
            for j in ind:
                outfile.write(str(j) + ',')
            outfile.write('\n')
            for g in nodes:
                print(g)

                validation = plot_accuracy_scvelo(data_test.loc[list(set(samples).intersection(data_test.index))],
                                                  data_t1_test.loc[list(set(samples).intersection(data_test.index))],
                                                  g, regulators_dict,
                                                  rules, save_plots=cluster,
                                                  plot=False, plot_clusters=False, save_df=True,
                                                  dir_prefix=dir_prefix + brcd + os.sep + str(cluster) + os.sep)
                tprs, fprs, area = roc(validation, g, n_thresholds=50, save_plots=cluster, plot=True, save=True,
                                       dir_prefix=dir_prefix + brcd + os.sep  + os.sep)
                tpr_all[g] = tprs
                fpr_all[g] = fprs
                outfile.write(f"{g},tprs,{tprs}\n")
                outfile.write(f"{g},fprs,{fprs}\n")
                area_all.append(area)
            outfile.close()

            for c_comp in sorted(list(set(clusters['class']))):
                if cluster == c_comp: continue
                else:
                    print(f"...{c_comp}")
                    try:
                        os.mkdir(f"{dir_prefix}{brcd}/{cluster}/{c_comp}")
                    except FileExistsError:
                        pass
                    samples_comp = clusters.loc[clusters['class'] == c_comp].index

                    outfile = open(f"{dir_prefix}{brcd}/{cluster}/{c_comp}/tprs_fprs_{brcd}_{cluster}_{c_comp}.csv", 'w+')
                    ind = [x for x in np.linspace(0, 1, 50)]
                    tpr_all = pd.DataFrame(index=ind)
                    fpr_all = pd.DataFrame(index=ind)
                    area_all = []

                    outfile.write(f",,")
                    for j in ind:
                        outfile.write(str(j) + ',')
                    outfile.write('\n')
                    for g in nodes:
                        print(g)

                        validation = plot_accuracy_scvelo(data_test.loc[list(set(samples_comp).intersection(data_test.index))],
                                                          data_t1_test.loc[list(set(samples_comp).intersection(data_test.index))],
                                                          g, regulators_dict,
                                                          rules, save_plots=cluster,
                                                          plot=False, plot_clusters=False, save_df=True,
                                                          dir_prefix=dir_prefix + brcd + os.sep + os.sep)
                        tprs, fprs, area = roc(validation, g, n_thresholds=50, save_plots=cluster, plot=True, save=True,
                                               dir_prefix=dir_prefix + brcd + os.sep + os.sep)
                        tpr_all[g] = tprs
                        fpr_all[g] = fprs
                        outfile.write(f"{g},tprs,{tprs}\n")
                        outfile.write(f"{g},fprs,{fprs}\n")
                        area_all.append(area)
                    outfile.close()

            # save AUC values by gene for each cross-validation fold and barcode
            outfile = open(f"{dir_prefix}{brcd}/{cluster}/auc_{brcd}_{cluster}.csv", 'w+')
            for n, a in enumerate(area_all):
                outfile.write(f"{nodes[n]},{a} \n")
                aucs.loc[nodes[n]][str(cluster)] = a
            outfile.close()
        aucs.to_csv(f"{dir_prefix}{brcd}/aucs.csv")

def crossval_rules(data, data_t1, clusters, nodes, dir_prefix, brcd,
                       vertex_dict, node_threshold):
    n_splits = 3
    kf = ms.StratifiedKFold(n_splits=n_splits)
    test_set = 'validation_set'
    os.mkdir(f"{dir_prefix}{brcd}/{test_set}/")



    i = 0  # Index of fold
    aucs = pd.DataFrame(index=nodes, columns=[str(i) for i in range(n_splits)])
    for train_index, test_index in kf.split(data.index, clusters.loc[data.index, 'class']):
        print(f'Generating Rules for K-Fold {i}')
        try:
            os.mkdir(f"{dir_prefix}{brcd}/{test_set}/{i}/")
        except FileExistsError:
            pass

        # save rules for each barcode and cross-validation fold
        rules, regulators_dict,strengths = get_rules_scvelo(data.iloc[train_index],data_t1.iloc[train_index], vertex_dict,
                                                         directory=dir_prefix + brcd + os.sep + "rules_" + brcd + '_' + str(i),
                                                         plot=False, threshold=node_threshold)
        save_rules(rules, regulators_dict, fname=f"{dir_prefix}{brcd}/{test_set}/{i}/rules_{brcd}_{i}.txt")

        # make one file of tprs (true positive rate) and fprs for each cross-validation fold


        outfile = open(f"{dir_prefix}{brcd}/{test_set}/{i}/tprs_fprs_{brcd}_{i}.csv", 'w+')
        ind = [x for x in np.linspace(0, 1, 50)]
        tpr_all = pd.DataFrame(index=ind)
        fpr_all = pd.DataFrame(index=ind)
        area_all = []

        outfile.write(f",,")
        for j in ind:
            outfile.write(str(j)+',')
        outfile.write('\n')
        for g in nodes:
            print(g)

            validation = plot_accuracy_scvelo(data.iloc[test_index],data_t1.iloc[test_index], g, regulators_dict, rules, save_plots=i,
                                       plot=False, plot_clusters=False, save_df=True,
                                       dir_prefix=dir_prefix + brcd + os.sep + str(test_set) + os.sep)
            tprs, fprs, area = roc(validation, g, n_thresholds=50, save_plots=i, plot=True, save=True,
                                   dir_prefix=dir_prefix + brcd + os.sep + str(test_set) + os.sep)
            tpr_all[g] = tprs
            fpr_all[g] = fprs
            outfile.write(f"{g},tprs,{tprs}\n")
            outfile.write(f"{g},fprs,{fprs}\n")
            area_all.append(area)
        outfile.close()

        # save AUC values by gene for each cross-validation fold and barcode
        outfile = open(f"{dir_prefix}{brcd}/{test_set}/{i}/auc_{brcd}_{i}.csv", 'w+')
        for n, a in enumerate(area_all):
            outfile.write(f"{nodes[n]},{a} \n")
            aucs.loc[nodes[n]][str(i)] = a
        outfile.close()
        i += 1
    aucs.to_csv(f"{dir_prefix}{brcd}/{test_set}/aucs.csv")