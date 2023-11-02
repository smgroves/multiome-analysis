import random
import time

import numpy as np
import seaborn as sns
import booleabayes as bb
from bb_utils import *
import sys
from arboreto.algo import grnboost2, genie3
from arboreto.utils import load_tf_names
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold

## Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
data_path = f'data/adata_imputed_combined_v3.csv'
data = pd.read_csv(op.join(dir_prefix, data_path), index_col=0, header = 0)
data.columns = [col.upper() for col in data.columns]

###############################################
# Arboreto package (used in SCENIC) for GRN inference
###############################################
# Load the TF names
# tfs = set(network['source']).intersection(set(network['target']))
#
# if __name__ == '__main__':
#     grn_net = grnboost2(expression_data=data,
#                     tf_names=tfs)
#
# print(grn_net.head())

###############################################
# Lasso regression for choosing top TFs
###############################################
def lasso_feature_selection(network, data, network_name, dir_prefix, save_network = True, plot = True):
    if not os.path.exists(f"{dir_prefix}/networks/feature_selection/{network_name}"):
        os.makedirs(f"{dir_prefix}/networks/feature_selection/{network_name}")
    targets = list(np.unique(network['target']))
    #for each target, run lasso regression to find top tfs
    for target in targets:
        #get tfs
        self_target = False
        tfs = list(np.unique(network[network['target'] == target]['source']))
        if target in tfs:
            self_target = True
            tfs.remove(target)
        X = data[tfs]
        y = data[target]
        #split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        # parameters to be tested on GridSearchCV
        params = {"alpha": np.arange(0.00001, 10, 500)}

        # Number of Folds and adding the random state for replication
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initializing the Model
        lasso = Lasso()

        # GridSearchCV with model, params and folds.
        lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
        lasso_cv.fit(X, y)
        print("Best Params {}".format(lasso_cv.best_params_))
        print("Best Score {}".format(lasso_cv.best_score_))

        for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
            #if directory doesn't exist, create it
            feature_sel_folder = f"{dir_prefix}/networks/feature_selection/{network_name}/Lasso_alpha_{alpha}"
            if not os.path.exists(feature_sel_folder):
                os.makedirs(feature_sel_folder)


            lasso1 = Lasso(alpha=alpha)
            lasso1.fit(X_train, y_train)
            score = lasso1.score(X_test, y_test)

            # Using np.abs() to make coefficients positive.
            lasso1_coef = np.abs(lasso1.coef_)

            if plot:
                # plotting the Column Names and Importance of Columns.
                plt.bar(tfs, lasso1_coef)
                plt.xticks(rotation=90)
                plt.grid()
                plt.title("Feature Selection Based on Lasso for Target: " + target + " with alpha: " + str(alpha)
                          + "\n and score: " + str(np.round(score, 3))
                          + "\n Percentage of best score: " + str(np.round(score/lasso_cv.best_score_*100, 1)) + "%")
                plt.xlabel("Features")
                plt.ylabel("Importance")
                plt.tight_layout()
                if alpha == lasso_cv.best_params_['alpha']:
                    plt.savefig(f"{feature_sel_folder}/{target}_best.png")
                else:
                    plt.savefig(f"{feature_sel_folder}/{target}.png")
                plt.close()

            #save feature subset to csv
            if save_network:
                feature_subset = np.array(tfs)[lasso1_coef > 0]
                if self_target == True:
                    feature_subset = np.append(feature_subset, target)
                subset = network[(network['target'] == target) & (network['source'].isin(feature_subset))]
                if not os.path.isfile(feature_sel_folder + f'/{network_name}_Lasso_{alpha}.csv'):
                    subset.to_csv(feature_sel_folder + f'/{network_name}_Lasso_{alpha}.csv', index = False)
                else:
                    with open(feature_sel_folder + f'/{network_name}_Lasso_{alpha}.csv', 'a') as f:
                        subset.to_csv(f, header=False, index = False)

# network_name = "DIRECT-NET_network_2020db_0.1_top8targets"
# network_path = f'networks/{network_name}.csv'
# network = pd.read_csv(op.join(dir_prefix, network_path), index_col=None, header = None)
# network.columns = ['source', 'target', 'weight', 'evidence']
# lasso_feature_selection(network, data, network_name, dir_prefix)
# lasso_feature_selection(network, data, network_name, dir_prefix, save_network = False, plot = True)

network_name = "DIRECT-NET_network_2020db_0.1"
network_path = f'networks/{network_name}.csv'
network = pd.read_csv(op.join(dir_prefix, network_path), index_col=None, header = None)
network.columns = ['source', 'target', 'weight', 'evidence']
# lasso_feature_selection(network, data, network_name, dir_prefix)

#comparing the networks

def compare_networks(network, new_net, alpha, save = True,save_dir = ""):

    both = network.merge(new_net, on=['source','target'],
                       how='left', indicator=True)
    difference = both[both["_merge"]!="both"]
    print("alpha: ", alpha)
    print("Dropped edges: ", difference.shape[0])
    targets = list(np.unique(network['target']))
    most_parents = 0
    most_parents_target = ""
    parents = []
    most_parents_tfs = []
    for target in targets:
        tmp = new_net[new_net['target'] == target]
        num_parents = len(np.unique(tmp['source']))
        parents.append(num_parents)
        if num_parents > most_parents:
            most_parents = num_parents
            most_parents_target = target
            most_parents_tfs = np.unique(tmp['source'])
    print("Target with most parents: ", most_parents_target, " with ", most_parents, " parents")
    print(most_parents_tfs)
    bins = np.arange(most_parents+2) - 0.5
    plt.hist(parents, bins = bins, edgecolor='#e0e0e0')
    plt.xticks(range(most_parents + 1))
    plt.xlim([-1, most_parents + 1])
    plt.title("Number of parents for each target with alpha " + str(alpha))
    if save:
        plt.savefig(f"{save_dir}/parent_hist_{alpha}.png")
        plt.close()
    else:
        plt.show()

#
# orig_network_path = network_path = f'networks/DIRECT-NET_network_with_FIGR_threshold_0_no_NEUROG2_top8regs_expanded_v2.csv'
# original_network = pd.read_csv(op.join(dir_prefix, network_path), index_col=None, header = None)
# original_network.columns = ['source', 'target']

for alpha in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]:
    new_network_path = f"{dir_prefix}/networks/feature_selection/{network_name}/Lasso_alpha_{alpha}/{network_name}_Lasso_{alpha}.csv"
    new_net  = pd.read_csv(new_network_path, index_col=None, header = 0)
    # compare_networks(network, new_network, alpha, save_dir=f"{dir_prefix}/networks/feature_selection/{network_name}")
    compare_networks(network, new_net, alpha, save = True, save_dir=f"{dir_prefix}/networks/feature_selection/{network_name}")

