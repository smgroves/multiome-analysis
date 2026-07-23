#%%
import bobaT as bb
import numpy as np
import sys
import scipy.stats as stats
import matplotlib.pyplot as plt
#%%
dir_prefix = '/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
brcd = 6667
validation_fname = f'validation/human_tumor_MSK/'
network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"

#%%
# Set variables for computation
remove_sinks = False
remove_selfloops = False
remove_sources = False

VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"

samples = ["RU1065", "RU1066", "RU1080", "RU1108", "RU1124", "RU1144", "RU1145", "RU1152", "RU1181", "RU1195", "RU1215", "RU1229", "RU1231", "RU1293", "RU1311", "RU1322"]
# "1L","2L","2LR","3L","5B","mt2","mt3","mt4","mt4Rf","mt5","mt6"


################
# Network
################
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                          remove_sources=remove_sources)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

#%%
##################
# Validation stats
##################
sample_fprs, sample_tprs, avg_aucs = [], [], []

for sample_id in samples:
    print(f"Processing sample: {sample_id}")
    tprs_all, fprs_all, area_all = bb.tl.roc_from_file(
            f'{VAL_DIR}/{sample_id}/accuracy_plots', nodes, save=False, save_dir=VAL_DIR)
    num_nodes = len(nodes)
    fpr_avg, tpr_avg, auc_avg = bb.tl.get_sample_avg_curve(
        fprs_all, tprs_all, num_nodes, area_all, remove_sources = True, graph = graph, vertex_dict = vertex_dict
    )
    print(f"\tSample {sample_id} - Average AUC: {auc_avg}")
    avg_aucs.append(auc_avg)
    sample_fprs.append(fpr_avg)
    sample_tprs.append(tpr_avg)

fpr_matrix = np.vstack(sample_fprs)  # same shape as tpr_matrix: (n_patients, n_thresholds)
tpr_matrix = np.vstack(sample_tprs)
 
# Direct look at patient-to-patient spread, no bootstrap involved
patient_std = tpr_matrix.std(axis=0)
patient_range = tpr_matrix.max(axis=0) - tpr_matrix.min(axis=0)
print("Std across patients at each threshold:", patient_std)
print("Range across patients at each threshold:", patient_range)


n = tpr_matrix.shape[0]
mean_tpr = tpr_matrix.mean(axis=0)
mean_fpr = fpr_matrix.mean(axis=0)   
se = tpr_matrix.std(axis=0, ddof=1) / np.sqrt(n)
t_crit = stats.t.ppf(0.975, df=n-1)
lower_param = mean_tpr - t_crit * se
upper_param = mean_tpr + t_crit * se

mean_tpr = tpr_matrix.mean(axis=0)

order = np.argsort(mean_fpr)
mean_fpr = mean_fpr[order]
mean_tpr = mean_tpr[order]
lower_tpr = lower_param[order]  # after computing these too, or reorder at the end
upper_tpr = upper_param[order]
#%%
################
# Plotting
################
plt.figure()
ax = plt.subplot()
for fpr_i, tpr_i in zip(sample_fprs, sample_tprs):
    plt.plot(fpr_i, tpr_i, "-", color="gray", alpha=0.3, lw=1)
plt.plot(mean_fpr, mean_tpr, "-o", color="C0", label="Mean ROC", lw=2)
plt.fill_between(mean_fpr, lower_tpr, upper_tpr, alpha=0.3, color="C0", label="95% CI")
ax.plot([0, 1], [0, 1], ls="--", c=".3")
plt.xlim(0, 1); plt.ylim(0, 1)
plt.legend()
plt.title(f"n={n} patients — mean AUC={np.mean(avg_aucs):.3f} ± {np.std(avg_aucs):.3f}")
plt.suptitle(f"Mean ROC with 95% CI across all samples", fontsize=16)
# plt.show()
plt.savefig(f"{VAL_DIR}/mean_roc_with_ci_all_samples.pdf", dpi=300)
plt.close()

