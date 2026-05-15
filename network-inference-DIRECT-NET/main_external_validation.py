import pandas as pd
import os
import bobaT as bb
import seaborn as sns
import numpy as np
import time
import random
print("Start")


customPalette = sns.color_palette('tab10')

# =============================================================================
# Set variables and csvs
# To modulate which parts of the pipeline need to be computed, use the following variables
# =============================================================================
plot_network = False
split_train_test = False
write_binarized_data = False
fit_rules = False
run_validation = True
validation_averages = True
find_average_states = False
find_attractors = False
# if -1, use average distance between clusters for search basin for attractors.
tf_basin = 2
# otherwise use the same size basin for all phenotypes. For single cell data, there may be so many samples that average distance is small.
filter_attractors = False
perturbations = True
stability = False
on_nodes = []
off_nodes = []

# Set variables for computation
remove_sinks = False
remove_selfloops = False
remove_sources = False

node_normalization = 0.3
node_threshold = 0  # don't remove any parents
transpose = True

# sample = sys.argv[1]
sample = "TKO-luc"
validation_fname = f'validation/allografts/{sample}'
fname = f"{sample}"
notes_for_log = "External validation"

# Set paths
dir_prefix = '/Users/smgroves/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET'
network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"

t1 = False
data_t1_path = None  # if no T1 (i.e. single dataset), replace with None

# Set metadata information
cellID_table = f'data/allografts/{sample}_clusters.csv'
# Assign headers to cluster csv, with one called "class"
cluster_header_list = ["class"]
# the below headers go with metadata_final
# cluster_header_list = ["orig.ident","nCount_RNA","nFeature_RNA","nCount_ATAC","nFeature_ATAC","nucleosome_signal",
#                        "nucleosome_percentile","TSS.enrichment","TSS.percentile","barcode","sample","ATAC_snn_res.0.5",
#                        "seurat_clusters","nCount_peaks","nFeature_peaks","peaks_snn_res.0.5","percent.mt","nCount_SCT",
#                        "nFeature_SCT","SCT_snn_res.0.5","SCT.weight","peaks.weight","nCount_Imputed_counts",
#                        "nFeature_Imputed_counts","nCount_gene_activity","nFeature_gene_activity","NE_score1",
#                        "class","non.NE_score1","comb.score","S.Score","G2M.Score","Phase","old.ident","wsnn_res.0.5"
#                        ]

# Set brcd and train/test data if rerun
brcd = str(6667)
print(brcd)

data_test_t0_path = f'data/allografts/adata_{sample}_allografts_v3_RORA_RORB_ave.csv'

# use a job brcd to keep track of multiple jobs for the same brcd
job_brcd = str(random.randint(0, 99999))
print(f"Job barcode: {job_brcd}")

random_state = 1234

# Append the results to a MasterResults file
#########################################

# =============================================================================
# Start timer and check paths
# =============================================================================

if not os.path.exists(f"{dir_prefix}/{brcd}"):
    # Create a new directory because it does not exist
    os.makedirs(f"{dir_prefix}/{brcd}")

if not os.path.exists(f"{dir_prefix}/{brcd}/jobs"):
    # Create a new directory because it does not exist
    os.makedirs(f"{dir_prefix}/{brcd}/jobs")

# sys.stdout = open(f'{dir_prefix}/{brcd}/jobs/{job_brcd}_log.txt','wt')


time1 = time.time()

# =======================================================================
# Load the network
# =============================================================================
graph, vertex_dict = bb.load.load_network(f'{dir_prefix}/{network_path}', remove_sinks=remove_sinks, remove_selfloops=remove_selfloops,
                                          remove_sources=remove_sources)

v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

# =============================================================================
# Load the data and clusters
# =============================================================================
print('Reading in data')

data_test_t0 = bb.load.load_data(f'{dir_prefix}/{data_test_t0_path}', nodes, norm=node_normalization, delimiter=',',
                                 log1p=False, transpose=True, sample_order=False, fillna=0)

# clusters = bb.utils.get_clusters(data_test_t0_path, is_data_split=False,
#                                  cellID_table=f"{dir_prefix}/{cellID_table}", cluster_header_list=cluster_header_list)

# # combine RORA and RORB into "RORA_RORB" because they are similar TFs and we don't have both in the human dataset; take maverage
# if "RORA" in data_test_t0.index and "RORB" in data_test_t0.index:
#     data_test_t0.loc["RORA_RORB"] = np.mean(
#         data_test_t0.loc["RORA"], data_test_t0.loc["RORB"])
#     data_test_t0 = data_test_t0.drop(["RORA", "RORB"], axis=0)

# # =============================================================================
# # Read in binarized data
# # =============================================================================
# print('Binarizing data')
# if write_binarized_data:
#     save = True
# else:
#     save = False
# if not os.path.exists(f"{dir_prefix}/{brcd}/binarized_data"):
#     # Create a new directory because it does not exist
#     os.makedirs(f"{dir_prefix}/{brcd}/binarized_data")

# print('Binarizing test data')
# binarized_data_test = bb.proc.binarize_data(data_test_t0, phenotype_labels=clusters, save=save,
#                                             save_dir=f"{dir_prefix}/{brcd}/binarized_data", fname=f'binarized_data_test_t0_{fname}')


print("Reading in pre-generated rules...")
rules, regulators_dict = bb.load.load_rules(
    fname=f"{dir_prefix}/{brcd}/rules/rules_{brcd}.txt")

# =============================================================================
# Calculate AUC for test dataset for a true error calculation
# =============================================================================

if run_validation:
    print("Running validation step...")
    VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"
    try:
        os.makedirs(VAL_DIR)
    except FileExistsError:
        pass

    validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(data_test_t0, data_test_t1=None, nodes=nodes,
                                                                    regulators_dict=regulators_dict, rules=rules,
                                                                    save=True, save_dir=VAL_DIR, plot=True,
                                                                    show_plots=False, save_df=True, fname=fname)
    # Saves auc values for each gene (node) in the passed directory as 'aucs.csv'
    bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)


else:
    print("Skipping validation step...")

if validation_averages:
    # TODO: remove any nodes that are sources and are skewing the AUCs artificially
    print("Calculating validation averages...")
    VAL_DIR = f"{dir_prefix}/{brcd}/{validation_fname}"

    if run_validation == False:
        # Function to calculate roc and tpr, fpr, area from saved validation files
        # if validation == False, read in values from files instead of from above
        tprs_all, fprs_all, area_all = bb.tl.roc_from_file(
            f'{VAL_DIR}/accuracy_plots', nodes, save=True, save_dir=VAL_DIR)

    aucs = pd.read_csv(f'{VAL_DIR}/aucs.csv', header=None, index_col=0)
    print("AUC means: ", aucs.mean())

    # bb.plot.plot_aucs(aucs, save=True, save_dir=VAL_DIR, show_plot=True)
    # once BB > 0.0.7, change to this line
    bb.plot.plot_aucs(VAL_DIR, save=True, show_plot=True)

    bb.plot.plot_validation_avgs(fprs_all, tprs_all, len(
        nodes), area_all, save=True, save_dir=VAL_DIR, show_plot=True)

    # bb version > 0.1.7
    summary_stats = bb.tl.get_sklearn_metrics(
        VAL_DIR)  # TODO need to fix append here
    bb.plot.plot_sklearn_metrics(VAL_DIR)
    bb.plot.plot_sklearn_summ_stats(summary_stats.drop(
        "max_error", axis=1), VAL_DIR, fname="")

else:
    print("Skipping validation averaging...")


time2 = time.time()
time_for_job = (time2 - time1) / 60.
print("Time for job: ", time_for_job)

log_job(dir_prefix, brcd, random_state, network_path, data_test_t0_path, data_t1_path, cellID_table, node_normalization,
        node_threshold, split_train_test, write_binarized_data, fit_rules, run_validation, validation_averages,
        find_average_states, find_attractors, tf_basin, filter_attractors, on_nodes, off_nodes, perturbations, stability,
        time=time_for_job, job_barcode=job_brcd,
        notes_for_job=notes_for_log)
