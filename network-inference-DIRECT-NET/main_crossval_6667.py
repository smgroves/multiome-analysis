# 5-fold cross-validation for the 6667 network (DIRECT-NET 2020db + independent LASSO models,
# wo sinks, RORA/RORB combined). Fits rules independently on each fold's training split and
# validates on that fold's held-out test split, to estimate how sensitive the fitted rules are
# to which cells happened to be used for training (as opposed to bootstrap-resampling a fixed
# rule set's test-set evaluation, which is cheap but doesn't capture this).
#
# All outputs are kept under 6667/cross-validation/ so they never mix with the full-data fit in
# 6667/rules/ and 6667/validation/in_sample_validation/.
#
# Uses the same network/data/normalization settings as
# main_all_data_remove_selfloops_6667.py so fold-level results are comparable to the full fit.

import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import bobaT as bb

if __name__ == "__main__":
    # =============================================================================
    # Config -- must match main_all_data_remove_selfloops_6667.py so folds are comparable
    # to the full-data fit
    # =============================================================================
    dir_prefix = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
    network_path = "networks/feature_selection/DIRECT-NET_network_2020db_0.1/combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
    data_path = "/data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
    cellID_table = "data/AA_clusters_splitgen.csv"
    cluster_header_list = ["class"]

    remove_sinks = False
    remove_selfloops = True
    remove_sources = False

    node_normalization = 0.3
    node_threshold = 0
    transpose = True

    brcd = "6667"
    n_folds = 5
    random_state = 1234
    split_fname = "combined"

    if dir_prefix[-1] != os.sep:
        dir_prefix = dir_prefix + os.sep

    CV_DIR = f"{dir_prefix}{brcd}/cross-validation"
    SPLIT_DIR = f"{CV_DIR}/data_split"
    RULES_DIR = f"{CV_DIR}/rules"
    VAL_ROOT = f"{CV_DIR}/validation"
    for d in (CV_DIR, SPLIT_DIR, RULES_DIR, VAL_ROOT):
        os.makedirs(d, exist_ok=True)

    time1 = time.time()

    # =============================================================================
    # Load network + full dataset (used only to build the folds)
    # =============================================================================
    print("Loading network...")
    graph, vertex_dict = bb.load.load_network(
        f"{dir_prefix}{network_path}",
        remove_sinks=remove_sinks,
        remove_selfloops=remove_selfloops,
        remove_sources=remove_sources,
    )
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)

    print("Loading full dataset...")
    data_t0 = bb.load.load_data(
        f"{dir_prefix}{data_path}",
        nodes,
        norm=node_normalization,
        delimiter=",",
        log1p=False,
        transpose=transpose,
        sample_order=False,
        fillna=0,
    )
    clusters = bb.utils.get_clusters(
        data_t0,
        cellID_table=f"{dir_prefix}{cellID_table}",
        cluster_header_list=cluster_header_list,
    )

    # =============================================================================
    # Build (or reuse) the 5 stratified folds
    # =============================================================================
    fold_paths = [
        f"{SPLIT_DIR}/train_t0_{split_fname}_{idx}.csv" for idx in range(n_folds)
    ]
    if all(os.path.isfile(p) for p in fold_paths):
        print("Fold splits already exist in data_split/, reusing them...")
    else:
        print(f"Splitting data into {n_folds} stratified folds...")
        bb.utils.split_train_test_crossval(
            data_t0,
            None,
            clusters,
            SPLIT_DIR,
            folds=n_folds,
            fname=split_fname,
            random_state=random_state,
        )

    # =============================================================================
    # Fit rules + validate on each fold
    # =============================================================================
    for idx in range(n_folds):
        fold_t1 = time.time()
        print(f"\n=== Fold {idx} ===")

        train_path = f"{SPLIT_DIR}/train_t0_{split_fname}_{idx}.csv"
        test_path = f"{SPLIT_DIR}/test_t0_{split_fname}_{idx}.csv"

        data_train = bb.load.load_data(
            train_path,
            nodes,
            norm=node_normalization,
            delimiter=",",
            log1p=False,
            transpose=True,
            sample_order=False,
            fillna=0,
        )
        data_test = bb.load.load_data(
            test_path,
            nodes,
            norm=node_normalization,
            delimiter=",",
            log1p=False,
            transpose=True,
            sample_order=False,
            fillna=0,
        )

        rules_fname = f"{RULES_DIR}/rules_fold{idx}.txt"
        if os.path.isfile(rules_fname):
            print(f"Rules for fold {idx} already exist, reading them in...")
            rules, regulators_dict = bb.load.load_rules(fname=rules_fname)
        else:
            print(f"Fitting rules on fold {idx} training data ({data_train.shape[0]} cells)...")
            rules, regulators_dict, strengths, signed_strengths = bb.tl.get_rules(
                data=data_train,
                vertex_dict=vertex_dict,
                plot=False,
                threshold=node_threshold,
            )
            bb.tl.save_rules(rules, regulators_dict, fname=rules_fname)
            strengths.to_csv(f"{RULES_DIR}/strengths_fold{idx}.csv")
            signed_strengths.to_csv(f"{RULES_DIR}/signed_strengths_fold{idx}.csv")

        VAL_DIR = f"{VAL_ROOT}/fold_{idx}"
        os.makedirs(VAL_DIR, exist_ok=True)

        print(f"Validating fold {idx} on held-out test data ({data_test.shape[0]} cells)...")
        validation, tprs_all, fprs_all, area_all = bb.tl.fit_validation(
            data_test,
            data_test_t1=None,
            nodes=nodes,
            regulators_dict=regulators_dict,
            rules=rules,
            save=True,
            save_dir=VAL_DIR,
            plot=True,
            show_plots=False,
            save_df=True,
            fname=f"fold{idx}",
        )
        bb.tl.save_auc_by_gene(area_all, nodes, VAL_DIR)

        summary_stats = bb.tl.get_sklearn_metrics(VAL_DIR)
        summary_stats.to_csv(f"{VAL_DIR}/summary_stats.csv")

        print(f"Fold {idx} time: {timedelta(seconds=time.time() - fold_t1)}")

    # =============================================================================
    # Combine folds: per-fold average ROC curve (genes averaged, sources excluded),
    # then bootstrap a 95% CI across folds and plot them together
    # =============================================================================
    print("\nCombining folds into a single ROC plot with CI...")
    fold_fprs, fold_tprs, fold_aucs = [], [], []
    per_gene_aucs = {}

    for idx in range(n_folds):
        VAL_DIR = f"{VAL_ROOT}/fold_{idx}"
        tprs_all, fprs_all, area_all = bb.tl.roc_from_file(
            f"{VAL_DIR}/accuracy_plots", nodes, save=False
        )
        fpr_avg, tpr_avg, auc_avg = bb.tl.get_sample_avg_curve(
            fprs_all,
            tprs_all,
            len(nodes),
            area_all,
            remove_sources=True,
            graph=graph,
            vertex_dict=vertex_dict,
        )
        fold_fprs.append(fpr_avg)
        fold_tprs.append(tpr_avg)
        fold_aucs.append(auc_avg)
        per_gene_aucs[f"fold_{idx}"] = pd.Series(area_all, index=nodes)

    bb.tl.plot_cohort_roc_with_ci(
        fold_fprs,
        fold_tprs,
        n_boot=2000,
        ci=95,
        save=True,
        save_dir=VAL_ROOT,
        show_plot=False,
        fname="5fold_crossval",
    )

    fold_auc_df = pd.DataFrame({"fold": list(range(n_folds)), "avg_auc": fold_aucs})
    fold_auc_df.to_csv(f"{VAL_ROOT}/fold_avg_aucs.csv", index=False)
    print("Per-fold average AUC (sources excluded):")
    print(fold_auc_df)
    print(f"Mean AUC across folds: {np.mean(fold_aucs):.3f} +/- {np.std(fold_aucs):.3f}")

    per_gene_df = pd.DataFrame(per_gene_aucs)
    per_gene_df["mean"] = per_gene_df.mean(axis=1)
    per_gene_df["std"] = per_gene_df.drop(columns=["mean"]).std(axis=1)
    per_gene_df.to_csv(f"{VAL_ROOT}/combined_aucs_by_gene.csv")

    time2 = time.time()
    print(f"\nTotal cross-validation time: {timedelta(seconds=time2 - time1)}")
