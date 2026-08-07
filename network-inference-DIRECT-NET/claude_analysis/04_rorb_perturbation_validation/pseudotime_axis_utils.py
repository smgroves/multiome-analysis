"""Shared helper: an NE1<->nonNE1 combinatorial axis (like diagnose_walk_axis_position.py's
NE<->nonNE axis) but with two changes, both driven by real data rather than assumption:

1. Poles are `Arc_5`/NE1 and `Arc_4`/nonNE1, not `Generalist_NE`/`Generalist_nonNE`.
   Checked directly against real GEMM cells' own palantir pseudotime
   (`comparisons/organoid_walks/palantir_data.csv` joined to `data/AA_clusters_splitgen.csv`
   phenotype labels): mean pseudotime per archetype is Arc_5 (0.013) < Arc_6 (0.031) <
   Generalist_NE (0.057) < Arc_3 (0.058) < Arc_1 (0.243) < Arc_2 (0.382) <
   Generalist_nonNE (0.466) < Arc_4 (0.532) -- the Generalists sit well short of the real
   pseudotime extremes; Arc_5/Arc_4 are the actual poles. (Arc_1/"Intermediate" landing
   near the middle is a reassuring consistency check, not part of the pole choice.)
2. The raw 0-100 %-agreement position is passed through a monotonic calibration curve
   (isotonic regression, decreasing: higher %NE1-agreement -> lower pseudotime) fit on
   every real GEMM cell's own (axis position, palantir_pseudotime) pair, so the plotted
   y-axis is real pseudotime, not a raw bit-count fraction.

Run standalone to print the fitted calibration's face-validity check (do all 8 archetypes'
own average states land in the same order as their real empirical mean pseudotime?):
    /opt/anaconda3/envs/bobaT_env_py3.13/bin/python claude_analysis/04_rorb_perturbation_validation/pseudotime_axis_utils.py
"""

import numpy as np

if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid
import bobaT as bb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

DIR_PREFIX = "/Users/xpz5km/Documents/GitHub/multiome-analysis/network-inference-DIRECT-NET"
NETWORK_PATH = (
    "networks/feature_selection/DIRECT-NET_network_2020db_0.1/"
    "combined_DIRECT-NET_network_2020db_0.1_Lasso_wo_sinks_RORA_RORB_combined.csv"
)
GEMM_DATA_PATH = f"{DIR_PREFIX}/data/adata_imputed_combined_v3_RORA_RORB_ave.csv"
GEMM_CLUSTERS_PATH = f"{DIR_PREFIX}/data/AA_clusters_splitgen.csv"
PALANTIR_PATH = f"{DIR_PREFIX}/comparisons/organoid_walks/palantir_data.csv"
NE1_POLE, NONNE1_POLE = "Arc_5", "Arc_4"
GEMM_NORM = 0.3  # matches main_all_data_remove_selfloops_6667.py's node_normalization


def load_axis(nodes):
    """Same construction as diagnose_walk_axis_position.py::load_axis, but poled on
    NE1(Arc_5)/nonNE1(Arc_4) instead of the Generalists."""
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    ne1_row = avg_states.loc[NE1_POLE]
    nonne1_row = avg_states.loc[NONNE1_POLE]
    ne1_idx = int("".join(str(int(v)) for v in ne1_row), 2)

    axis_genes = [g for g in nodes if int(ne1_row[g]) != int(nonne1_row[g])]
    n = len(nodes)
    mask = 0
    for i, g in enumerate(nodes):
        if g in axis_genes:
            mask |= 1 << (n - 1 - i)
    return ne1_idx, mask, len(axis_genes)


def axis_position(state_idx, ne1_idx, mask, n_axis_genes):
    mismatches = ((state_idx ^ ne1_idx) & mask).bit_count()
    return 100.0 * (n_axis_genes - mismatches) / n_axis_genes


def _load_real_cell_axis_and_pseudotime(nodes, ne1_idx, mask, n_axis_genes):
    """Binarizes every real GEMM cell (same convention used to build average_states.txt/
    attractors: norm=0.3, threshold=0.5), joins to its own palantir_pseudotime via
    CellID -> barcode+sample, returns (axis_position array, pseudotime array), plus a
    per-archetype-label breakdown for the face-validity check."""
    data = bb.load.load_data(GEMM_DATA_PATH, nodes, norm=GEMM_NORM, delimiter=",",
                              log1p=False, transpose=True, sample_order=False, fillna=0)
    binaries = bb.utils.binarize_data_df(data, nodes, threshold=0.5)
    cell_idx = binaries.apply(lambda row: int("".join(str(int(v)) for v in row), 2), axis=1)
    positions = cell_idx.apply(lambda idx: axis_position(idx, ne1_idx, mask, n_axis_genes))

    clusters = pd.read_csv(GEMM_CLUSTERS_PATH)
    extracted = clusters["CellID"].str.extract(r":([ACGT]{16})x-M(\d+)")
    clusters["barcode"] = extracted[0]
    clusters["sample"] = extracted[1].astype(float)
    clusters = clusters.set_index("CellID")
    clusters["axis_position"] = positions.reindex(clusters.index)
    clusters["phenotype"] = clusters["S_0.5threshold_splitgen"]

    palantir = pd.read_csv(PALANTIR_PATH)[["barcode", "sample", "palantir_pseudotime"]]
    palantir["sample"] = palantir["sample"].astype(float)
    merged = clusters.reset_index().merge(palantir, on=["barcode", "sample"], how="inner")
    merged = merged.dropna(subset=["axis_position", "palantir_pseudotime"])
    return merged


def fit_calibration(nodes, ne1_idx, mask, n_axis_genes, n_bins=40, verbose=True):
    """Fits the axis_position -> palantir_pseudotime calibration on real GEMM cells.

    Raw per-cell isotonic regression produces a step function with plateaus wherever
    per-cell noise breaks the monotonic constraint -- when applied to a walk's per-step
    positions, that turns smoothly-drifting individual walks into vertical spikes each
    time a step crosses a plateau edge. Instead: bin real cells' axis_position into
    `n_bins` equal-width bins, take each bin's mean pseudotime (averaging out per-cell
    noise first), fit isotonic regression on those bin means (still enforces the
    decreasing constraint, now on much less noisy inputs), then linearly interpolate
    between bin centers for a smooth, monotonic, continuous calibration curve.

    Returns a callable: array of axis_position (0-100) -> calibrated pseudotime-scale
    y-values.
    """
    merged = _load_real_cell_axis_and_pseudotime(nodes, ne1_idx, mask, n_axis_genes)

    bin_edges = np.linspace(0, 100, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    merged["bin"] = np.clip(np.digitize(merged["axis_position"], bin_edges[1:-1]), 0, n_bins - 1)
    bin_means = merged.groupby("bin")["palantir_pseudotime"].mean()
    bin_counts = merged.groupby("bin")["palantir_pseudotime"].size()
    # Empty bins (no real cells land there) get dropped -- isotonic + interp only needs
    # the bins that actually have data; np.interp extrapolates flatly beyond the ends.
    valid_bins = bin_means.index.to_numpy()
    x = bin_centers[valid_bins]
    y = bin_means.to_numpy()

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    y_smooth = iso.fit_transform(x, y)

    if verbose:
        by_phenotype = merged.groupby("phenotype")["palantir_pseudotime"].mean().sort_values()
        print(f"Fitted calibration on {len(merged)} real GEMM cells, binned into {len(x)} "
              f"populated bins (axis_position range {merged['axis_position'].min():.1f}-{merged['axis_position'].max():.1f}, "
              f"bin sizes {bin_counts.min()}-{bin_counts.max()}).")
        print("Real mean pseudotime per phenotype (face-validity reference):")
        print(by_phenotype)

    def calibrate(positions):
        return np.interp(positions, x, y_smooth)

    return calibrate


def archetype_axis_positions(nodes, ne1_idx, mask, n_axis_genes):
    avg_states = pd.read_csv(f"{DIR_PREFIX}/6667/attractors/average_states.txt", index_col=0)[nodes]
    positions = {}
    for archetype, row in avg_states.iterrows():
        idx = int("".join(str(int(v)) for v in row), 2)
        positions[archetype] = axis_position(idx, ne1_idx, mask, n_axis_genes)
    return positions


if __name__ == "__main__":
    graph, vertex_dict = bb.load.load_network(f"{DIR_PREFIX}/{NETWORK_PATH}", remove_sinks=False, remove_selfloops=True, remove_sources=False)
    v_names, nodes = bb.utils.get_nodes(vertex_dict, graph)
    ne1_idx, mask, n_axis_genes = load_axis(nodes)
    print(f"NE1(Arc_5) vs nonNE1(Arc_4) axis: {n_axis_genes} genes")

    calibrate = fit_calibration(nodes, ne1_idx, mask, n_axis_genes)

    print("\n=== Face-validity: each archetype's OWN average state, calibrated ===")
    positions = archetype_axis_positions(nodes, ne1_idx, mask, n_axis_genes)
    rows = []
    for archetype, pos in positions.items():
        rows.append({"archetype": archetype, "axis_position_pct": pos, "calibrated_pseudotime": calibrate(np.array([pos]))[0]})
    df = pd.DataFrame(rows).sort_values("calibrated_pseudotime")
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))
