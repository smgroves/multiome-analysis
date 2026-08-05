"""Loaders: each returns a canonical edge table.

Method loaders produce an inferred network per GRN method; ground-truth loaders
produce the reference / gold-standard network used for scoring. Add a new method by
writing a loader and registering it in METHOD_LOADERS.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import BEELINE_GSD, CFG, DIRECT_NET
from .edges import _finalize_edges, matrix_to_edges


# ---------------------------------------------------------------------------
# Method loaders
# ---------------------------------------------------------------------------

def load_bobat(run: str = CFG.bobat_run, signed: bool = True) -> pd.DataFrame:
    """boba-T / BooleaBayes network from its fitted strength matrix.

    Uses signed_strengths.csv (activation/repression sign preserved) by default.
    edge_weights.csv is an alternative (per-input regulatory weights).
    """
    fname = "signed_strengths.csv" if signed else "edge_weights.csv"
    path = os.path.join(DIRECT_NET, run, "rules", fname)
    mat = pd.read_csv(path, header=0, index_col=0)
    return matrix_to_edges(mat, regulators_on_columns=True)


def load_bobat_topology(network_csv: str) -> pd.DataFrame:
    """boba-T topology from a plain edge-list CSV (source,target[,score,evidence])."""
    df = pd.read_csv(os.path.join(DIRECT_NET, network_csv), header=None)
    df = df.rename(columns={0: "source", 1: "target"})
    if df.shape[1] > 2:
        df = df.rename(columns={2: "score"})
    return _finalize_edges(df[["source", "target"] + (["score"] if "score" in df else [])])


def load_celloracle(links_path: Optional[str] = None) -> pd.DataFrame:
    """CellOracle inferred GRN -> canonical edges.

    CellOracle stores per-cluster filtered GRNs in a Links object; each
    links.filtered_links[cluster] is a DataFrame with columns
    ['source', 'target', 'coef_mean', 'coef_abs', '-logp', 'p'].
    Aggregate across clusters (mean coef) or pass a single cluster's table.

        import celloracle as co
        links = co.load_hdf5(links_path)          # or oracle.get_links(...)
        df = pd.concat(links.filtered_links.values())
        agg = (df.groupby(['source', 'target'])['coef_mean']
                 .mean().reset_index().rename(columns={'coef_mean': 'weight'}))
        return _finalize_edges(agg)

    Alternatively, from the simulation coefficient matrix (genes x genes):
        return matrix_to_edges(oracle.coef_matrix, regulators_on_columns=?)  # verify axis
    """
    raise NotImplementedError(
        "Point load_celloracle at a saved Links object / coef_matrix. "
        "See docstring for the aggregation snippet."
    )


def load_scenic(adjacencies_path: Optional[str] = None) -> pd.DataFrame:
    """SCENIC/pySCENIC GRN. adjacencies.tsv has columns [TF, target, importance]."""
    if adjacencies_path is None:
        raise NotImplementedError("Provide SCENIC adjacencies.tsv path.")
    df = pd.read_csv(adjacencies_path, sep="\t")
    df = df.rename(columns={"TF": "source", "importance": "weight"})
    return _finalize_edges(df[["source", "target", "weight"]])


def load_genie3(link_list_path: Optional[str] = None) -> pd.DataFrame:
    """GENIE3 weighted link list: columns [regulatoryGene, targetGene, weight]."""
    if link_list_path is None:
        raise NotImplementedError("Provide GENIE3 link-list path.")
    df = pd.read_csv(link_list_path)
    df = df.rename(
        columns={"regulatoryGene": "source", "targetGene": "target"}
    )
    return _finalize_edges(df[["source", "target", "weight"]])


def load_wgcna(edge_path: Optional[str] = None) -> pd.DataFrame:
    """WGCNA co-expression edges (undirected; sign from correlation)."""
    raise NotImplementedError("Provide WGCNA edge table path.")


# Registry: name -> (loader, kwargs). Extend freely.
METHOD_LOADERS: dict[str, tuple[Callable[..., pd.DataFrame], dict]] = {
    "boba-T": (load_bobat, {}),
    "CellOracle": (load_celloracle, {}),
    "SCENIC": (load_scenic, {}),
    "GENIE3": (load_genie3, {}),
    "WGCNA": (load_wgcna, {}),
}


def load_all_methods(names: Optional[list[str]] = None) -> dict[str, pd.DataFrame]:
    """Load every registered method that has a working loader; skip stubs."""
    names = names or list(METHOD_LOADERS)
    out = {}
    for name in names:
        loader, kw = METHOD_LOADERS[name]
        try:
            out[name] = loader(**kw)
            print(f"[load] {name}: {len(out[name])} edges")
        except NotImplementedError as e:
            print(f"[skip] {name}: {e}")
    return out


# ---------------------------------------------------------------------------
# Ground-truth / reference-network loaders
# ---------------------------------------------------------------------------

def load_beeline_network(path: str) -> pd.DataFrame:
    """BEELINE reference network (GroundTruthNetwork.csv / refNetwork.csv).

    Format: columns Gene1, Gene2, Type where Type is '+' (activation) or '-'
    (repression). Edge = Gene1 -> Gene2. Used as the reference / gold standard for
    both the structure and BEELINE comparisons.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={"Gene1": "source", "Gene2": "target"})
    if "Type" in df.columns:
        df["sign"] = df["Type"].map({"+": 1, "-": -1}).fillna(0).astype(int)
        df["weight"] = df["sign"].astype(float)
    df = df.drop_duplicates(subset=["source", "target"])
    keep = ["source", "target"] + [c for c in ("weight", "sign") if c in df.columns]
    return _finalize_edges(df[keep])


def load_beeline_ranked(path: str) -> pd.DataFrame:
    """BEELINE algorithm output (rankedEdges.csv): Gene1, Gene2, EdgeWeight.

    Edge = Gene1 -> Gene2; score = EdgeWeight (unsigned ranking confidence). Sign is
    unknown for most methods, so it is left as 0. Handles comma- or tab-separated.
    """
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.rename(columns={"Gene1": "source", "Gene2": "target", "EdgeWeight": "score"})
    df["weight"] = np.nan   # magnitude only; no signed strength
    df["sign"] = 0
    return _finalize_edges(df[["source", "target", "weight", "score", "sign"]])


def load_celloracle_base_grn(
    path: str,
    gene_col: str = "gene_short_name",
    drop_cols: tuple = ("peak_id", "gene_short_name"),
) -> pd.DataFrame:
    """CellOracle base GRN (TF-binding prior) -> canonical edges.

    The base GRN is a table: rows = peaks, columns = [peak_id, gene_short_name,
    <TF_1>, <TF_2>, ...] with 0/1 entries (1 = that TF's motif is present in a
    regulatory region assigned to gene_short_name). Edge = TF -> gene_short_name for
    every 1, aggregated (de-duplicated) across peaks.

    Build it in CellOracle with:
        base_GRN = co.data.load_human_promoter_base_GRN()   # or a custom scATAC GRN
        base_GRN.to_parquet(path)
    then point this loader at `path` (.parquet or .csv).
    """
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    tf_cols = [c for c in df.columns if c not in drop_cols]
    long = df.melt(id_vars=[gene_col], value_vars=tf_cols,
                   var_name="source", value_name="v")
    long = long[long["v"] == 1].rename(columns={gene_col: "target"})
    long = long.drop_duplicates(subset=["source", "target"])
    long["weight"] = 1.0   # binary prior; no direction of effect
    long["sign"] = 0
    return _finalize_edges(long[["source", "target", "weight", "sign"]])


def load_sclc_chipseq_gt(path: str) -> pd.DataFrame:
    """Real, independent SCLC ChIP-seq ground truth -> canonical edges.

    Expects a pre-extracted (source, target, sign) CSV -- see benchmarking/README.md
    ("Sourcing a real SCLC ground truth") for how borromeo2016_ascl1_human_chip.csv and
    pozo2021_ascl1_direct.csv were built from the papers' supplementary tables. Both are
    ASCL1-only (single source TF); sign is 0 for binding-only evidence (Borromeo),
    +/-1 for binding + knockdown-confirmed direction (Pozo).
    """
    df = pd.read_csv(path)
    df["weight"] = df["sign"].astype(float)   # signed if known, else 0 -> weight 0 (unsigned)
    return _finalize_edges(df[["source", "target", "weight", "sign"]])


def load_ground_truth(source: str = "beeline", path: Optional[str] = None) -> pd.DataFrame:
    """Reference network for structure (point 1) and gold standard (point 2).

      - 'beeline': BEELINE ground-truth network (default: bundled GSD example).
      - 'celloracle_base': CellOracle base GRN (scATAC/promoter prior) -> requires a
        saved base_GRN parquet/csv path.
      - 'chip' / 'curated': ChIP-Atlas / ENCODE / TRRUST / DoRothEA (not wired yet).
    """
    if source == "beeline":
        path = path or os.path.join(BEELINE_GSD, "GroundTruthNetwork.csv")
        return load_beeline_network(path)
    if source == "celloracle_base":
        if path is None:
            raise NotImplementedError(
                "Provide a saved CellOracle base_GRN parquet/csv path."
            )
        return load_celloracle_base_grn(path)
    raise NotImplementedError(
        f"Ground-truth source '{source}' not wired up (try 'beeline' or 'celloracle_base')."
    )


def common_node_universe(method_edges: dict[str, pd.DataFrame]) -> set[str]:
    """Intersection of node sets across all loaded methods."""
    node_sets = []
    for e in method_edges.values():
        node_sets.append(set(e["source"]) | set(e["target"]))
    return set.intersection(*node_sets) if node_sets else set()
