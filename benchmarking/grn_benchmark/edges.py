"""Canonical edge representation shared by every method and comparison.

CANONICAL EDGE TABLE  (one row per directed edge):
    source : str    regulator (TF)
    target : str    regulated gene
    weight : float  signed edge strength if available, else magnitude; NaN if binary
    score  : float  non-negative confidence used for ranking (|weight| by default)
    sign   : int    +1 activator, -1 repressor, 0/NaN unknown
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# networkx keeps this portable (bb itself uses graph-tool, which is heavier to install).
import networkx as nx

CANON_COLS = ["source", "target", "weight", "score", "sign"]


def _finalize_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a partial edge frame into the canonical schema."""
    df = df.copy()
    if "weight" not in df:
        df["weight"] = np.nan
    if "sign" not in df:
        sign_series = pd.Series(np.sign(df["weight"]), index=df.index)
        df["sign"] = sign_series.fillna(0).astype(int)
    if "score" not in df:
        # Ranking confidence: magnitude of the signed weight by default.
        df["score"] = df["weight"].abs()
    df = df.dropna(subset=["source", "target"])
    # Self-loops are kept (BooleaBayes rules include auto-regulation); drop here if unwanted.
    return df[CANON_COLS].reset_index(drop=True)


def matrix_to_edges(
    mat: pd.DataFrame, regulators_on_columns: bool = True
) -> pd.DataFrame:
    """Melt a weighted adjacency matrix into a canonical edge table.

    With regulators_on_columns=True (BooleaBayes convention), rows are targets and
    columns are regulators; edge = column(regulator) -> row(target).
    """
    m = mat.copy()
    m.index.name = "target" if regulators_on_columns else "source"
    id_vars = [m.index.name] if m.index.name is not None else None
    long = m.reset_index().melt(
        id_vars=id_vars,
        var_name="regulator" if regulators_on_columns else "target",
        value_name="weight",
    )
    long = long.dropna(subset=["weight"])
    long = long[long["weight"] != 0]
    if regulators_on_columns:
        long = long.rename(columns={"regulator": "source"})
    else:
        long = long.rename(columns={m.index.name: "source"})
    return _finalize_edges(long)


def restrict_to_nodes(edges: pd.DataFrame, nodes: set[str] | list[str]) -> pd.DataFrame:
    node_set = set(nodes)
    return edges[edges["source"].isin(node_set) & edges["target"].isin(node_set)].copy()


def edges_to_graph(edges: pd.DataFrame, directed: bool = True) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for s, t, w, sc, sg in zip(
        edges["source"], edges["target"], edges["weight"], edges["score"], edges["sign"]
    ):
        G.add_edge(s, t, weight=w, score=sc, sign=sg)
    return G


def edge_key(source, target, directed: bool) -> tuple:
    """Canonical (source, target) key; sorted when undirected."""
    return (source, target) if directed else tuple(sorted((source, target)))


def edge_set(edges: pd.DataFrame, directed: bool = True) -> set[tuple]:
    return {edge_key(s, t, directed) for s, t in zip(edges["source"], edges["target"])}
