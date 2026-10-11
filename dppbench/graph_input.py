"""Dense graph-model input boundary; no dependency on torch."""
import numpy as np


def prepare_graph_features(values, train_mask, feature_names):
    """Impute missing covariates using supervised training nodes only.

    An undefined aggregate (e.g. sample std of a singleton group) stays
    missing in the pipeline. GNNs require a dense finite tensor: use the
    training median, or zero when the training column has no observations.
    Labels and split membership are never changed.
    """
    x = np.asarray(values, dtype=np.float64).copy()
    mask = np.asarray(train_mask, dtype=bool)
    if x.ndim != 2 or len(mask) != len(x) or not mask.any():
        raise ValueError("graph features require a nonempty, aligned training mask")
    if np.isinf(x).any():
        bad = np.isinf(x).any(axis=0)
        raise ValueError(f"graph features contain infinity: {np.asarray(feature_names)[bad].tolist()}")
    for idx in np.flatnonzero(np.isnan(x).any(axis=0)):
        observed = x[mask, idx]
        observed = observed[~np.isnan(observed)]
        fill = float(np.median(observed)) if observed.size else 0.0
        x[np.isnan(x[:, idx]), idx] = fill
    if (np.abs(x) > np.finfo(np.float32).max).any():
        raise ValueError("graph features exceed float32 range")
    return x.astype(np.float32)


def validate_graph_training(graph, metrics):
    """Reject invalid graph inputs before optimisation or sklearn metrics."""
    x = np.asarray(graph["x"])
    n = int(graph.get("num_nodes", len(x)))
    if x.ndim != 2 or len(x) != n or len(graph["y"]) != n:
        raise ValueError("graph num_nodes, feature rows and labels must agree")
    edges = np.asarray(graph.get("edge_index", np.empty((2, 0), dtype=np.int64)))
    if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
        raise ValueError("graph edge_index must be a 2-by-E integer array")
    if edges.size and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("graph edge_index references a node outside num_nodes")
    for name in ("train_mask", "val_mask", "test_mask", "std_test_mask"):
        if name in graph and np.asarray(graph[name]).shape != (n,):
            raise ValueError(f"graph {name} must align with num_nodes")
    if not np.isfinite(graph["x"]).all():
        raise ValueError("graph model input contains non-finite features")
    y = np.asarray(graph["y"])
    for name in ("train_mask", "val_mask"):
        selected = y[np.asarray(graph[name], dtype=bool)]
        if not selected.size:
            raise ValueError(f"graph {name} contains no labeled nodes")
        if not np.isfinite(selected).all() or not np.isin(selected, [0, 1]).all():
            raise ValueError(f"graph {name} contains invalid binary labels")
        if name == "val_mask" and "auc" in metrics and np.unique(selected).size < 2:
            raise ValueError("graph validation AUC requires both label classes")
