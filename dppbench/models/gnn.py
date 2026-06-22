"""Pure-PyTorch implementations of GCN, GraphSAGE, GAT and a transductive
full-batch training loop. No PyG / DGL dependency.
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Helpers: build symmetric normalised sparse adjacency / build CSR-style
# index buffers for SAGE-mean / GAT.
# ----------------------------------------------------------------------
def _to_undirected(edge_index):
    """Symmetrise and add self-loops, return (src, dst) numpy int64 1-D."""
    src = np.concatenate([edge_index[0], edge_index[1]])
    dst = np.concatenate([edge_index[1], edge_index[0]])
    # de-duplicate (sym edges may repeat)
    pairs = np.stack([src, dst], axis=0)
    _, unique_idx = np.unique(pairs.T, axis=0, return_index=True)
    unique_idx.sort()
    src = pairs[0, unique_idx]
    dst = pairs[1, unique_idx]
    return src, dst


def _build_norm_sparse_adj(edge_index, num_nodes, device):
    src, dst = _to_undirected(edge_index)
    # add self loops
    self_idx = np.arange(num_nodes, dtype=np.int64)
    src = np.concatenate([src, self_idx])
    dst = np.concatenate([dst, self_idx])

    deg = np.bincount(src, minlength=num_nodes).astype(np.float32)
    deg = np.maximum(deg, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    vals = (d_inv_sqrt[src] * d_inv_sqrt[dst]).astype(np.float32)

    indices = torch.from_numpy(np.stack([src, dst], axis=0)).long().to(device)
    values = torch.from_numpy(vals).to(device)
    A = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)).coalesce()
    return A


def _build_undirected(edge_index, num_nodes):
    """Return (src, dst) torch.long for SAGE / GAT (undirected, no self-loop here)."""
    src, dst = _to_undirected(edge_index)
    return torch.from_numpy(src).long(), torch.from_numpy(dst).long()


# ----------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------
class _GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, A):
        # A: sparse (N, N), x: (N, D)
        return torch.sparse.mm(A, self.lin(x))


class _SAGELayer(nn.Module):
    """Mean-aggregator GraphSAGE layer (no neighbour sampling — full batch)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.neigh_lin = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, src, dst, num_nodes):
        # mean over incoming neighbours: aggregate x[src] into dst
        msg = x[src]  # (E, D)
        out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)
        deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        mean = out / deg
        return self.self_lin(x) + self.neigh_lin(mean)


class _GATLayer(nn.Module):
    """Single-head GAT layer (sufficient for benchmarking without PyG)."""

    def __init__(self, in_dim, out_dim, dropout=0.0, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_dim, 1))
        self.a_dst = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.alpha = alpha
        self.dropout = dropout

    def forward(self, x, src, dst, num_nodes):
        h = self.W(x)  # (N, D)
        e_src = (h @ self.a_src).squeeze(-1)  # (N,)
        e_dst = (h @ self.a_dst).squeeze(-1)  # (N,)

        # Per-edge unnormalised attention.
        e = e_src[src] + e_dst[dst]  # (E,)
        e = F.leaky_relu(e, negative_slope=self.alpha)

        # softmax over destination nodes (incoming edges aggregated at dst).
        # numerical stability: subtract per-dst max
        e_max = torch.full((num_nodes,), float("-inf"), device=x.device, dtype=e.dtype)
        e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=True)
        e_max = torch.where(torch.isinf(e_max), torch.zeros_like(e_max), e_max)
        e_norm = (e - e_max[dst]).exp()

        denom = torch.zeros(num_nodes, device=x.device, dtype=e.dtype)
        denom.index_add_(0, dst, e_norm)
        denom = denom.clamp(min=1e-12)
        attn = e_norm / denom[dst]
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        msg = h[src] * attn.unsqueeze(-1)
        out = torch.zeros(num_nodes, h.size(1), device=x.device, dtype=h.dtype)
        out.index_add_(0, dst, msg)
        return out


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
class GNNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5,
                 model_type="sage", out_dim=1):
        super().__init__()
        self.model_type = model_type
        self.dropout = float(dropout)
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            din, dout = dims[i], dims[i + 1]
            if model_type == "gcn":
                layers.append(_GCNLayer(din, dout))
            elif model_type == "sage":
                layers.append(_SAGELayer(din, dout))
            elif model_type == "gat":
                layers.append(_GATLayer(din, dout, dropout=self.dropout))
            else:
                raise ValueError(f"Unknown model_type {model_type}")
        self.layers = nn.ModuleList(layers)

    def forward(self, x, ctx):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, _GCNLayer):
                x = layer(x, ctx["A"])
            else:
                x = layer(x, ctx["src"], ctx["dst"], ctx["num_nodes"])
            if i < len(self.layers) - 1:
                x = F.relu(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x.squeeze(-1) if x.dim() > 1 and x.size(-1) == 1 else x


class GCN(GNNModel):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, dropout, model_type="gcn", out_dim=1)


class GraphSAGE(GNNModel):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, dropout, model_type="sage", out_dim=1)


class GAT(GNNModel):
    def __init__(self, in_dim, hidden_dim=64, num_layers=2, dropout=0.5, **kwargs):
        super().__init__(in_dim, hidden_dim, num_layers, dropout, model_type="gat", out_dim=1)


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def _eval_metrics(logits, y, mask, metrics, threshold=0.5):
    from sklearn.metrics import roc_auc_score, f1_score
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    preds = (probs >= threshold).astype(np.int64)
    y_true = y[mask].detach().cpu().numpy()
    out = {}
    for m in metrics:
        try:
            if m == "auc":
                out[m] = float(roc_auc_score(y_true, probs))
            elif m == "f1":
                out[m] = float(f1_score(y_true, preds, zero_division=0))
            elif m == "acc":
                out[m] = float((preds == y_true).mean())
            else:
                out[m] = float("nan")
        except (ValueError, ZeroDivisionError):
            out[m] = float("nan")
    return out


def _best_f1_threshold(logits, y, mask):
    from sklearn.metrics import precision_recall_curve
    probs = torch.sigmoid(logits[mask]).detach().cpu().numpy()
    y_true = y[mask].detach().cpu().numpy()
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    )
    best_idx = int(np.nanargmax(f1))
    if best_idx >= len(thresholds):
        return 1.0
    return float(thresholds[best_idx])


def train_graph(model, graph, model_params, train_cfg, device=None):
    """Full-batch transductive training loop.

    Args:
        model: GCN / GraphSAGE / GAT instance.
        graph: dict with keys x, edge_index, y, train_mask, val_mask, test_mask, num_nodes.
        model_params: dict (lr, epochs, weight_decay, seed, ...).
        train_cfg: dict (metrics, ...).
        device: optional torch device string. If None, derived from
            model_params['device'] then CUDA availability.
    """
    seed = int(model_params.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = model_params.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    x = torch.from_numpy(graph["x"]).float().to(device)
    y = torch.from_numpy(graph["y"]).long().to(device)
    train_mask = torch.from_numpy(graph["train_mask"]).bool().to(device)
    val_mask = torch.from_numpy(graph["val_mask"]).bool().to(device)
    test_mask = torch.from_numpy(graph["test_mask"]).bool().to(device)
    num_nodes = int(graph["num_nodes"])

    # Build aggregation context once.
    ctx = {"num_nodes": num_nodes}
    if any(isinstance(layer, _GCNLayer) for layer in model.layers):
        ctx["A"] = _build_norm_sparse_adj(graph["edge_index"], num_nodes, device)
    if any(isinstance(layer, (_SAGELayer, _GATLayer)) for layer in model.layers):
        src, dst = _build_undirected(graph["edge_index"], num_nodes)
        ctx["src"] = src.to(device)
        ctx["dst"] = dst.to(device)

    epochs = int(model_params.get("epochs", 100))
    lr = float(model_params.get("lr", 0.01))
    weight_decay = float(model_params.get("weight_decay", 5e-4))
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    metrics = train_cfg.get("metrics", ["auc"])
    monitor = "auc" if "auc" in metrics else metrics[0]
    best_val = -float("inf")
    best_state = None

    pos_weight = None
    if bool(train_cfg.get("auto_pos_weight", True)):
        y_train = y[train_mask].float()
        positives = y_train.sum()
        negatives = y_train.numel() - positives
        if positives.item() > 0 and negatives.item() > 0:
            pos_weight = (negatives / positives).clamp(max=100.0)
    if "pos_weight" in train_cfg:
        pos_weight = torch.tensor(
            float(train_cfg["pos_weight"]), dtype=torch.float32, device=device,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        optim.zero_grad()
        logits = model(x, ctx)
        # Only labeled training nodes contribute.
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], y[train_mask].float(), pos_weight=pos_weight
        )
        loss.backward()
        optim.step()

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                logits_eval = model(x, ctx)
                tr = _eval_metrics(logits_eval, y, train_mask, metrics)
                va = _eval_metrics(logits_eval, y, val_mask, metrics)
            msg_extras = " ".join(f"val_{k}={v:.4f}" for k, v in va.items())
            print(f"  [{model.model_type.upper()}] epoch {epoch}/{epochs} loss={loss.item():.4f} {msg_extras}")
            if va.get(monitor, -float("inf")) > best_val:
                best_val = va[monitor]
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits_eval = model(x, ctx)
        threshold = 0.5
        if "f1" in metrics and bool(train_cfg.get("calibrate_threshold", True)):
            threshold = _best_f1_threshold(logits_eval, y, val_mask)
        test_result = _eval_metrics(
            logits_eval, y, test_mask, metrics, threshold=threshold,
        )
        if "f1" in metrics:
            test_result["f1_threshold"] = threshold
        std_test_mask_np = graph.get("std_test_mask")
        if std_test_mask_np is not None and np.asarray(std_test_mask_np).any():
            std_test_mask = torch.from_numpy(np.asarray(std_test_mask_np)).bool().to(device)
            std_result = _eval_metrics(
                logits_eval, y, std_test_mask, metrics, threshold=threshold,
            )
            test_result["std_test_n_nodes"] = int(std_test_mask.sum().item())
            for key, value in std_result.items():
                test_result[f"std_test_{key}"] = value
    print("Test:", {k: f"{v:.4f}" for k, v in test_result.items()})
    return test_result
