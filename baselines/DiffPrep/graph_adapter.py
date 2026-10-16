"""Transductive graph extension of the DiffPrep surrogate search protocol."""
import numpy as np
import torch
import torch.nn.functional as F

from dppbench.models.gnn import GNNModel, _build_undirected
from dppbench.graph_input import validate_graph_training


def graph_search_inputs(data, max_features, device, hidden_dim, tau, hard_sample):
    # The loader has already excluded frozen standard-test nodes during search.
    # Preserve chronological masks and unknown nodes; never random-split labels.
    if '__split__' in data.train_df and data.train_df['__split__'].eq('std_test').any():
        raise ValueError('DiffPrep graph search must not load standard-test nodes')
    graph = data.build_graph()
    validate_graph_training(graph, ['auc'])
    names = [c for c in data.train_df if c.startswith('feat_')]
    if max_features:
        names = names[:max_features]
    if not names:
        raise ValueError('DiffPrep graph search requires node features')
    train_mask = np.asarray(graph['train_mask'], dtype=bool)
    values = graph['x'][:, :len(names)].astype(np.float64)
    mean = values[train_mask].mean(axis=0)
    scale = values[train_mask].std(axis=0)
    scale = np.where(scale < 1e-6, 1., scale)
    values = np.clip((values - mean) / scale, -5., 5.)
    feat = torch.tensor(values, dtype=torch.float32, device=device)
    mask = torch.zeros_like(feat)  # build_graph imputes using training nodes only
    labels = torch.tensor(graph['y'], dtype=torch.float32, device=device)
    tr = torch.tensor(train_mask, device=device)
    va = torch.tensor(graph['val_mask'], dtype=torch.bool, device=device)
    src, dst = _build_undirected(graph['edge_index'], graph['num_nodes'])
    context = {'src': src.to(device), 'dst': dst.to(device), 'num_nodes': graph['num_nodes']}
    surrogate = GNNModel(len(names), hidden_dim=hidden_dim, num_layers=2,
                         dropout=0., model_type='sage').to(device)

    def loss_for(continuous, model, selected):
        # All nodes participate in message passing, but transform statistics
        # are fitted exclusively on supervised training nodes.
        continuous.fit_rows = tr
        out = continuous(feat, mask, gumbel_tau=tau, hard_sample=hard_sample)
        logits = model(out, context).reshape(-1)
        loss = F.binary_cross_entropy_with_logits(logits[selected], labels[selected])
        accuracy = ((logits[selected] >= 0) == labels[selected].bool()).float().mean()
        return loss, float(accuracy.detach().cpu())

    def train_step(continuous, model):
        return loss_for(continuous, model, tr)[0]

    def val_step(continuous, model):
        return loss_for(continuous, model, va)

    return feat, mask, names, surrogate, train_step, val_step
