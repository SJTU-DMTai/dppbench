"""DiffPrep model families, with DAGPipe task contexts and train-only supervision."""
import numpy as np
import pandas as pd
import torch

from baselines.DiffPrep.surrogate import TabularSurrogate, RecSurrogate
from dppbench.models.gnn import GNNModel, _build_undirected


def make_surrogate(task_type, in_dim, n_outputs, regression, cfg, adapter,
                   device, dtype):
    if task_type == 'graph':
        model = GNNModel(in_dim, hidden_dim=int(cfg.get('surrogate_hidden_dim', 32)),
                         num_layers=2, dropout=0., model_type='sage')
        name = 'diffprep_graphsage'
    elif task_type == 'rec':
        model = RecSurrogate(
            n_users=len(adapter['id_maps'][adapter['user']]) + 1,
            n_items=len(adapter['id_maps'][adapter['item']]) + 1,
            feat_dim=in_dim, emb_dim=int(cfg.get('rec_emb_dim', 16)),
            hidden=int(cfg.get('surrogate_hidden_dim', 32)))
        name = 'diffprep_rec_embedding_mlp'
    else:
        model = TabularSurrogate(in_dim, n_outputs, regression=regression)
        name = 'diffprep_linear'
    return model.to(device=device, dtype=dtype), name


def rec_ids(frame, adapter):
    return tuple(torch.tensor(frame[col].astype('string').map(adapter['id_maps'][col])
                             .fillna(0).to_numpy(dtype=np.int64))
                 for col in (adapter['user'], adapter['item']))


def graph_context(data, splits, columns, encoder):
    """Keep every legal context node, select supervised positions by stable ID."""
    graph = data.build_graph()
    from dppbench.graph_input import validate_graph_training
    validate_graph_training(graph, ['auc'])
    frame = data.train_df
    if '__split__' in frame and frame['__split__'].eq('std_test').any():
        raise ValueError('DAGPipe graph proxy cannot contain standard-test nodes')
    ids = pd.Index(frame[data.id_col])
    if not ids.is_unique:
        raise ValueError('Graph proxy requires unique node IDs')
    selected = {}
    for name in ('train', 'val'):
        positions = ids.get_indexer(splits[name][data.id_col])
        if (positions < 0).any() or not np.asarray(graph[name + '_mask'])[positions].all():
            raise ValueError('Graph proxy supervision crossed canonical masks')
        selected[name] = torch.tensor(positions, dtype=torch.long)
    src, dst = _build_undirected(graph['edge_index'], graph['num_nodes'])
    return {'features': torch.from_numpy(encoder.transform(frame[columns])),
            'src': src, 'dst': dst, 'num_nodes': graph['num_nodes'], **selected}
