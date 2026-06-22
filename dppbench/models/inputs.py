from collections import namedtuple, OrderedDict
from itertools import chain

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


DEFAULT_GROUP_NAME = "default_group"


class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'embedding_name', 'group_name'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=8, embedding_name=None, group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        return super().__new__(cls, name, vocabulary_size, embedding_dim, embedding_name, group_name)

    def __hash__(self):
        return self.name.__hash__()


class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1):
        return super().__new__(cls, name, dimension)

    def __hash__(self):
        return self.name.__hash__()


class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'is_attention'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, is_attention=False):
        return super().__new__(cls, sparsefeat, maxlen, combiner, length_name, is_attention)

    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()


def build_input_features(feature_columns):
    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        name = feat.name
        if name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[name] = (start, start + feat.dimension)
            start += feat.dimension
        elif isinstance(feat, VarLenSparseFeat):
            features[name] = (start, start + feat.maxlen)
            start += feat.maxlen
            if feat.length_name is not None and feat.length_name not in features:
                features[feat.length_name] = (start, start + 1)
                start += 1
    return features


def create_embedding_matrix(feature_columns, init_std=0.0001, sparse=False, device='cpu'):
    sparse_feats = [f for f in feature_columns if isinstance(f, SparseFeat)]
    varlen_feats = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]
    embedding_dict = nn.ModuleDict({
        feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim, sparse=sparse)
        for feat in sparse_feats + varlen_feats
    })
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
    return embedding_dict.to(device)


def embedding_lookup(X, embedding_dict, feature_index, sparse_feature_columns,
                     return_feat_list=(), to_list=False):
    group_embedding = {}
    for fc in sparse_feature_columns:
        if return_feat_list and fc.name not in return_feat_list:
            continue
        idx = feature_index[fc.name]
        input_tensor = X[:, idx[0]:idx[1]].long()
        emb = embedding_dict[fc.embedding_name](input_tensor)
        group_embedding.setdefault(fc.group_name, []).append(emb)
    if to_list:
        return list(chain.from_iterable(group_embedding.values()))
    return group_embedding


def varlen_embedding_lookup(X, embedding_dict, feature_index, varlen_sparse_feature_columns):
    result = {}
    for fc in varlen_sparse_feature_columns:
        idx = feature_index[fc.name]
        result[fc.name] = embedding_dict[fc.embedding_name](X[:, idx[0]:idx[1]].long())
    return result


def get_varlen_pooling_list(embedding_dict, X, feature_index, varlen_sparse_feature_columns, device):
    pooling_list = []
    for fc in varlen_sparse_feature_columns:
        seq_emb = embedding_dict[fc.name]
        if fc.length_name is not None:
            idx = feature_index[fc.length_name]
            seq_len = X[:, idx[0]:idx[1]].long().squeeze(1)
            mask = torch.arange(fc.maxlen, device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        else:
            idx = feature_index[fc.name]
            mask = X[:, idx[0]:idx[1]].long() != 0
        mask = mask.unsqueeze(2).float()
        if fc.combiner == "mean":
            length = mask.sum(dim=1, keepdim=True).clamp(min=1)
            emb = (seq_emb * mask).sum(dim=1, keepdim=True) / length
        elif fc.combiner == "sum":
            emb = (seq_emb * mask).sum(dim=1, keepdim=True)
        elif fc.combiner == "max":
            emb = seq_emb + (1 - mask) * (-1e9)
            emb = emb.max(dim=1, keepdim=True).values
        else:
            raise ValueError(f"Unknown combiner: {fc.combiner}")
        pooling_list.append(emb)
    return pooling_list


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if sparse_embedding_list and dense_value_list:
        sparse = torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        dense = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        return torch.cat([sparse, dense], dim=-1)
    elif sparse_embedding_list:
        return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    elif dense_value_list:
        return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
    raise ValueError("No input for DNN")


def maxlen_lookup(X, feature_index, maxlen_column):
    idx = feature_index[maxlen_column[0]]
    return X[:, idx[0]:idx[1]].long()


def pad_sequences(sequences, max_len, padding_value=0):
    result = np.full((len(sequences), max_len), padding_value, dtype=np.int64)
    lengths = np.zeros(len(sequences), dtype=np.int64)
    for i, seq in enumerate(sequences):
        if not isinstance(seq, (list, np.ndarray)):
            seq = []
        truncated = seq[-max_len:]
        length = len(truncated)
        lengths[i] = length
        if length > 0:
            result[i, :length] = truncated
    return result, lengths


def build_feature_columns(df, cfg, col_types=None):
    feat_cfg = cfg["feature"]
    embedding_dim = cfg["model_params"]["embedding_dim"]
    seq_max_len = feat_cfg["seq_max_len"]
    target_col = feat_cfg["target_col"]
    seq_cols = feat_cfg.get("seq_cols", {})

    col_types = col_types or {}

    exclude_cols = {"__split__", target_col}

    feature_columns = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        col_type = col_types.get(col)
        if col_type is None:
            continue

        if col_type == "categorical":
            vocab_size = int(df[col].max()) + 1 if df[col].dtype.kind in ('i', 'u') else int(df[col].nunique()) + 1
            feature_columns.append(
                SparseFeat(col, vocabulary_size=vocab_size, embedding_dim=embedding_dim)
            )
        elif col_type == "numeric":
            feature_columns.append(DenseFeat(col, dimension=1))
        elif col_type == "categorical_list":
            is_attention = col in seq_cols
            all_values = []
            for v in df[col]:
                if isinstance(v, list):
                    all_values.extend(v)
            if is_attention:
                ref_col = seq_cols[col]
                ref_fc = next((fc for fc in feature_columns if isinstance(fc, SparseFeat) and fc.name == ref_col), None)
                if ref_fc:
                    seq_vocab_size = (
                        int(max(all_values)) + 1
                        if all_values and isinstance(all_values[0], (int, np.integer))
                        else len(set(all_values)) + 1
                    )
                    vocab_size = max(ref_fc.vocabulary_size, seq_vocab_size)
                    embedding_name = ref_fc.embedding_name
                else:
                    vocab_size = int(max(all_values)) + 1 if all_values and isinstance(all_values[0], (int, np.integer)) else len(set(all_values)) + 1
                    embedding_name = ref_col
            else:
                if all_values and isinstance(all_values[0], (int, np.integer)):
                    vocab_size = int(max(all_values)) + 1
                else:
                    vocab_size = len(set(all_values)) + 1
                embedding_name = col
            feature_columns.append(
                VarLenSparseFeat(
                    SparseFeat(col, vocabulary_size=vocab_size,
                               embedding_name=embedding_name, embedding_dim=embedding_dim),
                    maxlen=seq_max_len,
                    combiner="mean",
                    length_name=f"{col}_length",
                    is_attention=is_attention,
                )
            )

    return feature_columns


def df_to_input(df, feature_columns, cfg):
    feat_cfg = cfg["feature"]
    seq_max_len = feat_cfg["seq_max_len"]
    target_col = feat_cfg["target_col"]

    x = {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            col_vals = df[fc.name]
            if col_vals.dtype == object or col_vals.dtype.name == 'category':
                x[fc.name] = col_vals.fillna("__UNK__").astype("category").cat.codes.values.astype(np.int64)
            else:
                vals = pd.to_numeric(col_vals, errors="coerce").fillna(0).values.astype(np.int64)
                vals[vals < 0] = 0
                vals[vals >= fc.vocabulary_size] = 0
                x[fc.name] = vals
        elif isinstance(fc, DenseFeat):
            x[fc.name] = pd.to_numeric(df[fc.name], errors="coerce").fillna(0).values.astype(np.float32)
        elif isinstance(fc, VarLenSparseFeat):
            raw_seqs = df[fc.name].tolist()
            padded, lengths = pad_sequences(raw_seqs, fc.maxlen, padding_value=0)
            padded[padded < 0] = 0
            padded[padded >= fc.vocabulary_size] = 0
            x[fc.name] = padded
            if fc.length_name:
                x[fc.length_name] = lengths

    y = df[target_col].values.astype(np.float32)
    return x, y
