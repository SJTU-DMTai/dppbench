import torch
import torch.nn as nn

from .base_model import RecModel
from .inputs import (
    embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list,
    combined_dnn_input, maxlen_lookup,
)
from .layers import DNN, AttentionSequencePoolingLayer


class DIEN(RecModel):
    """Lightweight DIEN-style model with GRU interest evolution."""

    def __init__(self, dnn_feature_columns,
                 dnn_use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False,
                 gru_hidden_size=None, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu',
                 **kwargs):
        super().__init__(dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                         init_std=init_std, seed=seed, task=task, device=device)
        self.attention_feature_columns = []
        self.pooling_varlen_feature_columns = []
        for fc in self.varlen_sparse_feature_columns:
            if fc.is_attention:
                self.attention_feature_columns.append(fc)
            else:
                self.pooling_varlen_feature_columns.append(fc)

        self.interest_dim = self._compute_interest_dim()
        hidden = int(gru_hidden_size or self.interest_dim)
        self.gru = nn.GRU(self.interest_dim, hidden, batch_first=True).to(device)
        self.interest_proj = (
            nn.Linear(hidden, self.interest_dim).to(device)
            if hidden != self.interest_dim else nn.Identity()
        )
        self.attention = AttentionSequencePoolingLayer(
            att_hidden_units=att_hidden_size,
            embedding_dim=self.interest_dim,
            att_activation=att_activation,
            weight_normalization=att_weight_normalization,
        )
        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)

    def forward(self, X):
        _, dense_value_list = self.input_from_feature_columns(
            X, self.feature_columns, self.embedding_dict
        )
        dnn_input_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.sparse_feature_columns, to_list=True,
        )
        sequence_embed_dict = varlen_embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.pooling_varlen_feature_columns,
        )
        dnn_input_emb_list += get_varlen_pooling_list(
            sequence_embed_dict, X, self.feature_index,
            self.pooling_varlen_feature_columns, self.device,
        )

        query_emb = self._query_embedding(X)
        keys_emb = self._keys_embedding(X)
        keys_len = self._keys_length(X)
        gru_out, _ = self.gru(keys_emb)
        evolved_keys = self.interest_proj(gru_out)
        hist = self.attention(query_emb, evolved_keys, keys_len)

        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1).view(X.size(0), -1)
        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        y_pred = self.out(self.dnn_linear(self.dnn(dnn_input)))
        return y_pred

    def _query_embedding(self, X):
        query_emb_list = []
        for fc in self.attention_feature_columns:
            query_emb_list.append(
                embedding_lookup(
                    X, self.embedding_dict, self.feature_index,
                    self.sparse_feature_columns,
                    return_feat_list=[fc.embedding_name], to_list=True,
                )
            )
        return torch.cat([e for sublist in query_emb_list for e in sublist], dim=-1)

    def _keys_embedding(self, X):
        keys_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.attention_feature_columns,
            return_feat_list=[fc.name for fc in self.attention_feature_columns],
            to_list=True,
        )
        return torch.cat(keys_emb_list, dim=-1)

    def _keys_length(self, X):
        keys_length_feature_name = [
            fc.length_name for fc in self.attention_feature_columns
            if fc.length_name is not None
        ]
        return torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)

    def _compute_interest_dim(self):
        return sum(feat.embedding_dim for feat in self.attention_feature_columns)
