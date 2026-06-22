import torch
import torch.nn as nn

from .base_model import RecModel
from .inputs import (
    SparseFeat, VarLenSparseFeat,
    embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list,
    combined_dnn_input, maxlen_lookup,
)
from .layers import DNN, AttentionSequencePoolingLayer


class DIN(RecModel):
    def __init__(self, dnn_feature_columns,
                 dnn_use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', **kwargs):
        super().__init__(dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                         init_std=init_std, seed=seed, task=task, device=device)

        self.attention_feature_columns = []
        self.pooling_varlen_feature_columns = []

        for fc in self.varlen_sparse_feature_columns:
            if fc.is_attention:
                self.attention_feature_columns.append(fc)
            else:
                self.pooling_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(
            att_hidden_units=att_hidden_size,
            embedding_dim=att_emb_dim,
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

        query_emb_list = []
        for fc in self.attention_feature_columns:
            ref_feat_name = fc.embedding_name
            query_emb_list.append(
                embedding_lookup(
                    X, self.embedding_dict, self.feature_index,
                    self.sparse_feature_columns,
                    return_feat_list=[ref_feat_name], to_list=True,
                )
            )

        keys_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.attention_feature_columns,
            return_feat_list=[fc.name for fc in self.attention_feature_columns], to_list=True,
        )

        dnn_input_emb_list = embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.sparse_feature_columns, to_list=True,
        )

        sequence_embed_dict = varlen_embedding_lookup(
            X, self.embedding_dict, self.feature_index,
            self.pooling_varlen_feature_columns,
        )
        sequence_embed_list = get_varlen_pooling_list(
            sequence_embed_dict, X, self.feature_index,
            self.pooling_varlen_feature_columns, self.device,
        )
        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        query_emb = torch.cat([e for sublist in query_emb_list for e in sublist], dim=-1)
        keys_emb = torch.cat(keys_emb_list, dim=-1)

        keys_length_feature_name = [
            fc.length_name for fc in self.attention_feature_columns
            if fc.length_name is not None
        ]
        keys_length = torch.squeeze(
            maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1
        )

        hist = self.attention(query_emb, keys_emb, keys_length)

        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)
        return y_pred

    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.attention_feature_columns:
            interest_dim += feat.embedding_dim
        return interest_dim
