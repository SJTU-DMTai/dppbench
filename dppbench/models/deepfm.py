import torch
import torch.nn as nn

from .base_model import RecModel
from .inputs import combined_dnn_input
from .layers import DNN


class DeepFM(RecModel):
    """DeepFM baseline with a DNN tower plus first/second-order FM terms."""

    def __init__(self, dnn_feature_columns,
                 dnn_use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu', l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu',
                 **kwargs):
        super().__init__(dnn_feature_columns, l2_reg_embedding=l2_reg_embedding,
                         init_std=init_std, seed=seed, task=task, device=device)
        self.dnn = DNN(
            inputs_dim=self.compute_input_dim(dnn_feature_columns),
            hidden_units=dnn_hidden_units,
            activation=dnn_activation,
            dropout_rate=dnn_dropout,
            use_bn=dnn_use_bn,
        )
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.linear = nn.Linear(self.compute_input_dim(dnn_feature_columns), 1).to(device)
        self.to(device)

    @staticmethod
    def _fm_second_order(sparse_embedding_list):
        if not sparse_embedding_list:
            return 0.0
        emb = torch.cat(sparse_embedding_list, dim=1)
        square_of_sum = torch.sum(emb, dim=1) ** 2
        sum_of_square = torch.sum(emb ** 2, dim=1)
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        deep_logit = self.dnn_linear(self.dnn(dnn_input))
        linear_logit = self.linear(dnn_input)
        fm_logit = self._fm_second_order(sparse_embedding_list)
        y_pred = self.out(deep_logit + linear_logit + fm_logit)
        return y_pred
