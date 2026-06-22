import torch.nn as nn

from .base_model import RecModel
from .inputs import combined_dnn_input
from .layers import DNN


class FNN(RecModel):
    """Feed-forward recommendation baseline over sparse/dense/sequence features."""

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
        self.to(device)

    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(
            X, self.feature_columns, self.embedding_dict
        )
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        dnn_output = self.dnn(dnn_input)
        y_pred = self.out(self.dnn_linear(dnn_output))
        return y_pred
