import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu',
                 dropout_rate=0.0, use_bn=False, init_std=0.0001,
                 dice_dim=2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn

        hidden_units = [inputs_dim] + list(hidden_units)
        self.linears = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(hidden_units) - 1)
        ])
        if use_bn:
            self.bn = nn.ModuleList([
                nn.BatchNorm1d(hidden_units[i + 1])
                for i in range(len(hidden_units) - 1)
            ])
        self.activation_layers = nn.ModuleList([
            self._get_activation(activation, hidden_units[i + 1], dice_dim)
            for i in range(len(hidden_units) - 1)
        ])
        self.dropout = nn.Dropout(dropout_rate)

        for linear in self.linears:
            nn.init.normal_(linear.weight, mean=0, std=init_std)

    @staticmethod
    def _get_activation(name, hidden_size=None, dice_dim=2):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'sigmoid':
            return nn.Sigmoid()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'prelu':
            return nn.PReLU()
        elif name.lower() == 'dice':
            return Dice(hidden_size, dim=dice_dim)
        return nn.ReLU(inplace=True)

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
            if self.use_bn:
                x = self.bn[i](x)
            x = self.activation_layers[i](x)
            x = self.dropout(x)
        return x


class Dice(nn.Module):
    def __init__(self, emb_size, dim=2, epsilon=1e-8):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("Dice only supports 2D or 3D inputs")
        if emb_size is None:
            raise ValueError("Dice requires emb_size")
        self.epsilon = epsilon
        self.dim = dim
        self.bn = nn.BatchNorm1d(emb_size, eps=epsilon)
        if dim == 2:
            self.alpha = nn.Parameter(torch.zeros(emb_size))
        else:
            self.alpha = nn.Parameter(torch.zeros(emb_size, 1))

    def forward(self, x):
        if x.dim() != self.dim:
            raise ValueError(f"Dice expected {self.dim}D input, got {x.dim()}D")
        if self.dim == 2:
            p = torch.sigmoid(self.bn(x))
            return p * x + (1 - p) * self.alpha * x

        x_t = x.transpose(1, 2)
        p = torch.sigmoid(self.bn(x_t))
        out = p * x_t + (1 - p) * self.alpha * x_t
        return out.transpose(1, 2)


class PredictionLayer(nn.Module):
    def __init__(self, task='binary'):
        super().__init__()
        self.task = task

    def forward(self, x):
        if self.task == 'binary':
            return torch.sigmoid(x)
        return x


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, att_hidden_units=(64, 16), embedding_dim=4,
                 att_activation='sigmoid', weight_normalization=False):
        super().__init__()
        self.local_att = LocalActivationUnit(
            hidden_units=att_hidden_units,
            embedding_dim=embedding_dim,
            activation=att_activation,
        )
        self.weight_normalization = weight_normalization

    def forward(self, query, keys, keys_length):
        keys_length = keys_length.long()
        att_score = self.local_att(query, keys)
        att_score = att_score.transpose(1, 2)

        mask = torch.arange(keys.size(1), device=keys.device).unsqueeze(0) < keys_length.unsqueeze(1)
        mask = mask.unsqueeze(1)

        if self.weight_normalization:
            paddings = torch.ones_like(att_score) * (-2 ** 32 + 1)
        else:
            paddings = torch.zeros_like(att_score)

        att_score = torch.where(mask, att_score, paddings)
        if self.weight_normalization:
            att_score = F.softmax(att_score, dim=-1)

        output = torch.matmul(att_score, keys)
        return output


class LocalActivationUnit(nn.Module):
    """DIN local activation unit over target item and user behavior sequence."""

    def __init__(self, hidden_units=(64, 32), embedding_dim=4,
                 activation='sigmoid', dropout_rate=0.0, use_bn=False):
        super().__init__()
        self.dnn = DNN(
            inputs_dim=4 * embedding_dim,
            hidden_units=hidden_units,
            activation=activation,
            dropout_rate=dropout_rate,
            use_bn=use_bn,
            dice_dim=3,
        )
        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavior):
        user_behavior_len = user_behavior.size(1)
        queries = query.expand(-1, user_behavior_len, -1)
        attention_input = torch.cat(
            [queries, user_behavior, queries - user_behavior, queries * user_behavior],
            dim=-1,
        )
        return self.dense(self.dnn(attention_input))
