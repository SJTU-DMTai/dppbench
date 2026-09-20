import torch
import torch.nn as nn
from .acyclicity import acyclicity_constraint
from .routing import FeatureRouter


class HierarchicalDAGPipe(nn.Module):
    def __init__(self, num_features, operation_layers: list[list[nn.Module]]):
        super().__init__()
        self.num_features = num_features
        self.num_layers = len(operation_layers)
        self.layer_sizes = [len(layer) for layer in operation_layers]

        self.op_layers = nn.ModuleList([
            nn.ModuleList(layer)
            for layer in operation_layers
        ])

        self.layer_adj_logits = nn.Parameter(torch.zeros(self.num_layers, self.num_layers))

        self.intra_adj_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(size, size))
            for size in self.layer_sizes
        ])

        self.routers = nn.ModuleList([
            FeatureRouter(num_features, size)
            for size in self.layer_sizes
        ])


    def forward(self, x):
        layer_adj = torch.sigmoid(self.layer_adj_logits)

        layer_outputs = []

        current_input = x

        for layer_idx in range(self.num_layers):
            routing = self.routers[layer_idx]()

            intra_adj = torch.sigmoid(self.intra_adj_logits[layer_idx])

            layer_ops = self.op_layers[layer_idx]
            layer_size = self.layer_sizes[layer_idx]

            op_outputs = []

            for op_idx in range(layer_size):
                op = layer_ops[op_idx]

                weighted_input = current_input * routing[:, op_idx].unsqueeze(0)  # [B, D] * [D] -> [B, D]

                out = op(weighted_input)

                for prev_op_idx in range(op_idx):
                    weight = intra_adj[prev_op_idx, op_idx]
                    out = out + weight * op_outputs[prev_op_idx]

                op_outputs.append(out)

            op_outputs_tensor = torch.stack(op_outputs, dim=0)  # [layer_size, B, D]
            layer_output = torch.mean(op_outputs_tensor, dim=0)  # [B, D]

            if layer_idx > 0:
                prev_outputs_tensor = torch.stack(layer_outputs[:layer_idx], dim=0)  # [prev_layers, B, D]
                prev_weights = layer_adj[:layer_idx, layer_idx].view(-1, 1, 1)  # [prev_layers, 1, 1]
                weighted_prev = torch.sum(prev_outputs_tensor * prev_weights, dim=0)  # [B, D]
                layer_output = layer_output + weighted_prev

            layer_outputs.append(layer_output)
            current_input = layer_output

        return layer_outputs[-1]

    def dag_penalty(self):
        layer_adj = torch.sigmoid(self.layer_adj_logits)
        layer_penalty = acyclicity_constraint(layer_adj)

        intra_penalties = []
        for layer_idx in range(self.num_layers):
            intra_adj = torch.sigmoid(self.intra_adj_logits[layer_idx])
            intra_penalties.append(acyclicity_constraint(intra_adj))

        return layer_penalty + sum(intra_penalties)
