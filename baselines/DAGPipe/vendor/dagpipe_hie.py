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


    def forward(self, x, reference_rows=None):
        from ..categorical import FrozenCategoryBranch

        layer_adj = torch.triu(torch.sigmoid(self.layer_adj_logits), diagonal=1)

        layer_outputs = []

        current_input = x.nan_to_num() if getattr(self, 'operator_space', None) in ('legacy_daghie', 'typed_public_v4') else x
        if getattr(self, 'operator_space', None) == 'typed_public_v4' and reference_rows is None:
            raise ValueError('Typed DAG requires its frozen training reference')
        trace = getattr(self, 'forward_observer', None)
        if trace is not None:
            trace('network_input', current_input)

        for layer_idx in range(self.num_layers):
            routing = self.routers[layer_idx]()

            intra_adj = torch.triu(torch.sigmoid(self.intra_adj_logits[layer_idx]), diagonal=1)

            layer_ops = self.op_layers[layer_idx]
            layer_size = self.layer_sizes[layer_idx]

            op_outputs = []

            for op_idx in range(layer_size):
                op = layer_ops[op_idx]

                routing_weight = routing[:, op_idx].unsqueeze(0)
                weighted_input = current_input * routing_weight
                observe = None
                if trace is not None:
                    def observe(stage, value, **details):
                        trace(stage, value, layer=layer_idx, node=op_idx, routing=routing_weight, **details)
                    observe('before_routing', current_input)
                    observe('after_routing', weighted_input)

                if reference_rows is None:
                    out = (op(current_input) * routing_weight if isinstance(op, FrozenCategoryBranch)
                           else op(weighted_input))
                else:
                    from ..runtime import reference_operation
                    try:
                        out = reference_operation(op, weighted_input, reference_rows,
                                                  getattr(self, "scale_protocol", None),
                                                  unrouted_input=current_input, routing_weight=routing_weight,
                                                  source_input=x, observe=observe)
                    except (ValueError, RuntimeError) as exc:
                        if getattr(self, 'operator_space', None) == 'typed_public_v4':
                            raise RuntimeError(f'v4 layer={layer_idx} node={op_idx} operator={op.name}: {exc}') from exc
                        raise

                if observe is not None:
                    observe('operator_output', out)

                observer = getattr(self, "node_observer", None)
                if observer is not None:
                    node_input = current_input if isinstance(op, FrozenCategoryBranch) else weighted_input
                    observer(layer_idx, op_idx, op, node_input.detach(), out.detach())

                for prev_op_idx in range(op_idx):
                    weight = intra_adj[prev_op_idx, op_idx]
                    contribution = weight * op_outputs[prev_op_idx]
                    if observe is not None:
                        observe('node_edge_contribution', contribution, source_node=prev_op_idx, weight=weight)
                    out = out + contribution

                if observe is not None:
                    observe('node_aggregate', out)
                op_outputs.append(out)

            op_outputs_tensor = torch.stack(op_outputs, dim=0)  # [layer_size, B, D]
            layer_output = torch.mean(op_outputs_tensor, dim=0)  # [B, D]
            if trace is not None:
                trace('layer_mean', layer_output, layer=layer_idx)

            if layer_idx > 0:
                prev_outputs_tensor = torch.stack(layer_outputs[:layer_idx], dim=0)  # [prev_layers, B, D]
                prev_weights = layer_adj[:layer_idx, layer_idx].view(-1, 1, 1)  # [prev_layers, 1, 1]
                contributions = prev_outputs_tensor * prev_weights
                if trace is not None:
                    for source, contribution in enumerate(contributions):
                        trace('layer_edge_contribution', contribution, layer=layer_idx,
                              source_layer=source, weight=prev_weights[source].squeeze())
                weighted_prev = torch.sum(contributions, dim=0)  # [B, D]
                layer_output = layer_output + weighted_prev

            layer_outputs.append(layer_output)
            if getattr(self, 'operator_space', None) == 'typed_public_v4' and not torch.isfinite(layer_output).all():
                raise ValueError(f'Nonfinite v4 DAG aggregate at layer {layer_idx}; no clipping applied')
            if trace is not None:
                trace('layer_aggregate', layer_output, layer=layer_idx)
            current_input = layer_output

        return layer_outputs[-1]

    def dag_penalty(self):
        # All forward edges are strictly upper triangular, so tr(exp(A*A))-K
        # and its parameter gradient are identically zero. Keep API compatibility.
        return 0.0 * (self.layer_adj_logits.sum() + sum(a.sum() for a in self.intra_adj_logits))
