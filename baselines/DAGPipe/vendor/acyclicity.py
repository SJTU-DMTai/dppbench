import torch

def acyclicity_constraint(adj_matrix: torch.Tensor):
    """
    adj_matrix: [K, K], non-negative
    """
    K = adj_matrix.size(0)
    expm = torch.matrix_exp(adj_matrix * adj_matrix)  # h(A) ^ 2 ?
    return torch.trace(expm) - K
