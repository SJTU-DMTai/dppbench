import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureRouter(nn.Module):
    def __init__(self, num_features, num_ops):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_features, num_ops))

    def forward(self):
        return F.softmax(self.logits, dim=-1)
