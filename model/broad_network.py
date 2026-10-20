"""Causal nine-channel CfC orientation predictor with explicit carried state."""

import torch
from torch import nn
from ncps.torch import CfCCell


class BroadPredictor(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError('hidden_size must be a positive integer')
        self.hidden_size = hidden_size
        self.cell = CfCCell(9, hidden_size, mode='default', backbone_layers=0)
        self.readout = nn.Linear(hidden_size, 4)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = inputs.new_zeros((inputs.shape[0], self.hidden_size))
        predictions = []
        for step in range(inputs.shape[1]):
            output, hidden = self.cell(inputs[:, step], hidden, 1.0)
            raw = self.readout(output)
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
            identity = raw.new_tensor([1., 0., 0., 0.])
            predictions.append(torch.where(norm > 1e-8, raw / norm.clamp_min(1e-8), identity))
        return torch.stack(predictions, dim=1), hidden
