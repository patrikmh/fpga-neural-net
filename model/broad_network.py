"""Causal nine-channel CfC orientation predictor with explicit carried state."""

import torch
from torch import nn
from ncps.torch import CfCCell


class BroadPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = CfCCell(9, 8, mode='default', backbone_layers=0)
        self.readout = nn.Linear(8, 4)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            hidden = inputs.new_zeros((inputs.shape[0], 8))
        predictions = []
        for step in range(inputs.shape[1]):
            output, hidden = self.cell(inputs[:, step], hidden, 1.0)
            raw = self.readout(output)
            norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
            identity = raw.new_tensor([1., 0., 0., 0.])
            predictions.append(torch.where(norm > 1e-8, raw / norm.clamp_min(1e-8), identity))
        return torch.stack(predictions, dim=1), hidden
