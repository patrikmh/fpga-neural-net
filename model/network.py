"""Eight-neuron CfC next-sample predictor with fresh state per trajectory."""

import torch
from torch import nn
from ncps.torch import CfCCell


class SignalPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = CfCCell(1, 8, mode="default", backbone_layers=0)
        self.readout = nn.Linear(8, 1)

    def forward(self, inputs):
        hidden = inputs.new_zeros((inputs.shape[0], 8))
        outputs = []
        for step in range(inputs.shape[1]):
            output, hidden = self.cell(inputs[:, step, :], hidden, 1.0)
            outputs.append(self.readout(output))
        return torch.stack(outputs, dim=1)
