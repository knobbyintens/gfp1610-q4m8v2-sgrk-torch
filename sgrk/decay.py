"""Horizon-decay schedule with a learned half-life per channel."""

import torch
import torch.nn as nn


class HorizonDecay(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.log_half_life = nn.Parameter(torch.zeros(channels))

    def forward(self, h, steps):
        half_life = torch.exp(self.log_half_life)
        factor = torch.pow(0.5, steps / half_life)
        return h * factor
