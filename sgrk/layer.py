"""SGRK layer: recurrent kernel bank with a learned sparsity gate."""

import torch
import torch.nn as nn


class KernelSparsityGate(nn.Module):
    def __init__(self, num_kernels, channels):
        super().__init__()
        self.score = nn.Linear(channels, num_kernels)

    def forward(self, h, temperature=1.0):
        logits = self.score(h) / temperature
        mask = torch.sigmoid(logits)
        # straight-through estimator: hard mask in forward pass
        hard = (mask > 0.5).float()
        return hard + mask - mask.detach()


class SGRKLayer(nn.Module):
    def __init__(self, kernel_bank=8, channels=64, sparsity=4):
        super().__init__()
        self.channels = channels
        self.kernel_bank = kernel_bank
        self.sparsity = sparsity
        self.kernels = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=2 * i + 1, padding=i, groups=channels)
            for i in range(kernel_bank)
        ])
        self.gate = KernelSparsityGate(kernel_bank, channels)

    def forward(self, h_prev, temperature=1.0):
        mask = self.gate(h_prev, temperature)
        mixed = torch.stack([k(h_prev) for k in self.kernels], dim=1)
        return (mask.unsqueeze(2) * mixed).sum(dim=1)
