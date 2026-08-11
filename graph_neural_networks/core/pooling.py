"""Permutation-invariant graph pooling modules."""

from __future__ import annotations

import torch
from torch import nn


class GatedGlobalPooling(nn.Module):
    """Graph-level gated output from Li et al. (2015), equation 7."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("pooling dimensions must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gate = nn.Linear(in_channels, out_channels)
        self.transform = nn.Linear(in_channels, out_channels)

    def forward(
        self, node_features: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        if node_features.ndim != 3 or node_features.shape[-1] != self.in_channels:
            raise ValueError(
                f"node_features must have shape [batch, nodes, {self.in_channels}]"
            )
        if node_mask.shape != node_features.shape[:2]:
            raise ValueError("node_mask must match the first two feature dimensions")
        contributions = torch.sigmoid(self.gate(node_features)) * torch.tanh(
            self.transform(node_features)
        )
        contributions = contributions * node_mask.unsqueeze(-1).to(
            dtype=contributions.dtype, device=contributions.device
        )
        return torch.tanh(contributions.sum(dim=1))

