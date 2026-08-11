"""Utilities shared by dense graph neural networks."""

from __future__ import annotations

import torch
from torch import nn


def validate_dense_graph(
    adjacency: torch.Tensor,
    edge_features: torch.Tensor,
    node_features: torch.Tensor,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate a dense graph batch and return a boolean node mask."""
    if node_features.ndim != 3:
        raise ValueError("node_features must have shape [batch, nodes, features]")
    batch_size, num_nodes, _ = node_features.shape
    if adjacency.shape != (batch_size, num_nodes, num_nodes):
        raise ValueError("adjacency must have shape [batch, nodes, nodes]")
    if edge_features.ndim != 4 or edge_features.shape[:3] != (
        batch_size,
        num_nodes,
        num_nodes,
    ):
        raise ValueError(
            "edge_features must have shape [batch, nodes, nodes, features]"
        )
    if node_mask is None:
        node_mask = torch.ones(
            batch_size, num_nodes, dtype=torch.bool, device=node_features.device
        )
    elif node_mask.shape != (batch_size, num_nodes):
        raise ValueError("node_mask must have shape [batch, nodes]")
    return node_mask.to(device=node_features.device, dtype=torch.bool)


class MaskedBatchNorm1d(nn.Module):
    """Batch-normalize only valid nodes and keep padded nodes equal to zero."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, values: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[-1] != self.channels:
            raise ValueError("values must have shape [batch, nodes, channels]")
        mask = node_mask.to(device=values.device, dtype=torch.bool)
        valid = values[mask]
        output = torch.zeros_like(values)
        if valid.numel() == 0:
            return output
        if self.training and valid.shape[0] == 1:
            normalized = nn.functional.batch_norm(
                valid,
                self.norm.running_mean,
                self.norm.running_var,
                self.norm.weight,
                self.norm.bias,
                training=False,
                eps=self.norm.eps,
            )
        else:
            normalized = self.norm(valid)
        output[mask] = normalized
        return output

