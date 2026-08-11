"""Dense edge-conditioned convolution."""

from __future__ import annotations

import torch
from torch import nn

from ..core.graph import validate_dense_graph


class ECCConv(nn.Module):
    """Degree-normalized ECC with a linear filter-generating network.

    ``adjacency[:, i, j]`` and ``edge_features[:, i, j]`` describe edge
    ``j -> i``. Identity connections are deliberately handled by GraphEncoder.
    """

    def __init__(self, in_channels: int, out_channels: int, edge_feature_dim: int):
        super().__init__()
        if min(in_channels, out_channels, edge_feature_dim) <= 0:
            raise ValueError("ECC dimensions must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim
        self.filter_network = nn.Linear(
            edge_feature_dim, in_channels * out_channels
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(
        self,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = validate_dense_graph(adjacency, edge_features, node_features, node_mask)
        if node_features.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} node features")
        if edge_features.shape[-1] != self.edge_feature_dim:
            raise ValueError(f"expected {self.edge_feature_dim} edge features")

        batch_size, num_nodes, _ = node_features.shape
        weights = self.filter_network(edge_features).view(
            batch_size,
            num_nodes,
            num_nodes,
            self.in_channels,
            self.out_channels,
        )
        sources = node_features[:, None, :, :].expand(-1, num_nodes, -1, -1)
        messages = torch.einsum("bijk,bijko->bijo", sources, weights)
        valid_pairs = mask[:, :, None] & mask[:, None, :]
        edge_mask = adjacency.to(messages.dtype) * valid_pairs.to(messages.dtype)
        degree = edge_mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        output = (messages * edge_mask.unsqueeze(-1)).sum(dim=2) / degree
        output = output + self.bias
        return output * mask.unsqueeze(-1).to(output.dtype)

