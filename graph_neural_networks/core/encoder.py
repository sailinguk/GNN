"""Composable graph-to-vector encoder."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn

from .graph import MaskedBatchNorm1d, validate_dense_graph


class GraphEncoder(nn.Module):
    """Compose replaceable message-passing layers and graph pooling.

    Each message-passing layer must expose ``in_channels`` and ``out_channels``
    and accept ``(adjacency, edge_features, node_features, node_mask)``.
    """

    def __init__(
        self,
        layers: Sequence[nn.Module],
        pooling: nn.Module,
        condition_dim: int = 0,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("GraphEncoder requires at least one message-passing layer")
        if condition_dim < 0:
            raise ValueError("condition_dim cannot be negative")
        self.layers = nn.ModuleList(layers)
        self.condition_dim = condition_dim
        self.pooling = pooling
        self.residuals = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.activations = nn.ModuleList()
        previous_out = None
        for layer in self.layers:
            if not hasattr(layer, "in_channels") or not hasattr(layer, "out_channels"):
                raise TypeError("message-passing layers need in_channels/out_channels")
            if previous_out is not None and layer.in_channels != previous_out:
                raise ValueError("adjacent message-passing layer dimensions do not match")
            self.residuals.append(
                nn.Identity()
                if layer.in_channels == layer.out_channels
                else nn.Linear(layer.in_channels, layer.out_channels, bias=False)
            )
            self.normalizations.append(MaskedBatchNorm1d(layer.out_channels))
            self.activations.append(activation())
            previous_out = layer.out_channels
        expected_pool_input = previous_out + condition_dim
        if getattr(pooling, "in_channels", None) != expected_pool_input:
            raise ValueError(
                f"pooling input must be {expected_pool_input}, including condition_dim"
            )
        self.node_embedding_dim = previous_out
        self.graph_embedding_dim = getattr(pooling, "out_channels", None)

    def forward(
        self,
        adjacency: torch.Tensor,
        edge_features: torch.Tensor,
        node_features: torch.Tensor,
        node_mask: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = validate_dense_graph(adjacency, edge_features, node_features, node_mask)
        hidden = node_features * mask.unsqueeze(-1).to(node_features.dtype)
        for layer, residual, normalization, activation in zip(
            self.layers, self.residuals, self.normalizations, self.activations
        ):
            updated = layer(adjacency, edge_features, hidden, mask)
            hidden = activation(normalization(updated + residual(hidden), mask))
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)

        batch_size, num_nodes, _ = hidden.shape
        if self.condition_dim:
            if condition is None or condition.shape != (batch_size, self.condition_dim):
                raise ValueError(
                    f"condition must have shape [{batch_size}, {self.condition_dim}]"
                )
            expanded = condition.to(device=hidden.device, dtype=hidden.dtype)
            expanded = expanded.unsqueeze(1).expand(-1, num_nodes, -1)
            hidden = torch.cat((hidden, expanded), dim=-1)
        elif condition is not None:
            raise ValueError("This graph encoder was created without conditioning")
        return self.pooling(hidden, mask)

