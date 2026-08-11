"""Padding-free gated graph pooling."""

from __future__ import annotations

import torch
from torch import nn


class GatedGlobalPooling(nn.Module):
    def __init__(self, in_channels: int, graph_embedding_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.graph_embedding_dim = graph_embedding_dim
        self.gate = nn.Linear(in_channels, graph_embedding_dim)
        self.transform = nn.Linear(in_channels, graph_embedding_dim)

    def forward(self, node_embeddings, *, batch):
        if node_embeddings.ndim != 2:
            raise ValueError("node_embeddings must have shape [total_nodes, features]")
        if batch.shape != node_embeddings.shape[:1]:
            raise ValueError("batch must identify the graph of every real node")
        values = torch.sigmoid(self.gate(node_embeddings)) * torch.tanh(
            self.transform(node_embeddings)
        )
        try:
            from torch_geometric.nn import global_add_pool
        except ImportError as error:
            raise ImportError("pooling requires torch-geometric") from error
        pooled = global_add_pool(values, batch)
        return torch.tanh(pooled)
