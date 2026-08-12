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
        if batch.ndim != 1 or batch.dtype != torch.long:
            raise ValueError("batch must be a one-dimensional torch.long tensor")
        if batch.numel() == 0:
            return values.new_empty((0, self.graph_embedding_dim))
        num_graphs = int(batch.max()) + 1
        pooled = values.new_zeros((num_graphs, self.graph_embedding_dim))
        pooled.index_add_(0, batch, values)
        return torch.tanh(pooled)
