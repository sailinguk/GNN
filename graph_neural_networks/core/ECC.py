"""Padding-free edge-conditioned convolution with native and PyG backends."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def adjacency_to_edges(A: torch.Tensor, E: torch.Tensor):
    """Convert one real-size ``A[n,n]`` graph to sparse edge tensors."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must have shape [nodes, nodes]")
    if E.ndim != 3 or E.shape[:2] != A.shape:
        raise ValueError("E must have shape [nodes, nodes, edge_feature_dim]")
    source, target = A.nonzero(as_tuple=True)
    edge_index = torch.stack((source, target), dim=0)
    return edge_index, E[source, target]


class NativeECCLayer(nn.Module):
    """Pure-PyTorch ECC over only the real nodes and edges."""

    def __init__(self, in_channels: int, out_channels: int, edge_feature_dim: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim
        self.edge_network = nn.Linear(edge_feature_dim, in_channels * out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, F, E, edge_index):
        source, target = edge_index
        weights = self.edge_network(E).view(-1, self.in_channels, self.out_channels)
        messages = torch.einsum("mi,mio->mo", F[source], weights)
        output = F.new_zeros((F.shape[0], self.out_channels))
        output.index_add_(0, target, messages)
        degree = F.new_zeros(F.shape[0])
        degree.index_add_(0, target, torch.ones_like(target, dtype=F.dtype))
        return output / degree.clamp_min(1).unsqueeze(-1) + self.bias


class PyGECCLayer(nn.Module):
    """Equivalent ECC implemented with PyG ``NNConv``."""

    def __init__(self, in_channels: int, out_channels: int, edge_feature_dim: int):
        super().__init__()
        try:
            from torch_geometric.nn import NNConv
        except ImportError as error:
            raise ImportError("backend='pyg' requires torch-geometric") from error
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_feature_dim = edge_feature_dim
        self.edge_network = nn.Linear(edge_feature_dim, in_channels * out_channels)
        self.conv = NNConv(
            in_channels,
            out_channels,
            self.edge_network,
            aggr="mean",
            root_weight=False,
            bias=True,
        )

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, F, E, edge_index):
        return self.conv(F, edge_index, E)


class ECC(nn.Module):
    """Padding-free ECC stack with residual, BatchNorm and ReLU blocks."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        channels: Sequence[int] = (32, 64),
        backend: str = "native",
    ):
        super().__init__()
        if backend not in {"native", "pyg"}:
            raise ValueError("backend must be 'native' or 'pyg'")
        if not channels or any(width <= 0 for width in channels):
            raise ValueError("channels must contain positive dimensions")
        layer_type = NativeECCLayer if backend == "native" else PyGECCLayer
        dimensions = (node_feature_dim, *channels)
        self.layers = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        for in_channels, out_channels in zip(dimensions[:-1], dimensions[1:]):
            self.layers.append(layer_type(in_channels, out_channels, edge_feature_dim))
            self.residuals.append(
                nn.Identity()
                if in_channels == out_channels
                else nn.Linear(in_channels, out_channels, bias=False)
            )
            self.normalizations.append(nn.BatchNorm1d(out_channels))
        self.backend = backend
        self.channels = tuple(channels)
        self.out_channels = channels[-1]

    def forward(self, F, E, *, edge_index=None, A=None):
        if edge_index is None:
            if A is None:
                raise ValueError("provide edge_index, or one real-size A and E graph")
            edge_index, E = adjacency_to_edges(A, E)
        hidden = F
        for layer, residual, normalization in zip(
            self.layers, self.residuals, self.normalizations
        ):
            hidden = torch.relu(
                normalization(layer(hidden, E, edge_index) + residual(hidden))
            )
        return hidden


ECCLayer = NativeECCLayer
