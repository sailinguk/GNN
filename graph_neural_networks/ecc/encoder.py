"""GraphVAE-style ECC graph encoder."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from ..core.encoder import GraphEncoder
from ..core.pooling import GatedGlobalPooling
from .layer import ECCConv


class ECCGraphEncoder(GraphEncoder):
    """ECC layers followed by gated global pooling.

    Defaults reproduce the GraphVAE QM9 graph-to-vector architecture:
    two ECC layers with 32/64 channels and a 128-dimensional pooled output.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        channels: Sequence[int] = (32, 64),
        graph_embedding_dim: int = 128,
        condition_dim: int = 0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        if not channels or any(channel <= 0 for channel in channels):
            raise ValueError("channels must contain positive dimensions")
        dimensions = (node_feature_dim, *channels)
        layers = [
            ECCConv(in_channels, out_channels, edge_feature_dim)
            for in_channels, out_channels in zip(dimensions[:-1], dimensions[1:])
        ]
        pooling = GatedGlobalPooling(channels[-1] + condition_dim, graph_embedding_dim)
        super().__init__(layers, pooling, condition_dim, activation)
        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.channels = tuple(channels)

