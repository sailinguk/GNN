"""Shared graph validation, encoder composition, and pooling."""

from .encoder import GraphEncoder
from .graph import MaskedBatchNorm1d, validate_dense_graph
from .pooling import GatedGlobalPooling

__all__ = [
    "GatedGlobalPooling",
    "GraphEncoder",
    "MaskedBatchNorm1d",
    "validate_dense_graph",
]

