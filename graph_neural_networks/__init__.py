"""Public interface for reusable dense graph encoders."""

from .core import GatedGlobalPooling, GraphEncoder
from .ecc import ECCConv, ECCGraphEncoder

__all__ = ["ECCConv", "ECCGraphEncoder", "GatedGlobalPooling", "GraphEncoder"]

