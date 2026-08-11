"""Optional PyTorch Geometric-to-dense adapter.

The core package does not import or depend on PyTorch Geometric. This helper
uses the public attributes of a PyG Data object when callers choose that input.
"""

from __future__ import annotations

import torch


def pyg_graph_to_dense(data, max_nodes: int | None = None):
    """Convert one PyG-like Data object to the core dense tensor convention."""
    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise TypeError("data must expose x and edge_index")
    num_nodes = int(data.x.shape[0])
    size = num_nodes if max_nodes is None else max_nodes
    if size < num_nodes:
        raise ValueError("max_nodes is smaller than the graph")
    edge_attr = getattr(data, "edge_attr", None)
    edge_dim = 1 if edge_attr is None else int(edge_attr.shape[-1])
    adjacency = data.x.new_zeros((size, size))
    edge_features = data.x.new_zeros((size, size, edge_dim))
    node_features = data.x.new_zeros((size, data.x.shape[-1]))
    node_mask = torch.zeros(size, dtype=torch.bool, device=data.x.device)
    node_features[:num_nodes] = data.x
    node_mask[:num_nodes] = True
    for edge_id, (src, dst) in enumerate(data.edge_index.t().tolist()):
        adjacency[dst, src] = 1
        if edge_attr is None:
            edge_features[dst, src, 0] = 1
        else:
            edge_features[dst, src] = edge_attr[edge_id]
    return adjacency, edge_features, node_features, node_mask

