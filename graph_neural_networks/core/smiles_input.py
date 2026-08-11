"""Download and prepare a heavy-atom QM9 graph dataset once."""

from __future__ import annotations

import torch

try:
    from torch_geometric.data import InMemoryDataset
except ImportError as error:
    raise ImportError("QM9 input requires torch-geometric") from error


def qm9_heavy_atom_transform(data):
    """Remove hydrogen and retain C/N/O/F node and four bond features.

    PyG QM9 stores the atom one-hot encoding in the first five columns of ``x``
    in H/C/N/O/F order. The returned graph has four-dimensional C/N/O/F nodes.
    """
    try:
        from torch_geometric.utils import subgraph
    except ImportError as error:
        raise ImportError("QM9 input requires torch-geometric") from error
    data = data.clone()
    atom_type = data.x[:, :5].argmax(dim=-1)
    keep = atom_type != 0
    edge_index, edge_attr = subgraph(
        keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
    )
    data.x = data.x[keep, 1:5].to(torch.float32)
    if getattr(data, "pos", None) is not None:
        data.pos = data.pos[keep]
    if getattr(data, "z", None) is not None:
        data.z = data.z[keep]
    data.edge_index = edge_index
    data.edge_attr = edge_attr[:, :4].to(torch.float32)
    return data


class PreparedQM9(InMemoryDataset):
    """Cached QM9 with hydrogen removed and C/N/O/F graph features.

    On first construction, PyG downloads/processes QM9 and this class performs
    the heavy-atom conversion once. Later training epochs load the cached
    ``qm9_heavy_atoms.pt`` without repeating molecular preprocessing.
    """

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["qm9_heavy_atoms.pt"]

    def __init__(self, root):
        super().__init__(root)
        self.load(self.processed_paths[0])

    def process(self):
        from torch_geometric.datasets import QM9
        from tqdm import tqdm

        source = QM9(root=self.root)
        prepared = [
            qm9_heavy_atom_transform(source[index])
            for index in tqdm(range(len(source)), desc="Preparing heavy-atom QM9")
        ]
        self.save(prepared, self.processed_paths[0])


def prepare_qm9(root):
    """Prepare the cached dataset and return it."""
    return PreparedQM9(root)

