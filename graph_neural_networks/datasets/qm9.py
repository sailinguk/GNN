"""QM9 SDF adapter matching the dense heavy-atom representation in GraphVAE."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset


ATOM_NUMBERS = (6, 7, 8, 9)  # C, N, O, F
MAX_HEAVY_ATOMS = 9
NUM_BOND_TYPES = 4


class DenseQM9(Dataset):
    """Read the original QM9 SDF/CSV files as padded dense graph tensors."""

    def __init__(self, root: str | Path, target_column: str = "mu") -> None:
        self.root = Path(root)
        data_root = self.root
        if not (data_root / "gdb9.sdf").is_file() and (
            data_root / "raw" / "gdb9.sdf"
        ).is_file():
            data_root = data_root / "raw"
        self.sdf_path = data_root / "gdb9.sdf"
        self.csv_path = data_root / "gdb9.sdf.csv"
        if not self.sdf_path.is_file() or not self.csv_path.is_file():
            raise FileNotFoundError(
                "Expected gdb9.sdf and gdb9.sdf.csv in the QM9 data directory"
            )
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if target_column not in (reader.fieldnames or []):
                raise ValueError(f"unknown QM9 target column: {target_column}")
            self._targets = torch.tensor(
                [float(row[target_column]) for row in reader], dtype=torch.float32
            )
        self.target_column = target_column
        self._supplier = None

    def _molecules(self):
        if self._supplier is None:
            try:
                from rdkit import Chem
            except ImportError as error:
                raise ImportError("QM9 example requires RDKit") from error
            self._supplier = Chem.SDMolSupplier(
                str(self.sdf_path), removeHs=False, sanitize=False
            )
            if len(self._supplier) != len(self._targets):
                raise RuntimeError("QM9 SDF and CSV contain different row counts")
        return self._supplier

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_supplier"] = None
        return state

    def __len__(self) -> int:
        return len(self._targets)

    @property
    def targets(self) -> torch.Tensor:
        return self._targets

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        try:
            from rdkit.Chem.rdchem import BondType
        except ImportError as error:
            raise ImportError("QM9 example requires RDKit") from error
        molecule = self._molecules()[index]
        if molecule is None:
            raise ValueError(f"RDKit could not parse QM9 molecule {index}")

        heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 1]
        if len(heavy_atoms) > MAX_HEAVY_ATOMS:
            raise ValueError("QM9 molecule exceeds nine heavy atoms")
        atom_to_class = {number: position for position, number in enumerate(ATOM_NUMBERS)}
        old_to_new = {atom.GetIdx(): i for i, atom in enumerate(heavy_atoms)}

        adjacency = torch.zeros(MAX_HEAVY_ATOMS, MAX_HEAVY_ATOMS)
        edge_features = torch.zeros(
            MAX_HEAVY_ATOMS, MAX_HEAVY_ATOMS, NUM_BOND_TYPES
        )
        node_features = torch.zeros(MAX_HEAVY_ATOMS, len(ATOM_NUMBERS))
        node_mask = torch.zeros(MAX_HEAVY_ATOMS, dtype=torch.bool)
        for new_index, atom in enumerate(heavy_atoms):
            atomic_number = atom.GetAtomicNum()
            if atomic_number not in atom_to_class:
                raise ValueError(f"unexpected QM9 atom type: {atomic_number}")
            node_features[new_index, atom_to_class[atomic_number]] = 1.0
            node_mask[new_index] = True

        bond_to_class = {
            BondType.SINGLE: 0,
            BondType.DOUBLE: 1,
            BondType.TRIPLE: 2,
            BondType.AROMATIC: 3,
        }
        for bond in molecule.GetBonds():
            old_src, old_dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if old_src not in old_to_new or old_dst not in old_to_new:
                continue
            if bond.GetBondType() not in bond_to_class:
                raise ValueError(f"unexpected QM9 bond type: {bond.GetBondType()}")
            src, dst = old_to_new[old_src], old_to_new[old_dst]
            bond_class = bond_to_class[bond.GetBondType()]
            # Core convention: A[target, source] represents source -> target.
            adjacency[dst, src] = adjacency[src, dst] = 1.0
            edge_features[dst, src, bond_class] = 1.0
            edge_features[src, dst, bond_class] = 1.0
        return {
            "adjacency": adjacency,
            "edge_features": edge_features,
            "node_features": node_features,
            "node_mask": node_mask,
            "target": self._targets[index],
        }


def collate_dense_graphs(samples: list[dict[str, torch.Tensor]]):
    return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}


def split_qm9(
    dataset_size: int,
    validation_size: int = 10_000,
    test_size: int = 10_000,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if validation_size < 1 or test_size < 1:
        raise ValueError("validation_size and test_size must be positive")
    if validation_size + test_size >= dataset_size:
        raise ValueError("validation and test sets leave no training samples")
    order = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(seed))
    test = order[:test_size]
    validation = order[test_size : test_size + validation_size]
    train = order[test_size + validation_size :]
    return train, validation, test
