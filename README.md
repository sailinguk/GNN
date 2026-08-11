# GNN

Reusable padding-free ECC and gated graph pooling implemented with PyTorch. ECC
can run through either the native PyTorch implementation or PyG's `NNConv`;
both use a linear edge filter, mean aggregation, an external residual,
BatchNorm and ReLU.

## Requirements

- `torch`
- `torch-geometric`

## Structure

```text
graph_neural_networks/
├── core/
│   ├── smiles_input.py  # download and cache heavy-atom QM9 graphs
│   ├── ECC.py           # compatible native and PyG ECC backends
│   └── Pooling.py       # gated pooling over real nodes
└── example/QM9/
    └── Predict_property.py
```

## ECC parameters

- `node_feature_dim`: input node-vector length
- `edge_feature_dim`: input edge-vector length
- `channels`: ECC widths; its length is the number of ECC layers
- `backend`: `"native"` or `"pyg"`
- `graph_embedding_dim`: gated pooling output length

```python
from graph_neural_networks import ECC, GatedGlobalPooling

ecc = ECC(4, 4, channels=(32, 64), backend="native")
pooling = GatedGlobalPooling(64, graph_embedding_dim=128)

nodes = ecc(data.x, data.edge_attr, edge_index=data.edge_index)
graph_embedding = pooling(nodes, batch=data.batch)
```

## QM9 example

On the first run, `smiles_input.py` downloads QM9, removes hydrogen, prepares
the C/N/O/F graphs and caches them as `processed/qm9_heavy_atoms.pt`. Training
epochs only read this prepared data.

```bash
python3 -m graph_neural_networks.example.QM9.Predict_property \
  --backend pyg \
  --property mu \
  --channels 32 64 \
  --embedding-dim 128 \
  --epochs 100 \
  --batch-size 64 \
  --eval-batch-size 20 \
  --learning-rate 0.0001 \
  --num-workers 0 \
  --seed 56145612
```

Use `--backend native` or `--backend pyg`; neither backend pads nodes. On an HPC
server, the command may need an `sbatch` or `job.sh` file.

The target is the dipole moment `mu` in Debye. Validation also reports
`MAE / 0.066513725`, the chemically normalized MAE used by the reference
example.
