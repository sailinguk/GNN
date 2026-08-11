# GNN

Reusable graph neural network components for GraphVAE. The first implementation
follows the GraphVAE encoder: edge-conditioned convolution (ECC), residual
connections, batch normalization, ReLU, and gated global pooling.

## Requirements

- Python 3.10+
- PyTorch
- RDKit (QM9 example only)

Install versions that match your operating system and CUDA environment.
PyTorch Geometric is not required. An optional adapter is provided for callers
that already use PyG data objects.

## Public interface

```python
from graph_neural_networks import ECCConv, GatedGlobalPooling, GraphEncoder
from graph_neural_networks import ECCGraphEncoder

encoder = ECCGraphEncoder(
    node_feature_dim=4,
    edge_feature_dim=4,
    channels=(32, 64),
    graph_embedding_dim=128,
)

embedding = encoder(A, E, X, node_mask)  # [batch, 128]
```

The dense input convention is:

- `A`: `[batch, nodes, nodes]`, where `A[:, i, j]` is edge `j -> i`
- `E`: `[batch, nodes, nodes, edge_features]`
- `X`: `[batch, nodes, node_features]`
- `node_mask`: `[batch, nodes]`

`GraphEncoder` accepts any message-passing layers exposing `in_channels`,
`out_channels`, and the same dense forward signature. It also accepts a
replaceable pooling module. An optional graph condition can be concatenated to
every node immediately before pooling.

## QM9 example

Show all parameters:

```bash
python3 -m graph_neural_networks.example_trainers.qm9_regression --help
```

Train the GraphVAE-style encoder to predict molecular dipole moment:

```bash
python3 -m graph_neural_networks.example_trainers.qm9_regression \
  --device cuda \
  --epochs 25 \
  --batch-size 32 \
  --learning-rate 0.001
```

The main options control the ECC channel widths, graph embedding size, optimizer,
dataset directory, output directory, data split sizes, random seed, and device.
If running on an HPC server, the command may need to be placed in a Slurm
`sbatch` file or another cluster-specific script such as `job.sh`.
