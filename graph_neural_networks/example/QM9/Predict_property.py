"""Download QM9 with PyG and train an ECC molecular property predictor."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from graph_neural_networks import ECC, GatedGlobalPooling, PreparedQM9


QM9_TARGETS = (
    "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "u0",
    "u", "h", "g", "cv", "u0_atom", "u_atom", "h_atom", "g_atom",
    "a", "b", "c",
)

CHEMICAL_ACCURACY_DEBYE = 0.066513725


class PropertyPredictor(nn.Module):
    def __init__(self, backend, channels, embedding_dim):
        super().__init__()
        self.backend = backend
        self.ecc = ECC(4, 4, channels, backend=backend)
        self.pooling = GatedGlobalPooling(channels[-1], embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, data):
        nodes = self.ecc(data.x, data.edge_attr, edge_index=data.edge_index)
        graph = self.pooling(nodes, batch=data.batch)
        return self.head(graph).squeeze(-1)


def copy_native_initialization(native_model, pyg_model):
    """Give the PyG implementation exactly the native model's initial weights."""
    for native_layer, pyg_layer in zip(native_model.ecc.layers, pyg_model.ecc.layers):
        pyg_layer.edge_network.load_state_dict(native_layer.edge_network.state_dict())
        pyg_layer.bias.data.copy_(native_layer.bias.data)
    for native_residual, pyg_residual in zip(
        native_model.ecc.residuals, pyg_model.ecc.residuals
    ):
        pyg_residual.load_state_dict(native_residual.state_dict())
    for native_norm, pyg_norm in zip(
        native_model.ecc.normalizations, pyg_model.ecc.normalizations
    ):
        pyg_norm.load_state_dict(native_norm.state_dict())
    pyg_model.pooling.load_state_dict(native_model.pooling.state_dict())
    pyg_model.head.load_state_dict(native_model.head.state_dict())


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/QM9"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/QM9"))
    parser.add_argument("--backend", choices=("native", "pyg"), default="pyg")
    parser.add_argument("--property", choices=QM9_TARGETS, default="mu")
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=56145612)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def run_epoch(model, loader, target_index, device, optimizer=None):
    model.train(optimizer is not None)
    squared_error = absolute_error = 0.0
    count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for data in loader:
            data = data.to(device)
            target = data.y[:, target_index]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(data)
            loss = nn.functional.mse_loss(prediction, target)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            squared_error += nn.functional.mse_loss(
                prediction, target, reduction="sum"
            ).item()
            absolute_error += nn.functional.l1_loss(
                prediction, target, reduction="sum"
            ).item()
            count += target.numel()
    mse = squared_error / count
    mae_debye = absolute_error / count
    return mse, mae_debye, mae_debye / CHEMICAL_ACCURACY_DEBYE


def main():
    try:
        from torch_geometric.loader import DataLoader
    except ImportError as error:
        raise ImportError("this example requires torch-geometric") from error

    args = arguments()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    # smiles_input prepares this representation once and caches it on disk.
    dataset = PreparedQM9(root=str(args.data_dir))
    if args.validation_size + args.test_size >= len(dataset):
        raise ValueError("validation and test sets leave no training samples")
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
    test_indices = order[: args.test_size]
    valid_indices = order[args.test_size : args.test_size + args.validation_size]
    train_indices = order[args.test_size + args.validation_size :]
    target_index = QM9_TARGETS.index(args.property)
    common_loader_args = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        dataset[train_indices],
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        **common_loader_args,
    )
    valid_loader = DataLoader(
        dataset[valid_indices],
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        dataset[test_indices],
        batch_size=args.eval_batch_size,
        shuffle=False,
        **common_loader_args,
    )

    # Always create the same native reference first. For the PyG experiment,
    # explicitly copy every trainable tensor so both runs start identically.
    native_initialization = PropertyPredictor(
        "native", args.channels, args.embedding_dim
    ).to(device)
    if args.backend == "native":
        model = native_initialization
    else:
        model = PropertyPredictor("pyg", args.channels, args.embedding_dim).to(device)
        copy_native_initialization(native_initialization, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_mse_debye2",
                "train_mae_debye",
                "train_mae_chemical_normalized",
                "valid_mse_debye2",
                "valid_mae_debye",
                "valid_mae_chemical_normalized",
            ]
        )
        initial_valid = run_epoch(model, valid_loader, target_index, device)
        writer.writerow([0, "", "", "", initial_valid[0], initial_valid[1], initial_valid[2]])
        handle.flush()
        print(
            f"epoch=000 valid_mse_D2={initial_valid[0]:.6f} "
            f"valid_mae_D={initial_valid[1]:.6f} "
            f"valid_mae_chemical={initial_valid[2]:.6f}",
            flush=True,
        )
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(model, train_loader, target_index, device, optimizer)
            valid = run_epoch(model, valid_loader, target_index, device)
            writer.writerow([epoch, *train, *valid])
            handle.flush()
            print(
                f"epoch={epoch:03d} train_mse_D2={train[0]:.6f} "
                f"train_mae_D={train[1]:.6f} "
                f"valid_mse_D2={valid[0]:.6f} valid_mae_D={valid[1]:.6f} "
                f"valid_mae_chemical={valid[2]:.6f}",
                flush=True,
            )

    test = run_epoch(model, test_loader, target_index, device)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backend": args.backend,
            "channels": args.channels,
            "embedding_dim": args.embedding_dim,
            "property": args.property,
            "chemical_accuracy_debye": CHEMICAL_ACCURACY_DEBYE,
            "test_mse_debye2": test[0],
            "test_mae_debye": test[1],
            "test_mae_chemical_normalized": test[2],
            "config": {key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()},
        },
        args.output_dir / "model.pt",
    )
    print(
        f"test_mse_D2={test[0]:.6f} test_mae_D={test[1]:.6f} "
        f"test_mae_chemical={test[2]:.6f}"
    )


if __name__ == "__main__":
    main()
