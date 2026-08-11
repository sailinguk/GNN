"""Train a GraphVAE-style ECC encoder for QM9 dipole-moment regression."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from graph_neural_networks.datasets import DenseQM9, collate_dense_graphs, split_qm9
from graph_neural_networks.ecc import ECCGraphEncoder


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/qm9"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qm9_mu"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--target-column", default="mu")
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--train-subset", type=int)
    parser.add_argument("--validation-subset", type=int)
    parser.add_argument("--test-subset", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    return args


class QM9Regressor(nn.Module):
    def __init__(self, channels: list[int], embedding_dim: int) -> None:
        super().__init__()
        self.encoder = ECCGraphEncoder(4, 4, channels, embedding_dim)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        embedding = self.encoder(
            batch["adjacency"],
            batch["edge_features"],
            batch["node_features"],
            batch["node_mask"],
        )
        return self.head(embedding).squeeze(-1)


def limited(indices: torch.Tensor, count: int | None) -> torch.Tensor:
    return indices if count is None else indices[: min(count, len(indices))]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, target_mean, target_std, device):
    training = optimizer is not None
    model.train(training)
    total_squared_error = 0.0
    total_absolute_error = 0.0
    examples = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)
            normalized_target = (batch["target"] - target_mean) / target_std
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = F.mse_loss(prediction, normalized_target)
            if training:
                loss.backward()
                optimizer.step()
            count = batch["target"].numel()
            total_squared_error += F.mse_loss(
                prediction, normalized_target, reduction="sum"
            ).item()
            prediction_original = prediction * target_std + target_mean
            total_absolute_error += F.l1_loss(
                prediction_original, batch["target"], reduction="sum"
            ).item()
            examples += count
    mse = total_squared_error / examples
    mae = total_absolute_error / examples
    if not math.isfinite(mse) or not math.isfinite(mae):
        raise RuntimeError(f"non-finite metrics: mse={mse}, mae={mae}")
    return mse, mae


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = DenseQM9(args.data_dir, args.target_column)
    train_indices, validation_indices, test_indices = split_qm9(
        len(dataset), args.validation_size, args.test_size, args.seed
    )
    train_indices = limited(train_indices, args.train_subset)
    validation_indices = limited(validation_indices, args.validation_subset)
    test_indices = limited(test_indices, args.test_subset)
    target_values = dataset.targets[train_indices]
    target_mean = target_values.mean().to(device)
    target_std = target_values.std(unbiased=False).clamp_min(1e-8).to(device)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate_dense_graphs,
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices.tolist()), shuffle=False, **loader_options
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices.tolist()), shuffle=False, **loader_options
    )

    model = QM9Regressor(args.channels, args.embedding_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, 0.999),
        weight_decay=args.weight_decay,
    )
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_mse", "train_mae", "validation_mse", "validation_mae"])
        for epoch in range(1, args.epochs + 1):
            train = run_epoch(
                model, train_loader, optimizer, target_mean, target_std, device
            )
            validation = run_epoch(
                model, validation_loader, None, target_mean, target_std, device
            )
            writer.writerow([epoch, *train, *validation])
            handle.flush()
            print(
                f"epoch={epoch:03d} train_mse={train[0]:.6f} train_mae={train[1]:.6f} "
                f"validation_mse={validation[0]:.6f} validation_mae={validation[1]:.6f}",
                flush=True,
            )

    test_mse, test_mae = run_epoch(
        model, test_loader, None, target_mean, target_std, device
    )
    test_metrics = {"test_mse": test_mse, "test_mae": test_mae}
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "channels": list(args.channels),
        "embedding_dim": args.embedding_dim,
        "node_feature_dim": 4,
        "edge_feature_dim": 4,
        "target_column": args.target_column,
        "target_mean": target_mean.item(),
        "target_std": target_std.item(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, args.output_dir / "model.pt")
    print(f"test_mse={test_mse:.6f} test_mae={test_mae:.6f}", flush=True)


if __name__ == "__main__":
    main()
