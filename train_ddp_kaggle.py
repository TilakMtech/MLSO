"""
Kaggle-ready CIFAR-10 training with optional single-node multi-GPU DDP.

This script is intentionally written as an assignment-grade reference
implementation: the logic is simple, reproducible, and heavily commented so a
reviewer can see how the system is initialized, how data is partitioned across
workers, and how performance measurements are collected.

Usage examples
--------------
1 GPU baseline:
    CUDA_VISIBLE_DEVICES=0 python train_ddp_kaggle.py \
        --data-dir /kaggle/working/data \
        --output-dir /kaggle/working/outputs \
        --epochs 30 \
        --batch-size 128 \
        --amp \
        --run-name resnet18_1gpu_bs128

2 GPU DDP:
    torchrun --standalone --nproc_per_node=2 train_ddp_kaggle.py \
        --data-dir /kaggle/working/data \
        --output-dir /kaggle/working/outputs \
        --epochs 30 \
        --batch-size 128 \
        --amp \
        --run-name resnet18_2gpu_bs128
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18


def is_distributed() -> bool:
    """Return True when the script is launched under torchrun/DDP."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    """
    Initialize the NCCL process group and bind the current process to its GPU.

    Returns
    -------
    rank : int
        Global process rank across all workers.
    world_size : int
        Total number of training processes (GPUs in this single-node setup).
    local_rank : int
        GPU index local to this node.
    device : torch.device
        CUDA device assigned to the current process.
    """
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup():
    """Destroy the process group cleanly at the end of training."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    """
    Seed all relevant RNGs.

    We offset the base seed by rank so each worker has a deterministic but
    distinct RNG stream where appropriate.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(num_classes: int = 10) -> nn.Module:
    """
    Construct a CIFAR-10-adapted ResNet-18.

    ResNet-18 was originally designed for ImageNet (224x224 images). CIFAR-10
    uses 32x32 images, so we replace the large 7x7 / stride-2 stem with a 3x3 /
    stride-1 convolution and remove the initial max-pooling layer. This keeps
    more spatial information in early layers and is standard practice for
    CIFAR-sized inputs.
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(data_dir: str, batch_size: int, workers: int, distributed: bool):
    """
    Create training and test data loaders.

    Important design note:
    - In DDP mode, DistributedSampler partitions the dataset so every worker
      sees a different shard of the epoch. This is essential; otherwise each GPU
      would redundantly process the same mini-batches and effective scaling would
      be invalid.
    - The test loader also uses a DistributedSampler in distributed mode so
      evaluation is sharded and then reduced across workers.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tfms = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_tfms)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        drop_last=False,
    )
    return train_loader, test_loader, train_sampler


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, distributed: bool):
    """
    Evaluate the model and optionally aggregate metrics across all workers.

    Accuracy and loss are reduced across ranks so the reported test accuracy is
    for the full CIFAR-10 test set rather than for a single GPU shard.
    """
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    loss_meter = torch.tensor(0.0, device=device)
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum()
        total += labels.size(0)
        loss_meter += loss.detach() * labels.size(0)

    if distributed:
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_meter, op=dist.ReduceOp.SUM)

    acc = (correct / total).item()
    loss = (loss_meter / total).item()
    return loss, acc


def save_json(path: Path, payload):
    """Write a JSON file, creating parent folders if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv(path: Path, row: dict):
    """Append one row to a CSV file, writing a header on first creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Kaggle-ready CIFAR-10 DDP training")
    parser.add_argument("--data-dir", type=str, default="/kaggle/working/data")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--run-name", type=str, default="exp")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    distributed = is_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Kaggle workflow.")

    if distributed:
        rank, world_size, local_rank, device = setup_distributed()
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0")

    # Rank-offset seeds keep the run reproducible while preserving distinct
    # worker-level randomness where needed.
    seed_everything(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir) / args.run_name
    metrics_csv = output_dir / "epoch_metrics.csv"
    summary_json = output_dir / "summary.json"

    train_loader, test_loader, train_sampler = build_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        distributed=distributed,
    )

    model = build_model().to(device)
    if distributed:
        # DDP wraps one replica per GPU. Gradient all-reduce happens during
        # backward, keeping parameter replicas synchronized across workers.
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        # Linear LR scaling is a common rule when global batch size grows with
        # world_size. It preserves approximately similar optimizer dynamics.
        lr=args.lr * world_size,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Use the modern AMP API to avoid deprecation warnings on recent PyTorch.
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        # Ensure rank 0 has created the output directory before any worker
        # reaches code paths that might assume it exists.
        dist.barrier()

    run_start = time.perf_counter()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if distributed and train_sampler is not None:
            # Required so every epoch gets a different shuffled partition.
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_count = 0
        compute_time = 0.0
        comm_time_proxy = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            step_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # In DDP, gradient synchronization is triggered during backward.
            scaler.scale(loss).backward()
            torch.cuda.synchronize(device)
            backward_done = time.perf_counter()

            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            step_done = time.perf_counter()

            # Measurement note:
            # - compute_time counts forward + loss + backward until gradients are ready
            # - comm_time_proxy counts the tail of the step after backward. In DDP
            #   this includes communication plus some optimizer/scaler overhead, so
            #   it should be interpreted as a communication-dominated proxy rather
            #   than an exact NCCL-only timer.
            compute_time += backward_done - step_start
            comm_time_proxy += step_done - backward_done

            batch_size_actual = labels.size(0)
            train_loss_sum += loss.item() * batch_size_actual
            train_count += batch_size_actual

            if rank == 0 and step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        scheduler.step()

        train_loss = train_loss_sum / max(train_count, 1)
        test_loss, test_acc = evaluate(model, test_loader, device, distributed)
        epoch_time = time.perf_counter() - epoch_start
        best_acc = max(best_acc, test_acc)

        # train_count is per rank, so multiply by world_size for global throughput.
        samples_seen = train_count * world_size
        throughput = samples_seen / epoch_time

        row = {
            "run_name": args.run_name,
            "epoch": epoch,
            "world_size": world_size,
            "per_gpu_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * world_size,
            "train_loss": round(train_loss, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "epoch_time_sec": round(epoch_time, 6),
            "compute_time_sec": round(compute_time, 6),
            "comm_time_sec": round(comm_time_proxy, 6),
            "comm_fraction": round(comm_time_proxy / max(epoch_time, 1e-9), 6),
            "throughput_samples_per_sec": round(throughput, 3),
            "lr": optimizer.param_groups[0]["lr"],
            "amp": int(args.amp),
        }

        if rank == 0:
            append_csv(metrics_csv, row)
            print(
                f"[epoch {epoch:02d}] "
                f"acc={test_acc*100:.2f}% time={epoch_time:.2f}s "
                f"throughput={throughput:.1f} img/s comm_frac={row['comm_fraction']:.3f}",
                flush=True,
            )

            ckpt = {
                "epoch": epoch,
                "model": model.module.state_dict() if distributed else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            }
            torch.save(ckpt, output_dir / "last_checkpoint.pt")
            if test_acc >= best_acc:
                torch.save(ckpt, output_dir / "best_checkpoint.pt")

    total_time = time.perf_counter() - run_start
    summary = {
        "run_name": args.run_name,
        "world_size": world_size,
        "per_gpu_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "epochs": args.epochs,
        "best_test_acc": best_acc,
        "total_train_time_sec": total_time,
        "amp": bool(args.amp),
    }

    if rank == 0:
        save_json(summary_json, summary)
        print(json.dumps(summary, indent=2), flush=True)

    cleanup()


if __name__ == "__main__":
    main()
