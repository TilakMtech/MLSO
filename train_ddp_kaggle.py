
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet18


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, local_rank, device


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_model(num_classes: int = 10) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(data_dir: str, batch_size: int, workers: int, distributed: bool):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_csv(path: Path, row: dict):
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
    parser.add_argument("--amp", action="store_true")
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
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr * world_size,  # linear LR scaling for global batch size
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    if distributed:
        dist.barrier()

    run_start = time.perf_counter()
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.perf_counter()
        train_loss_sum = 0.0
        train_count = 0
        compute_time = 0.0
        comm_time = 0.0

        for step, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            start_compute = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.cuda.synchronize(device)
            end_backward = time.perf_counter()

            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize(device)
            end_step = time.perf_counter()

            # Approximate communication time as step time after backward-ready compute.
            compute_time += end_backward - start_compute
            comm_time += end_step - end_backward

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
            "comm_time_sec": round(comm_time, 6),
            "comm_fraction": round(comm_time / max(epoch_time, 1e-9), 6),
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
