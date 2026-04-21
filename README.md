
# Kaggle 2-GPU CIFAR-10 DDP Assignment Package

This package is rewritten specifically for **Kaggle free infrastructure with 2 GPUs**.

## Final system choice

- **Dataset:** CIFAR-10
- **Model:** CIFAR-10 adapted ResNet-18
- **Parallelism:** Synchronous data parallelism
- **Framework:** PyTorch DistributedDataParallel (DDP)
- **Communication backend:** NCCL
- **Platform:** Single-node Kaggle notebook with 2 GPUs

## Folder contents

- `train_ddp_kaggle.py` - training script for both 1-GPU baseline and 2-GPU DDP
- `analyze_results.py` - computes speedup, efficiency, communication fraction, and accuracy gap
- `plot_results.py` - generates convergence and timing plots
- `kaggle_notebook_cells.md` - copy-paste notebook cells
- `experiment_table_template.md` - report-ready tables
- `results_template.csv` - fill-in template for measured results

---

## Exact Kaggle notebook commands

### Cell 1 - verify hardware
```bash
!nvidia-smi
```

### Cell 2 - optional environment check
```python
import os, torch
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

### Cell 3 - install/update dependencies if needed
```bash
!pip install -q torch torchvision pandas matplotlib
```

### Cell 4 - baseline 1-GPU training
```bash
!CUDA_VISIBLE_DEVICES=0 python /kaggle/working/train_ddp_kaggle.py \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/outputs \
  --epochs 30 \
  --batch-size 128 \
  --workers 4 \
  --lr 0.1 \
  --amp \
  --run-name resnet18_1gpu_bs128
```

### Cell 5 - 2-GPU DDP training
```bash
!torchrun --standalone --nproc_per_node=2 /kaggle/working/train_ddp_kaggle.py \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/outputs \
  --epochs 30 \
  --batch-size 128 \
  --workers 4 \
  --lr 0.1 \
  --amp \
  --run-name resnet18_2gpu_bs128
```

### Cell 6 - second 2-GPU configuration for batch-size sensitivity
```bash
!torchrun --standalone --nproc_per_node=2 /kaggle/working/train_ddp_kaggle.py \
  --data-dir /kaggle/working/data \
  --output-dir /kaggle/working/outputs \
  --epochs 30 \
  --batch-size 256 \
  --workers 4 \
  --lr 0.1 \
  --amp \
  --run-name resnet18_2gpu_bs256
```

### Cell 7 - compute assignment metrics
```bash
!python /kaggle/working/analyze_results.py \
  --baseline /kaggle/working/outputs/resnet18_1gpu_bs128/epoch_metrics.csv \
  --parallel /kaggle/working/outputs/resnet18_2gpu_bs128/epoch_metrics.csv \
  --output /kaggle/working/outputs/analysis_summary.json
```

### Cell 8 - generate plots
```bash
!python /kaggle/working/plot_results.py \
  --baseline /kaggle/working/outputs/resnet18_1gpu_bs128/epoch_metrics.csv \
  --parallel /kaggle/working/outputs/resnet18_2gpu_bs128/epoch_metrics.csv \
  --output-dir /kaggle/working/outputs/plots
```

---

## Suggested report experiments

### Experiment A - scaling by worker count
- 1 GPU, per-GPU batch size 128, global batch size 128
- 2 GPUs, per-GPU batch size 128, global batch size 256

### Experiment B - scaling by batch size on 2 GPUs
- 2 GPUs, per-GPU batch size 128, global batch size 256
- 2 GPUs, per-GPU batch size 256, global batch size 512

---

## Metrics to report

- Speedup: `S(N) = T1 / TN`
- Parallel efficiency: `E(N) = S(N) / N * 100`
- Communication cost: `sync_time / step_time` or `sync_time / epoch_time`
- Response time: wall-clock epoch time
- Accuracy gap: `Acc_1GPU - Acc_2GPU`

---

## Reproducibility notes

- Baseline and DDP runs should use the same epoch count and augmentation pipeline.
- The script uses:
  - CIFAR-10 download via `torchvision.datasets.CIFAR10`
  - CIFAR-style ResNet-18 stem
  - SGD + momentum + cosine decay
  - AMP for better GPU utilization
- For a fair comparison, compare the **mean epoch time** after the first epoch or compare the full-run average consistently across all runs.
