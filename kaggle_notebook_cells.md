
# Kaggle Notebook Cells

## 1. Verify that Kaggle attached 2 GPUs
```bash
!nvidia-smi
```

## 2. Check PyTorch can see both GPUs
```python
import torch
print("CUDA:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## 3. Place scripts in /kaggle/working
If your zip is uploaded as a dataset or notebook input, copy the scripts:
```bash
!cp -r /kaggle/input/YOUR_DATASET_FOLDER/* /kaggle/working/
!ls /kaggle/working/
```

## 4. Single-GPU baseline
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

## 5. Two-GPU DDP main run
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

## 6. Two-GPU larger-batch run
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

## 7. Analyze 1-GPU vs 2-GPU scaling
```bash
!python /kaggle/working/analyze_results.py \
  --baseline /kaggle/working/outputs/resnet18_1gpu_bs128/epoch_metrics.csv \
  --parallel /kaggle/working/outputs/resnet18_2gpu_bs128/epoch_metrics.csv \
  --output /kaggle/working/outputs/analysis_summary.json
```

## 8. Generate figures for the report
```bash
!python /kaggle/working/plot_results.py \
  --baseline /kaggle/working/outputs/resnet18_1gpu_bs128/epoch_metrics.csv \
  --parallel /kaggle/working/outputs/resnet18_2gpu_bs128/epoch_metrics.csv \
  --output-dir /kaggle/working/outputs/plots
```
