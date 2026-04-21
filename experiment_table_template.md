
# Experimental Table Templates

## Table 1. System Configuration
| Item | Choice | Justification |
|---|---|---|
| Dataset | CIFAR-10 | Small images and moderate model size make data parallelism easy to evaluate |
| Model | CIFAR-10 ResNet-18 | Strong baseline without excessive complexity |
| Parallelism | Synchronous data parallelism | Best match for replicated small vision model |
| Backend | NCCL | Optimized GPU collectives |
| Platform | Kaggle single node with 2 GPUs | Directly supports 1 vs 2 GPU comparison |
| Framework | PyTorch DDP | Standard and reproducible multi-GPU training |

## Table 2. Measured Performance
| Run Name | GPUs | Per-GPU Batch | Global Batch | Avg Epoch Time (s) | Total Time (s) | Best Test Acc | Avg Comm Fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| resnet18_1gpu_bs128 | 1 | 128 | 128 |  |  |  | 0.000 |
| resnet18_2gpu_bs128 | 2 | 128 | 256 |  |  |  |  |
| resnet18_2gpu_bs256 | 2 | 256 | 512 |  |  |  |  |

## Table 3. Scaling Metrics
| Comparison | T1 (s) | Tn (s) | Speedup S(N) | Efficiency E(N) | Accuracy Gap |
|---|---:|---:|---:|---:|---:|
| 1 GPU vs 2 GPU DDP |  |  |  |  |  |

## Table 4. Deviation Analysis
| Observation | Expected | Measured | Likely Cause | Proposed Mitigation |
|---|---|---|---|---|
| Speedup below 2x | Near-linear scaling |  | Gradient synchronization overhead | Larger per-GPU batch and fewer small steps |
| Communication fraction non-zero | Low but visible |  | All-reduce cost | AMP and better compute/comm ratio |
| Accuracy difference | Near-zero |  | Global batch change / LR scaling | Tune LR and warmup |
