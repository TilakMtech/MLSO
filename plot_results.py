"""
Generate report-ready plots from 1-GPU and 2-GPU experiment logs.

The assignment asks for convergence curves and timing-oriented comparisons, so
this script deliberately emphasizes:
- test accuracy vs training time
- epoch time vs epoch
- communication fraction vs epoch
- summary bars for response time and best accuracy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def cumulative_time(df: pd.DataFrame) -> pd.Series:
    """Convert per-epoch durations into wall-clock time elapsed since run start."""
    return df["epoch_time_sec"].cumsum()


def main():
    parser = argparse.ArgumentParser(description="Plot CIFAR-10 DDP experiment results")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--parallel", required=True)
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.baseline)
    par = pd.read_csv(args.parallel)

    plt.figure(figsize=(7, 4.5))
    plt.plot(cumulative_time(base), base["test_acc"], marker="o", label="1 GPU")
    plt.plot(cumulative_time(par), par["test_acc"], marker="o", label="2 GPU DDP")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Test Accuracy")
    plt.title("Convergence: Accuracy vs Training Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy_vs_time.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(base["epoch"], base["epoch_time_sec"], marker="o", label="1 GPU")
    plt.plot(par["epoch"], par["epoch_time_sec"], marker="o", label="2 GPU DDP")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Time (s)")
    plt.title("Response Time per Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "epoch_time_vs_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(par["epoch"], par["comm_fraction"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Communication Fraction")
    plt.title("2-GPU DDP Communication-Dominated Fraction")
    plt.tight_layout()
    plt.savefig(out / "comm_fraction.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(["1 GPU", "2 GPU"], [base["epoch_time_sec"].mean(), par["epoch_time_sec"].mean()])
    plt.ylabel("Average Epoch Time (s)")
    plt.title("Average Response Time by Configuration")
    plt.tight_layout()
    plt.savefig(out / "avg_epoch_time_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.bar(["1 GPU", "2 GPU"], [base["test_acc"].max(), par["test_acc"].max()])
    plt.ylabel("Best Test Accuracy")
    plt.title("Best Accuracy by Configuration")
    plt.tight_layout()
    plt.savefig(out / "best_accuracy_bar.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
