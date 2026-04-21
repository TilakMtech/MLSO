
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    plt.plot(base["epoch"], base["test_acc"], marker="o", label="1 GPU")
    plt.plot(par["epoch"], par["test_acc"], marker="o", label="2 GPU DDP")
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "accuracy_vs_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(base["epoch"], base["epoch_time_sec"], marker="o", label="1 GPU")
    plt.plot(par["epoch"], par["epoch_time_sec"], marker="o", label="2 GPU DDP")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Time (s)")
    plt.title("Epoch Time vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "epoch_time_vs_epoch.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(par["epoch"], par["comm_fraction"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Communication Fraction")
    plt.title("2-GPU DDP Communication Fraction")
    plt.tight_layout()
    plt.savefig(out / "comm_fraction.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
