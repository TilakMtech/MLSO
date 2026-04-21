
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Analyze 1-GPU vs 2-GPU CIFAR-10 DDP results")
    parser.add_argument("--baseline", required=True, help="Path to 1-GPU epoch_metrics.csv")
    parser.add_argument("--parallel", required=True, help="Path to 2-GPU epoch_metrics.csv")
    parser.add_argument("--output", default="analysis_summary.json")
    args = parser.parse_args()

    base = pd.read_csv(args.baseline)
    par = pd.read_csv(args.parallel)

    t1 = float(base["epoch_time_sec"].mean())
    t2 = float(par["epoch_time_sec"].mean())
    acc1 = float(base["test_acc"].max())
    acc2 = float(par["test_acc"].max())
    comm_fraction = float(par["comm_fraction"].mean())

    speedup = t1 / t2
    efficiency = speedup / 2.0 * 100.0
    accuracy_gap = acc1 - acc2

    payload = {
        "avg_epoch_time_1gpu_sec": round(t1, 4),
        "avg_epoch_time_2gpu_sec": round(t2, 4),
        "speedup_2gpu_vs_1gpu": round(speedup, 4),
        "parallel_efficiency_percent": round(efficiency, 2),
        "avg_comm_fraction_2gpu": round(comm_fraction, 4),
        "best_acc_1gpu": round(acc1, 4),
        "best_acc_2gpu": round(acc2, 4),
        "accuracy_gap": round(accuracy_gap, 4),
    }

    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
