"""Train the fragment-graph GIN on scaffold splits (3 seeds by default).

    python train_gin.py                       # BBBP, BACE, ESOL
    python train_gin.py --datasets BBBP --seeds 0

Outputs: checkpoints/{DATASET}_gin_seed{S}.pt and gin_results.csv
"""

import argparse

import numpy as np
import pandas as pd
import torch

from fragment_graphs import DATASETS
from frag_gin import train_one

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--ionic", action="store_true", help="use the BRICS + ionic graphs (frag_graphs_ions/)")
    args = ap.parse_args()
    graph_dir = "frag_graphs_ions" if args.ionic else "frag_graphs"
    ckpt_dir = "checkpoints_ions" if args.ionic else "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    rows = []
    for name in args.datasets:
        task = DATASETS[name][1]
        metric = "ROC-AUC" if task == "classification" else "RMSE"
        tests = []
        for seed in args.seeds:
            val, test, ep = train_one(name, task, seed, device, graph_dir=graph_dir, ckpt_dir=ckpt_dir)
            tests.append(test)
            print(f"{name} seed {seed}: valid {metric} {val:.3f} | test {metric} {test:.3f} | {ep} epochs")
        rows.append({"dataset": name, "metric": metric, "test_mean": np.mean(tests), "test_std": np.std(tests)})
    df = pd.DataFrame(rows)
    df.to_csv("gin_results_ions.csv" if args.ionic else "gin_results.csv", index=False)
    print(df.round(3).to_string(index=False))
