"""One command: get the MoleculeNet CSVs, scaffold-split them, build fragment graphs,
and write the statistics table.

    python prepare_data.py                     # downloads BBBP, BACE, ESOL into ./data
    python prepare_data.py --data_dir my_csvs  # use CSVs you already have

    python prepare_data.py --ionic             # keep counter-ions as nodes (BRICS + ionic)

Outputs: frag_graphs/{DATASET}_{train,valid,test}.pt and fragment_stats.csv
         (frag_graphs_ions/ and fragment_stats_ions.csv with --ionic)
"""

import argparse
import os
import urllib.request

import pandas as pd

import fragment_graphs as fg

URLS = {
    "BBBP": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "BACE": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "ESOL": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
}
# File names tried, in order, when --data_dir is given
LOCAL_NAMES = {
    "BBBP": ["BBBP.csv", "bbbp.csv"],
    "BACE": ["bace.csv", "BACE.csv"],
    "ESOL": ["esol.csv", "ESOL.csv", "delaney-processed.csv", "delaney.csv"],
}
LABEL_CANDIDATES = {
    "BBBP": ["p_np", "label", "y"],
    "BACE": ["Class", "class", "label", "y"],
    "ESOL": ["measured log solubility in mols per litre", "logSolubility", "measured_log_solubility",
             "logS", "solubility", "y"],
}


def locate(name, data_dir):
    for fname in LOCAL_NAMES[name]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            return path
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, os.path.basename(URLS[name]))
    print(f"downloading {name} -> {path}")
    urllib.request.urlretrieve(URLS[name], path)
    return path


def detect_columns(name, path):
    cols = list(pd.read_csv(path, nrows=1).columns)
    smi = next((c for c in cols if c.lower() in ("smiles", "mol")), None)
    lab = next((c for c in LABEL_CANDIDATES[name] if c in cols), None)
    if smi is None or lab is None:
        raise KeyError(f"{name}: can't find SMILES/label columns in {path}; header is {cols}")
    return smi, lab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--datasets", nargs="*", default=list(fg.DATASETS))
    ap.add_argument("--ionic", action="store_true", help="BRICS + ionic decomposition -> frag_graphs_ions/")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    for name in args.datasets:
        path = locate(name, args.data_dir)
        fg.CSV_COLUMNS[name] = detect_columns(name, path)
        fg.CSV_PATHS[name] = path
        print(f"{name}: {path}  columns={fg.CSV_COLUMNS[name]}")
    fg.run(args.datasets, build=True, out_dir=args.out, ionic=args.ionic)
