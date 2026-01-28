import argparse
import pickle
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-pkl",
        default="results/raw/tuning_n_input_approach1_config8.pkl",
        help="Input pickle produced by the offline sweep.",
    )
    parser.add_argument(
        "--replica",
        type=int,
        default=0,
        help="Which replica to export (default: 0).",
    )
    parser.add_argument(
        "--out-csv",
        default="results/tables/03_fields_tuning_ninput_config8.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_pkl)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path, "rb") as f:
        results_ens = pickle.load(f)

    if not isinstance(results_ens, dict):
        raise TypeError(
            f"Expected dict replica->DataFrame/list, got {type(results_ens)} from {in_path}"
        )

    if args.replica not in results_ens:
        raise KeyError(
            f"Replica {args.replica} not found. Available keys: {list(results_ens.keys())}"
        )

    obj = results_ens[args.replica]

    # Accept either DataFrame or list-of-dicts
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = pd.DataFrame(obj)

    # Normalize column names (support legacy names)
    rename_map = {
        "n_input": "n_input",
        "mae": "mae_mm_day",
        "rmse": "rmse_mm_day",
        "mean_r_spatial": "corr_spatial_mean",
        "execution_time": "execution_time_s",
        "execution_time_s": "execution_time_s",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["n_input", "mae_mm_day", "rmse_mm_day", "corr_spatial_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in replica {args.replica}: {missing}. "
            f"Available: {list(df.columns)}"
        )

    keep = ["n_input", "mae_mm_day", "rmse_mm_day", "corr_spatial_mean"]
    if "execution_time_s" in df.columns:
        keep.append("execution_time_s")

    df = df[keep].sort_values("n_input")

    # Add metadata columns for consistency across results tables
    df["config_id"] = "config_8"
    df["target_type"] = "fields"
    df["split"] = "val"
    df["replica"] = args.replica

    df.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()