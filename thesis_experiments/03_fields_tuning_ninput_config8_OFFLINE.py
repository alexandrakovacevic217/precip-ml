"""
OFFLINE THESIS SCRIPT (NOT meant to be run by default)

Sweep over the number of input PCs (N_input) for a *fixed* configuration
(legacy "config16", corresponding to thesis "config_8") in the APPROACH 1 / target=fields case.

This script:
  1) assumes you already have:
     - CPC precipitation loaded (PRcpc)
     - flattened CPC target array (PRzm)
     - CPC land/sea (valid-data) mask (lsmask_cpc)
     - precomputed ERA5 PCs pickles loaded into EOFS
     - time-alignment indices EOFS_timeind (aligned to CPC dates)
  2) trains a model for each N_input
  3) predicts validation fields
  4) computes MAE / RMSE / mean spatial correlation
  5) saves results to pickles (raw, lightweight compared to netCDF)

⚠️ NOT reproducible out-of-the-box:
- requires large datasets and precomputed EOF/PC artifacts
- long runtime (hours/days depending on machine)
- the "build_and_train", "predict_and_reshape", "correlation_spatial" and
  "build_X_given_ninput" functions must exist in your codebase

Recommended workflow for the GitHub repo:
- Run this offline when needed
- Export final tables to CSV using an export script (pkl -> csv)
- Plot from CSV for the README
"""

import os
import gc
import copy
import pickle
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

# These imports must exist in your project.
# Adjust module names if they differ.
from src.training import build_and_train
from src.inference import predict_and_reshape
from src.metrics import correlation_spatial
from src.features import build_X_given_ninput  # expected signature like in your legacy code


def mem_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline sweep: N_input PCs (fields, config16).")
    p.add_argument("--out-dir", required=True, help="Output directory where pickles will be saved.")
    p.add_argument("--replicas", type=int, default=1, help="Number of replicas (default: 1).")
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility within a run (default: 42).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512).",
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Model activation (default: relu).",
    )
    p.add_argument(
        "--n-input-list",
        type=int,
        nargs="*",
        default=[5, 15, 30, 50, 80, 120, 200, 300, 400, 600, 900, 1300, 1800, 2300, 2800],
        help="List of N_input PCs to sweep (default: thesis list).",
    )
    p.add_argument(
        "--tag",
        type=str,
        default="tuning_n_input_approach1_config16",
        help="Filename tag for outputs.",
    )
    return p.parse_args()


def run_sweep(
    *,
    out_dir: str,
    tag: str,
    n_input_list: List[int],
    replicas: int,
    seed: int,
    batch_size: int,
    activation: str,
    # --- Required objects from your pipeline (must be available in globals or injected) ---
    PRzm: np.ndarray,               # (time, Z) flattened CPC target
    PRcpc: Any,                     # xarray DataArray (time, lat, lon) CPC field (for val truth)
    lsmask_cpc: np.ndarray,         # (lat, lon) mask with 1 / nan like your legacy code
    EOFS: Dict[str, dict],          # precomputed PCs dict: EOFS[var]['PC'] etc.
    vname: List[str],               # list of variables for input
    EOFS_timeind: np.ndarray,       # indices aligning EOFS time to CPC time
    multi_time: bool,               # whether you concatenate t-2,t-1,t
    idx_train_end: int,
    idx_val_end: int,
) -> Dict[int, pd.DataFrame]:
    """
    Returns:
      results_ens: dict[replica_idx] -> DataFrame with columns:
        n_input, mae, rmse, mean_r_spatial, execution_time_s
    """
    os.makedirs(out_dir, exist_ok=True)

    # Targets (fields)
    Y = PRzm  # (time, Z)
    y_train = Y[:idx_train_end, :]
    y_val = Y[idx_train_end:idx_val_end, :]

    # Mask indices for reshape back to (lat, lon)
    imask = np.argwhere(~np.isnan(lsmask_cpc.flatten())).flatten()

    results_ens: Dict[int, pd.DataFrame] = {}

    for k in range(replicas):
        print(f"\n🔁 Replica {k}/{replicas-1} — RAM: {mem_gb():.2f} GB")
        results_list = []

        # (optional) vary seed per replica deterministically
        seed_k = seed + k

        for n_input in n_input_list:
            print(f"▶️ N_input = {n_input}")

            # Keep local copies (avoid side effects)
            y_train_n = y_train.copy()
            y_val_n = y_val.copy()

            # Deepcopy EOFS like legacy (heavy but safe)
            eofs_local = copy.deepcopy(EOFS)

            # Seeds
            tf.random.set_seed(seed_k)
            np.random.seed(seed_k)

            # Build inputs
            # Expected to return X_all with shape (time, features)
            X_all = build_X_given_ninput(n_input, eofs_local, vname, EOFS_timeind, multi_time)
            X_train_n = X_all[:idx_train_end, :]
            X_val_n = X_all[idx_train_end:idx_val_end, :]

            # Train
            model, history = build_and_train(
                X_train_n,
                y_train_n,
                X_val_n,
                y_val_n,
                batch_size=batch_size,
                activation=activation,
            )

            # Predict validation fields
            # CPC shape (time, lat, lon)
            t_cpc, lat_cpc, lon_cpc = PRcpc.shape

            # In your legacy code you reduced val length if multi_time
            t_val = idx_val_end - idx_train_end - (2 if multi_time else 0)

            PR_val = predict_and_reshape(
                model,
                X_val_n,
                imask,
                output_shape=(t_val, lat_cpc, lon_cpc),
                multi_time=multi_time,
            )

            PR_cpc_true_val = PRcpc[idx_train_end:idx_val_end, :, :]

            # Metrics
            mae = float(np.nanmean(np.abs(PR_val - PR_cpc_true_val)))
            rmse = float(np.sqrt(np.nanmean((PR_val - PR_cpc_true_val) ** 2)))

            r_spatial = correlation_spatial(PR_val, PR_cpc_true_val)
            mean_r_spatial = float(np.nanmean(r_spatial))

            results_list.append(
                {
                    "n_input": int(n_input),
                    "mae": mae,
                    "rmse": rmse,
                    "mean_r_spatial": mean_r_spatial,
                    # legacy kept execution time; you can add timing if you have tic/toc
                    "execution_time_s": np.nan,
                }
            )

            # Cleanup to reduce GPU/CPU memory issues
            tf.keras.backend.clear_session()
            del history, model, PR_val, PR_cpc_true_val, X_all, X_train_n, X_val_n
            gc.collect()

        results_df = pd.DataFrame(results_list).sort_values("n_input")
        # Save per-replica pickle
        per_rep_path = os.path.join(out_dir, f"ML_results_bs_Rep{k}.pkl")
        results_df.to_pickle(per_rep_path)
        print(f"✅ Saved replica table: {per_rep_path}")

        results_ens[k] = results_df
        print(f"✅ End replica {k} — RAM: {mem_gb():.2f} GB")

    # Save ensemble dict
    ens_path = os.path.join(out_dir, f"{tag}.pkl")
    with open(ens_path, "wb") as f:
        pickle.dump(results_ens, f)
    print(f"\n📦 Saved ensemble pickle: {ens_path}")

    return results_ens


def main():
    args = parse_args()

    # -------------------------------------------------------------------------
    # IMPORTANT: The following objects must exist in your environment.
    # This script is intentionally "OFFLINE": you should import or build them
    # from your pipeline before calling run_sweep().
    #
    # You can do that by:
    #   - running inside an environment where you've already created them, or
    #   - adding your project-specific loading code here (not recommended for repo)
    # -------------------------------------------------------------------------

    required_names = [
        "PRzm", "PRcpc", "lsmask_cpc",
        "EOFS", "vname", "EOFS_timeind",
        "multi_time", "idx_train_end", "idx_val_end",
    ]

    missing = [n for n in required_names if n not in globals()]
    if missing:
        raise RuntimeError(
            "Missing required in-memory objects: "
            + ", ".join(missing)
            + "\n\nThis is an OFFLINE script: you must create these objects "
              "using your pipeline (data loading + PC loading + alignment) "
              "before running the sweep."
        )

    run_sweep(
        out_dir=args.out_dir,
        tag=args.tag,
        n_input_list=args.n_input_list,
        replicas=args.replicas,
        seed=args.seed,
        batch_size=args.batch_size,
        activation=args.activation,
        PRzm=globals()["PRzm"],
        PRcpc=globals()["PRcpc"],
        lsmask_cpc=globals()["lsmask_cpc"],
        EOFS=globals()["EOFS"],
        vname=globals()["vname"],
        EOFS_timeind=globals()["EOFS_timeind"],
        multi_time=globals()["multi_time"],
        idx_train_end=globals()["idx_train_end"],
        idx_val_end=globals()["idx_val_end"],
    )


if __name__ == "__main__":
    main()