import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import EarlyStopping

from src.config import load_config
from src.data_loading import prep_input_filled
from src.inference import predict_and_reshape
from src.training import build_and_train


def idx_bounds_from_cfg(cfg: dict) -> Tuple[int, int, int]:
    s = cfg["split"]
    start_year = int(s["start_year"])
    train_end = int(s["train_end"])
    val_end = int(s["val_end"])
    test_end = int(s["test_end"])
    days_per_year = int(s["days_per_year"])

    idx_train_end = (train_end - start_year + 1) * days_per_year
    idx_val_end = (val_end - start_year + 1) * days_per_year
    idx_test_end = (test_end - start_year + 1) * days_per_year
    return idx_train_end, idx_val_end, idx_test_end


def load_era5_pcs_pickles(dir_eof: str, variables: List[str]) -> Dict[str, dict]:
    eofs: Dict[str, dict] = {}

    for var in variables:
        path = os.path.join(dir_eof, f"ERA5_PCs_{var}_1979-2020_N8000_split.pickle")
        with open(path, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict) and var in obj:
            eofs[var] = obj[var]
        else:
            eofs[var] = obj

    return eofs


def build_X(
    eofs: Dict[str, dict],
    variables: List[str],
    n_input: int,
    eofs_time_idx: np.ndarray,
    multi_time: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    if not multi_time:
        X = eofs[variables[0]]["PC"][0:n_input, eofs_time_idx]
        for var in variables[1:]:
            X = np.concatenate((X, eofs[var]["PC"][0:n_input, eofs_time_idx]), axis=0)
        return X.T, eofs_time_idx

    t_m2 = eofs_time_idx[:-2]
    t_m1 = eofs_time_idx[1:-1]
    t0 = eofs_time_idx[2:]

    X = np.concatenate(
        (
            eofs[variables[0]]["PC"][0:n_input, t_m2],
            eofs[variables[0]]["PC"][0:n_input, t_m1],
            eofs[variables[0]]["PC"][0:n_input, t0],
        ),
        axis=0,
    )

    for var in variables[1:]:
        X = np.concatenate(
            (
                X,
                eofs[var]["PC"][0:n_input, t_m2],
                eofs[var]["PC"][0:n_input, t_m1],
                eofs[var]["PC"][0:n_input, t0],
            ),
            axis=0,
        )

    return X.T, t0


def build_xy_and_splits(cfg: dict, paths: dict) -> dict:
    ini_day = cfg["time"]["start"]
    end_day = cfg["time"]["end"]

    cpc_start = ini_day + "T12:00:00.000000000"
    cpc_end = end_day + "T12:00:00.000000000"

    variables = cfg["inputs"]["variables"]
    n_input = int(cfg["inputs"]["n_pcs"])
    multi_time = bool(cfg["inputs"]["multi_time"])

    target_type = cfg["target"]["type"]
    n_output = int(cfg["target"]["n_pcs"])

    dir_data = paths["DIR_DATA"]
    dir_eof = paths["DIR_EOF"]
    cpc_pcs_pickle = paths["CPC_PCS_PICKLE"]

    ds_cpc = xr.open_dataset(os.path.join(dir_data, "CPC_pr_filled_1979-2021.nc"))
    pr_cpc = ds_cpc["pr"].sel(time=slice(cpc_start, cpc_end))

    lsmask_cpc = np.where(~np.isnan(pr_cpc.isel(time=0).data), 1.0, np.nan)
    pr_flat = prep_input_filled(pr_cpc, lsmask_cpc)  # (time, Z)

    eofs = load_era5_pcs_pickles(dir_eof, variables)

    time_eofs = eofs[variables[0]].get("time", None)
    if time_eofs is None:
        raise KeyError(
            f"Missing 'time' in ERA5 PCs pickle for variable '{variables[0]}'. "
            "A time coordinate is required to align inputs with CPC."
        )

    _, eofs_time_idx, pr_time_idx = np.intersect1d(
        np.array(time_eofs).astype("datetime64[D]"),
        pr_cpc["time"].data.astype("datetime64[D]"),
        return_indices=True,
    )

    X, _aligned_eofs_time_idx = build_X(
        eofs=eofs,
        variables=variables,
        n_input=n_input,
        eofs_time_idx=eofs_time_idx,
        multi_time=multi_time,
    )

    if target_type == "fields":
        if not multi_time:
            Y = pr_flat[pr_time_idx, :]
        else:
            Y = pr_flat[pr_time_idx[2:], :]

    elif target_type == "pcs":
        with open(cpc_pcs_pickle, "rb") as f:
            cpc_obj = pickle.load(f)

        pc = cpc_obj["PC_unscaled"]  # expected shape (N, T)
        std = np.std(pc, axis=1, keepdims=True)
        std[std == 0] = np.nan
        pc_scaled = pc / std

        Y_all = pc_scaled.T  # (time, N)
        if not multi_time:
            Y = Y_all[pr_time_idx, :n_output]
        else:
            Y = Y_all[pr_time_idx[2:], :n_output]

    else:
        raise ValueError("cfg['target']['type'] must be 'fields' or 'pcs'")

    idx_train_end, idx_val_end, idx_test_end = idx_bounds_from_cfg(cfg)

    X_train = X[:idx_train_end, :]
    X_val = X[idx_train_end:idx_val_end, :]
    X_test = X[idx_val_end:idx_test_end, :]

    y_train = Y[:idx_train_end, :]
    y_val = Y[idx_train_end:idx_val_end, :]
    y_test = Y[idx_val_end:idx_test_end, :]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "target_type": target_type,
        "multi_time": multi_time,
        "lsmask_cpc": lsmask_cpc,
    }


def make_run_dir(out_dir: str, run_name: str | None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = out / run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # fail if already exists
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dir-data", required=True)
    parser.add_argument("--dir-eof", required=True)
    parser.add_argument("--cpc-pcs-pickle", required=True)

    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default=None)

    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--train-subset-steps", type=int, default=200)

    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = {
        "DIR_DATA": args.dir_data,
        "DIR_EOF": args.dir_eof,
        "CPC_PCS_PICKLE": args.cpc_pcs_pickle,
    }

    run_dir = make_run_dir(args.out_dir, args.run_name)

    # Save config snapshot
    config_text = Path(args.config).read_text(encoding="utf-8")
    (run_dir / "config_used.yaml").write_text(config_text, encoding="utf-8")

    res = build_xy_and_splits(cfg, paths)

    X_train, y_train = res["X_train"], res["y_train"]
    X_val, y_val = res["X_val"], res["y_val"]
    X_test, y_test = res["X_test"], res["y_test"]

    target_type = res["target_type"]
    multi_time = res["multi_time"]
    lsmask_cpc = res["lsmask_cpc"]

    print(f"Run dir: {run_dir}")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    if args.mixed_precision:
        mixed_precision.set_global_policy("mixed_float16")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )

    model, history = build_and_train(
        X_train, y_train,
        X_val, y_val,
        activation="relu",
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )

    # Save model + history
    model.save(run_dir / "model.keras")
    (run_dir / "history.json").write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    # Inference
    n_subset_steps = min(args.train_subset_steps, X_train.shape[0])

    if target_type == "fields":
        imask = np.argwhere(~np.isnan(lsmask_cpc.flatten())).flatten()
        lat_cpc, lon_cpc = lsmask_cpc.shape

        pred_train_subset = predict_and_reshape(
            model,
            X_train[:n_subset_steps],
            imask,
            output_shape=(n_subset_steps, lat_cpc, lon_cpc),
            multi_time=multi_time,
        )

        pred_val = predict_and_reshape(
            model,
            X_val,
            imask,
            output_shape=(X_val.shape[0], lat_cpc, lon_cpc),
            multi_time=multi_time,
        )

        np.save(run_dir / "pred_train_subset.npy", pred_train_subset)
        np.save(run_dir / "pred_val.npy", pred_val)

    elif target_type == "pcs":
        pred_train_subset_pc = model.predict(X_train[:n_subset_steps], verbose=0)
        pred_val_pc = model.predict(X_val, verbose=0)

        np.save(run_dir / "pred_train_subset_pc.npy", pred_train_subset_pc)
        np.save(run_dir / "pred_val_pc.npy", pred_val_pc)

    else:
        raise ValueError("cfg['target']['type'] must be 'fields' or 'pcs'")

    print("Done.")


if __name__ == "__main__":
    main()


