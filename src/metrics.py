import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


def correlation_temporal(X, Y):
    """Pointwise temporal correlation map over time -> returns (lat, lon)."""
    if isinstance(X, xr.DataArray):
        mean_X = X.mean(dim="time", skipna=True)
        fluct_X = X - mean_X
        fluct_X_sq = (fluct_X ** 2).sum(dim="time", skipna=True)
    else:
        mean_X = np.nanmean(X, axis=0)
        fluct_X = X - mean_X
        fluct_X_sq = np.nansum(fluct_X ** 2, axis=0)

    if isinstance(Y, xr.DataArray):
        mean_Y = Y.mean(dim="time", skipna=True)
        fluct_Y = Y - mean_Y
        fluct_Y_sq = (fluct_Y ** 2).sum(dim="time", skipna=True)
    else:
        mean_Y = np.nanmean(Y, axis=0)
        fluct_Y = Y - mean_Y
        fluct_Y_sq = np.nansum(fluct_Y ** 2, axis=0)

    if isinstance(X, xr.DataArray) or isinstance(Y, xr.DataArray):
        numerator = (fluct_X * fluct_Y).sum(dim="time", skipna=True)
    else:
        numerator = np.nansum(fluct_X * fluct_Y, axis=0)

    denominator = np.sqrt(fluct_X_sq * fluct_Y_sq)

    if np.any(denominator == 0):
        print("Attenzione: divisione per zero nella correlazione temporale.")

    r_map = numerator / denominator
    r_map = np.where(denominator == 0, np.nan, r_map)
    return r_map


def correlation_spatial(X, Y):
    """Spatial correlation time series -> returns (time,)."""
    if isinstance(X, xr.DataArray):
        mean_X = X.mean(dim=["lat", "lon"], skipna=True)
        fluct_X = X - mean_X.broadcast_like(X)
        fluct_X_sq = (fluct_X ** 2).sum(dim=["lat", "lon"], skipna=True)
    else:
        mean_X = np.nanmean(X, axis=(1, 2))
        fluct_X = X - mean_X[:, None, None]
        fluct_X_sq = np.nansum(fluct_X ** 2, axis=(1, 2))

    if isinstance(Y, xr.DataArray):
        mean_Y = Y.mean(dim=["lat", "lon"], skipna=True)
        fluct_Y = Y - mean_Y.broadcast_like(Y)
        fluct_Y_sq = (fluct_Y ** 2).sum(dim=["lat", "lon"], skipna=True)
    else:
        mean_Y = np.nanmean(Y, axis=(1, 2))
        fluct_Y = Y - mean_Y[:, None, None]
        fluct_Y_sq = np.nansum(fluct_Y ** 2, axis=(1, 2))

    if isinstance(X, xr.DataArray) or isinstance(Y, xr.DataArray):
        numerator = (fluct_X * fluct_Y).sum(dim=["lat", "lon"], skipna=True)
    else:
        numerator = np.nansum(fluct_X * fluct_Y, axis=(1, 2))

    denominator = np.sqrt(fluct_X_sq * fluct_Y_sq)

    if np.any(denominator == 0):
        print("Attenzione: divisione per zero nella correlazione spaziale.")

    return numerator / denominator


def easy_metrics(X_ml, X_obs, name_suffix: str, DIR_ML_OUT: str):
    mae = np.nanmean(np.abs(X_ml - X_obs))
    rmse = np.sqrt(np.nanmean((X_ml - X_obs) ** 2))
    r_spatial = correlation_spatial(X_ml, X_obs)
    mean_r_spatial = np.nanmean(r_spatial)

    metrics = {
        "MAE [mm/day]": mae,
        "RMSE [mm/day]": rmse,
        "Mean Spatial Corr.": mean_r_spatial,
    }

    os.makedirs(DIR_ML_OUT, exist_ok=True)
    pkl_path = os.path.join(DIR_ML_OUT, f"metrics_experiment_{name_suffix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(metrics, f)

    df_metrics = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
    fig, ax = plt.subplots(figsize=(4, 1.5))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_metrics.values,
        colLabels=df_metrics.columns,
        loc="center",
        cellLoc="left",
    )
    table.scale(1, 1.5)
    plt.title(f"Metrics Summary - {name_suffix}", fontsize=10)
    plt.tight_layout()
    plt.show(block=True)

    return metrics


def metrics(X_ml, X_obs, name_suffix: str, DIR_ML_OUT: str):
    mae = np.nanmean(np.abs(X_ml - X_obs))
    rmse = np.sqrt(np.nanmean((X_ml - X_obs) ** 2))
    bias = np.nanmean(X_ml - X_obs)

    ss_res = np.nansum((X_obs - X_ml) ** 2)
    ss_tot = np.nansum((X_obs - np.nanmean(X_obs)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    r_spatial = correlation_spatial(X_ml, X_obs)
    mean_r_spatial = np.nanmean(r_spatial)

    r_temporal = correlation_temporal(X_ml, X_obs)
    mean_r_temporal = np.nanmean(r_temporal)

    metrics_dict = {
        "MAE [mm/day]": mae,
        "RMSE [mm/day]": rmse,
        "Bias [mm/day]": bias,
        "R²": r2,
        "Mean Spatial Corr.": mean_r_spatial,
        "Mean Temporal Corr.": mean_r_temporal,
    }

    os.makedirs(DIR_ML_OUT, exist_ok=True)
    pkl_path = os.path.join(DIR_ML_OUT, f"metrics_experiment_{name_suffix}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(metrics_dict, f)

    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df_metrics.values,
        colLabels=df_metrics.columns,
        loc="center",
        cellLoc="left",
    )
    table.scale(1, 1.5)
    fig.suptitle(f"Metrics Summary – {name_suffix}", fontsize=12, y=1.15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.75)
    plt.show(block=True)

    return metrics_dict
