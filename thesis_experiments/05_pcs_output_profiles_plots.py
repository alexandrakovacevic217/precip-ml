# thesis_experiments/05_pcs_output_profiles_plots.py
#
# Make "profiles 3x2" plots from precomputed CSV matrices (PC-output approach).
# - TOTAL metrics plot (optionally with constant ERA5 baselines)
# - RECONSTRUCTED metrics plot (ERA5 varies with n_out, read from matrices)
#
# Usage:
#   python thesis_experiments/05_pcs_output_profiles_plots.py \
#     --matrices-dir results/tables/pc_matrices \
#     --out-dir results/figures/pc_profiles
#
# Optional (TOTAL only): use constant ERA5 baselines like in thesis
#   --use-era5-constants --era5-mae 1.20 --era5-rmse 3.27 --era5-corr 0.66

import argparse
import sys
from pathlib import Path

# make repo root importable so `from src...` works even without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------- IO helpers -----------------------------
def read_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_numeric(pd.Index(df.index).astype(str).str.strip(), errors="coerce")
    df.columns = pd.to_numeric(pd.Index(df.columns).astype(str).str.strip(), errors="coerce")
    df = df.sort_index().sort_index(axis=1)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df


def style_global():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })


def style_axes(ax):
    ax.grid(True, which="both", linewidth=0.4, alpha=0.5)
    ax.tick_params(labelsize=10)
    for s in ax.spines.values():
        s.set_alpha(0.4)


def safe_present(requested, available_set):
    return [v for v in requested if v in available_set]


def pick_extreme_idx(y: np.ndarray, metric_key: str) -> int:
    if metric_key.lower() in ("corr", "correlation", "r", "pearson"):
        return int(np.nanargmax(y))
    return int(np.nanargmin(y))


# ----------------------------- TOTAL plot (ERA5 constant line) -----------------------------
def plot_metric_row_total(
    ax_left,
    ax_right,
    df_raw: pd.DataFrame,
    y_label: str,
    era5_value: float | None,
    n_in_trend,
    n_in_best,
    n_out_trend,
    n_out_best,
    metric_key: str,
):
    df = df_raw.copy()
    n_out_vals = df.index.to_numpy()
    n_in_vals = df.columns.to_numpy()

    n_in_trend = safe_present(n_in_trend, set(n_in_vals))
    n_out_trend = safe_present(n_out_trend, set(n_out_vals))
    n_in_best_ok = (n_in_best in set(n_in_vals))
    n_out_best_ok = (n_out_best in set(n_out_vals))

    # palette (come nel tuo codice)
    COLORS_LEFT = {30:"#1f77b4", 200:"#2ca02c", 1800:"#9467bd", "best":"#d62728"}
    COLORS_RIGHT = {30:"#1f77b4", 200:"#2ca02c", 1800:"#9467bd", "best":"#d62728"}
    COLOR_ERA = "#4d4d4d"
    LW_TREND, LW_BEST, LW_ERA, MS_EXT = 1.4, 1.8, 0.9, 5

    ymins, ymaxs = [], []

    # LEFT: fixed n_in, x=n_out (columns)
    for ni in n_in_trend:
        y = df[ni].to_numpy()
        ax_left.plot(n_out_vals, y, lw=LW_TREND, color=COLORS_LEFT.get(ni), label=f"n_in={ni}")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    if n_in_best_ok:
        y = df[n_in_best].to_numpy()
        ax_left.plot(n_out_vals, y, lw=LW_BEST, color=COLORS_LEFT["best"], label=f"n_in={n_in_best} (best)")
        j = pick_extreme_idx(y, metric_key)
        ax_left.plot(n_out_vals[j], y[j], marker="o", ms=MS_EXT, color="black")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    if era5_value is not None:
        ax_left.axhline(era5_value, ls="--", lw=LW_ERA, color=COLOR_ERA)
        ymins.append(era5_value); ymaxs.append(era5_value)

    ax_left.set_xlabel("$n_{out}$", fontsize=11)
    ax_left.set_ylabel(y_label, fontsize=11)
    style_axes(ax_left)
    ax_left.margins(x=0.02)

    # RIGHT: fixed n_out, x=n_in (rows)
    for no in n_out_trend:
        y = df.loc[no, :].to_numpy()
        ax_right.plot(n_in_vals, y, lw=LW_TREND, color=COLORS_RIGHT.get(no), label=f"n_out={no}")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    if n_out_best_ok:
        y = df.loc[n_out_best, :].to_numpy()
        ax_right.plot(n_in_vals, y, lw=LW_BEST, color=COLORS_RIGHT["best"], label=f"n_out={n_out_best} (best)")
        i = pick_extreme_idx(y, metric_key)
        ax_right.plot(n_in_vals[i], y[i], marker="o", ms=MS_EXT, color="black")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    if era5_value is not None:
        ax_right.axhline(era5_value, ls="--", lw=LW_ERA, color=COLOR_ERA)
        ymins.append(era5_value); ymaxs.append(era5_value)

    ax_right.set_xlabel("$n_{in}$", fontsize=11)
    ax_right.set_ylabel(y_label, fontsize=11)
    style_axes(ax_right)
    ax_right.margins(x=0.02)

    # shared y-lims
    ylo, yhi = np.nanmin(ymins), np.nanmax(ymaxs)
    pad = 0.05 * (yhi - ylo if (yhi - ylo) > 0 else 1.0)
    ax_left.set_ylim(ylo - pad, yhi + pad)
    ax_right.set_ylim(ylo - pad, yhi + pad)

    # legend handles
    leg_left = [Line2D([0],[0], color=COLORS_LEFT.get(v, "#777"), lw=LW_TREND, label=f"n_in={v}") for v in n_in_trend]
    if n_in_best_ok:
        leg_left += [Line2D([0],[0], color=COLORS_LEFT["best"], lw=LW_BEST, label=f"n_in={n_in_best} (best)")]
    if era5_value is not None:
        leg_left += [Line2D([0],[0], color=COLOR_ERA, lw=LW_ERA, ls="--", label="ERA5")]

    leg_right = [Line2D([0],[0], color=COLORS_RIGHT.get(v, "#777"), lw=LW_TREND, label=f"n_out={v}") for v in n_out_trend]
    if n_out_best_ok:
        leg_right += [Line2D([0],[0], color=COLORS_RIGHT["best"], lw=LW_BEST, label=f"n_out={n_out_best} (best)")]
    if era5_value is not None:
        leg_right += [Line2D([0],[0], color=COLOR_ERA, lw=LW_ERA, ls="--", label="ERA5")]

    return leg_left, leg_right


def make_profiles_total(
    df_mae_tot, df_rmse_tot, df_cov_tot,
    outpath: Path,
    n_in_trend, n_in_best, n_out_trend, n_out_best,
    era5_constants: dict | None,
):
    fig = plt.figure(figsize=(14, 13.5))
    gs = plt.GridSpec(3, 2, hspace=0.34, wspace=0.20)

    ax1L, ax1R = plt.subplot(gs[0,0]), plt.subplot(gs[0,1])
    legL1, legR1 = plot_metric_row_total(
        ax1L, ax1R, df_mae_tot, "MAE (mm/day)",
        None if era5_constants is None else era5_constants.get("mae"),
        n_in_trend, n_in_best, n_out_trend, n_out_best, "mae"
    )

    ax2L, ax2R = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
    _, _ = plot_metric_row_total(
        ax2L, ax2R, df_rmse_tot, "RMSE (mm/day)",
        None if era5_constants is None else era5_constants.get("rmse"),
        n_in_trend, n_in_best, n_out_trend, n_out_best, "rmse"
    )

    ax3L, ax3R = plt.subplot(gs[2,0]), plt.subplot(gs[2,1])
    _, _ = plot_metric_row_total(
        ax3L, ax3R, df_cov_tot, "Correlation",
        None if era5_constants is None else era5_constants.get("corr"),
        n_in_trend, n_in_best, n_out_trend, n_out_best, "corr"
    )

    fig.subplots_adjust(top=0.78)
    fig.text(0.255, 0.92, "Profiles at fixed $n_{in}$",  ha="center", va="center")
    fig.text(0.745, 0.92, "Profiles at fixed $n_{out}$", ha="center", va="center")

    legend_left = fig.legend(handles=legL1, loc="upper center",
                             bbox_to_anchor=(0.255, 0.88), frameon=False, ncol=4)
    legend_right = fig.legend(handles=legR1, loc="upper center",
                              bbox_to_anchor=(0.745, 0.88), frameon=False, ncol=4)
    fig.add_artist(legend_left)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show(block=True)


# ----------------------------- RECONSTRUCTED plot (ERA5 depends on n_out) -----------------------------
def plot_metric_row_reconstructed(
    axL,
    axR,
    df_metric: pd.DataFrame,
    df_era5: pd.DataFrame,
    y_label: str,
    metric_key: str,
    n_in_trend,
    n_in_best,
    n_out_trend,
    n_out_best,
):
    # Okabe–Ito palette (come nella tua seconda parte)
    COL_LEFT  = {30:"#0072B2", 1800:"#009E73", "best":"#D55E00"}
    COL_RIGHT = {120:"#0072B2", 1800:"#009E73", "best":"#D55E00"}
    COL_ERA   = "#4d4d4d"
    LW_TREND, LW_BEST, LW_ERA, MS_EXT = 1.4, 1.8, 0.9, 5

    df = df_metric.copy()
    dfE = df_era5.copy()

    x_out = df.index.to_numpy()
    x_in = df.columns.to_numpy()

    n_in_tr = safe_present(n_in_trend, set(x_in))
    n_out_tr = safe_present(n_out_trend, set(x_out))
    n_in_best_ok = (n_in_best in set(x_in))
    n_out_best_ok = (n_out_best in set(x_out))

    ymins, ymaxs = [], []

    # LEFT: fixed n_in (x=n_out)
    for ni in n_in_tr:
        y = df[ni].to_numpy()
        axL.plot(x_out, y, lw=LW_TREND, color=COL_LEFT.get(ni, "#777"), label=f"n_in={ni}")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    if n_in_best_ok:
        y = df[n_in_best].to_numpy()
        axL.plot(x_out, y, lw=LW_BEST, color=COL_LEFT["best"], label=f"n_in={n_in_best} (best)")
        j = pick_extreme_idx(y, metric_key)
        axL.plot(x_out[j], y[j], marker="o", ms=MS_EXT, color="black")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

    # ERA5 curve (varies with n_out): use first column (replicated across n_in)
    era_curve = dfE.iloc[:, 0].to_numpy()
    axL.plot(x_out, era_curve, lw=LW_ERA, ls="--", color=COL_ERA, label="ERA5")
    ymins.append(np.nanmin(era_curve)); ymaxs.append(np.nanmax(era_curve))

    axL.set_xlabel("$n_{out}$", fontsize=11)
    axL.set_ylabel(y_label, fontsize=11)
    axL.margins(x=0.02)
    style_axes(axL)

    # RIGHT: fixed n_out (x=n_in)
    for no in n_out_tr:
        y = df.loc[no, :].to_numpy()
        axR.plot(x_in, y, lw=LW_TREND, color=COL_RIGHT.get(no, "#777"), label=f"n_out={no}")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

        # ERA5 horizontal line for each chosen n_out (same color)
        era_val = float(dfE.loc[no, :].iloc[0])
        axR.axhline(era_val, lw=LW_ERA, ls="--", color=COL_RIGHT.get(no, "#777"))

    if n_out_best_ok:
        y = df.loc[n_out_best, :].to_numpy()
        axR.plot(x_in, y, lw=LW_BEST, color=COL_RIGHT["best"], label=f"n_out={n_out_best} (best)")
        i = pick_extreme_idx(y, metric_key)
        axR.plot(x_in[i], y[i], marker="o", ms=MS_EXT, color="black")
        ymins.append(np.nanmin(y)); ymaxs.append(np.nanmax(y))

        era_val = float(dfE.loc[n_out_best, :].iloc[0])
        axR.axhline(era_val, lw=LW_ERA, ls="--", color=COL_RIGHT["best"])

    axR.set_xlabel("$n_{in}$", fontsize=11)
    axR.set_ylabel(y_label, fontsize=11)
    axR.margins(x=0.02)
    style_axes(axR)

    ylo, yhi = np.nanmin(ymins), np.nanmax(ymaxs)
    pad = 0.05 * max(yhi - ylo, 1e-9)
    axL.set_ylim(ylo - pad, yhi + pad)
    axR.set_ylim(ylo - pad, yhi + pad)

    legL = [
        Line2D([0],[0], color=COL_LEFT.get(n_in_tr[0], "#777"), lw=LW_TREND, label=f"n_in={n_in_tr[0]}") if len(n_in_tr) > 0 else None,
        Line2D([0],[0], color=COL_LEFT.get(n_in_tr[1], "#777"), lw=LW_TREND, label=f"n_in={n_in_tr[1]}") if len(n_in_tr) > 1 else None,
        Line2D([0],[0], color=COL_LEFT["best"], lw=LW_BEST, label=f"n_in={n_in_best} (best)") if n_in_best_ok else None,
        Line2D([0],[0], color=COL_ERA, lw=LW_ERA, ls="--", label="ERA5"),
    ]
    legL = [h for h in legL if h is not None]

    legR = [
        Line2D([0],[0], color=COL_RIGHT.get(n_out_tr[0], "#777"), lw=LW_TREND, label=f"n_out={n_out_tr[0]}") if len(n_out_tr) > 0 else None,
        Line2D([0],[0], color=COL_RIGHT.get(n_out_tr[1], "#777"), lw=LW_TREND, label=f"n_out={n_out_tr[1]}") if len(n_out_tr) > 1 else None,
        Line2D([0],[0], color=COL_RIGHT["best"], lw=LW_BEST, label=f"n_out={n_out_best} (best)") if n_out_best_ok else None,
    ]
    legR = [h for h in legR if h is not None]

    return legL, legR


def make_profiles_reconstructed(
    df_mae_rec, df_rmse_rec, df_cov_rec,
    df_mae_era5_rec, df_rmse_era5_rec, df_cov_era5_rec,
    outpath: Path,
    n_in_trend, n_in_best, n_out_trend, n_out_best,
):
    fig = plt.figure(figsize=(14, 13))
    gs = plt.GridSpec(3, 2, hspace=0.34, wspace=0.20)

    ax1L, ax1R = plt.subplot(gs[0,0]), plt.subplot(gs[0,1])
    legL1, legR1 = plot_metric_row_reconstructed(
        ax1L, ax1R, df_mae_rec, df_mae_era5_rec, "MAE (mm/day)", "mae",
        n_in_trend, n_in_best, n_out_trend, n_out_best
    )

    ax2L, ax2R = plt.subplot(gs[1,0]), plt.subplot(gs[1,1])
    _, _ = plot_metric_row_reconstructed(
        ax2L, ax2R, df_rmse_rec, df_rmse_era5_rec, "RMSE (mm/day)", "rmse",
        n_in_trend, n_in_best, n_out_trend, n_out_best
    )

    ax3L, ax3R = plt.subplot(gs[2,0]), plt.subplot(gs[2,1])
    _, _ = plot_metric_row_reconstructed(
        ax3L, ax3R, df_cov_rec, df_cov_era5_rec, "Correlation", "corr",
        n_in_trend, n_in_best, n_out_trend, n_out_best
    )

    fig.subplots_adjust(top=0.80, left=0.07, right=0.98)
    fig.text(0.255, 0.975, "Profiles at fixed $n_{in}$",  ha="center", va="center")
    fig.text(0.745, 0.975, "Profiles at fixed $n_{out}$", ha="center", va="center")

    legend_left = fig.legend(handles=legL1, loc="upper center",
                             bbox_to_anchor=(0.255, 0.91), frameon=False, ncol=4)
    legend_right = fig.legend(handles=legR1, loc="upper center",
                              bbox_to_anchor=(0.745, 0.91), frameon=False, ncol=3)
    fig.add_artist(legend_left)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show(block=True)


# ----------------------------- main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices-dir", default="results/tables/pc_matrices")
    parser.add_argument("--out-dir", default="results/figures/pc_profiles")

    # TOTAL profiles selection
    parser.add_argument("--total-n-in-trend", nargs="+", type=int, default=[30, 200, 1800])
    parser.add_argument("--total-n-in-best", type=int, default=400)
    parser.add_argument("--total-n-out-trend", nargs="+", type=int, default=[30, 200, 1800])
    parser.add_argument("--total-n-out-best", type=int, default=50)

    # RECON profiles selection (your second block)
    parser.add_argument("--rec-n-in-trend", nargs="+", type=int, default=[30, 1800])
    parser.add_argument("--rec-n-in-best", type=int, default=400)
    parser.add_argument("--rec-n-out-trend", nargs="+", type=int, default=[120, 1800])
    parser.add_argument("--rec-n-out-best", type=int, default=10)

    # optional TOTAL era5 constants
    parser.add_argument("--use-era5-constants", action="store_true")
    parser.add_argument("--era5-mae", type=float, default=1.20)
    parser.add_argument("--era5-rmse", type=float, default=3.27)
    parser.add_argument("--era5-corr", type=float, default=0.66)

    args = parser.parse_args()

    style_global()

    matrices_dir = Path(args.matrices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load matrices
    df_mae_tot = read_matrix(matrices_dir / "mae_matrix_total_config8_true.csv")
    df_rmse_tot = read_matrix(matrices_dir / "rmse_matrix_total_config8_true.csv")
    df_cov_tot = read_matrix(matrices_dir / "cov_matrix_total_config8_true.csv")

    df_mae_rec = read_matrix(matrices_dir / "mae_matrix_reconstructed_config8_true.csv")
    df_rmse_rec = read_matrix(matrices_dir / "rmse_matrix_reconstructed_config8_true.csv")
    df_cov_rec = read_matrix(matrices_dir / "cov_matrix_reconstructed_config8_true.csv")

    df_mae_era5_rec = read_matrix(matrices_dir / "mae_matrix_era5_reconstructed_coast.csv")
    df_rmse_era5_rec = read_matrix(matrices_dir / "rmse_matrix_era5_reconstructed_coast.csv")
    df_cov_era5_rec = read_matrix(matrices_dir / "cov_matrix_era5_reconstructed_coast.csv")

    # TOTAL: either constant era5 baselines or None
    era5_constants = None
    if args.use_era5_constants:
        era5_constants = {"mae": args.era5_mae, "rmse": args.era5_rmse, "corr": args.era5_corr}

    make_profiles_total(
        df_mae_tot, df_rmse_tot, df_cov_tot,
        outpath=out_dir / "metrics_profiles_3x2_total.pdf",
        n_in_trend=args.total_n_in_trend,
        n_in_best=args.total_n_in_best,
        n_out_trend=args.total_n_out_trend,
        n_out_best=args.total_n_out_best,
        era5_constants=era5_constants,
    )

    make_profiles_reconstructed(
        df_mae_rec, df_rmse_rec, df_cov_rec,
        df_mae_era5_rec, df_rmse_era5_rec, df_cov_era5_rec,
        outpath=out_dir / "metrics_profiles_3x2_reconstructed.pdf",
        n_in_trend=args.rec_n_in_trend,
        n_in_best=args.rec_n_in_best,
        n_out_trend=args.rec_n_out_trend,
        n_out_best=args.rec_n_out_best,
    )

    print("Saved plots to:", out_dir)


if __name__ == "__main__":
    main()