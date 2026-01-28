import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


CFG_ORDER = ["config_1", "config_2", "config_5", "config_8"]
NINPUT_ORDER = [80, 400, 1000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/tables/02_fields_configs_vs_ninput_subset.csv")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--fmt", choices=["png", "pdf", "both"], default="both")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    needed = ["config_id", "n_input", "mae_mm_day", "rmse_mm_day", "corr_spatial_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")

    # Filter to the exact subset used in the thesis plot
    df = df[df["config_id"].isin(CFG_ORDER) & df["n_input"].isin(NINPUT_ORDER)].copy()

    # Make sure we have a complete table (some combos may be missing; handle gracefully)
    # Pivot: rows=config, cols=n_input
    mae = df.pivot_table(index="config_id", columns="n_input", values="mae_mm_day")
    rmse = df.pivot_table(index="config_id", columns="n_input", values="rmse_mm_day")
    corr = df.pivot_table(index="config_id", columns="n_input", values="corr_spatial_mean")

    # Reindex to enforce consistent order
    mae = mae.reindex(index=CFG_ORDER, columns=NINPUT_ORDER)
    rmse = rmse.reindex(index=CFG_ORDER, columns=NINPUT_ORDER)
    corr = corr.reindex(index=CFG_ORDER, columns=NINPUT_ORDER)

    # Font style (similar to yours)
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    # Colors (keep it simple and readable)
    color_map = {
        80: "tab:blue",
        400: "tab:orange",
        1000: "tab:green",
    }

    x = np.arange(len(CFG_ORDER))
    width = 0.25

    fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    def set_ylim(ax, values, margin=0.02):
        values = [v for v in values if np.isfinite(v)]
        if not values:
            return
        vmin, vmax = min(values), max(values)
        if vmax > vmin:
            ax.set_ylim(vmin - margin * (vmax - vmin), vmax + margin * (vmax - vmin))

    # --- MAE ---
    axs[0].bar(x - width, mae[80].values, width=width, label="80 PCs", color=color_map[80], edgecolor="black", linewidth=0.4)
    axs[0].bar(x,         mae[400].values, width=width, label="400 PCs", color=color_map[400], edgecolor="black", linewidth=0.4)
    axs[0].bar(x + width, mae[1000].values, width=width, label="1000 PCs", color=color_map[1000], edgecolor="black", linewidth=0.4)
    axs[0].set_title("MAE [mm/day]")
    axs[0].set_ylabel("Value")
    axs[0].legend(loc="upper right", frameon=False)
    set_ylim(axs[0], list(mae[80].values) + list(mae[400].values) + list(mae[1000].values))

    # --- RMSE ---
    axs[1].bar(x - width, rmse[80].values, width=width, label="80 PCs", color=color_map[80], edgecolor="black", linewidth=0.4)
    axs[1].bar(x,         rmse[400].values, width=width, label="400 PCs", color=color_map[400], edgecolor="black", linewidth=0.4)
    axs[1].bar(x + width, rmse[1000].values, width=width, label="1000 PCs", color=color_map[1000], edgecolor="black", linewidth=0.4)
    axs[1].set_title("RMSE [mm/day]")
    axs[1].set_ylabel("Value")
    axs[1].legend(loc="upper right", frameon=False)
    set_ylim(axs[1], list(rmse[80].values) + list(rmse[400].values) + list(rmse[1000].values))

    # --- Spatial Corr ---
    axs[2].bar(x - width, corr[80].values, width=width, label="80 PCs", color=color_map[80], edgecolor="black", linewidth=0.4)
    axs[2].bar(x,         corr[400].values, width=width, label="400 PCs", color=color_map[400], edgecolor="black", linewidth=0.4)
    axs[2].bar(x + width, corr[1000].values, width=width, label="1000 PCs", color=color_map[1000], edgecolor="black", linewidth=0.4)
    axs[2].set_title("Mean Spatial Corr.")
    axs[2].set_ylabel("Value")
    axs[2].legend(loc="upper left", frameon=False)
    set_ylim(axs[2], list(corr[80].values) + list(corr[400].values) + list(corr[1000].values))

    axs[2].set_xticks(x)
    axs[2].set_xticklabels(["config 1", "config 2", "config 5", "config 8"])

    fig.tight_layout()

    base = out_dir / "02_fields_configs_vs_ninput_subset"
    if args.fmt in ("png", "both"):
        fig.savefig(f"{base}.png", dpi=250)
    if args.fmt in ("pdf", "both"):
        fig.savefig(f"{base}.pdf", dpi=300)

    plt.close(fig)
    print("Saved:", base.with_suffix(".png") if args.fmt in ("png", "both") else "", base.with_suffix(".pdf") if args.fmt in ("pdf", "both") else "")


if __name__ == "__main__":
    main()