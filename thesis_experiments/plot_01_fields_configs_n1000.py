import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ORDER_9 = [
    "config_1", "config_7", "config_2", "config_3",
    "config_8", "config_5", "config_4", "config_6",
    "config_9",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/tables/01_fields_configs_n1000.csv")
    parser.add_argument("--out-dir", default="results/figures")
    parser.add_argument("--highlight", default="config_8")
    parser.add_argument("--fmt", choices=["png", "pdf", "both"], default="both")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Keep only n_input=1000 if present, otherwise assume file is already filtered
    if "n_input" in df.columns:
        df = df[df["n_input"] == 1000].copy()

    # Ensure expected columns exist
    needed = ["config_id", "mae_mm_day", "rmse_mm_day", "corr_spatial_mean"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")

    # Order configs like your thesis plot
    df["order"] = df["config_id"].apply(lambda c: ORDER_9.index(c) if c in ORDER_9 else 999)
    df = df.sort_values("order")

    configs = df["config_id"].tolist()

    # Font style (similar to yours)
    mpl.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
    })

    # Colors: highlight config_8 in green, others light gray
    colors = ["green" if c == args.highlight else "lightgray" for c in configs]

    def bar_panel(ax, values, ylabel, title):
        ax.bar(configs, values, color=colors, edgecolor="black", linewidth=0.5)
        vmin, vmax = float(min(values)), float(max(values))
        if vmax > vmin:
            ax.set_ylim(vmin - 0.05 * (vmax - vmin), vmax + 0.05 * (vmax - vmin))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    bar_panel(
        axs[0],
        df["mae_mm_day"].values,
        "MAE [mm/day]",
        "MAE per configurazione (N_input = 1000)",
    )
    bar_panel(
        axs[1],
        df["rmse_mm_day"].values,
        "RMSE [mm/day]",
        "RMSE per configurazione (N_input = 1000)",
    )
    bar_panel(
        axs[2],
        df["corr_spatial_mean"].values,
        "Mean Spatial Corr.",
        "Correlazione spaziale media (N_input = 1000)",
    )

    fig.tight_layout()

    base = out_dir / "01_fields_configs_n1000"
    if args.fmt in ("png", "both"):
        fig.savefig(f"{base}.png", dpi=250)
    if args.fmt in ("pdf", "both"):
        fig.savefig(f"{base}.pdf", dpi=300)

    plt.close(fig)
    print("Saved:", base.with_suffix(".png") if args.fmt in ("png", "both") else "", base.with_suffix(".pdf") if args.fmt in ("pdf", "both") else "")


if __name__ == "__main__":
    main()