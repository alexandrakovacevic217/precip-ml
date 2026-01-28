import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def default_ticks(n_inputs):
    """
    Mimic your old logic:
      - show all ticks >= 120
      - for smaller values, show only multiples of 10
    """
    ticks = []
    for x in n_inputs:
        if x >= 120 or (x % 10 == 0):
            ticks.append(x)
    return ticks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="results/tables/03_fields_tuning_ninput_config8.csv",
        help="Input CSV produced by the export script.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/figures",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--fmt",
        choices=["png", "pdf", "both"],
        default="both",
        help="Output format.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path).sort_values("n_input")

    required = ["n_input", "mae_mm_day", "rmse_mm_day", "corr_spatial_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {csv_path}: {missing}")

    # Style similar to your thesis plots
    mpl.rcParams.update(
        {
            "text.usetex": False,
            "mathtext.fontset": "cm",
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
        }
    )

    n_inputs = df["n_input"].tolist()
    xticks_to_show = default_ticks(n_inputs)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(df["n_input"], df["mae_mm_day"], marker="o")
    axs[0].set_title("MAE")
    axs[0].set_ylabel("MAE [mm/day]")
    axs[0].set_xticks(xticks_to_show)

    axs[1].plot(df["n_input"], df["rmse_mm_day"], marker="o")
    axs[1].set_title("RMSE")
    axs[1].set_ylabel("RMSE [mm/day]")
    axs[1].set_xticks(xticks_to_show)

    axs[2].plot(df["n_input"], df["corr_spatial_mean"], marker="o")
    axs[2].set_title("Mean Spatial Correlation")
    axs[2].set_xlabel("Number of PCs")
    axs[2].set_ylabel("Mean spatial corr.")
    axs[2].set_xticks(xticks_to_show)

    plt.setp(axs[2].get_xticklabels(), rotation=45)
    plt.tight_layout()

    base = out_dir / "03_fields_tuning_ninput_config8"
    if args.fmt in ("png", "both"):
        fig.savefig(f"{base}.png", dpi=250)
    if args.fmt in ("pdf", "both"):
        fig.savefig(f"{base}.pdf", dpi=300)

    plt.close(fig)
    print("Saved:", base)


if __name__ == "__main__":
    main()