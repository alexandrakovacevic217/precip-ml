# thesis_experiments/04_pcs_output_tables.py
#
# Create thesis-ready TABLES (PDF) for the PC-output approach, using precomputed CSV matrices.
# This script is lightweight: it reads CSVs and renders tables. No heavy data needed.
#
# Usage (from repo root):
#   python thesis_experiments/04_pcs_output_tables.py \
#     --matrices-dir results/tables/pc_matrices \
#     --out-dir results/figures/pc_tables
#
# It uses src/table_utils.py (the functions you just provided).

import argparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib as mpl
import pandas as pd

from src.table_utils import (
    generate_table_image_comparison_perc,
)


def _read_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.sort_index().sort_index(axis=1)


def _set_table_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "Liberation Serif"],
        "mathtext.fontset": "cm",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrices-dir",
        default="results/tables/pc_matrices",
        help="Folder containing the CSV matrices (ML + ERA5).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/figures/pc_tables",
        help="Where to save the generated table PDFs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saved figures.",
    )
    parser.add_argument(
        "--fontsize",
        type=int,
        default=10,
        help="Base font size in tables.",
    )
    args = parser.parse_args()

    matrices_dir = Path(args.matrices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_table_style()

    # --- ML vs CPC ---
    df_cov_tot = _read_matrix(matrices_dir / "cov_matrix_total_config8_true.csv")
    df_rmse_tot = _read_matrix(matrices_dir / "rmse_matrix_total_config8_true.csv")
    df_mae_tot = _read_matrix(matrices_dir / "mae_matrix_total_config8_true.csv")

    df_cov_rec = _read_matrix(matrices_dir / "cov_matrix_reconstructed_config8_true.csv")
    df_rmse_rec = _read_matrix(matrices_dir / "rmse_matrix_reconstructed_config8_true.csv")
    df_mae_rec = _read_matrix(matrices_dir / "mae_matrix_reconstructed_config8_true.csv")

    # --- ERA5 vs CPC (coast) ---
    df_cov_era5_tot = _read_matrix(matrices_dir / "cov_matrix_era5_total_coast.csv")
    df_rmse_era5_tot = _read_matrix(matrices_dir / "rmse_matrix_era5_total_coast.csv")
    df_mae_era5_tot = _read_matrix(matrices_dir / "mae_matrix_era5_total_coast.csv")

    df_cov_era5_rec = _read_matrix(matrices_dir / "cov_matrix_era5_reconstructed_coast.csv")
    df_rmse_era5_rec = _read_matrix(matrices_dir / "rmse_matrix_era5_reconstructed_coast.csv")
    df_mae_era5_rec = _read_matrix(matrices_dir / "mae_matrix_era5_reconstructed_coast.csv")

    # ============= TOTAL field tables =============
    generate_table_image_comparison_perc(
        name="MAE (mm/day): comparison with CPC total field",
        ml_df=df_mae_tot,
        era5_df=df_mae_era5_tot,
        higher_is_better=False,
        savepath=str(out_dir / "table_mae_total_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    generate_table_image_comparison_perc(
        name="RMSE (mm/day): comparison with CPC total field",
        ml_df=df_rmse_tot,
        era5_df=df_rmse_era5_tot,
        higher_is_better=False,
        savepath=str(out_dir / "table_rmse_total_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    generate_table_image_comparison_perc(
        name="Correlation: comparison with CPC total field",
        ml_df=df_cov_tot,
        era5_df=df_cov_era5_tot,
        higher_is_better=True,
        savepath=str(out_dir / "table_cov_total_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    # ============= RECONSTRUCTED field tables =============
    generate_table_image_comparison_perc(
        name="MAE (mm/day): comparison with CPC reconstructed field",
        ml_df=df_mae_rec,
        era5_df=df_mae_era5_rec,
        higher_is_better=False,
        savepath=str(out_dir / "table_mae_reconstructed_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    generate_table_image_comparison_perc(
        name="RMSE (mm/day): comparison with CPC reconstructed field",
        ml_df=df_rmse_rec,
        era5_df=df_rmse_era5_rec,
        higher_is_better=False,
        savepath=str(out_dir / "table_rmse_reconstructed_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    generate_table_image_comparison_perc(
        name="Correlation: comparison with CPC reconstructed field",
        ml_df=df_cov_rec,
        era5_df=df_cov_era5_rec,
        higher_is_better=True,
        savepath=str(out_dir / "table_cov_reconstructed_vs_era5_coast.pdf"),
        dpi=args.dpi,
        fontsize=args.fontsize,
        show=False,
    )

    print("Saved table PDFs to:", out_dir)


if __name__ == "__main__":
    main()