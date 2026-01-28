from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_table_image_comparison(
    name: str,
    ml_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    higher_is_better: bool = True,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Simple table image comparing ML vs ERA5.
    Marks cells where ML is better with an arrow.

    - Rows = n_out (index)
    - Cols = n_in  (columns)
    """
    # align on common rows/cols
    common_rows = ml_df.index.intersection(era5_df.index)
    common_cols = ml_df.columns.intersection(era5_df.columns)
    ml_df = ml_df.loc[common_rows, common_cols]
    era5_df = era5_df.loc[common_rows, common_cols]

    table_text = []
    for i in ml_df.index:
        row = []
        for j in ml_df.columns:
            ml_val = ml_df.loc[i, j]
            era_val = era5_df.loc[i, j]
            if pd.isna(ml_val) or pd.isna(era_val):
                cell = ""
            else:
                improved = (ml_val > era_val) if higher_is_better else (ml_val < era_val)
                marker = " \u2190" if improved else ""
                cell = f"{float(ml_val):.3f}{marker}"
            row.append(cell)
        table_text.append(row)

    text_df = pd.DataFrame(table_text, index=ml_df.index, columns=ml_df.columns)

    fig, ax = plt.subplots(figsize=(1 + len(text_df.columns), 1 + len(text_df.index)))
    ax.axis("off")

    table = ax.table(
        cellText=text_df.values,
        rowLabels=[f"n_out={i}" for i in text_df.index],
        colLabels=[f"n_in={j}" for j in text_df.columns],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    ax.set_title(name, fontsize=14)
    plt.tight_layout()

    if show:
        plt.show()
    return fig, ax


def generate_table_image_comparison_perc(
    name: str,
    ml_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    higher_is_better: bool = True,
    show_arrow: bool = True,
    savepath: Optional[str] = None,
    dpi: int = 300,
    fontsize: int = 10,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Thesis-style table comparing ML vs ERA5 with (optional) percentage improvement.

    - Rows = n_out, Cols = n_in.
    - If ML improves vs ERA5: print two lines:
        '0.873 ←'
        '(+12.4%)'

    Parameters
    ----------
    higher_is_better:
        True for correlation; False for MAE/RMSE.
    savepath:
        If provided, saves the figure (PDF/PNG). Parent folders are created.
    show:
        If True, displays the figure.
    """

    # ---------- sanitize + align ----------
    def sanitize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx = pd.Index([str(x).strip() for x in df.index])
        cols = pd.Index([str(x).strip() for x in df.columns])
        df.index = pd.to_numeric(idx, errors="coerce")
        df.columns = pd.to_numeric(cols, errors="coerce")
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        return df.sort_index().sort_index(axis=1)

    ml_df = sanitize(ml_df)
    era5_df = sanitize(era5_df)

    common_rows = ml_df.index.intersection(era5_df.index)
    common_cols = ml_df.columns.intersection(era5_df.columns)
    ml_df = ml_df.loc[common_rows, common_cols]
    era5_df = era5_df.loc[common_rows, common_cols]

    # ---------- build cell texts ----------
    table_text = []
    col_max_chars = [0] * ml_df.shape[1]
    any_improved = False

    for i in ml_df.index:
        row = []
        for c, j in enumerate(ml_df.columns):
            ml_val = ml_df.loc[i, j]
            era_val = era5_df.loc[i, j]

            if pd.isna(ml_val) or pd.isna(era_val):
                cell = ""
            else:
                ml_val = float(ml_val)
                era_val = float(era_val)

                improved = (ml_val > era_val) if higher_is_better else (ml_val < era_val)
                if improved:
                    denom = abs(era_val) if era_val != 0 else np.nan
                    if np.isnan(denom):
                        cell = f"{ml_val:.3f}"
                    else:
                        delta = (ml_val - era_val) if higher_is_better else (era_val - ml_val)
                        perc = 100.0 * (delta / denom)
                        arrow = " \u2190" if show_arrow else ""
                        cell = f"{ml_val:.3f}{arrow}\n(+{perc:.1f}%)"
                        any_improved = True
                else:
                    cell = f"{ml_val:.3f}"

            row.append(cell)
            col_max_chars[c] = max(col_max_chars[c], len(cell.split("\n")[0]))
        table_text.append(row)

    text_df = pd.DataFrame(table_text, index=ml_df.index, columns=ml_df.columns)

    # ---------- adaptive column widths ----------
    per_char = 1.0
    min_unit = 10.0
    units = [max(min_unit, per_char * n) for n in col_max_chars]
    total_units = sum(units)
    col_widths = [u / total_units for u in units]

    # ---------- figure size ----------
    n_rows, n_cols = text_df.shape
    row_h = 0.46 if any_improved else 0.36
    fig_h = 1.0 + row_h * (n_rows + 1.4)
    fig_w = 0.8 + 0.20 * total_units

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    table = ax.table(
        cellText=text_df.values,
        rowLabels=[f"n_out={i}" for i in text_df.index],
        colLabels=[f"n_in={j}" for j in text_df.columns],
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.05, 2.3 if any_improved else 1.20)

    try:
        table.auto_set_column_width(col=list(range(n_cols)))
    except Exception:
        pass

    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        txt = cell.get_text()
        txt.set_ha("center")
        txt.set_va("center")
        try:
            txt.set_wrap(True)
            txt.set_linespacing(1.1)
        except Exception:
            pass

    ax.set_title(name, fontsize=13, loc="left", pad=6)
    plt.tight_layout(pad=0.6)

    if savepath:
        savepath_p = Path(savepath)
        savepath_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(savepath_p), bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    return fig, ax


def generate_table_image(
    name: str,
    ml_df: pd.DataFrame,
    savepath: Optional[str] = None,
    dpi: int = 300,
    fontsize: int = 10,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Renders a matrix DataFrame as a formatted numeric table (3 decimals).

    If savepath is provided, saves the figure.
    """
    table_text = [
        [f"{float(ml_df.loc[i, j]):.3f}" if pd.notna(ml_df.loc[i, j]) else "" for j in ml_df.columns]
        for i in ml_df.index
    ]
    text_df = pd.DataFrame(table_text, index=ml_df.index, columns=ml_df.columns)

    fig, ax = plt.subplots(
        figsize=(1 + len(text_df.columns), 1 + len(text_df.index)),
        dpi=dpi,
    )
    ax.axis("off")
    table = ax.table(
        cellText=text_df.values,
        rowLabels=[f"n_out={i}" for i in text_df.index],
        colLabels=[f"n_in={j}" for j in text_df.columns],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.2)

    ax.set_title(name, fontsize=14)
    plt.tight_layout()

    if savepath:
        savepath_p = Path(savepath)
        savepath_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(savepath_p), bbox_inches="tight", dpi=dpi)

    if show:
        plt.show()
    return fig, ax