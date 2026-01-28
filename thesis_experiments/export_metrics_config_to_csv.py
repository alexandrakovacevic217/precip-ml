import argparse
import pickle
import re
from pathlib import Path

import pandas as pd


CFG_RE = re.compile(r"^(config_\d+)_([0-9]+)pcs$")


def parse_config_full(name: str):
    """
    Examples:
      config_8_1000pcs -> ("config_8", 1000)
      config_1_80pcs   -> ("config_1", 80)
      config_7         -> ("config_7", None)  # if no suffix
    """
    m = CFG_RE.match(name)
    if not m:
        return name, None
    return m.group(1), int(m.group(2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-pkl", required=True, help="Path to cumulative_metrics_config.pkl")
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()

    in_path = Path(args.in_pkl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "rb") as f:
        cumulative_metrics = pickle.load(f)

    rows = []
    for d in cumulative_metrics:
        cfg_full = d["Config"]
        cfg_id, n_input = parse_config_full(cfg_full)

        rows.append(
            {
                "config_id": cfg_id,                 # e.g. config_8
                "config_full": cfg_full,             # e.g. config_8_1000pcs
                "n_input": n_input,                  # 80/400/1000 or None
                "mae_mm_day": d.get("MAE [mm/day]"),
                "rmse_mm_day": d.get("RMSE [mm/day]"),
                "corr_spatial_mean": d.get("Mean Spatial Corr."),
                "corr_temporal_mean": d.get("Mean Temporal Corr."),
                "corr_temporal_full": d.get("Full Temporal Corr."),
            }
        )

    df = pd.DataFrame(rows)

    # 0) MASTER: tutto quello che c’è nel PKL (tutte le config / pcs disponibili)
    master_path = out_dir / "metrics_config_master.csv"
    df.to_csv(master_path, index=False)

    # 1) Plot #1 (tesi): tutte le configurazioni con 1000 pcs
    df_1000 = df[df["n_input"] == 1000].copy()
    out_01 = out_dir / "01_fields_configs_n1000.csv"
    df_1000.to_csv(out_01, index=False)

    # 2) Plot #2 (tesi): subset config 1/2/5/8 per 80/400/1000 pcs
    subset_cfg = ["config_1", "config_2", "config_5", "config_8"]
    df_subset = df[
        (df["config_id"].isin(subset_cfg))
        & (df["n_input"].isin([80, 400, 1000]))
    ].copy()
    out_02 = out_dir / "02_fields_configs_vs_ninput_subset.csv"
    df_subset.to_csv(out_02, index=False)

    print("Wrote:")
    print(" -", master_path)
    print(" -", out_01)
    print(" -", out_02)


if __name__ == "__main__":
    main()
