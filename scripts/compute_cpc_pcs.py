import argparse
import pickle
from pathlib import Path

import xarray as xr

from src.eof import get_only_pcs


def main():
    p = argparse.ArgumentParser(description="Compute CPC PCs (classic EOF, no split). Requires 
local CPC netcdf.")
    p.add_argument("--cpc-file", required=True, help="Path to CPC netcdf (e.g. 
CPC_pr_filled_1979-2021.nc)")
    p.add_argument("--var", default="pr", help="Variable name inside dataset (default: pr)")
    p.add_argument("--out-dir", required=True, help="Output directory for pickle.")
    p.add_argument("--start", default="1979-01-01T12:00:00.000000000", help="Start date (CPC 
time format ok).")
    p.add_argument("--end", default="2020-12-31T12:00:00.000000000", help="End date (CPC time 
format ok).")
    p.add_argument("--N", type=int, default=5, help="Number of EOF/PC modes.")
    p.add_argument("--zpc", action="store_true", help="Standardize PCs (your method).")
    p.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = p.parse_args()

    cpc_file = Path(args.cpc_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(cpc_file)

    if args.var not in ds.data_vars:
        raise KeyError("Variable '{}' not found. data_vars={}".format(args.var, 
list(ds.data_vars)))

    da = ds[args.var]

    # CPC di solito ha dim 'time'
    if "time" not in da.dims and "valid_time" in da.dims:
        time_dim = "valid_time"
    elif "time" in da.dims:
        time_dim = "time"
    else:
        raise ValueError("Time dimension not found in da.dims={}".format(da.dims))

    da_sel = da.sel({time_dim: slice(args.start, args.end)})

    if args.verbose:
        print("[CPC] Loaded:", cpc_file)
        print("[CPC] var={} dims={} shape={}".format(args.var, da_sel.dims, da_sel.shape))

    zff = get_only_pcs(da_sel, N=args.N, verbose=args.verbose, zPC=args.zpc)

    payload = {
        args.var: zff,
        "time": da_sel[time_dim],
        "lat": ds.get("lat", ds.get("latitude", None)),
        "lon": ds.get("lon", ds.get("longitude", None)),
    }

    out_name = "CPC_PCs_{}_{}-{}_N{}_classic.pickle".format(
        args.var, args.start[:4], args.end[:4], args.N
    )
    out_path = out_dir / out_name

    with open(out_path, "wb") as f:
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)

    print("[OK] Saved:", out_path)


if __name__ == "__main__":
    main()
