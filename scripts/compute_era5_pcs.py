import argparse
import pickle
from pathlib import Path

import xarray as xr

from src.eof import get_only_pcs_split_weight


GROUP_A = [
    "msl",
    "ta250", "ta500", "ta700", "ta850",
    "q250", "q500", "q700", "q850",
    "u250", "u500", "u700", "u850",
    "v250", "v500", "v700", "v850",
]
GROUP_B = ["w250", "w500", "w700", "w850"]


def infer_time_dim(da):
    if "valid_time" in da.dims:
        return "valid_time"
    if "time" in da.dims:
        return "time"
    raise ValueError("Time dimension not found in da.dims={}".format(da.dims))


def select_dataarray(ds, varname):
    # Caso normale: variabile uguale al varname (es. ta850)
    if varname in ds.data_vars:
        return ds[varname]

    # Caso w*: spesso la variabile nel file si chiama solo "w"
    if varname.startswith("w") and "w" in ds.data_vars:
        return ds["w"]

    raise KeyError("Variabile '{}' non trovata. data_vars={}".format(varname, 
list(ds.data_vars)))


def main():
    p = argparse.ArgumentParser(description="Compute EOF PCs for ERA5 variables (requires local 
ERA5 netcdf files).")
    p.add_argument("--era5-dir", required=True, help="Directory containing ERA5 netcdf files.")
    p.add_argument("--out-dir", required=True, help="Directory to save pickle outputs.")
    p.add_argument("--group", default="all", choices=["A", "B", "all"], help="Which variable 
group to compute.")
    p.add_argument("--pattern", required=True, help="Filename pattern with {var}, e.g. 
ERA5_{var}_...nc")
    p.add_argument("--start", default="1979-01-01", help="Full-period start date for slicing.")
    p.add_argument("--end", default="2020-12-31", help="Full-period end date for slicing.")
    p.add_argument("--train-start", default="1979-01-01", help="Training start date.")
    p.add_argument("--train-end", default="2012-12-31", help="Training end date.")
    p.add_argument("--test-start", default="2013-01-01", help="Test start date.")
    p.add_argument("--N", type=int, default=8000, help="Number of EOF/PC modes.")
    p.add_argument("--zpc", action="store_true", help="Standardize PCs using training std.")
    p.add_argument("--verbose", action="store_true", help="Verbose output.")
    args = p.parse_args()

    era5_dir = Path(args.era5_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.group == "A":
        variables = GROUP_A
    elif args.group == "B":
        variables = GROUP_B
    else:
        variables = GROUP_A + GROUP_B

    for var in variables:
        fpath = era5_dir / args.pattern.format(var=var)
        if not fpath.exists():
            print("[SKIP] Missing file:", fpath)
            continue

        if args.verbose:
            print("\n[LOAD]", var, "->", fpath)

        ds = xr.open_dataset(fpath)
        da = select_dataarray(ds, var)

        time_dim = infer_time_dim(da)
        da_sel = da.sel({time_dim: slice(args.start, args.end)})

        if args.verbose:
            print("[DATA] dims={} shape={} time_dim={}".format(da_sel.dims, da_sel.shape, 
time_dim))

        zff = get_only_pcs_split_weight(
            da_sel,
            N=args.N,
            train_start=args.train_start,
            train_end=args.train_end,
            test_start=args.test_start,
            zPC=args.zpc,
            verbose=args.verbose,
        )

        payload = {
            var: zff,
            "time": da_sel[time_dim],
            "lat": ds.get("lat", ds.get("latitude", None)),
            "lon": ds.get("lon", ds.get("longitude", None)),
        }

        out_name = "ERA5_PCs_{}_{}-{}_N{}_split_weight.pickle".format(var, args.start[:4], 
args.end[:4], args.N)
        out_path = out_dir / out_name

        with open(out_path, "wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)

        print("[OK] Saved:", out_path)


if __name__ == "__main__":
    main()
