# src/eof.py
import numpy as np
import xeofs.xarray as xe


def get_only_pcs_split_weight(
    xrVar,
    N=None,
    train_start="1979-01-01",
    train_end="2012-12-31",
    test_start="2013-01-01",
    zPC=True,
    verbose=False
):
    """
    Fit EOF sul periodo training con pesi cos(lat), poi proietta l'intero dataset (train + test)
    sulle EOF del training.

    Ritorna un dict con:
      - PC: (N, T_all)
      - fldmean: (lat, lon) media del training
      - ExpVar: (N,) explained variance ratio
      - std_train: (N,) oppure None
    """

    # 1) Rimuovi eventuale dimensione verticale
    for d in list(xrVar.dims):
        if d in ("plev", "pressure_level", "level"):
            xrVar = xrVar.squeeze(d, drop=True)
            break

    # 2) Identifica dimensione tempo
    if "valid_time" in xrVar.dims:
        time_dim = "valid_time"
    elif "time" in xrVar.dims:
        time_dim = "time"
    else:
        raise ValueError("Dimensione tempo non trovata. dims={}".format(xrVar.dims))

    # 3) Identifica dimensione latitudine
    lat_candidates = [d for d in xrVar.dims if "lat" in d.lower()]
    if len(lat_candidates) != 1:
        raise ValueError("Dimensione lat ambigua/assente: {} in dims={}".format(lat_candidates, 
xrVar.dims))
    lat_dim = lat_candidates[0]

    # 4) Split train/test
    xrTrain = xrVar.sel({time_dim: slice(train_start, train_end)})
    xrVar_all = xrVar  # alias più leggibile

    nTrain = int(xrTrain.sizes[time_dim])
    if nTrain < 2:
        raise ValueError("Training troppo corto: nTrain={}. Controlla date e asse 
tempo.".format(nTrain))

    # 5) Default robusto per N
    nSpace = int(np.prod([xrTrain.sizes[d] for d in xrTrain.dims if d != time_dim]))
    if N is None:
        N = min(nTrain, nSpace)
    else:
        N = int(N)
        if N < 1:
            raise ValueError("N deve essere >= 1")
        N = min(N, nTrain, nSpace)

    # 6) Pesi cos(lat) (normalizzati rispetto alla media)
    coslat = np.cos(np.deg2rad(xrVar_all[lat_dim]))
    weights = coslat / coslat.mean()

    # 7) Media su TRAIN e anomalie
    fldmean = xrTrain.mean(dim=time_dim).values
    xrTrain_anom = xrTrain - fldmean

    # 8) Fit EOF sul training
    eof = xe.single.EOF(n_modes=N, use_coslat=True)
    eof.fit(xrTrain_anom, dim=time_dim)

    comps = eof.components()                    # (mode, lat, lon)
    expvar = eof.explained_variance_ratio().values  # (mode,)

    # 9) Proiezione di tutto il periodo sulle EOF train
    anom_all = xrVar_all - fldmean
    anom_weighted = anom_all * weights  # broadcast su lon

    T_all = int(anom_weighted.sizes[time_dim])
    anom_2d = anom_weighted.values.reshape(T_all, -1)  # (T, space)
    eof_2d = comps.values.reshape(N, -1)               # (N, space)

    PCs_all = np.dot(anom_2d, eof_2d.T)  # (T, N)

    # 10) Standardizzazione su TRAIN
    PCs_train = PCs_all[:nTrain, :]
    PCs_test = PCs_all[nTrain:, :]

    std_train = None
    if zPC:
        std_train = PCs_train.std(axis=0)  # (N,)
        std_train = np.where(std_train == 0, 1.0, std_train)
        PCs_train = PCs_train / std_train
        PCs_test = PCs_test / std_train
        PCs_all = np.vstack([PCs_train, PCs_test])

    PC_out = PCs_all.T  # (N, T_all)

    if verbose:
        print("[EOF] time_dim={} nTrain={} T_all={} N={}".format(time_dim, nTrain, T_all, N))
        print("[EOF] PC shape (N,T):", PC_out.shape)

    return {
        "PC": PC_out,
        "fldmean": fldmean,
        "ExpVar": expvar,
        "std_train": std_train if zPC else None,
        "time_dim": time_dim,
        "n_train": nTrain,
        "train_range": (train_start, train_end),
        "test_start": test_start,
        "use_coslat": True,
        "zPC": zPC,
    }
def get_only_pcs(xrVar, N=None, verbose=True, zPC=True):
    """
    Versione 'classica' (no split): fit EOF/PC su tutto il periodo.
    Restituisce solo PC + mean + ExpVar.

    Output:
      - 'PC': array come in xeofs (lasciato com'è) oppure standardizzato se zPC=True
      - 'fldmean': mean su asse tempo
      - 'ExpVar': explained variance ratio
    """

    # Squeeze eventuale dimensione verticale
    for d in list(xrVar.dims):
        if d in ("plev", "pressure_level", "level"):
            xrVar = xrVar.squeeze(d, drop=True)
            break

    # Decide time dimension
    if "valid_time" in xrVar.dims:
        time_dim = "valid_time"
    elif "time" in xrVar.dims:
        time_dim = "time"
    else:
        raise ValueError("Time dimension not found in xrVar.dims={}".format(xrVar.dims))

    nT = int(xrVar.sizes[time_dim])
    fldeofs = {}

    if N is None:
        N = nT
    else:
        N = int(N)

    eof = xe.single.EOF(n_modes=N, use_coslat=True)
    eof.fit(xrVar, dim=time_dim)

    var_mean = xrVar.mean(dim=time_dim).values
    scores = eof.scores()  # PCs
    expvar = eof.explained_variance_ratio()

    fldeofs["PC"] = scores.values
    fldeofs["fldmean"] = var_mean

    # >>> LASCIO LA TUA STANDARDIZZAZIONE IDENTICA <<<
    if zPC is True:
        fldeofs["PC"] = np.matmul(
            scores.values.T,
            np.eye(N, N) * (1 / scores.values.std(axis=1))
        ).T
        if verbose is True:
            print("Standardized PC")

    fldeofs["ExpVar"] = expvar.values
    if verbose is True:
        print("EOF completed")

    return fldeofs
