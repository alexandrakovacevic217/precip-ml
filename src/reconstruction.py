import numpy as np


def reconstruct_field(PCs, EOFs, field_mean, n_modes: int, n_lat: int, n_lon: int):
    """
    Reconstruct (time, lat, lon) field from PCs and EOFs plus mean field.

    PCs: (n_modes, time)
    EOFs: (n_modes, lat, lon) or compatible
    field_mean: (lat, lon)
    """
    EOFs_flat = EOFs[:n_modes].reshape(n_modes, n_lat * n_lon)
    PCs_cut = PCs[:n_modes, :]
    recon = np.matmul(EOFs_flat.T, PCs_cut).T.reshape(PCs_cut.shape[1], n_lat, n_lon)
    return recon + np.tile(field_mean, [PCs_cut.shape[1], 1, 1])
