import numpy as np


def prep_input(PRcpc, lsmask):
    """
    Flatten PRcpc using lsmask and REMOVE days with any missing value among valid cells.
    Returns:
      - PRzms: (t_kept, n_valid_cells)
      - PRcpc_kept: (t_kept, lat, lon)
    """
    tPR, yPR, xPR = PRcpc.shape
    PRz = PRcpc.values.reshape(tPR, yPR * xPR)
    imask = np.argwhere(~np.isnan(lsmask.flatten())).flatten()
    PRzm = PRz[:, imask]

    cpc_missingdata_idx = []
    cpc_fulldata_idx = []
    k = 0
    for i in range(tPR):
        miss = np.isnan(PRzm[i, :]).sum()
        if miss:
            cpc_missingdata_idx.append(i)
            print("missing " + str(miss) + " data grid points")
            print("index missing data: " + str(i))
            k += 1
        else:
            cpc_fulldata_idx.append(i)

    print("Total days with missing data: " + str(k))
    PRzms = PRzm[cpc_fulldata_idx, :]
    return PRzms, PRcpc[cpc_fulldata_idx, :, :]


def prep_input_filled(PRcpc, lsmask):
    """
    Flatten PRcpc using lsmask WITHOUT removing days.
    Returns PRzm: (time, n_valid_cells)
    """
    tPR, yPR, xPR = PRcpc.shape
    PRz = PRcpc.values.reshape(tPR, yPR * xPR)
    imask = np.argwhere(~np.isnan(lsmask.flatten())).flatten()
    PRzm = PRz[:, imask]
    return PRzm
