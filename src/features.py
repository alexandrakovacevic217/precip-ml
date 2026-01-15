import numpy as np


def build_X_given_ninput(n_input: int, EOFS: dict, vname: list, EOFS_timeind, multi_time: bool):
    """
    Build input matrix X by concatenating the first n_input PCs for each variable in vname.
    If multi_time=True, concatenates PCs at t-2, t-1, t0 (reduces time by 2).
    """
    if multi_time:
        t_minus2_indices = EOFS_timeind[:-2]
        t_minus1_indices = EOFS_timeind[1:-1]
        t0_indices = EOFS_timeind[2:]

        X = np.concatenate((
            EOFS[vname[0]]["PC"][0:n_input, t_minus2_indices],
            EOFS[vname[0]]["PC"][0:n_input, t_minus1_indices],
            EOFS[vname[0]]["PC"][0:n_input, t0_indices],
        ), axis=0)

        for var in vname[1:]:
            X = np.concatenate((
                X,
                EOFS[var]["PC"][0:n_input, t_minus2_indices],
                EOFS[var]["PC"][0:n_input, t_minus1_indices],
                EOFS[var]["PC"][0:n_input, t0_indices],
            ), axis=0)

        X = X.T
    else:
        X = EOFS[vname[0]]["PC"][0:n_input, EOFS_timeind]
        for var in vname[1:]:
            X = np.concatenate((X, EOFS[var]["PC"][0:n_input, EOFS_timeind]), axis=0)
        X = X.T

    return X
