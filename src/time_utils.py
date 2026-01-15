import numpy as np


def get_index_from_date(time_array, target_date: str) -> int:
    """
    Return the index in an xarray time coordinate matching target_date (YYYY-MM-DD).
    Raises IndexError if not found.
    """
    return int(np.where(time_array.dt.date == np.datetime64(target_date))[0][0])
