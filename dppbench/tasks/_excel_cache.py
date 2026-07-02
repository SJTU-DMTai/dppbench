import os

import pandas as pd


def read_excel_cached(excel_path, cache_path=None, **kwargs):
    """Read an Excel file through a local pickle cache.

    Parsing .xls/.xlsx files is much slower than loading a pickled DataFrame.
    The cache is invalidated when the source Excel file is newer than the
    cache file.
    """
    if cache_path is None:
        cache_path = f"{excel_path}.pkl"

    if os.path.exists(cache_path):
        cache_mtime = os.path.getmtime(cache_path)
        excel_mtime = os.path.getmtime(excel_path)
        if cache_mtime >= excel_mtime:
            return pd.read_pickle(cache_path)

    df = pd.read_excel(excel_path, **kwargs)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_pickle(cache_path)
    return df
