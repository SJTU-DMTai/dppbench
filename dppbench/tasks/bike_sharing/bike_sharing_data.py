import os
import urllib.request
import zipfile
import pandas as pd

from ...dataset import TabularData


class BikeSharingData(TabularData):
    """UCI Bike Sharing dataset (Capital Bikeshare 2011-2012).

    Hourly demand prediction with daily-level auxiliary table. Provides:
    - ``train_df``: hour.csv (17379 rows) with target ``cnt``
    - ``auxiliary_dfs['day']``: day.csv (731 rows) for day-level summary

    Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
    """

    DATA_URL = (
        "https://archive.ics.uci.edu/static/public/275/"
        "bike+sharing+dataset.zip"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    ZIP_NAME = "bike+sharing+dataset.zip"
    HOUR_NAME = "hour.csv"
    DAY_NAME = "day.csv"

    def __init__(self, data_dir=None):
        super().__init__(name="BikeSharing")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = "instant"
        self.target_col = "cnt"
        self._sort_col = "_sort_idx"

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        hour_path = os.path.join(self.data_dir, self.HOUR_NAME)
        day_path = os.path.join(self.data_dir, self.DAY_NAME)
        if os.path.exists(hour_path) and os.path.exists(day_path):
            return

        zip_path = os.path.join(self.data_dir, self.ZIP_NAME)
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000):
            print(f"Downloading Bike Sharing data from {self.DATA_URL} ...")
            req = urllib.request.Request(
                self.DATA_URL,
                headers={"User-Agent": self.USER_AGENT, "Accept": "*/*"},
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp, \
                        open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        f.write(chunk)
            except Exception as e:
                if os.path.exists(zip_path):
                    os.remove(zip_path)
                raise RuntimeError(
                    f"Failed to download {self.DATA_URL}: {e}. "
                    f"Place {self.ZIP_NAME} in {self.data_dir} manually."
                )

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

    def load_data(self):
        self._download_if_missing()

        hour_path = os.path.join(self.data_dir, self.HOUR_NAME)
        day_path = os.path.join(self.data_dir, self.DAY_NAME)

        hour = pd.read_csv(hour_path)
        day = pd.read_csv(day_path)

        # Build a numeric chronological sort index for time-series ops.
        if "instant" in hour.columns:
            hour[self._sort_col] = hour["instant"].astype("int64")

        # Keep day-level table lean for JoinTable aggregation.
        self.train_df = hour.reset_index(drop=True)
        self.test_df = None
        self.auxiliary_dfs["day"] = day.reset_index(drop=True)

        return self.train_df, self.test_df

    def split(self, val_ratio=0.2, seed=42):
        df = self.train_df

        # Peel std-test rows (carried via __split__) before chronological split
        # so they neither pollute train/val nor leak the marker column into
        # downstream model training.
        std_test = None
        if "__split__" in df.columns and (df["__split__"] == "std_test").any():
            std_test = (
                df[df["__split__"] == "std_test"]
                .drop(columns="__split__").reset_index(drop=True)
            )
            df = (
                df[df["__split__"] != "std_test"]
                .drop(columns="__split__").reset_index(drop=True)
            )

        if self._sort_col in df.columns:
            df = df.sort_values(self._sort_col, kind="mergesort").reset_index(drop=True)
        elif "instant" in df.columns:
            df = df.sort_values("instant", kind="mergesort").reset_index(drop=True)

        n = len(df)
        cut = int(n * (1 - val_ratio))
        train = df.iloc[:cut].reset_index(drop=True)
        val = df.iloc[cut:].reset_index(drop=True)
        print(
            f"Split: train={len(train)}, val={len(val)}, test=0 (chronological)"
            + (f", std_test={len(std_test)}" if std_test is not None else "")
        )
        self.train_df = df  # keep sorted state
        out = {"train": train, "val": val, "test": None}
        if std_test is not None:
            out["std_test"] = std_test
        return out
