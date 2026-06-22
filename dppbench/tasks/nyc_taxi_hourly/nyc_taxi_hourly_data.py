import os
import urllib.request
import pandas as pd
import numpy as np

from ...dataset import TabularData


class NycTaxiHourlyData(TabularData):
    """NYC TLC Yellow Taxi trip records (raw event-level log).

    Trip-level parquet from the official TLC release. The data is *raw* and
    *dirty* — it contains negative fares, zero distances, (0, 0) pickup
    coordinates, fare_amount > 1000, and timestamps that fall outside the
    nominal month. The preprocessing pipeline must clean these via
    ``HandleError(action=delete)`` and then bucket the events into a (PULocationID,
    hour) panel via ``ResampleTimeSeries``. The downstream regression
    target is the per-(zone, hour) trip count.

    Source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    """

    DATA_URL = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/"
        "yellow_tripdata_2023-01.parquet"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    PARQUET_NAME = "yellow_tripdata_2023-01.parquet"

    KEEP_COLS = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "tip_amount",
        "total_amount",
        "PULocationID",
        "DOLocationID",
    ]

    def __init__(self, data_dir=None, sample_zones=20, max_rows=None):
        super().__init__(name="NycTaxiHourly")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = None
        self.target_col = "trip_count"
        # Restrict to top-N busiest zones to keep training tractable.
        self.sample_zones = int(sample_zones) if sample_zones else None
        self.max_rows = int(max_rows) if max_rows else None
        self._sort_col = "_hour_idx"

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        path = os.path.join(self.data_dir, self.PARQUET_NAME)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            return path

        print(f"Downloading NYC TLC Yellow Taxi data from {self.DATA_URL} ...")
        req = urllib.request.Request(
            self.DATA_URL,
            headers={"User-Agent": self.USER_AGENT, "Accept": "*/*"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, \
                    open(path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(
                f"Failed to download {self.DATA_URL}: {e}. "
                f"Place {self.PARQUET_NAME} in {self.data_dir} manually."
            )
        return path

    def load_data(self):
        path = self._download_if_missing()
        df = pd.read_parquet(path)
        keep = [c for c in self.KEEP_COLS if c in df.columns]
        df = df[keep].copy()

        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=42).reset_index(drop=True)

        # Restrict to busiest zones to keep panel manageable.
        if self.sample_zones and "PULocationID" in df.columns:
            top = (
                df["PULocationID"].value_counts()
                .head(self.sample_zones).index.tolist()
            )
            df = df[df["PULocationID"].isin(top)].reset_index(drop=True)

        self.train_df = df
        self.test_df = None
        return self.train_df, self.test_df

    def split(self, val_ratio=0.2, seed=42):
        df = self.train_df

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

        if self._sort_col not in df.columns:
            raise ValueError(
                f"Expected '{self._sort_col}' from ResampleTimeSeries before split."
            )
        df = df.sort_values(self._sort_col, kind="mergesort").reset_index(drop=True)
        unique_ts = np.sort(df[self._sort_col].unique())
        cut_idx = int(len(unique_ts) * (1 - val_ratio))
        cut_ts = unique_ts[cut_idx]
        train = df[df[self._sort_col] < cut_ts].reset_index(drop=True)
        val = df[df[self._sort_col] >= cut_ts].reset_index(drop=True)
        print(
            f"Split: train={len(train)}, val={len(val)}, test=0 (chronological)"
            + (f", std_test={len(std_test)}" if std_test is not None else "")
        )
        self.train_df = df
        out = {"train": train, "val": val, "test": None}
        if std_test is not None:
            out["std_test"] = std_test
        return out
