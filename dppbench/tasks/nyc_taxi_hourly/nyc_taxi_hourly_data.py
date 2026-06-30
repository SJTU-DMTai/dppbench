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
    hour) panel in the loader. The downstream regression target is the per-zone,
    per-hour trip count.

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

        df = self._clean_raw_events(df)
        df = self._build_hourly_panel(df)

        self.train_df = df
        self.test_df = None
        return self.train_df, self.test_df

    def _clean_raw_events(self, df):
        df = df.copy()
        if "tpep_pickup_datetime" in df.columns:
            df["tpep_pickup_datetime"] = pd.to_datetime(
                df["tpep_pickup_datetime"], errors="coerce"
            )
        if "tpep_dropoff_datetime" in df.columns:
            df["tpep_dropoff_datetime"] = pd.to_datetime(
                df["tpep_dropoff_datetime"], errors="coerce"
            )
        if "passenger_count" in df.columns:
            df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce")
            bad_passenger = (df["passenger_count"] < 0) | (df["passenger_count"] > 9)
            df.loc[bad_passenger.fillna(False), "passenger_count"] = pd.NA
        for col in ["trip_distance", "fare_amount", "tip_amount", "total_amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        rules = [
            ("trip_distance", 0, 200),
            ("fare_amount", 0, 1000),
            ("total_amount", 0, 2000),
        ]
        mask = pd.Series(True, index=df.index)
        for col, lo, hi in rules:
            if col in df.columns:
                mask = mask & df[col].between(lo, hi, inclusive="both")
        if "tpep_pickup_datetime" in df.columns:
            start = pd.Timestamp("2022-12-15")
            end = pd.Timestamp("2023-02-15")
            mask = mask & df["tpep_pickup_datetime"].between(start, end, inclusive="both")
        return df.loc[mask.fillna(False)].reset_index(drop=True)

    def _build_hourly_panel(self, df):
        required = {"tpep_pickup_datetime", "PULocationID"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"Missing required NYC taxi columns for hourly panel: {missing}")
        df = df.copy()
        df["_hour_ts"] = df["tpep_pickup_datetime"].dt.floor("h")
        agg_spec = {}
        for col, funcs in {
            "trip_distance": ["mean", "sum"],
            "fare_amount": ["mean", "sum"],
            "tip_amount": ["mean"],
            "total_amount": ["mean"],
            "passenger_count": ["mean"],
        }.items():
            if col in df.columns:
                agg_spec[col] = funcs
        if agg_spec:
            out = df.groupby(["PULocationID", "_hour_ts"], dropna=False).agg(agg_spec).reset_index()
            out.columns = [
                "_".join([str(x) for x in col if x]) if isinstance(col, tuple) else col
                for col in out.columns
            ]
            counts = (
                df.groupby(["PULocationID", "_hour_ts"], dropna=False)
                .size()
                .reset_index(name="trip_count")
            )
            out = out.merge(counts, on=["PULocationID", "_hour_ts"], how="left")
        else:
            out = (
                df.groupby(["PULocationID", "_hour_ts"], dropna=False)
                .size()
                .reset_index(name="trip_count")
            )
        out["_hour_idx"] = pd.to_datetime(out["_hour_ts"]).astype("datetime64[ns]").astype("int64") // 10 ** 9
        return out.sort_values(["PULocationID", "_hour_idx"], kind="mergesort").reset_index(drop=True)

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
                f"Expected '{self._sort_col}' from hourly panel built in load_data()."
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
