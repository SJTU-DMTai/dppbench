import os
import glob
import urllib.request
import zipfile
import pandas as pd
import numpy as np

from ...dataset import TabularData


class BeijingAirQualityData(TabularData):
    """UCI Beijing Multi-Site Air Quality dataset (2013-03 .. 2017-02).

    Twelve monitoring stations, hourly readings (~35064 rows each).
    Concatenated into a single train_df with a categorical ``station`` column.
    Provides:
    - ``train_df``: concatenated multi-station hourly data (target ``PM2.5``)
    - ``auxiliary_dfs['station_meta']``: per-station numeric summary

    Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
    """

    DATA_URL = (
        "https://archive.ics.uci.edu/static/public/501/"
        "beijing+multi+site+air+quality+data.zip"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    ZIP_NAME = "beijing+multi+site+air+quality+data.zip"
    INNER_ZIP = "PRSA2017_Data_20130301-20170228.zip"
    POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    METEO = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]

    def __init__(self, data_dir=None):
        super().__init__(name="BeijingAirQuality")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = None
        self.target_col = "PM2.5"
        self._sort_col = "_sort_idx"

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # If station csvs already there, skip.
        existing = glob.glob(os.path.join(self.data_dir, "**", "PRSA_Data_*.csv"), recursive=True)
        if len(existing) >= 12:
            return

        zip_path = os.path.join(self.data_dir, self.ZIP_NAME)
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000):
            print(f"Downloading Beijing air quality data from {self.DATA_URL} ...")
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

        # The outer zip contains another zip with the actual CSVs.
        inner = os.path.join(self.data_dir, self.INNER_ZIP)
        if os.path.exists(inner):
            with zipfile.ZipFile(inner, "r") as zf:
                zf.extractall(self.data_dir)

    def _find_csv_files(self):
        return sorted(glob.glob(
            os.path.join(self.data_dir, "**", "PRSA_Data_*.csv"),
            recursive=True,
        ))

    def load_data(self):
        self._download_if_missing()
        csvs = self._find_csv_files()
        if len(csvs) == 0:
            raise RuntimeError(
                f"No PRSA_Data_*.csv files found under {self.data_dir}"
            )
        frames = [pd.read_csv(p) for p in csvs]
        df = pd.concat(frames, axis=0, ignore_index=True)

        # Parse hourly sort index from (year, month, day, hour).
        df[self._sort_col] = (
            df["year"].astype("int64") * 1_000_000
            + df["month"].astype("int64") * 10_000
            + df["day"].astype("int64") * 100
            + df["hour"].astype("int64")
        )

        # Drop original "No" id col if present (per-station numbering).
        if "No" in df.columns:
            df = df.drop(columns=["No"])

        # Build station_meta aux table: per-station numeric summary.
        meta_cols = [c for c in self.POLLUTANTS + self.METEO if c in df.columns]
        meta = (
            df[["station"] + meta_cols]
            .groupby("station")
            .mean(numeric_only=True)
            .reset_index()
        )

        self.train_df = df.reset_index(drop=True)
        self.test_df = None
        self.auxiliary_dfs["station_meta"] = meta

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

        if self._sort_col in df.columns:
            df = df.sort_values([self._sort_col], kind="mergesort").reset_index(drop=True)
        # Hold out the most recent val_ratio of timestamps as validation.
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
