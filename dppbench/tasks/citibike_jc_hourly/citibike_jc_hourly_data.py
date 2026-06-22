import os
import io
import urllib.request
import zipfile
import glob
import pandas as pd
import numpy as np

from ...dataset import TabularData


class CitibikeJcHourlyData(TabularData):
    """Citi Bike Jersey City trip records (raw event-level CSV).

    Each ride is one row; the schema changed across years so this loader
    normalises both old and new column names to ``started_at`` /
    ``start_station_id`` / ``birth_year`` etc. The raw data is dirty:
    sentinel ``birth_year=1969`` for unknown, durations of a few seconds,
    rides crossing midnight, etc. Preprocessing must wash these via
    ``CustomClean`` + ``HandleError(action=delete)`` and then bucket the events into a
    (start_station, hour) panel via ``ResampleTimeSeries``. The downstream
    regression target is the per-(station, hour) rental count.

    Source: https://citibikenyc.com/system-data
    """

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    # Default to one month of JC data which is small (~50 MB unzipped).
    DEFAULT_URLS = [
        "https://s3.amazonaws.com/tripdata/JC-202301-citibike-tripdata.csv.zip",
    ]

    # Map old (pre-2021) names to new schema.
    OLD_TO_NEW = {
        "starttime": "started_at",
        "stoptime": "ended_at",
        "start station id": "start_station_id",
        "end station id": "end_station_id",
        "start station name": "start_station_name",
        "end station name": "end_station_name",
        "tripduration": "trip_duration",
        "birth year": "birth_year",
        "usertype": "member_casual",
    }

    def __init__(self, data_dir=None, urls=None, sample_stations=15, max_rows=None):
        super().__init__(name="CitibikeJcHourly")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = None
        self.target_col = "rental_count"
        self.urls = urls or list(self.DEFAULT_URLS)
        self.sample_stations = int(sample_stations) if sample_stations else None
        self.max_rows = int(max_rows) if max_rows else None
        self._sort_col = "_hour_idx"

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        # If we already have any *.csv from a prior extract, skip.
        existing = glob.glob(os.path.join(self.data_dir, "*citibike*.csv")) \
            + glob.glob(os.path.join(self.data_dir, "JC-*.csv"))
        if existing:
            return existing

        all_csvs = []
        for url in self.urls:
            zip_name = url.rsplit("/", 1)[-1]
            zip_path = os.path.join(self.data_dir, zip_name)
            if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000):
                print(f"Downloading Citi Bike data from {url} ...")
                req = urllib.request.Request(
                    url, headers={"User-Agent": self.USER_AGENT, "Accept": "*/*"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=600) as resp, \
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
                        f"Failed to download {url}: {e}. "
                        f"Place {zip_name} in {self.data_dir} manually."
                    )

            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            all_csvs.extend(glob.glob(os.path.join(self.data_dir, "JC-*.csv")))
            all_csvs.extend(glob.glob(os.path.join(self.data_dir, "*citibike*.csv")))

        return sorted(set(all_csvs))

    def _normalise_schema(self, df):
        # rename old column names, lower / strip
        rename = {}
        for c in df.columns:
            key = c.strip()
            if key in self.OLD_TO_NEW:
                rename[c] = self.OLD_TO_NEW[key]
        if rename:
            df = df.rename(columns=rename)
        return df

    def load_data(self):
        csvs = self._download_if_missing()
        if not csvs:
            raise RuntimeError(f"No Citi Bike CSV found under {self.data_dir}")
        frames = []
        for p in csvs:
            try:
                frames.append(pd.read_csv(p, low_memory=False))
            except Exception as e:
                print(f"  [warn] skip {p}: {e}")
        if not frames:
            raise RuntimeError("All CSV reads failed")
        df = pd.concat(frames, axis=0, ignore_index=True)
        df = self._normalise_schema(df)

        # Coerce timestamp columns up front.
        for c in ("started_at", "ended_at"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

        # Compute trip_duration in seconds when missing.
        if "trip_duration" not in df.columns and {"started_at", "ended_at"}.issubset(df.columns):
            df["trip_duration"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()

        if self.max_rows is not None and len(df) > self.max_rows:
            df = df.sample(self.max_rows, random_state=42).reset_index(drop=True)

        if self.sample_stations and "start_station_id" in df.columns:
            top = (
                df["start_station_id"].astype(str).value_counts()
                .head(self.sample_stations).index.tolist()
            )
            df["start_station_id"] = df["start_station_id"].astype(str)
            df = df[df["start_station_id"].isin(top)].reset_index(drop=True)

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
