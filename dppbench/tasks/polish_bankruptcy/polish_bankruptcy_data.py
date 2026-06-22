import os
import urllib.request
import zipfile
import pandas as pd

from ...dataset import TabularData


class PolishBankruptcyData(TabularData):
    """UCI Polish Companies Bankruptcy dataset.

    5 ARFF files (``1year.arff`` ... ``5year.arff``), each containing 64
    financial ratios and a binary ``class`` (1 = bankrupt). The 5-year
    cohort is used as the main table; the earlier-year tables are exposed
    as auxiliary tables (without their ``class`` column) and broadcast to
    the main table via a constant ``_join_key`` so JoinTable can compute
    cross-cohort summary statistics — analogous to home_credit's bureau
    aggregates.

    Source: https://archive.ics.uci.edu/dataset/365/polish+companies+bankruptcy+data
    """

    DATA_URL = (
        "https://archive.ics.uci.edu/static/public/365/"
        "polish+companies+bankruptcy+data.zip"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    ZIP_NAME = "polish+companies+bankruptcy+data.zip"
    ARFF_FILES = ["1year.arff", "2year.arff", "3year.arff",
                  "4year.arff", "5year.arff"]

    def __init__(self, data_dir=None):
        super().__init__(name="PolishBankruptcy")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = None
        self.target_col = "TARGET"

    def _all_arff_present(self):
        return all(
            os.path.exists(os.path.join(self.data_dir, name))
            for name in self.ARFF_FILES
        )

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if self._all_arff_present():
            return

        zip_path = os.path.join(self.data_dir, self.ZIP_NAME)
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000):
            print(f"Downloading Polish bankruptcy data from {self.DATA_URL} ...")
            req = urllib.request.Request(
                self.DATA_URL,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": "*/*",
                },
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
                    f"Failed to download {self.DATA_URL}: {e}"
                )

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

    @staticmethod
    def _read_arff(path):
        from scipy.io import arff
        data, _ = arff.loadarff(path)
        df = pd.DataFrame(data)
        # ARFF "class" arrives as bytes in numeric attribute encoding.
        if "class" in df.columns:
            df["class"] = df["class"].apply(
                lambda v: int(v.decode()) if isinstance(v, (bytes, bytearray))
                else int(v)
            )
        # Numeric attribute columns may be object dtype with b"?" missing
        # markers. Coerce to float.
        for c in df.columns:
            if c == "class":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def load_data(self):
        self._download_if_missing()

        dfs = {}
        for name in self.ARFF_FILES:
            n = int(name[0])
            dfs[n] = self._read_arff(os.path.join(self.data_dir, name))

        main = dfs[5].rename(columns={"class": "TARGET"}).copy()
        main["_join_key"] = 1
        self.train_df = main.reset_index(drop=True)
        self.test_df = None

        # Earlier-year cohorts as aux tables. Drop the per-year label so it
        # never leaks into the aggregated features. Add the constant
        # ``_join_key`` so JoinTable can broadcast.
        for n in range(1, 5):
            aux = dfs[n].drop(columns=["class"]).copy()
            aux["_join_key"] = 1
            self.auxiliary_dfs[f"year{n}"] = aux

        return self.train_df, self.test_df
