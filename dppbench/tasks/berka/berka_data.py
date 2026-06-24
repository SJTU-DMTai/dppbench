import os
import zipfile
import urllib.request
import pandas as pd

from ...dataset import TabularData


class BerkaData(TabularData):
    """PKDD'99 Czech bank financial dataset (8 relational tables).

    Loan default prediction. The main table is ``loan`` (~700 rows, target =
    binary good/bad), with auxiliary tables: account, client, disp, card,
    order, trans, district.

    Source: http://lisp.vse.cz/pkdd99/
    """

    DATA_URL = "http://lisp.vse.cz/pkdd99/DATA/data_berka.zip"
    ZIP_CANDIDATES = [
        "http://lisp.vse.cz/pkdd99/DATA/data_berka.zip",
        "http://sorry.vse.cz/~berka/challenge/pkdd1999/data_berka.zip",
    ]
    ASC_BASE_URLS = [
        "https://raw.githubusercontent.com/zhouxu-ds/ds-projects/master/loan_default_prediction/data",
        "https://raw.githubusercontent.com/anttttti/Berka-Dataset/master/data",
    ]
    USER_AGENT = "Wget/1.21.4"
    AUX_TABLES = ["account", "client", "disp", "card", "order", "trans", "district"]
    ASC_FILES = ["account", "card", "client", "disp", "district", "loan", "order", "trans"]
    MAIN_TABLE = "loan"

    def __init__(self, data_dir=None):
        super().__init__(name="Berka")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.id_col = "loan_id"
        self.target_col = "target"

    # ------------------------------------------------------------------
    # Download / extraction
    # ------------------------------------------------------------------
    def _all_asc_present(self):
        return all(
            os.path.exists(os.path.join(self.data_dir, f"{name}.asc"))
            for name in self.ASC_FILES
        )

    def _try_download_zip(self, url, zip_path):
        req = urllib.request.Request(
            url, headers={"User-Agent": self.USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=60) as resp, \
                open(zip_path, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)

    def _try_download_asc(self, base_url, name):
        url = f"{base_url}/{name}.asc"
        out = os.path.join(self.data_dir, f"{name}.asc")
        req = urllib.request.Request(
            url, headers={"User-Agent": self.USER_AGENT}
        )
        with urllib.request.urlopen(req, timeout=60) as resp, \
                open(out, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)

        if self._all_asc_present():
            return

        attempts = []

        # 1) Try per-file download from reliable GitHub mirrors first
        #    (original Czech servers lisp.vse.cz / sorry.vse.cz are offline)
        per_file_ok = True
        for name in self.ASC_FILES:
            target = os.path.join(self.data_dir, f"{name}.asc")
            if os.path.exists(target):
                continue
            ok = False
            for base in self.ASC_BASE_URLS:
                url = f"{base}/{name}.asc"
                print(f"Attempting Berka {name}.asc from {url} ...")
                try:
                    self._try_download_asc(base, name)
                    print(f"  -> success: {target}")
                    ok = True
                    break
                except Exception as e:
                    attempts.append(f"{url}: {e}")
                    if os.path.exists(target):
                        os.remove(target)
            if not ok:
                per_file_ok = False
                break

        if self._all_asc_present():
            return

        # 2) Try zip download as fallback
        zip_path = os.path.join(self.data_dir, "data_berka.zip")
        if not os.path.exists(zip_path):
            for url in self.ZIP_CANDIDATES:
                print(f"Attempting Berka zip download from {url} ...")
                try:
                    self._try_download_zip(url, zip_path)
                    print(f"  -> success: {zip_path}")
                    break
                except Exception as e:
                    attempts.append(f"{url}: {e}")
                    if os.path.exists(zip_path):
                        os.remove(zip_path)

        if os.path.exists(zip_path) and not self._all_asc_present():
            print(f"Extracting {zip_path} ...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(self.data_dir)
            except Exception as e:
                attempts.append(f"unzip {zip_path}: {e}")

        if self._all_asc_present():
            return

        raise RuntimeError(
            f"Failed to download Berka dataset. "
            f"Please place all 8 .asc files (account/card/client/"
            f"disp/district/loan/order/trans) under {self.data_dir} "
            f"manually. Attempts:\n" + "\n".join(attempts)
        )

    # ------------------------------------------------------------------
    # Reading helpers
    # ------------------------------------------------------------------
    def _read_asc(self, name):
        path = os.path.join(self.data_dir, f"{name}.asc")
        if not os.path.exists(path):
            return None
        try:
            return pd.read_csv(path, sep=";", quotechar='"', encoding="latin-1")
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=";", quotechar='"', encoding="utf-8",
                               errors="replace")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_data(self):
        self._download_if_missing()

        loan = self._read_asc(self.MAIN_TABLE)
        if loan is None:
            raise FileNotFoundError(
                f"Berka loan.asc missing under {self.data_dir}"
            )

        status_map = {"A": 0, "C": 0, "B": 1, "D": 1}
        loan["target"] = loan["status"].map(status_map)
        loan = loan.dropna(subset=["target"]).copy()
        loan["target"] = loan["target"].astype(int)
        loan = loan.drop(columns=["status"])

        self.train_df = loan.reset_index(drop=True)
        self.test_df = None

        for name in self.AUX_TABLES:
            df = self._read_asc(name)
            self.auxiliary_dfs[name] = df

        return self.train_df, self.test_df
