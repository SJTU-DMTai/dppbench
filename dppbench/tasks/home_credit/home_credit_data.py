import os
import shutil
import subprocess
import urllib.request
import zipfile
import pandas as pd
from ...dataset import TabularData


class HomeCreditData(TabularData):
    """Home Credit Default Risk dataset.

    The canonical source is the Kaggle competition:
    https://www.kaggle.com/competitions/home-credit-default-risk/data

    Kaggle competition downloads require accepted rules and API credentials.
    For environments with an internal/public mirror, set
    ``DPPBENCH_HOME_CREDIT_URL`` to one or more comma-separated archive URLs.
    """

    COMPETITION = "home-credit-default-risk"
    COMPETITION_URL = (
        "https://www.kaggle.com/competitions/home-credit-default-risk/data"
    )
    ENV_URLS = "DPPBENCH_HOME_CREDIT_URL"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    ARCHIVE_NAME = "home-credit-default-risk.zip"
    REQUIRED_FILES = (
        "application_train.csv",
        "application_test.csv",
    )
    AUX_FILES = {
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "previous_application": "previous_application.csv",
        "pos_cash": "POS_CASH_balance.csv",
        "credit_card": "credit_card_balance.csv",
        "installments": "installments_payments.csv",
    }

    def __init__(self, data_dir=None):
        super().__init__(name="HomeCredit")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.target_col = "TARGET"
        self.id_col = "SK_ID_CURR"

    def _download_archive(self, url, filename):
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return filepath

        print(f"{self.name} Downloading {url}")
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "*/*",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp, \
                    open(filepath, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise
        return filepath

    @staticmethod
    def _safe_extract_zip(zip_path, output_dir):
        abs_output = os.path.abspath(output_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                target = os.path.abspath(os.path.join(output_dir, member.filename))
                if os.path.commonpath([abs_output, target]) != abs_output:
                    raise RuntimeError(
                        f"Unsafe path in archive {zip_path}: {member.filename}"
                    )
            zf.extractall(output_dir)

    def _extract_archives(self):
        for name in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, name)
            if not name.lower().endswith(".zip"):
                continue
            if not zipfile.is_zipfile(path):
                continue
            print(f"{self.name} Extracting {path}")
            self._safe_extract_zip(path, self.data_dir)
        self._promote_required_files()

    def _promote_required_files(self):
        required = list(self.REQUIRED_FILES) + list(self.AUX_FILES.values())
        for filename in required:
            root_path = os.path.join(self.data_dir, filename)
            if os.path.exists(root_path):
                continue
            for dirpath, _, filenames in os.walk(self.data_dir):
                if filename not in filenames:
                    continue
                src = os.path.join(dirpath, filename)
                if os.path.abspath(src) == os.path.abspath(root_path):
                    break
                shutil.copy2(src, root_path)
                break

    def _try_download_from_urls(self):
        urls = [
            url.strip()
            for url in os.environ.get(self.ENV_URLS, "").split(",")
            if url.strip()
        ]
        for i, url in enumerate(urls, start=1):
            archive_name = os.path.basename(url.split("?", 1)[0]) or self.ARCHIVE_NAME
            if not archive_name.lower().endswith(".zip"):
                archive_name = f"home-credit-default-risk-{i}.zip"
            try:
                self._download_archive(url, archive_name)
                self._extract_archives()
                if not self._missing_required_files():
                    return True
            except Exception as exc:
                print(f"{self.name} mirror download failed: {type(exc).__name__}: {exc}")
        return False

    def _run_kaggle_download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        kaggle = shutil.which("kaggle")
        if not kaggle:
            return False

        cmd = [
            kaggle,
            "competitions",
            "download",
            "-c",
            self.COMPETITION,
            "-p",
            self.data_dir,
            "--force",
        ]
        print(f"{self.name} Downloading with Kaggle CLI: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.returncode != 0:
            return False
        self._extract_archives()
        return not self._missing_required_files()

    def _missing_required_files(self):
        self._promote_required_files()
        required = list(self.REQUIRED_FILES) + list(self.AUX_FILES.values())
        return [
            name
            for name in required
            if not os.path.exists(os.path.join(self.data_dir, name))
        ]

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not self._missing_required_files():
            return
        self._extract_archives()
        if not self._missing_required_files():
            return

        if self._try_download_from_urls():
            return
        if self._run_kaggle_download():
            return

        missing = self._missing_required_files()
        raise RuntimeError(
            f"{self.name} data files are missing: {missing}. "
            f"Set {self.ENV_URLS} to a direct archive URL, or install/configure "
            f"Kaggle CLI and accept the competition rules at {self.COMPETITION_URL}. "
            f"Then place/download the files under {self.data_dir}."
        )

    def load_data(self):
        self._download_if_missing()

        data_dir = self.data_dir
        train_path = os.path.join(data_dir, "application_train.csv")
        test_path = os.path.join(data_dir, "application_test.csv")
        self.train_df = pd.read_csv(train_path)
        if os.path.exists(test_path):
            self.test_df = pd.read_csv(test_path)
        else:
            self.test_df = None

        self._load_auxiliary_tables()
        return self.train_df, self.test_df

    def _load_auxiliary_tables(self):
        data_dir = self.data_dir

        for name, filename in self.AUX_FILES.items():
            filepath = os.path.join(data_dir, filename)
            if not filepath.endswith(".zip") and not os.path.exists(filepath):
                zip_path = filepath + ".zip"
                if os.path.exists(zip_path):
                    self._safe_extract_zip(zip_path, data_dir)

            if os.path.exists(filepath):
                self.auxiliary_dfs[name] = pd.read_csv(filepath)
            else:
                self.auxiliary_dfs[name] = None
