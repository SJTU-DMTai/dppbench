import gzip
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
import pandas as pd
from ...dataset import TabularData


class FraudDetectionData(TabularData):
    """IEEE-CIS Fraud Detection dataset.

    The canonical source is the Kaggle competition:
    https://www.kaggle.com/competitions/ieee-fraud-detection/data

    The default unauthenticated mirror below is used first. If it becomes
    unavailable, set ``DPPBENCH_FRAUD_DETECTION_URL`` to one or more direct
    archive URLs, or configure Kaggle CLI credentials.
    """

    COMPETITION = "ieee-fraud-detection"
    COMPETITION_URL = (
        "https://www.kaggle.com/competitions/ieee-fraud-detection/data"
    )
    ENV_URLS = "DPPBENCH_FRAUD_DETECTION_URL"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    REQUIRED_FILES = (
        "train_transaction.csv",
        "train_identity.csv",
        "test_transaction.csv",
        "test_identity.csv",
    )
    PUBLIC_FILE_URLS = {
        "train_transaction.csv.tar.gz": (
            "https://giskard-library-test-datasets.s3.eu-north-1.amazonaws.com/"
            "fraud_detection_classification_dataset-train_transaction.csv.tar.gz"
        ),
        "train_identity.csv.tar.gz": (
            "https://giskard-library-test-datasets.s3.eu-north-1.amazonaws.com/"
            "fraud_detection_classification_dataset-train_identity.csv.tar.gz"
        ),
        "test_transaction.csv.tar.gz": (
            "https://giskard-library-test-datasets.s3.eu-north-1.amazonaws.com/"
            "fraud_detection_classification_dataset-test_transaction.csv.tar.gz"
        ),
        "test_identity.csv.tar.gz": (
            "https://giskard-library-test-datasets.s3.eu-north-1.amazonaws.com/"
            "fraud_detection_classification_dataset-test_identity.csv.tar.gz"
        ),
    }

    def __init__(self, data_dir=None):
        super().__init__(name="FraudDetection")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.target_col = "isFraud"
        self.id_col = "TransactionID"

    def _download_file(self, url, filename):
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
    def _ensure_safe_path(output_dir, member_name, archive_path):
        abs_output = os.path.abspath(output_dir)
        target = os.path.abspath(os.path.join(output_dir, member_name))
        if os.path.commonpath([abs_output, target]) != abs_output:
            raise RuntimeError(f"Unsafe path in archive {archive_path}: {member_name}")

    def _extract_archive(self, archive_path):
        if zipfile.is_zipfile(archive_path):
            print(f"{self.name} Extracting {archive_path}")
            with zipfile.ZipFile(archive_path, "r") as zf:
                for member in zf.infolist():
                    self._ensure_safe_path(
                        self.data_dir, member.filename, archive_path
                    )
                zf.extractall(self.data_dir)
            self._promote_required_files()
            return

        if tarfile.is_tarfile(archive_path):
            print(f"{self.name} Extracting {archive_path}")
            with tarfile.open(archive_path, "r:*") as tf:
                for member in tf.getmembers():
                    self._ensure_safe_path(self.data_dir, member.name, archive_path)
                    if member.issym() or member.islnk():
                        raise RuntimeError(
                            f"Unsafe link in archive {archive_path}: {member.name}"
                        )
                tf.extractall(self.data_dir)
            self._promote_required_files()
            return

        if archive_path.endswith(".gz"):
            output_name = os.path.basename(archive_path[:-3])
            if output_name.endswith(".tar"):
                output_name = output_name[:-4]
            output_path = os.path.join(self.data_dir, output_name)
            print(f"{self.name} Decompressing {archive_path}")
            with gzip.open(archive_path, "rb") as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        self._promote_required_files()

    def _extract_existing_archives(self):
        for name in os.listdir(self.data_dir):
            path = os.path.join(self.data_dir, name)
            lower = name.lower()
            if lower.endswith((".zip", ".tar", ".tar.gz", ".tgz", ".csv.gz")):
                self._extract_archive(path)
        self._promote_required_files()

    def _promote_required_files(self):
        for filename in self.REQUIRED_FILES:
            root_path = os.path.join(self.data_dir, filename)
            if os.path.exists(root_path):
                continue
            for dirpath, _, filenames in os.walk(self.data_dir):
                match = None
                for candidate in filenames:
                    if (
                        candidate == filename
                        or candidate.endswith(f"-{filename}")
                        or candidate.endswith(f"_{filename}")
                    ):
                        match = candidate
                        break
                if match is None:
                    continue
                src = os.path.join(dirpath, match)
                if os.path.abspath(src) == os.path.abspath(root_path):
                    break
                shutil.copy2(src, root_path)
                break

    def _try_public_file_downloads(self):
        for filename, url in self.PUBLIC_FILE_URLS.items():
            target = filename.replace(".tar.gz", "")
            if os.path.exists(os.path.join(self.data_dir, target)):
                continue
            try:
                archive_path = self._download_file(url, filename)
                self._extract_archive(archive_path)
            except Exception as exc:
                print(f"{self.name} public download failed: {type(exc).__name__}: {exc}")
                return False
        return not self._missing_required_files()

    def _try_download_from_urls(self):
        urls = [
            url.strip()
            for url in os.environ.get(self.ENV_URLS, "").split(",")
            if url.strip()
        ]
        for i, url in enumerate(urls, start=1):
            name = os.path.basename(url.split("?", 1)[0]) or f"ieee-fraud-{i}.zip"
            try:
                archive_path = self._download_file(url, name)
                self._extract_archive(archive_path)
                if not self._missing_required_files():
                    return True
            except Exception as exc:
                print(f"{self.name} mirror download failed: {type(exc).__name__}: {exc}")
        return False

    def _run_kaggle_download(self):
        kaggle = shutil.which("kaggle")
        if not kaggle:
            return False

        os.makedirs(self.data_dir, exist_ok=True)
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
        self._extract_existing_archives()
        return not self._missing_required_files()

    def _missing_required_files(self):
        self._promote_required_files()
        return [
            name
            for name in self.REQUIRED_FILES
            if not os.path.exists(os.path.join(self.data_dir, name))
        ]

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not self._missing_required_files():
            return
        self._extract_existing_archives()
        if not self._missing_required_files():
            return

        if self._try_download_from_urls():
            return
        if self._try_public_file_downloads():
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

        train_trans = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
        train_id = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
        self.train_df = train_trans.merge(train_id, on="TransactionID", how="left")

        test_trans = pd.read_csv(os.path.join(data_dir, "test_transaction.csv"))
        test_id = pd.read_csv(os.path.join(data_dir, "test_identity.csv"))
        self.test_df = test_trans.merge(test_id, on="TransactionID", how="left")

        print(f"Train merged: {self.train_df.shape}, Test merged: {self.test_df.shape}")
        return self.train_df, self.test_df
