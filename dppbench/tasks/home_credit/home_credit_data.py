import os
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import pandas as pd
from ...dataset import TabularData

class _Redirect308Handler(urllib.request.HTTPRedirectHandler):
    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_301(req, fp, code, msg, headers)


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
    ENV_HF_ENDPOINTS = "DPPBENCH_HOME_CREDIT_HF_ENDPOINTS"
    HF_ENDPOINT_ENV = "HF_ENDPOINT"
    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    ARCHIVE_NAME = "home-credit-default-risk.zip"

    HF_REPO_PATH = "datasets/jamirc/home_credit_default_risk"
    HF_API_PATH = "api/datasets/jamirc/home_credit_default_risk"
    HF_DEFAULT_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
    HF_FILE_CANDIDATES = {
        "application_train.csv": ("application_tr.csv", "application_train.csv"),
        "application_test.csv": ("application_ts.csv", "application_test.csv"),
    }

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

    @staticmethod
    def _urlopen(req, timeout):
        opener = urllib.request.build_opener(_Redirect308Handler)
        try:
            return opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code != 308:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            redirected = urllib.request.Request(
                urllib.parse.urljoin(req.full_url, location),
                headers=dict(req.header_items()),
            )
            return opener.open(redirected, timeout=timeout)

    def _download_file(self, url, filename):
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            return filepath

        print(f"{self.name} Downloading {url}")
        tmp_path = filepath + ".part"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "*/*",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with self._urlopen(req, timeout=900) as resp, \
                    open(tmp_path, "wb") as f:
                expected = resp.headers.get("Content-Length")
                expected_size = int(expected) if expected and expected.isdigit() else None
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                actual_size = f.tell()
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"incomplete download for {url}: "
                    f"expected {expected_size} bytes, got {actual_size}"
                )
            if actual_size == 0:
                raise IOError(f"empty download for {url}")
            os.replace(tmp_path, filepath)
        except Exception:
            for path in (tmp_path, filepath):
                if os.path.exists(path):
                    os.remove(path)
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
            alias_names = set(self.HF_FILE_CANDIDATES.get(filename, ()))
            for dirpath, _, filenames in os.walk(self.data_dir):
                match = None
                for candidate in filenames:
                    if (
                        candidate == filename
                        or candidate in alias_names
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

    @staticmethod
    def _split_env_urls(value):
        return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]

    def _hf_endpoints(self):
        endpoints = []
        endpoints.extend(self._split_env_urls(os.environ.get(self.ENV_HF_ENDPOINTS, "")))
        hf_endpoint = os.environ.get(self.HF_ENDPOINT_ENV, "").strip().rstrip("/")
        if hf_endpoint:
            endpoints.append(hf_endpoint)
        endpoints.extend(self.HF_DEFAULT_ENDPOINTS)

        seen = set()
        unique = []
        for endpoint in endpoints:
            if not endpoint or endpoint in seen:
                continue
            seen.add(endpoint)
            unique.append(endpoint)
        return unique

    def _hf_api_url(self, endpoint):
        return f"{endpoint}/{self.HF_API_PATH}"

    def _hf_file_url(self, endpoint, remote_name):
        return f"{endpoint}/{self.HF_REPO_PATH}/resolve/main/{remote_name}"

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
                self._download_file(url, archive_name)
                self._extract_archives()
                if not self._missing_required_files():
                    return True
            except Exception as exc:
                print(f"{self.name} mirror download failed: {type(exc).__name__}: {exc}")
        return False

    def _huggingface_siblings(self):
        for endpoint in self._hf_endpoints():
            req = urllib.request.Request(
                self._hf_api_url(endpoint),
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": "application/json",
                },
            )
            try:
                with self._urlopen(req, timeout=60) as resp:
                    payload = json.load(resp)
            except Exception as exc:
                print(
                    f"{self.name} HuggingFace file listing failed at "
                    f"{endpoint}: {exc}"
                )
                continue
            siblings = {
                item.get("rfilename")
                for item in payload.get("siblings", [])
                if item.get("rfilename")
            }
            return siblings, endpoint
        return None, None

    def _huggingface_remote_candidates(self, local_name, siblings):
        candidates = self.HF_FILE_CANDIDATES.get(local_name, (local_name,))
        if siblings is None:
            return candidates
        present = [name for name in candidates if name in siblings]
        return present or candidates

    def _try_download_huggingface(self):
        required = list(self.REQUIRED_FILES) + list(self.AUX_FILES.values())
        os.makedirs(self.data_dir, exist_ok=True)
        siblings, listed_endpoint = self._huggingface_siblings()
        endpoints = [listed_endpoint] if listed_endpoint else self._hf_endpoints()
        for local_name in required:
            target = os.path.join(self.data_dir, local_name)
            if os.path.exists(target) and os.path.getsize(target) > 0:
                continue
            downloaded = False
            for endpoint in endpoints:
                for remote_name in self._huggingface_remote_candidates(local_name, siblings):
                    url = self._hf_file_url(endpoint, remote_name)
                    print(
                        f"{self.name} Downloading {local_name} from "
                        f"{endpoint} file {remote_name}..."
                    )
                    try:
                        self._download_file(url, local_name)
                        downloaded = True
                        break
                    except Exception as e:
                        print(
                            f"{self.name} HuggingFace download failed for "
                            f"{local_name} from {endpoint}/{remote_name}: {e}"
                        )
                if downloaded:
                    break
            if not downloaded:
                return False
        self._promote_required_files()
        return not self._missing_required_files()

    def _kaggle_credentials_available(self):
        if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
            return True
        config_dirs = []
        if os.environ.get("KAGGLE_CONFIG_DIR"):
            config_dirs.append(os.environ["KAGGLE_CONFIG_DIR"])
        config_dirs.extend(["~/.config/kaggle", "~/.kaggle"])
        return any(
            os.path.exists(os.path.expanduser(os.path.join(d, "kaggle.json")))
            for d in config_dirs
        )

    def _run_kaggle_download(self):
        os.makedirs(self.data_dir, exist_ok=True)
        kaggle = shutil.which("kaggle")
        if not kaggle:
            return False
        if not self._kaggle_credentials_available():
            print(
                f"{self.name} Kaggle CLI found but kaggle.json or "
                f"KAGGLE_USERNAME/KAGGLE_KEY is missing; skipping Kaggle download."
            )
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
        if self._try_download_huggingface():
            return
        if self._run_kaggle_download():
            return

        missing = self._missing_required_files()
        raise RuntimeError(
            f"{self.name} data files are missing: {missing}. "
            f"Set {self.ENV_URLS} to a direct archive URL, set "
            f"{self.ENV_HF_ENDPOINTS}=https://hf-mirror.com or HF_ENDPOINT to "
            f"a reachable HuggingFace-compatible endpoint, install/configure "
            f"Kaggle CLI and accept the competition rules at {self.COMPETITION_URL}, "
            f"or download the files manually from HuggingFace "
            f"(jamirc/home_credit_default_risk; application_tr.csv maps to "
            f"application_train.csv and application_ts.csv maps to application_test.csv)."
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
