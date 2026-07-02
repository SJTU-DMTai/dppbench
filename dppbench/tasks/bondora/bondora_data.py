import os
import urllib.request
import pandas as pd

from ...dataset import TabularData
from .._excel_cache import read_excel_cached


class BondoraData(TabularData):
    """Bondora P2P loan default prediction dataset.

    The original ``https://www.bondora.com/marketing/media/LoanData.zip``
    endpoint is no longer served (404 / redirect to marketing page). The
    public statistics page now exposes a single XLSX with 31 columns at
    the Azure blob URL below.

    Source: https://www.bondora.com/en/public-statistics
    Direct: https://sabanners001.blob.core.windows.net/statistics/public/loan_dataset_investor.xlsx
    """

    DATA_URL = (
        "https://sabanners001.blob.core.windows.net/statistics/public/"
        "loan_dataset_investor.xlsx"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    XLSX_NAME = "loan_dataset_investor.xlsx"
    SHEET_NAME = "Loan Dataset"
    # Drop fields whose value or missingness strongly reveals the terminal loan outcome. 
    LEAKAGE_COLS = [
        "loan_last_recorded_action_date_local",
        "principal_balance",
        "principal_debt",
        "principal_paid_total",
        "is_default",
        "debt_occured_date_local",
        "months_in_default",
        "loan_status_risk",
        "early_repaid_at",
        "is_early_repaid_within_14_days",
        "nr_of_payments",
    ]

    def __init__(self, data_dir=None):
        super().__init__(name="Bondora")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.id_col = "loan_id"
        self.target_col = "target"

    # ------------------------------------------------------------------
    # Download / extraction
    # ------------------------------------------------------------------
    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        xlsx_path = os.path.join(self.data_dir, self.XLSX_NAME)

        if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 1_000_000:
            return

        print(f"Downloading Bondora data from {self.DATA_URL} ...")
        req = urllib.request.Request(
            self.DATA_URL,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, \
                    open(xlsx_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:
            if os.path.exists(xlsx_path):
                os.remove(xlsx_path)
            raise RuntimeError(
                f"Failed to download Bondora data from {self.DATA_URL}. "
                f"Please put {self.XLSX_NAME} into {self.data_dir} manually. "
                f"Original error: {e}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_data(self):
        self._download_if_missing()

        xlsx_path = os.path.join(self.data_dir, self.XLSX_NAME)
        loans = read_excel_cached(xlsx_path, sheet_name=self.SHEET_NAME)

        # Keep only loans that have reached a terminal outcome (Repaid or
        # Defaulted). Active / Returned loans have no ground-truth label
        # for default prediction.
        loans = loans[loans["loan_status"].isin(["Repaid", "Defaulted"])].copy()
        loans["target"] = (loans["loan_status"] == "Defaulted").astype(int)
        loans = loans.drop(columns=["loan_status"])
        loans = loans.drop(columns=self.LEAKAGE_COLS, errors="ignore")

        # Build a per-country auxiliary table to demonstrate the multi-table
        # JoinTable aggregation pattern (same shape as home_credit's bureau / installments).
        # NB: only non-target features are aggregated to avoid leakage.
        aux_country = (
            loans.groupby("country")
            .agg(
                country_loan_cnt=("loan_id", "count"),
                country_avg_amount=("issued_amount", "mean"),
                country_avg_interest=("initial_interest_rate", "mean"),
                country_avg_duration=("initial_loan_duration", "mean"),
            )
            .reset_index()
        )

        self.train_df = loans.reset_index(drop=True)
        self.test_df = None
        self.auxiliary_dfs["country_stats"] = aux_country

        return self.train_df, self.test_df
