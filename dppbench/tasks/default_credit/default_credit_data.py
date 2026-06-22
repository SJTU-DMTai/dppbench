import os
import urllib.request
import zipfile
import pandas as pd

from ...dataset import TabularData


class DefaultCreditData(TabularData):
    """UCI Taiwan Default of Credit Card Clients dataset.

    30k card holders with 6 months of repayment-status / billing / payment
    panels. The original wide-format panel columns are pivoted into a long
    auxiliary table ``monthly_history`` so the multi-table JoinTable pattern
    (a la home_credit) can be applied via the YAML pipeline.

    Source: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
    """

    DATA_URL = (
        "https://archive.ics.uci.edu/static/public/350/"
        "default+of+credit+card+clients.zip"
    )
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    ZIP_NAME = "default+of+credit+card+clients.zip"
    XLS_NAME = "default of credit card clients.xls"

    PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    BILL_COLS = [
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    ]
    PAY_AMT_COLS = [
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    ]

    def __init__(self, data_dir=None):
        super().__init__(name="DefaultCredit")
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        self.id_col = "ID"
        self.target_col = "TARGET"

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        xls_path = os.path.join(self.data_dir, self.XLS_NAME)
        if os.path.exists(xls_path) and os.path.getsize(xls_path) > 100_000:
            return

        zip_path = os.path.join(self.data_dir, self.ZIP_NAME)
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 100_000):
            print(f"Downloading Taiwan default credit data from {self.DATA_URL} ...")
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
                    f"Failed to download {self.DATA_URL}: {e}. "
                    f"Place {self.ZIP_NAME} in {self.data_dir} manually."
                )

        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)

    def load_data(self):
        self._download_if_missing()

        xls_path = os.path.join(self.data_dir, self.XLS_NAME)
        # The first row is a category header; the actual column names are on row 1.
        df = pd.read_excel(xls_path, header=1)
        df = df.rename(columns={"default payment next month": "TARGET"})

        # Build monthly_history aux table (wide -> long).
        records = []
        for m, (pay, bill, pay_amt) in enumerate(zip(
            self.PAY_COLS, self.BILL_COLS, self.PAY_AMT_COLS
        )):
            sub = df[["ID", pay, bill, pay_amt]].rename(columns={
                pay: "PAY_STATUS",
                bill: "BILL_AMT",
                pay_amt: "PAY_AMT",
            })
            sub["MONTH_OFFSET"] = m
            records.append(sub)
        monthly = pd.concat(records, ignore_index=True)

        panel_cols = self.PAY_COLS + self.BILL_COLS + self.PAY_AMT_COLS
        self.train_df = df.drop(columns=panel_cols).reset_index(drop=True)
        self.test_df = None
        self.auxiliary_dfs["monthly_history"] = monthly

        return self.train_df, self.test_df
