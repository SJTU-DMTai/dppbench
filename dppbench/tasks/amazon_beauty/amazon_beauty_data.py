import os
import gzip
import json
import urllib.request
import pandas as pd
from ...dataset import RecData


class AmazonBeautyData(RecData):
    INTERACTION_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz"
    ITEM_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz"
    CANONICAL_K_CORE = 2
    CANONICAL_K_CORE_MAX_ITER = 100
    CANONICAL_DATA_PROTOCOL = {
        "canonical_k_core": {
            "user_col": "user_id",
            "item_col": "item_id",
            "k": CANONICAL_K_CORE,
            "max_iter": CANONICAL_K_CORE_MAX_ITER,
        }
    }

    # TODO: ignore text features
    INTERACTION_RENAME = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "reviewerName": "user_name",
        "unixReviewTime": "timestamp",
        # "reviewText": "review_text",
        "vote": "vote",
        # "summary": "summary",
        "overall": "rating",
    }

    ITEM_RENAME = {
        "asin": "item_id",
        "title": "item_name",
        # "feature": "item_feature",
        # "description": "item_description",
        "price": "item_price",
        "also_buy": "item_also_buy",
        "brand": "item_brand",
        "categories": "item_categories",
    }

    def __init__(self, data_dir=None):
        super().__init__(name="AmazonBeauty")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.interaction_df = None
        self.item_df = None
        self._item_id_related_cols = ["item_also_buy"]
        self.canonical_data_protocol = self.CANONICAL_DATA_PROTOCOL

    @staticmethod
    def _load_gz_json(filepath):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return pd.DataFrame(records)

    @staticmethod
    def _select_and_rename(raw_df, rename_map):
        wanted = [c for c in rename_map if c in raw_df.columns]
        return raw_df[wanted].rename(columns=rename_map)

    def _download_file(self, url, filename):
        filepath = os.path.join(self.data_dir, filename)
        if not os.path.exists(filepath):
            os.makedirs(self.data_dir, exist_ok=True)
            print(f"{self.name} Downloading {url}")
            urllib.request.urlretrieve(url, filepath)
        return filepath

    def load_data(self):
        interaction_file = self._download_file(self.INTERACTION_URL, "All_Beauty.json.gz")
        item_file = self._download_file(self.ITEM_URL, "meta_All_Beauty.json.gz")
        self.interaction_df = self._select_and_rename(self._load_gz_json(interaction_file), self.INTERACTION_RENAME)
        self.item_df = self._select_and_rename(self._load_gz_json(item_file), self.ITEM_RENAME)
        self._str2dense()
        self._apply_canonical_k_core()

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "user_name": self.CATEGORICAL,
            "timestamp": self.TIMESTAMP,
            "vote": self.NUMERIC,
            "rating": self.NUMERIC,
            "item_name": self.CATEGORICAL,
            "item_price": self.NUMERIC,
            "item_brand": self.CATEGORICAL,
            "item_categories": self.CATEGORICAL_LIST,
            "item_also_buy": self.CATEGORICAL_LIST,
        })

        self._apply_configured_label_rule("rating")

        return self.interaction_df, self.item_df

    def _apply_canonical_k_core(self):
        """Apply the fixed Amazon Beauty interaction core before std-test split.

        This is dataset protocol, not a tunable preprocessing operator. Keeping
        it here makes train/std-test construction independent of searched
        pipelines and prevents std-test size from changing with k-core params.
        """
        user_col = "user_id"
        item_col = "item_id"
        k = int(self.CANONICAL_K_CORE)
        if k <= 1:
            return
        if user_col not in self.interaction_df or item_col not in self.interaction_df:
            raise ValueError(
                f"AmazonBeauty canonical k-core requires {user_col}/{item_col}"
            )

        df = self.interaction_df.copy()
        before = len(df)
        prev_len = -1
        for _ in range(int(self.CANONICAL_K_CORE_MAX_ITER)):
            user_counts = df[user_col].value_counts()
            item_counts = df[item_col].value_counts()
            users = set(user_counts[user_counts >= k].index)
            items = set(item_counts[item_counts >= k].index)
            df = df[df[user_col].isin(users) & df[item_col].isin(items)]
            if len(df) == prev_len:
                break
            prev_len = len(df)

        self.interaction_df = df.reset_index(drop=True)
        print(
            f"{self.name} canonical k-core k={k}: "
            f"{before} -> {len(self.interaction_df)} interactions"
        )

    def _str2dense(self):
        # convert item_price and vote to dense features
        self.item_df["item_price"] = (
            self.item_df["item_price"]
            .astype(str)
            .str.replace(r"^\$", "", regex=True)
            .str.replace(",", "", regex=False)
        )
        self.item_df["item_price"] = pd.to_numeric(self.item_df["item_price"], errors="coerce")

        self.interaction_df["vote"] = (
            self.interaction_df["vote"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        self.interaction_df["vote"] = pd.to_numeric(self.interaction_df["vote"], errors="coerce").astype("Int64")
