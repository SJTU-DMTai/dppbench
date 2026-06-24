import os
import gzip
import json
import urllib.request
import pandas as pd
from ...dataset import RecData


class AmazonBeautyData(RecData):
    INTERACTION_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/All_Beauty.json.gz"
    ITEM_URL = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_All_Beauty.json.gz"

    INTERACTION_RENAME = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "unixReviewTime": "timestamp",
        "overall": "rating",
    }

    ITEM_RENAME = {
        "asin": "item_id",
        "brand": "item_brand",
    }

    def __init__(self, data_dir=None):
        super().__init__(name="AmazonBeauty")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.interaction_df = None
        self.item_df = None
        self._item_id_related_cols = []

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

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "timestamp": self.TIMESTAMP,
            "rating": self.NUMERIC,
            "item_brand": self.CATEGORICAL,
        })

        self._apply_configured_label_rule("rating")

        return self.interaction_df, self.item_df
