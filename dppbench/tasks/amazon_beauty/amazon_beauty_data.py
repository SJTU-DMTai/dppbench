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

        items = self.item_df.copy()

        # 商品维表必须有有效的商品键。
        missing_id = (
            items["item_id"].isna()
            | items["item_id"].astype("string").str.strip().eq("").fillna(False)
        )
        if missing_id.any():
            raise ValueError(
                f"AmazonBeauty metadata contains "
                f"{int(missing_id.sum())} rows without item_id"
            )

        # 缺失品牌统一为 NA，避免空字符串被当作另一个品牌。
        if "item_brand" in items.columns:
            items["item_brand"] = (
                items["item_brand"]
                .astype("string")
                .str.strip()
                .replace("", pd.NA)
            )

        # 先消除当前使用字段完全相同的重复记录。
        before = len(items)
        items = items.drop_duplicates()

        if "item_brand" in items.columns:
            # 相同商品出现多个不同的非空品牌，不能任意保留第一条。
            brand_counts = (
                items.groupby("item_id")["item_brand"]
                .nunique(dropna=True)
            )
            conflicts = brand_counts[brand_counts > 1]

            if not conflicts.empty:
                examples = conflicts.index[:10].tolist()
                raise ValueError(
                    f"AmazonBeauty metadata has conflicting brands for "
                    f"{len(conflicts)} items; examples={examples}"
                )

            # 此时每个商品最多一个非空品牌：
            # 合并重复记录，并优先保留该非空值。
            items = (
                items.groupby("item_id", sort=False, as_index=False)
                .agg(item_brand=("item_brand", "first"))
            )
        else:
            items = items.drop_duplicates(subset=["item_id"])

        self.item_df = items.reset_index(drop=True)

        print(
            f"[AmazonBeauty item metadata] "
            f"rows={before} -> {len(self.item_df)}, "
            f"unique_item_id={self.item_df['item_id'].is_unique}"
        )

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "timestamp": self.TIMESTAMP,
            "rating": self.NUMERIC,
            "item_brand": self.CATEGORICAL,
        })

        self._apply_configured_label_rule("rating")

        return self.interaction_df, self.item_df
