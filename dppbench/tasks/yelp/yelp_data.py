import os
import json
import tarfile
import zipfile
import urllib.request
import pandas as pd
from ...dataset import RecData


class YelpData(RecData):
    URL = "https://business.yelp.com/external-assets/files/Yelp-JSON.zip"

    INTERACTION_FIELDS = ["user_id", "business_id", "date", "stars", "useful", "funny", "cool"]
    INTERACTION_RENAME = {"business_id": "item_id"}

    USER_FIELDS = [
        "user_id", "name", "review_count", "yelping_since",
        "useful", "funny", "cool", "elite", "friends",
        "fans", "average_stars", "compliment_hot", "compliment_more",
        "compliment_profile", "compliment_cute", "compliment_list",
        "compliment_note", "compliment_plain", "compliment_cool",
        "compliment_funny", "compliment_writer", "compliment_photos",
    ]

    ITEM_FIELDS = ["business_id", "name", "postal_code", "stars", "is_open", "categories"]
    ITEM_RENAME = {
        "business_id": "item_id",
        "name": "item_name",
        "postal_code": "item_postal_code",
        "stars": "item_stars",
        "is_open": "item_is_open",
        "categories": "item_categories",
    }

    def __init__(self, data_dir=None):
        super().__init__(name="Yelp")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.interaction_df = None
        self.user_df = None
        self.item_df = None

    def _download_and_extract(self):
        zip_path = os.path.join(self.data_dir, "Yelp-JSON.zip")
        extract_dir = os.path.join(self.data_dir, "yelp")
        if not os.path.exists(extract_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            if not os.path.exists(zip_path):
                print(f"{self.name} Downloading {self.URL}")
                req = urllib.request.Request(
                    self.URL,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
                    out.write(resp.read())
            print(f"{self.name} Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
            tar_path = self._find_file(extract_dir, "yelp_dataset.tar")
            print(f"{self.name} Extracting {tar_path}")
            with tarfile.open(tar_path, "r") as tf:
                tf.extractall(extract_dir)
        return extract_dir

    @staticmethod
    def _load_json_lines(filepath, fields=None):
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if fields:
                    obj = {k: obj.get(k) for k in fields}
                records.append(obj)
        return pd.DataFrame(records)

    @staticmethod
    def _find_file(directory, filename):
        for root, _, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        raise FileNotFoundError(f"{filename} not found in {directory}")

    def load_data(self):
        extract_dir = self._download_and_extract()

        review_path = self._find_file(extract_dir, "yelp_academic_dataset_review.json")
        self.interaction_df = self._load_json_lines(review_path, self.INTERACTION_FIELDS)
        self.interaction_df = self.interaction_df.rename(columns=self.INTERACTION_RENAME)
        self.interaction_df["timestamp"] = (
            pd.to_datetime(self.interaction_df["date"], errors="coerce")
            .astype("int64") // 10**9
        )
        self.interaction_df = self.interaction_df.drop(columns=["date"])

        user_path = self._find_file(extract_dir, "yelp_academic_dataset_user.json")
        self.user_df = self._load_json_lines(user_path, self.USER_FIELDS)
        user_rename = {col: f"user_{col}" for col in self.user_df.columns if col != "user_id"}
        self.user_df = self.user_df.rename(columns=user_rename)
        self.user_df["user_yelping_since"] = (
            pd.to_datetime(self.user_df["user_yelping_since"], errors="coerce")
            .astype("int64") // 10**9
        )

        item_path = self._find_file(extract_dir, "yelp_academic_dataset_business.json")
        self.item_df = self._load_json_lines(item_path, self.ITEM_FIELDS)
        self.item_df = self.item_df.rename(columns=self.ITEM_RENAME)
        self.item_df["item_categories"] = self.item_df["item_categories"].apply(
            lambda x: [c.strip() for c in x.split(",")] if isinstance(x, str) else []
        )

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "stars": self.NUMERIC,
            "useful": self.NUMERIC,
            "funny": self.NUMERIC,
            "cool": self.NUMERIC,
            "timestamp": self.TIMESTAMP,
            "user_name": self.TEXT,
            "user_review_count": self.NUMERIC,
            "user_yelping_since": self.TIMESTAMP,
            "user_useful": self.NUMERIC,
            "user_funny": self.NUMERIC,
            "user_cool": self.NUMERIC,
            "user_elite": self.TEXT,
            "user_friends": self.TEXT,
            "user_fans": self.NUMERIC,
            "user_average_stars": self.NUMERIC,
            "user_compliment_hot": self.NUMERIC,
            "user_compliment_more": self.NUMERIC,
            "user_compliment_profile": self.NUMERIC,
            "user_compliment_cute": self.NUMERIC,
            "user_compliment_list": self.NUMERIC,
            "user_compliment_note": self.NUMERIC,
            "user_compliment_plain": self.NUMERIC,
            "user_compliment_cool": self.NUMERIC,
            "user_compliment_funny": self.NUMERIC,
            "user_compliment_writer": self.NUMERIC,
            "user_compliment_photos": self.NUMERIC,
            "item_name": self.TEXT,
            "item_postal_code": self.CATEGORICAL,
            "item_stars": self.NUMERIC,
            "item_is_open": self.CATEGORICAL,
            "item_categories": self.CATEGORICAL_LIST,
        })

        self._apply_configured_label_rule("stars")

        return self.interaction_df, self.user_df, self.item_df
