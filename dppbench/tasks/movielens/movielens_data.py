import os
import zipfile
import urllib.request
import pandas as pd
from ...dataset import RecData


class MovielensData(RecData):
    URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

    def __init__(self, data_dir=None):
        super().__init__(name="Movielens")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.interaction_df = None
        self.item_df = None
        self.user_df = None

    def _download_and_extract(self):
        zip_path = os.path.join(self.data_dir, "ml-1m.zip")
        extract_dir = os.path.join(self.data_dir, "ml-1m")
        if not os.path.exists(extract_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            if not os.path.exists(zip_path):
                print(f"{self.name} Downloading {self.URL}")
                urllib.request.urlretrieve(self.URL, zip_path)
            print(f"{self.name} Extracting {zip_path}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
        return extract_dir

    def load_data(self):
        extract_dir = self._download_and_extract()

        self.interaction_df = pd.read_csv(
            os.path.join(extract_dir, "ratings.dat"),
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "item_id", "rating", "timestamp"],
            encoding="latin-1",
        )

        self.user_df = pd.read_csv(
            os.path.join(extract_dir, "users.dat"),
            sep="::",
            engine="python",
            header=None,
            names=["user_id", "user_gender", "user_age", "user_occupation", "user_zip_code"],
            encoding="latin-1",
        )

        self.item_df = pd.read_csv(
            os.path.join(extract_dir, "movies.dat"),
            sep="::",
            engine="python",
            header=None,
            names=["item_id", "item_title", "item_genres"],
            encoding="latin-1",
        )
        self.item_df["item_genres"] = self.item_df["item_genres"].apply(
            lambda x: x.split("|") if isinstance(x, str) else []
        )

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "rating": self.NUMERIC,
            "timestamp": self.TIMESTAMP,
            "user_gender": self.CATEGORICAL,
            "user_age": self.CATEGORICAL,
            "user_occupation": self.CATEGORICAL,
            "user_zip_code": self.CATEGORICAL,
            "item_title": self.TEXT,
            "item_genres": self.CATEGORICAL_LIST,
        })

        self._apply_configured_label_rule("rating")

        return self.interaction_df, self.user_df, self.item_df
