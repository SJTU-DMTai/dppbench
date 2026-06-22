import ast
import os
import shutil
import urllib.request
import zipfile

import pandas as pd

from ...dataset import RecData


class KuairecData(RecData):
    """KuaiRec recommendation dataset.

    Official site: https://kuairec.com/
    Repository: https://github.com/chongminggao/KuaiRec

    The loader defaults to ``small_matrix.csv`` because it is the near fully
    observed matrix described by the paper. Missing local files are downloaded
    from the official Zenodo archive.
    """

    URL = "https://zenodo.org/records/18164998/files/KuaiRec.zip"
    ENV_URLS = "DPPBENCH_KUAIREC_URL"
    ARCHIVE_NAME = "KuaiRec.zip"

    MATRIX_FILES = {
        "small": "small_matrix.csv",
        "big": "big_matrix.csv",
    }
    SIDE_FILES = (
        "user_features.csv",
        "item_categories.csv",
        "item_daily_features.csv",
    )
    OPTIONAL_FILES = (
        "social_network.csv",
        "kuairec_caption_category.csv",
    )

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    ITEM_DAILY_COLS = [
        "video_id",
        "date",
        "author_id",
        "video_type",
        "upload_type",
        "visible_status",
        "video_duration",
        "video_width",
        "video_height",
        "music_id",
        "video_tag_id",
    ]
    ITEM_DAILY_RENAME = {
        "video_id": "item_id",
        "author_id": "item_author_id",
        "video_type": "item_video_type",
        "upload_type": "item_upload_type",
        "visible_status": "item_visible_status",
        "video_duration": "item_video_duration",
        "video_width": "item_video_width",
        "video_height": "item_video_height",
        "music_id": "item_music_id",
        "video_tag_id": "item_video_tag_id",
    }

    def __init__(self, data_dir=None, matrix="small"):
        super().__init__(name="KuaiRec")
        if matrix not in self.MATRIX_FILES:
            raise ValueError(
                f"matrix must be one of {sorted(self.MATRIX_FILES)}, got {matrix!r}"
            )
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.matrix = matrix
        self.interaction_df = None
        self.item_df = None
        self.user_df = None
        self._user_id_related_cols = ["user_friend_list"]

    @staticmethod
    def _parse_list(value):
        if isinstance(value, list):
            return value
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        if isinstance(parsed, list):
            return parsed
        return []

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

    def _required_files(self):
        return (self.MATRIX_FILES[self.matrix],) + self.SIDE_FILES

    def _file_path(self, filename):
        return os.path.join(self.data_dir, filename)

    def _missing_required_files(self):
        self._promote_data_files()
        return [
            name
            for name in self._required_files()
            if not os.path.exists(self._file_path(name))
        ]

    def _promote_data_files(self):
        wanted = set(self.MATRIX_FILES.values())
        wanted.update(self.SIDE_FILES)
        wanted.update(self.OPTIONAL_FILES)
        if not os.path.isdir(self.data_dir):
            return

        for filename in wanted:
            root_path = self._file_path(filename)
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

    def _download_file(self, url, filename):
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = self._file_path(filename)
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

    def _extract_archives(self):
        if not os.path.isdir(self.data_dir):
            return
        for name in os.listdir(self.data_dir):
            if not name.lower().endswith(".zip"):
                continue
            path = self._file_path(name)
            if not zipfile.is_zipfile(path):
                continue
            print(f"{self.name} Extracting {path}")
            self._safe_extract_zip(path, self.data_dir)
        self._promote_data_files()

    def _try_download_from_urls(self):
        urls = [
            url.strip()
            for url in os.environ.get(self.ENV_URLS, "").split(",")
            if url.strip()
        ]
        urls.append(self.URL)

        for i, url in enumerate(urls, start=1):
            archive_name = os.path.basename(url.split("?", 1)[0]) or self.ARCHIVE_NAME
            if not archive_name.lower().endswith(".zip"):
                archive_name = f"kuairec-{i}.zip"
            try:
                archive_path = self._download_file(url, archive_name)
                self._safe_extract_zip(archive_path, self.data_dir)
                self._promote_data_files()
                if not self._missing_required_files():
                    return True
            except Exception as exc:
                print(f"{self.name} download failed: {type(exc).__name__}: {exc}")
        return False

    def _download_if_missing(self):
        os.makedirs(self.data_dir, exist_ok=True)
        if not self._missing_required_files():
            return
        self._extract_archives()
        if not self._missing_required_files():
            return
        if self._try_download_from_urls():
            return

        missing = self._missing_required_files()
        raise RuntimeError(
            f"{self.name} data files are missing: {missing}. "
            f"Set {self.ENV_URLS} to a direct KuaiRec archive URL, or place "
            f"the official files under {self.data_dir}."
        )

    def _load_interactions(self):
        filename = self.MATRIX_FILES[self.matrix]
        df = pd.read_csv(
            self._file_path(filename),
            dtype={
                "user_id": "int64",
                "video_id": "int64",
                "play_duration": "float32",
                "video_duration": "float32",
                "timestamp": "float64",
                "watch_ratio": "float32",
            },
        )
        df = df.rename(columns={"video_id": "item_id"})
        return df

    def _load_users(self):
        user_df = pd.read_csv(self._file_path("user_features.csv"))

        social_path = self._file_path("social_network.csv")
        if os.path.exists(social_path):
            social_df = pd.read_csv(social_path)
            social_df = social_df.rename(columns={"friend_list": "user_friend_list"})
            social_df["user_friend_list"] = social_df["user_friend_list"].apply(
                self._parse_list
            )
            user_df = user_df.merge(social_df, on="user_id", how="left")
        else:
            user_df["user_friend_list"] = [[] for _ in range(len(user_df))]

        user_df["user_friend_list"] = user_df["user_friend_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        return user_df

    def _load_items(self):
        categories = pd.read_csv(self._file_path("item_categories.csv"))
        categories = categories.rename(
            columns={"video_id": "item_id", "feat": "item_categories"}
        )
        categories["item_categories"] = categories["item_categories"].apply(
            self._parse_list
        )

        daily = pd.read_csv(
            self._file_path("item_daily_features.csv"),
            usecols=self.ITEM_DAILY_COLS,
        )
        daily = daily.sort_values(["video_id", "date"]).drop_duplicates(
            "video_id", keep="last"
        )
        daily = daily.drop(columns=["date"]).rename(columns=self.ITEM_DAILY_RENAME)

        return categories.merge(daily, on="item_id", how="left")

    def load_data(self):
        self._download_if_missing()

        self.interaction_df = self._load_interactions()
        self.user_df = self._load_users()
        self.item_df = self._load_items()

        self.register_col_types({
            "user_id": self.CATEGORICAL,
            "item_id": self.CATEGORICAL,
            "play_duration": self.NUMERIC,
            "video_duration": self.NUMERIC,
            "time": self.TEXT,
            "date": self.TIMESTAMP,
            "timestamp": self.TIMESTAMP,
            "watch_ratio": self.NUMERIC,
            "user_active_degree": self.CATEGORICAL,
            "is_lowactive_period": self.CATEGORICAL,
            "is_live_streamer": self.CATEGORICAL,
            "is_video_author": self.CATEGORICAL,
            "follow_user_num": self.NUMERIC,
            "follow_user_num_range": self.CATEGORICAL,
            "fans_user_num": self.NUMERIC,
            "fans_user_num_range": self.CATEGORICAL,
            "friend_user_num": self.NUMERIC,
            "friend_user_num_range": self.CATEGORICAL,
            "register_days": self.NUMERIC,
            "register_days_range": self.CATEGORICAL,
            "user_friend_list": self.CATEGORICAL_LIST,
            "item_categories": self.CATEGORICAL_LIST,
            "item_author_id": self.CATEGORICAL,
            "item_video_type": self.CATEGORICAL,
            "item_upload_type": self.CATEGORICAL,
            "item_visible_status": self.CATEGORICAL,
            "item_video_duration": self.NUMERIC,
            "item_video_width": self.NUMERIC,
            "item_video_height": self.NUMERIC,
            "item_music_id": self.CATEGORICAL,
            "item_video_tag_id": self.CATEGORICAL,
        })
        self.register_col_types({
            f"onehot_feat{i}": self.CATEGORICAL for i in range(18)
        })

        self._apply_configured_label_rule("watch_ratio")
        return self.interaction_df, self.user_df, self.item_df
