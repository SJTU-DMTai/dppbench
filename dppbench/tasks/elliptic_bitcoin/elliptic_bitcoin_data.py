import os
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd

from ...dataset import TabularData


class EllipticBitcoinData(TabularData):
    """Elliptic Bitcoin Transaction dataset.

    Three raw CSVs:
      * elliptic_txs_features.csv — header-less, 167 columns
            [txId, time_step, feat_0, ..., feat_164]; ~203k rows.
      * elliptic_txs_classes.csv  — header (txId, class); class in
            {"unknown", "1" (illicit), "2" (licit)}.
      * elliptic_txs_edgelist.csv — header (txId1, txId2); ~234k directed
            edges over the same node id space.

    Dirty / messy aspects requiring preprocessing:
      * 77% of nodes have ``class == "unknown"`` and must be retained as
        unlabeled but mask-excluded from loss/metrics.
      * 165 numeric features have wildly different magnitudes (need z-score).
      * ``txId`` is a long stringified integer; must be remapped to a dense
        contiguous node index for tensor edge_index.
      * Edge list contains duplicates that should be deduplicated.
      * Time steps 1-49 define the standard chronological train/val/test split.
    """

    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    ZIP_URL = "https://data.pyg.org/datasets/elliptic/elliptic_bitcoin_dataset.zip"

    LFS_URLS = {
        "elliptic_txs_features.csv":
            "https://media.githubusercontent.com/media/GuyenSoto/BTC/master/elliptic_txs_features.csv",
        "elliptic_txs_classes.csv":
            "https://media.githubusercontent.com/media/GuyenSoto/BTC/master/elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv":
            "https://media.githubusercontent.com/media/GuyenSoto/BTC/master/elliptic_txs_edgelist.csv",
    }

    REQUIRED_FILES = (
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
    )

    CLASS_MAP = {
        "unknown": -1,
        "-1": -1,
        -1: -1,
        "1": 1,
        1: 1,
        "2": 0,
        2: 0,
        "0": 0,
        0: 0,
    }

    def __init__(self, data_dir=None):
        super().__init__(name="EllipticBitcoin")
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        self.target_col = "class"
        self.id_col = "txId"
        self.graph = None

    # ------------------------------------------------------------------
    # download
    # ------------------------------------------------------------------
    def _all_present(self):
        return all(
            os.path.exists(os.path.join(self.data_dir, f))
            for f in self.REQUIRED_FILES
        )

    def _flatten_after_extract(self):
        """The PyG zip extracts into a sub-folder; flatten if so."""
        for root, _dirs, files in os.walk(self.data_dir):
            if root == self.data_dir:
                continue
            for f in files:
                src = os.path.join(root, f)
                dst = os.path.join(self.data_dir, f)
                if not os.path.exists(dst):
                    os.replace(src, dst)

    def _try_zip(self):
        os.makedirs(self.data_dir, exist_ok=True)
        zip_path = os.path.join(self.data_dir, "elliptic_bitcoin_dataset.zip")
        if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 1_000_000):
            print(f"Downloading Elliptic zip from {self.ZIP_URL} ...")
            req = urllib.request.Request(
                self.ZIP_URL,
                headers={"User-Agent": self.USER_AGENT, "Accept": "*/*"},
            )
            with urllib.request.urlopen(req, timeout=600) as resp, open(zip_path, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.data_dir)
        self._flatten_after_extract()

    def _try_lfs(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for fname, url in self.LFS_URLS.items():
            target = os.path.join(self.data_dir, fname)
            if os.path.exists(target) and os.path.getsize(target) > 1024:
                continue
            print(f"Downloading {fname} from GitHub LFS mirror ...")
            req = urllib.request.Request(
                url, headers={"User-Agent": self.USER_AGENT, "Accept": "*/*"}
            )
            with urllib.request.urlopen(req, timeout=600) as resp, open(target, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)

    def _download_if_missing(self):
        if self._all_present():
            return
        last_err = None
        for attempt_name, attempt_fn in (("PyG zip", self._try_zip),
                                         ("GitHub LFS", self._try_lfs)):
            try:
                attempt_fn()
                if self._all_present():
                    return
            except Exception as e:
                print(f"  [warn] {attempt_name} download failed: {e}")
                last_err = e
        if not self._all_present():
            raise RuntimeError(
                f"Failed to download Elliptic Bitcoin dataset. Place these files "
                f"manually in {self.data_dir}: {list(self.REQUIRED_FILES)}. "
                f"Last error: {last_err}"
            )

    def _normalize_tx_id_column(self, df, col):
        if df is None or col not in df.columns:
            return
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().all():
            df[col] = numeric.astype(np.int64)
        else:
            df[col] = df[col].astype(str)

    @classmethod
    def _normalize_class_value(cls, value):
        if pd.isna(value):
            return np.nan
        key = str(value).strip()
        if key in cls.CLASS_MAP:
            return cls.CLASS_MAP[key]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return np.nan
        numeric = int(numeric)
        return cls.CLASS_MAP.get(numeric, np.nan)

    @classmethod
    def _normalize_class_series(cls, series):
        return series.map(cls._normalize_class_value)

    def _class_lookup(self):
        classes = self.auxiliary_dfs.get("classes")
        if classes is None or "txId" not in classes.columns or "class" not in classes.columns:
            return {}
        mapped = self._normalize_class_series(classes["class"])
        return dict(zip(classes["txId"].astype(str), mapped))

    def _attach_class_labels(self, df):
        if df is None or "txId" not in df.columns:
            return df
        lookup = self._class_lookup()
        if "class" in df.columns:
            labels = self._normalize_class_series(df["class"])
        else:
            labels = pd.Series(np.nan, index=df.index)
        if lookup:
            labels = labels.fillna(df["txId"].astype(str).map(lookup))
        df["class"] = labels.fillna(-1).astype(np.int64)
        return df

    def load_std_test_frozen(self):
        loaded = super().load_std_test_frozen()
        if loaded and self.train_df is not None:
            self._normalize_tx_id_column(self.train_df, "txId")
            self._attach_class_labels(self.train_df)
        return loaded

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    def load_data(self):
        self._download_if_missing()
        feat_path = os.path.join(self.data_dir, "elliptic_txs_features.csv")
        cls_path = os.path.join(self.data_dir, "elliptic_txs_classes.csv")
        edge_path = os.path.join(self.data_dir, "elliptic_txs_edgelist.csv")

        # features: header-less; first col = txId, second = time_step, rest = feat_*
        feats = pd.read_csv(feat_path, header=None)
        n_feat = feats.shape[1] - 2
        col_names = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_feat)]
        feats.columns = col_names
        self._normalize_tx_id_column(feats, "txId")

        classes = pd.read_csv(cls_path)
        self._normalize_tx_id_column(classes, "txId")
        self.auxiliary_dfs["classes"] = classes

        self._attach_class_labels(feats)
        self.train_df = feats

        edges = pd.read_csv(edge_path)
        self._normalize_tx_id_column(edges, "txId1")
        self._normalize_tx_id_column(edges, "txId2")
        edges = edges.drop_duplicates(subset=["txId1", "txId2"]).reset_index(drop=True)
        self.auxiliary_dfs["edges"] = edges

        self.test_df = None
        print(
            f"Loaded features: {self.train_df.shape}, "
            f"classes: {classes.shape}, edges: {edges.shape}"
        )
        return self.train_df, self.test_df

    # ------------------------------------------------------------------
    # graph build (post pre-process)
    # ------------------------------------------------------------------
    def build_graph(self):
        if self.train_df is None:
            raise RuntimeError("call load_data() and run_pre_process() before build_graph()")

        df = self.train_df.copy()
        self._normalize_tx_id_column(df, "txId")
        self._attach_class_labels(df)
        edges = self.auxiliary_dfs.get("edges")
        if edges is None:
            raise RuntimeError("auxiliary edges table is missing")
        edges = edges.copy()
        self._normalize_tx_id_column(edges, "txId1")
        self._normalize_tx_id_column(edges, "txId2")
        edges = edges.drop_duplicates(subset=["txId1", "txId2"]).reset_index(drop=True)

        # Build txId -> dense node index.
        node_ids = df["txId"].astype(str).tolist()
        node_to_idx = {tx: i for i, tx in enumerate(node_ids)}
        num_nodes = len(node_to_idx)

        # Map edge list, drop edges referencing unknown ids.
        e1 = edges["txId1"].astype(str).map(node_to_idx)
        e2 = edges["txId2"].astype(str).map(node_to_idx)
        valid = e1.notna() & e2.notna()
        src = e1[valid].astype(np.int64).to_numpy()
        dst = e2[valid].astype(np.int64).to_numpy()
        edge_index = np.stack([src, dst], axis=0)

        # Feature matrix.
        feat_cols = [c for c in df.columns if c.startswith("feat_")]
        x = df[feat_cols].to_numpy(dtype=np.float32)

        # Labels: -1 unknown, 1 illicit, 0 licit.
        y_raw = pd.to_numeric(df["class"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

        # Time-step masks.
        ts = pd.to_numeric(df["time_step"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
        labeled = y_raw >= 0
        std_test_mask = (
            (df["__split__"].astype(str).to_numpy() == "std_test") & labeled
            if "__split__" in df.columns else np.zeros(len(df), dtype=bool)
        )
        train_mask = labeled & (ts <= 34)
        val_mask = labeled & (ts >= 35) & (ts <= 39)
        test_mask = labeled & (ts >= 40)
        if std_test_mask.any():
            train_mask = train_mask & ~std_test_mask
            val_mask = val_mask & ~std_test_mask
            test_mask = test_mask & ~std_test_mask

        self.graph = {
            "x": x,
            "edge_index": edge_index,
            "y": y_raw,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "std_test_mask": std_test_mask,
            "num_nodes": num_nodes,
            "time_step": ts,
        }
        print(
            f"Graph: nodes={num_nodes}, feats={x.shape[1]}, edges={edge_index.shape[1]}, "
            f"train={int(train_mask.sum())}, val={int(val_mask.sum())}, "
            f"test={int(test_mask.sum())}, std_test={int(std_test_mask.sum())}"
        )
        return self.graph

    def split(self, val_ratio=0.2, seed=42):
        # Graph tasks bypass tabular split; return passthrough so downstream code
        # that erroneously calls split() still gets something sensible.
        return {"train": self.train_df, "val": None, "test": None}
