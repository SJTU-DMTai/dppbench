import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from torch.utils.data import DataLoader

from .inputs import (
    SparseFeat, DenseFeat, VarLenSparseFeat,
    build_input_features, create_embedding_matrix,
    embedding_lookup, varlen_embedding_lookup, get_varlen_pooling_list,
    combined_dnn_input, df_to_input,
)
from .layers import PredictionLayer


class BaseModel(nn.Module):
    def __init__(self, seed=1024, task='binary', device='cpu'):
        super().__init__()
        torch.manual_seed(seed)
        self.task = task
        self.device = device
        self.out = PredictionLayer(task)
        self.metrics = {}
        self.loss_func = None
        self.optim = None

    def compile(self, optimizer='adam', lr=0.001, loss=None, metrics=None, weight_decay=0.0):
        if optimizer == 'adam':
            self.optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'sgd':
            self.optim = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == 'adagrad':
            self.optim = torch.optim.Adagrad(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.optim = optimizer

        if loss is None:
            self.loss_func = F.binary_cross_entropy if self.task == 'binary' else F.mse_loss
        elif isinstance(loss, str):
            self.loss_func = {'binary_crossentropy': F.binary_cross_entropy,
                              'mse': F.mse_loss}[loss]
        else:
            self.loss_func = loss

        self.metrics = {}
        metrics = metrics or (['auc'] if self.task == 'binary' else ['mse'])
        metric_map = {
            'auc': roc_auc_score,
            'logloss': log_loss,
            'mse': mean_squared_error,
        }
        for m in metrics:
            if isinstance(m, str):
                self.metrics[m] = metric_map[m]
            else:
                self.metrics[m.__name__] = m

    def fit(self, x, y, batch_size=256, epochs=1, verbose=1,
            validation_data=None, shuffle=True,
            early_stopping_monitor=None, early_stopping_patience=0):
        if isinstance(x, dict):
            x = [x[feat] for feat in self.feature_index]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1)),
            torch.from_numpy(y),
        )
        train_loader = DataLoader(train_tensor, shuffle=shuffle, batch_size=batch_size)

        do_val = validation_data is not None
        if do_val:
            val_x, val_y = validation_data[0], validation_data[1]
            if isinstance(val_x, dict):
                val_x = [val_x[feat] for feat in self.feature_index]

        use_es = do_val and early_stopping_monitor is not None and early_stopping_patience > 0
        higher_better = {"auc": True, "logloss": False, "mse": False}
        if use_es:
            es_higher = higher_better.get(early_stopping_monitor, True)
            best_score = -np.inf if es_higher else np.inf
            wait = 0
            best_weights = None

        history = {"loss": []}

        for epoch in range(epochs):
            model = self.train()
            start = time.time()
            loss_epoch = 0
            n_samples = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                y_pred = model(x_batch).squeeze()
                loss = self.loss_func(y_pred, y_batch, reduction='mean')

                if hasattr(self, 'get_regularization_loss'):
                    loss = loss + self.get_regularization_loss().squeeze()

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss_epoch += loss.item() * len(y_batch)
                n_samples += len(y_batch)

            avg_loss = loss_epoch / n_samples
            history["loss"].append(avg_loss)

            if verbose > 0:
                elapsed = int(time.time() - start)
                msg = f"Epoch {epoch + 1}/{epochs} - {elapsed}s - loss: {avg_loss:.4f}"
                if do_val:
                    val_result = self.evaluate(val_x, val_y, batch_size)
                    for k, v in val_result.items():
                        # msg += f" - val_{k}: {v:.4f}"
                        history.setdefault(f"val_{k}", []).append(v)
                print(msg)

            if use_es:
                monitor_key = f"val_{early_stopping_monitor}"
                current = history[monitor_key][-1]
                improved = (current > best_score) if es_higher else (current < best_score)
                if improved:
                    best_score = current
                    wait = 0
                    best_weights = copy.deepcopy(self.state_dict())
                else:
                    wait += 1
                    if wait >= early_stopping_patience:
                        if verbose > 0:
                            print(f"Early stopping at epoch {epoch + 1}, "
                                  f"best val_{early_stopping_monitor}: {best_score:.4f}")
                        break

        if use_es and best_weights is not None:
            self.load_state_dict(best_weights)

        return history

    def evaluate(self, x, y, batch_size=256):
        pred = self.predict(x, batch_size)
        result = {}
        for name, metric_fn in self.metrics.items():
            try:
                result[name] = float(metric_fn(y, pred))
            except (ValueError, ZeroDivisionError):
                result[name] = float("nan")
        return result

    def predict(self, x, batch_size=256):
        if isinstance(x, dict):
            x = [x[feat] for feat in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(x, axis=-1)))
        loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=False)

        was_training = self.training
        self.eval()
        preds = []
        try:
            with torch.no_grad():
                for (x_batch,) in loader:
                    x_batch = x_batch.to(self.device).float()
                    y_pred = self.forward(x_batch).cpu().numpy()
                    preds.append(y_pred)
        finally:
            if was_training:
                self.train()
        return np.concatenate(preds).squeeze()

    @staticmethod
    def _sample_negative_item(rng, item_pool, seen, weights=None, max_tries=50):
        n_items = len(item_pool)
        if n_items == 0:
            return None
        for _ in range(max_tries):
            if weights is None:
                idx = int(rng.integers(0, n_items))
            else:
                idx = int(rng.choice(n_items, p=weights))
            item = item_pool[idx]
            if item not in seen:
                return item
        candidates = [item for item in item_pool if item not in seen]
        if not candidates:
            return None
        return candidates[int(rng.integers(0, len(candidates)))]

    @staticmethod
    def _build_item_feature_map(df, item_col, item_feature_cols):
        if not item_feature_cols:
            return {}
        item_df = df[[item_col] + item_feature_cols].drop_duplicates(
            subset=[item_col], keep="first"
        )
        return item_df.set_index(item_col).to_dict(orient="index")

    def _augment_with_negative_samples(self, train_df, test_df, cfg):
        train_cfg = cfg.get("train", {})
        neg_cfg = train_cfg.get("negative_sampling") or {}
        if not neg_cfg or not bool(neg_cfg.get("enabled", False)):
            return train_df

        feat_cfg = cfg.get("feature", {})
        target_col = feat_cfg.get("target_col")
        if not target_col or target_col not in train_df.columns:
            return train_df

        user_col = neg_cfg.get("user_col", "user_id")
        item_col = neg_cfg.get("item_col", "item_id")
        if user_col not in train_df.columns or item_col not in train_df.columns:
            return train_df

        label_rule = feat_cfg.get("label_rule", {}) or {}
        positive_label = label_rule.get("positive_label", 1)
        negative_label = label_rule.get("negative_label", 0)
        num_negatives = int(neg_cfg.get("num_negatives", 1))
        if num_negatives <= 0:
            return train_df

        full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
        item_pool = pd.unique(full_df[item_col].dropna())
        if len(item_pool) == 0:
            return train_df

        sampler = str(neg_cfg.get("sampler", "random")).lower()
        weights = None
        if sampler in {"popular", "popularity"}:
            pop = full_df[item_col].value_counts()
            alpha = float(neg_cfg.get("popularity_alpha", 0.75))
            weights = np.array([float(pop.get(item, 0)) ** alpha for item in item_pool])
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None

        seen_source = full_df
        if bool(neg_cfg.get("exclude_positive_only", True)):
            seen_source = seen_source[seen_source[target_col] == positive_label]
        user_seen = (
            seen_source.groupby(user_col)[item_col].apply(set).to_dict()
        )

        seq_cols = set(feat_cfg.get("seq_cols", {}).keys())
        item_feature_cols = neg_cfg.get("item_feature_cols")
        if item_feature_cols is None:
            prefixes = tuple(neg_cfg.get("item_feature_prefixes", ["item_"]))
            item_feature_cols = [
                col for col in full_df.columns
                if col != item_col and any(str(col).startswith(p) for p in prefixes)
            ]
        item_feature_cols = [
            col for col in item_feature_cols
            if col in full_df.columns and col not in seq_cols and col != target_col
        ]
        item_feature_map = self._build_item_feature_map(
            full_df, item_col, item_feature_cols
        )
        interaction_defaults = neg_cfg.get("interaction_defaults", {}) or {}

        anchors = train_df[train_df[target_col] == positive_label]
        rng = np.random.default_rng(int(neg_cfg.get("seed", train_cfg.get("seed", 42))))
        max_anchors = neg_cfg.get("max_anchors")
        if max_anchors is not None and len(anchors) > int(max_anchors):
            anchors = anchors.sample(
                n=int(max_anchors),
                random_state=int(neg_cfg.get("seed", train_cfg.get("seed", 42))),
            )

        neg_rows = []
        for _, anchor in anchors.iterrows():
            seen = set(user_seen.get(anchor[user_col], set()))
            for _ in range(num_negatives):
                neg_item = self._sample_negative_item(rng, item_pool, seen, weights=weights)
                if neg_item is None:
                    continue
                seen.add(neg_item)
                row = anchor.copy(deep=True)
                row[item_col] = neg_item
                row[target_col] = negative_label
                for col, value in item_feature_map.get(neg_item, {}).items():
                    row[col] = value
                for col, value in interaction_defaults.items():
                    if col in row.index:
                        row[col] = value
                neg_rows.append(row)

        if not neg_rows:
            return train_df

        neg_df = pd.DataFrame(neg_rows, columns=train_df.columns)
        augmented = pd.concat([train_df, neg_df], ignore_index=True, sort=False)
        print(
            "Added training negatives: "
            f"{len(neg_df)} ({num_negatives} per positive, sampler={sampler})"
        )
        return augmented.sample(
            frac=1.0,
            random_state=int(neg_cfg.get("seed", train_cfg.get("seed", 42))),
        ).reset_index(drop=True)

    def train_and_evaluate(self, train_df, test_df, feature_columns, cfg):
        """Train the model on ``train_df`` and evaluate on ``test_df``.

        Validation rows are sliced out of ``train_df`` here (rather than via
        a preprocessing operator) so baselines cannot reshape the train/val
        split. ``train.val_ratio`` (default ``0.1``) and ``train.seed``
        (default ``42``) in the model config control the slice; for rec
        tasks we use a stratified split on the binary label when available.
        """
        from sklearn.model_selection import train_test_split

        train_cfg = cfg["train"]
        val_ratio = float(train_cfg.get("val_ratio", 0.1))
        seed = int(train_cfg.get("seed", 42))

        train_df = self._augment_with_negative_samples(train_df, test_df, cfg)

        target_col = cfg.get("feature", {}).get("target_col")
        stratify = None
        if target_col and target_col in train_df.columns:
            tgt = train_df[target_col]
            if tgt.nunique() > 1 and tgt.nunique() <= 50:
                stratify = tgt

        train_split, val_split = train_test_split(
            train_df, test_size=val_ratio, random_state=seed, stratify=stratify,
        )
        train_split = train_split.reset_index(drop=True)
        val_split = val_split.reset_index(drop=True)

        self.compile(
            optimizer=train_cfg.get("optimizer", "adam"),
            lr=train_cfg.get("lr", 0.001),
            metrics=train_cfg.get("metrics", ["auc"]),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        train_x, train_y = df_to_input(train_split, feature_columns, cfg)
        val_x, val_y = df_to_input(val_split, feature_columns, cfg)
        test_x, test_y = df_to_input(test_df, feature_columns, cfg)

        print(f"Train samples: {len(train_y)}, Val samples: {len(val_y)}, Test samples: {len(test_y)}")

        history = self.fit(
            train_x, train_y,
            batch_size=train_cfg.get("batch_size", 256),
            epochs=train_cfg.get("epochs", 10),
            verbose=1,
            validation_data=(val_x, val_y),
            early_stopping_monitor=train_cfg.get("early_stopping_monitor"),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 0),
        )

        test_result = self.evaluate(test_x, test_y, batch_size=train_cfg.get("batch_size", 256))
        for metric_name, metric_val in test_result.items():
            print(f"Test {metric_name}: {metric_val:.4f}")

        return history, test_result


class RecModel(BaseModel):
    def __init__(self, feature_columns, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu'):
        super().__init__(seed=seed, task=task, device=device)

        self.feature_columns = feature_columns
        self.sparse_feature_columns = [f for f in feature_columns if isinstance(f, SparseFeat)]
        self.dense_feature_columns = [f for f in feature_columns if isinstance(f, DenseFeat)]
        self.varlen_sparse_feature_columns = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

        self.feature_index = build_input_features(feature_columns)
        self.embedding_dict = create_embedding_matrix(feature_columns, init_std, sparse=False, device=device)
        self.l2_reg_embedding = l2_reg_embedding

        self.to(device)

    def input_from_feature_columns(self, X, feature_columns, embedding_dict):
        sparse_feature_columns = [f for f in feature_columns if isinstance(f, SparseFeat)]
        varlen_feature_columns = [f for f in feature_columns if isinstance(f, VarLenSparseFeat)]

        sparse_emb_list = embedding_lookup(
            X, embedding_dict, self.feature_index, sparse_feature_columns, to_list=True
        )

        varlen_emb_dict = varlen_embedding_lookup(
            X, embedding_dict, self.feature_index, varlen_feature_columns
        )
        varlen_emb_list = get_varlen_pooling_list(
            varlen_emb_dict, X, self.feature_index, varlen_feature_columns, self.device
        )

        dense_value_list = []
        for fc in self.dense_feature_columns:
            idx = self.feature_index[fc.name]
            dense_value_list.append(X[:, idx[0]:idx[1]])

        return sparse_emb_list + varlen_emb_list, dense_value_list

    def compute_input_dim(self, feature_columns):
        sparse_dim = sum(
            f.embedding_dim for f in feature_columns if isinstance(f, (SparseFeat, VarLenSparseFeat))
        )
        dense_dim = sum(
            f.dimension for f in feature_columns if isinstance(f, DenseFeat)
        )
        return sparse_dim + dense_dim

    def get_regularization_loss(self):
        reg_loss = torch.zeros(1, device=self.device)
        for param in self.embedding_dict.parameters():
            reg_loss += torch.norm(param, 2)
        return self.l2_reg_embedding * reg_loss
