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


class _ColumnBatchCollator:
    """Assemble only the requested batch, retaining NumPy's original promotion."""
    def __init__(self, columns, target=None, sample_weight=None):
        size = len(columns[0])
        if any(len(column) != size for column in columns) or (
                target is not None and len(target) != size):
            raise ValueError("Feature and target arrays must have matching row counts")
        self.columns = columns
        self.target = target
        self.sample_weight = sample_weight

    def __call__(self, indices):
        batch = torch.from_numpy(np.concatenate(
            [column[indices] for column in self.columns], axis=-1))
        if self.target is None:
            return (batch,)
        result = (batch, torch.from_numpy(self.target[indices]))
        if self.sample_weight is not None:
            result += (torch.from_numpy(self.sample_weight[indices]),)
        return result


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
            early_stopping_monitor=None, early_stopping_patience=0, sample_weight=None):
        if isinstance(x, dict):
            x = [x[feat] for feat in self.feature_index]

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float32)
            if len(sample_weight) != len(y) or not np.isfinite(sample_weight).all() or (sample_weight < 0).any():
                raise ValueError('Training sample weights must be finite, nonnegative, and aligned')
        train_loader = DataLoader(range(len(y)), shuffle=shuffle, batch_size=batch_size,
                                  collate_fn=_ColumnBatchCollator(x, y, sample_weight))

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
        first_update = True

        for epoch in range(epochs):
            model = self.train()
            start = time.time()
            loss_epoch = 0
            n_samples = 0

            for batch in train_loader:
                x_batch, y_batch = batch[:2]
                x_batch = x_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                # Keep the batch axis when a small search sample leaves a
                # final batch of one row.
                y_pred = model(x_batch).reshape(-1)
                if sample_weight is None:
                    loss = self.loss_func(y_pred, y_batch, reduction='mean')
                else:
                    weights = batch[2].to(self.device).float()
                    losses = self.loss_func(y_pred, y_batch, reduction='none')
                    loss = (losses * weights).sum() / weights.sum().clamp(min=1e-8)

                if hasattr(self, 'get_regularization_loss'):
                    loss = loss + self.get_regularization_loss().squeeze()

                self.optim.zero_grad()
                loss.backward()
                if first_update:
                    from baselines.common.budget import memory_checkpoint
                    memory_checkpoint("rec.first_backward")
                self.optim.step()
                if first_update:
                    memory_checkpoint("rec.first_optimizer_step")
                    first_update = False

                loss_epoch += loss.item() * len(y_batch)
                n_samples += len(y_batch)

            avg_loss = loss_epoch / n_samples
            history["loss"].append(avg_loss)

            if do_val:
                val_result = self.evaluate(val_x, val_y, batch_size)
                for k, v in val_result.items():
                    history.setdefault(f"val_{k}", []).append(v)
            if verbose > 0:
                elapsed = int(time.time() - start)
                msg = f"Epoch {epoch + 1}/{epochs} - {elapsed}s - loss: {avg_loss:.4f}"
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

        loader = DataLoader(range(len(x[0])), batch_size=batch_size, shuffle=False,
                            collate_fn=_ColumnBatchCollator(x))

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

    def train_and_evaluate(self, train_df, test_df, feature_columns, cfg, evaluate_test=True):
        """Train the model on ``train_df`` and evaluate on ``test_df``.

        Consume the train/validation marker fixed before preprocessing.
        Direct callers without markers retain the legacy stratified split
        controlled by ``train.val_ratio`` and ``train.seed``.
        """
        from sklearn.model_selection import train_test_split

        train_cfg = cfg["train"]
        val_ratio = float(train_cfg.get("val_ratio", 0.1))
        seed = int(train_cfg.get("seed", 42))

        # print(f"Training pool before validation split: total={len(train_df)}")

        target_col = cfg.get("feature", {}).get("target_col")
        stratify = None
        if target_col and target_col in train_df.columns:
            tgt = train_df[target_col]
            if tgt.nunique() > 1 and tgt.nunique() <= 50:
                stratify = tgt

        if "__split__" in train_df:
            train_split = train_df.loc[train_df["__split__"] == "train"].drop(columns="__split__")
            val_split = train_df.loc[train_df["__split__"] == "val"].drop(columns="__split__")
            if train_split.empty or val_split.empty:
                raise ValueError("Preprocessing removed the fixed train/validation partition")
        else:
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
        if evaluate_test:
            test_x, test_y = df_to_input(test_df, feature_columns, cfg)
            test_size = len(test_y)
        else:
            test_x = test_y = None
            test_size = 0

        split_msg = f"Model fit split: train={len(train_y)}, val={len(val_y)}"
        if evaluate_test:
            split_msg += f", test={test_size}"
        split_msg += f" (val_ratio={val_ratio})"
        print(split_msg)

        history = self.fit(
            train_x, train_y,
            batch_size=train_cfg.get("batch_size", 256),
            epochs=train_cfg.get("epochs", 10),
            verbose=1,
            validation_data=(val_x, val_y),
            sample_weight=(pd.to_numeric(train_split['sample_weight'], errors='raise').fillna(1.).to_numpy()
                           if 'sample_weight' in train_split else None),
            early_stopping_monitor=train_cfg.get("early_stopping_monitor"),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 0),
        )

        test_result = {}
        if evaluate_test:
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
