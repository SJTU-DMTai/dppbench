"""HAIPipe alternative-evaluation network and iterative candidate selector."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn


class ScoreNN(nn.Module):
    """Match upstream released model/10, including scoring-head BatchNorm."""

    FORMAT_VERSION = 2

    def __init__(self, dataset_dim: int, primitive_count: int, sequence_length: int = 30) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        half = dataset_dim // 2
        self.dataset_nn = nn.Sequential(
            nn.Linear(dataset_dim, half), nn.BatchNorm1d(half), nn.LeakyReLU(),
            nn.Linear(half, half), nn.BatchNorm1d(half), nn.LeakyReLU(),
            nn.Linear(half, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),
            nn.Linear(300, 300), nn.BatchNorm1d(300), nn.LeakyReLU(),
        )
        self.dataset_dim = int(dataset_dim)
        self.seq_embedding = nn.Embedding(primitive_count + 2, 30)
        self.seq_lstm = nn.LSTM(30, 100, num_layers=5, dropout=0.5, batch_first=False)
        merged = 300 + 100 * self.sequence_length
        self.end_mlp = nn.Sequential(
            nn.Linear(merged, merged // 2), nn.BatchNorm1d(merged // 2), nn.LeakyReLU(),
            nn.Linear(merged // 2, 200), nn.BatchNorm1d(200), nn.LeakyReLU(),
            nn.Linear(200, 200), nn.BatchNorm1d(200), nn.LeakyReLU(),
            nn.Linear(200, 10), nn.BatchNorm1d(10), nn.LeakyReLU(),
            nn.Linear(10, 1), nn.BatchNorm1d(1), nn.Tanh(),
        )

    def metadata(self, vocabulary: list[str]) -> dict:
        return {
            "format_version": self.FORMAT_VERSION,
            "architecture": "upstream_model_10",
            "data_dim": self.dataset_dim,
            "primitive_vocabulary": list(vocabulary),
            "sequence_length": self.sequence_length,
        }

    def latent(self, pipeline_features: torch.Tensor) -> torch.Tensor:
        embedded = self.seq_embedding(pipeline_features).permute(1, 0, 2)
        output, _ = self.seq_lstm(embedded)
        return output.permute(1, 0, 2).flatten(start_dim=1)

    def forward(self, dataset_features: torch.Tensor, pipeline_features: torch.Tensor) -> torch.Tensor:
        return self.end_mlp(torch.cat((self.dataset_nn(dataset_features), self.latent(pipeline_features)), 1))


def pipeline_features(pipeline, vocabulary: list[str], sequence_length: int = 30) -> np.ndarray:
    ids = {name: idx + 1 for idx, name in enumerate(vocabulary)}
    result = [ids.get(name, len(vocabulary) + 1) for name in pipeline.op_names()[:sequence_length]]
    result.extend([0] * (sequence_length - len(result)))
    return np.asarray(result, dtype=np.int64)


class ActiveCandidateSelector:
    """Iterative representative/informative/diverse selection used by HAIPipe."""

    def __init__(self, model: ScoreNN, *, k_total: int = 20, iterations: int = 7, seed: int = 42) -> None:
        self.model = model
        self.committee = [copy.deepcopy(model) for _ in range(3)]
        self.k_total = int(k_total)
        self.iterations = int(iterations)
        self.rng = random.Random(seed)

    @staticmethod
    def _fit(model, dataset_matrix, candidate_features, ids, labels) -> None:
        if len(ids) < 2:
            return
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        targets = torch.as_tensor(labels, dtype=torch.float32)
        for _ in range(5):
            output = model(
                torch.as_tensor(dataset_matrix[ids], dtype=torch.float32),
                torch.as_tensor(candidate_features[ids], dtype=torch.long),
            ).flatten()
            loss = nn.functional.l1_loss(output, targets)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    def _informative(self, dataset_matrix, candidate_features, ids: list[int]) -> list[int]:
        if not ids:
            return []
        dataset = torch.as_tensor(dataset_matrix[ids], dtype=torch.float32)
        pipeline = torch.as_tensor(candidate_features[ids], dtype=torch.long)
        models = [self.model, *self.committee]
        outputs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                outputs.append(model(dataset, pipeline).flatten().cpu().numpy())
        base = outputs[0]
        scores = {}
        raw = np.concatenate((dataset_matrix[ids], candidate_features[ids]), axis=1)
        for position, candidate_id in enumerate(ids):
            change = sum(
                np.linalg.norm((base[position] - prediction[position]) * raw[position])
                for prediction in outputs[1:]
            ) / len(self.committee)
            scores[candidate_id] = float(change)
        return sorted(ids, key=lambda i: scores[i], reverse=True)

    def _representatives(self, latent: np.ndarray, available: list[int], count: int) -> list[int]:
        if len(available) <= count:
            return available
        clusters = min(2 * count, len(available))
        labels = KMeans(n_clusters=clusters, random_state=0, n_init=10).fit(latent[available])
        selected = []
        for cluster, center in enumerate(labels.cluster_centers_):
            members = [i for pos, i in enumerate(available) if labels.labels_[pos] == cluster]
            selected.append(min(members, key=lambda i: np.linalg.norm(latent[i] - center)))
        return selected

    def select(self, dataset_feature: np.ndarray, candidate_features: np.ndarray, evaluator, candidates) -> tuple[int, list[dict]]:
        if not candidates:
            raise ValueError("HAIPipe active selector received no candidates")
        if len(candidates) == 1:
            result = evaluator.evaluate(candidates[0])
            return 0, [{
                "candidate": 0,
                "fitness": float(result.fitness) if result.success else None,
                "success": result.success,
                "error": result.error,
            }]
        dataset_matrix = np.repeat(dataset_feature[None, :], len(candidates), axis=0)
        labeled: dict[int, float] = {}
        tried: set[int] = set()
        records: list[dict] = []
        batch = max(1, self.k_total // max(1, self.iterations))
        for _ in range(self.iterations):
            available = [i for i in range(len(candidates)) if i not in tried]
            if not available or len(tried) >= self.k_total:
                break
            self.model.eval()
            with torch.no_grad():
                latent = self.model.latent(torch.as_tensor(candidate_features, dtype=torch.long)).cpu().numpy()
            representative = self._representatives(latent, available, min(batch * 2, len(available)))
            diverse = list(representative)
            if labeled:
                labeled_latent = latent[list(labeled)]
                diverse.sort(
                    key=lambda i: min(np.linalg.norm(latent[i] - item) for item in labeled_latent),
                )
            informative = self._informative(dataset_matrix, candidate_features, representative)
            chosen = []
            while len(chosen) < min(batch, self.k_total - len(tried), len(representative)):
                source = diverse if len(chosen) % 2 == 0 else informative
                candidate = next((i for i in source if i not in chosen), None)
                if candidate is None:
                    break
                chosen.append(candidate)
            for index in chosen:
                result = evaluator.evaluate(candidates[index])
                score = float(result.fitness) if result.success else -1.0
                labeled[index] = score
                tried.add(index)
                records.append({"candidate": index, "fitness": score if result.success else None, "success": result.success, "error": result.error})
            valid = [(i, score) for i, score in labeled.items() if score >= 0]
            if valid:
                average = sum(score for _, score in valid) / len(valid)
                train_ids = [i for i, _ in valid]
                labels = torch.as_tensor([score - average for _, score in valid], dtype=torch.float32)
                if len(train_ids) > 1:
                    self._fit(self.model, dataset_matrix, candidate_features, train_ids, labels)
                    for committee_model in self.committee:
                        bootstrap = [self.rng.choice(train_ids) for _ in train_ids]
                        bootstrap_labels = [labeled[i] - average for i in bootstrap]
                        self._fit(
                            committee_model, dataset_matrix, candidate_features,
                            bootstrap, bootstrap_labels,
                        )
        if not labeled:
            return 0, records
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(
                torch.as_tensor(dataset_matrix, dtype=torch.float32),
                torch.as_tensor(candidate_features, dtype=torch.long),
            ).flatten().cpu().numpy()
        final_count = max(1, self.k_total - batch * self.iterations)
        ranked = sorted(range(len(candidates)), key=lambda i: float(predictions[i]), reverse=True)
        for index in ranked:
            if len([r for r in records if r.get("final")]) >= final_count:
                break
            if index not in labeled:
                result = evaluator.evaluate(candidates[index])
                labeled[index] = float(result.fitness) if result.success else -1.0
                records.append({"candidate": index, "fitness": result.fitness if result.success else None, "success": result.success, "error": result.error, "final": True})
        return max(labeled, key=labeled.get), records


__all__ = ["ActiveCandidateSelector", "ScoreNN", "pipeline_features"]


def pretrain_score_model(
    model: ScoreNN,
    dataset_matrix: np.ndarray,
    pipeline_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 10,
    learning_rate: float = 0.01,
    batch_size: int = 16,
) -> list[float]:
    """Offline pretraining entry point for the alternative evaluator."""
    if len(labels) < 2:
        raise ValueError("Alternative-evaluator pretraining requires at least two labeled pipelines")
    if epochs < 1 or batch_size < 2:
        raise ValueError("ScoreNN requires positive epochs and batch_size >= 2 for BatchNorm")
    device = next(model.parameters()).device
    dataset_tensor = torch.as_tensor(dataset_matrix, dtype=torch.float32, device=device)
    pipeline_tensor = torch.as_tensor(pipeline_matrix, dtype=torch.long, device=device)
    label_tensor = torch.as_tensor(labels, dtype=torch.float32, device=device).flatten()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    model.train()
    for _ in range(int(epochs)):
        for group in optimizer.param_groups:
            group["lr"] *= 0.9
        batches = list(torch.randperm(len(labels), device=device).split(batch_size))
        # Keep every example without a singleton batch, which BatchNorm rejects.
        if len(batches) > 1 and len(batches[-1]) == 1:
            last = batches.pop()
            batches[-1] = torch.cat((batches[-1], last))
        total_loss = 0.0
        for ids in batches:
            output = model(dataset_tensor[ids], pipeline_tensor[ids]).flatten()
            loss = nn.functional.l1_loss(output, label_tensor[ids])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss.item()) * len(ids)
        history.append(total_loss / len(labels))
    return history


__all__.append("pretrain_score_model")
