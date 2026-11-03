"""CSV 输入读取、特征构造和结果写出。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    genes: list[str]
    cell_ids: list[str]
    expression: np.ndarray  # genes × cells
    time: np.ndarray  # cells
    sources: np.ndarray
    targets: np.ndarray
    labels: np.ndarray | None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset(input_dir: str | Path, require_labels: bool = False) -> Dataset:
    """读取 expression.csv、time.csv、可选 candidates.csv 和 labels.csv。"""
    root = Path(input_dir)
    expression_rows = _read_rows(root / "expression.csv")
    if not expression_rows:
        raise ValueError("expression.csv 不能为空")
    fields = list(expression_rows[0])
    if fields[0] != "cell_id" or len(fields) < 3:
        raise ValueError("expression.csv 首列必须是 cell_id，至少包含两个基因列")
    genes = fields[1:]
    cell_ids = [row["cell_id"] for row in expression_rows]
    expression = np.asarray(
        [[float(row[gene]) for row in expression_rows] for gene in genes],
        dtype=np.float32,
    )
    times = {row["cell_id"]: float(row["pseudotime"]) for row in _read_rows(root / "time.csv")}
    if set(times) != set(cell_ids):
        raise ValueError("time.csv 的 cell_id 必须与 expression.csv 完全一致")
    time = np.asarray([times[cell_id] for cell_id in cell_ids], dtype=np.float32)

    candidates_path = root / "candidates.csv"
    if candidates_path.is_file():
        candidates = _read_rows(candidates_path)
        index = {gene: position for position, gene in enumerate(genes)}
        try:
            sources = np.asarray([index[row["source"]] for row in candidates], dtype=np.int64)
            targets = np.asarray([index[row["target"]] for row in candidates], dtype=np.int64)
        except KeyError as error:
            raise ValueError("candidates.csv 包含未知基因: %s" % error.args[0]) from error
    else:
        sources, targets = np.where(~np.eye(len(genes), dtype=bool))
        sources, targets = sources.astype(np.int64), targets.astype(np.int64)

    labels = None
    label_path = root / "labels.csv"
    if label_path.is_file():
        values = {(row["source"], row["target"]): float(row["label"]) for row in _read_rows(label_path)}
        labels = np.asarray(
            [values.get((genes[source], genes[target]), 0.0) for source, target in zip(sources, targets)],
            dtype=np.float32,
        )
    if require_labels and labels is None:
        raise FileNotFoundError("训练需要 labels.csv（source,target,label 三列）")
    return Dataset(genes, cell_ids, expression, time, sources, targets, labels)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    center = np.median(values, axis=0, keepdims=True)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0, keepdims=True)
    scale = np.where(scale > 1e-6, scale, values.std(axis=0, keepdims=True))
    return ((values - center) / np.where(scale > 1e-6, scale, 1.0)).astype(np.float32)


def make_features(data: Dataset, windows: int) -> dict[str, np.ndarray]:
    """仅从 RNA 表达和时间构建有向边特征；不读取标签。"""
    if windows < 2 or windows > len(data.cell_ids):
        raise ValueError("windows 必须在 2 和细胞数之间")
    order = np.argsort(data.time, kind="mergesort")
    expression = data.expression[:, order]
    time = data.time[order]
    groups = np.array_split(np.arange(len(time)), windows)
    if any(len(group) == 0 for group in groups):
        raise ValueError("每个时间窗至少需要一个细胞")
    means = np.stack([expression[:, group].mean(axis=1) for group in groups], axis=1)
    stds = np.stack([expression[:, group].std(axis=1) for group in groups], axis=1)
    centered = np.arange(windows, dtype=np.float32) - (windows - 1) / 2
    trend = (means * centered).sum(axis=1, keepdims=True) / max(float((centered ** 2).sum()), 1.0)
    node = _robust_scale(np.concatenate((means, stds, trend), axis=1))

    def cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        left = left - left.mean(axis=1, keepdims=True)
        right = right - right.mean(axis=1, keepdims=True)
        left = left / np.sqrt((left ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
        right = right / np.sqrt((right ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
        return (left[data.sources] * right[data.targets]).sum(axis=1)

    pair_raw = np.stack((
        cosine(expression, expression),
        cosine(means[:, :-1], means[:, 1:]),
        cosine(means[:, :-1], means[:, 1:]),
        cosine(stds[:, :-1], stds[:, 1:]),
    ), axis=1)
    # 第三个特征按反向边重算，以保留 source→target 方向性。
    reverse = (means[:, :-1] - means[:, :-1].mean(axis=1, keepdims=True))
    following = (means[:, 1:] - means[:, 1:].mean(axis=1, keepdims=True))
    reverse /= np.sqrt((reverse ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
    following /= np.sqrt((following ** 2).sum(axis=1, keepdims=True)).clip(1e-8)
    pair_raw[:, 2] = (reverse[data.targets] * following[data.sources]).sum(axis=1)
    pair = _robust_scale(pair_raw)

    deltas = np.diff(means, axis=1, append=means[:, -1:])
    window_pair = np.stack((
        means[data.sources].T,
        means[data.targets].T,
        deltas[data.targets].T - deltas[data.sources].T,
    ), axis=2).transpose(1, 0, 2)
    window_pair = _robust_scale(window_pair.reshape(-1, 3)).reshape(len(data.sources), windows, 3)
    durations = np.asarray([len(group) for group in groups], dtype=np.float32)
    durations /= durations.sum()
    return {
        "node": node, "pair": pair, "window_pair": window_pair,
        "sources": data.sources, "targets": data.targets,
        "durations": durations,
    }


def write_predictions(path: str | Path, data: Dataset, global_score: np.ndarray, dynamic_score: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source", "target", "global_probability"] + [
            "window_%d_probability" % (index + 1) for index in range(dynamic_score.shape[1])
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for edge, (source, target) in enumerate(zip(data.sources, data.targets)):
            row = {"source": data.genes[source], "target": data.genes[target], "global_probability": "%.8f" % global_score[edge]}
            row.update({"window_%d_probability" % (index + 1): "%.8f" % value for index, value in enumerate(dynamic_score[edge])})
            writer.writerow(row)
