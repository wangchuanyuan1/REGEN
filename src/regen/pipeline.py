"""训练与推理接口。"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import Dataset, make_features
from .model import REGEN


def _tensor_features(data: Dataset, windows: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device=device) for key, value in make_features(data, windows).items()}


def train(data: Dataset, checkpoint: str | Path, windows: int, epochs: int, device_name: str, seed: int = 0) -> dict:
    if data.labels is None:
        raise ValueError("训练需要 labels.csv")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device_name)
    features = _tensor_features(data, windows, device)
    labels = torch.as_tensor(data.labels, device=device)
    model = REGEN(features["node"].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7.5e-4, weight_decay=1e-4)
    prior = labels.mean().clamp(1e-4, 1 - 1e-4)
    for epoch in range(epochs):
        model.train()
        diffusion_time = torch.rand_like(labels)
        state = torch.bernoulli((1 - diffusion_time) * labels + diffusion_time * prior)
        logits, _ = model(features, state, diffusion_time)
        positive_weight = ((1 - prior) / prior).clamp(1.0, 100.0)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weight) + 1e-4 * model.last_kl
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if epoch == 0 or (epoch + 1) % max(1, epochs // 10) == 0:
            print("epoch=%d loss=%.5f" % (epoch + 1, float(loss.detach())), flush=True)
    destination = Path(checkpoint); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "node_dim": int(features["node"].shape[1]), "windows": windows, "class_prior": float(prior), "genes": data.genes}, destination)
    return {"checkpoint": str(destination), "class_prior": float(prior)}


@torch.no_grad()
def infer(data: Dataset, checkpoint: str | Path, output: str | Path, device_name: str, sampling_steps: int = 40) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload["genes"] != data.genes:
        raise ValueError("checkpoint 与输入数据的基因顺序不一致")
    features = _tensor_features(data, int(payload["windows"]), device)
    model = REGEN(int(payload["node_dim"])).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    state = torch.full((len(data.sources),), float(payload["class_prior"]), device=device)
    dynamic = None
    for step in range(sampling_steps, 0, -1):
        level = step / sampling_steps
        logits, dynamic = model(features, state, torch.full_like(state, level))
        probability = torch.sigmoid(logits)
        next_level = (step - 1) / sampling_steps
        state = (1 - next_level) * probability + next_level * float(payload["class_prior"])
    global_score = 1.0 - torch.prod(1.0 - dynamic, dim=1)
    from .data import write_predictions
    write_predictions(output, data, global_score.cpu().numpy(), dynamic.cpu().numpy())
    return global_score.cpu().numpy(), dynamic.cpu().numpy()

