"""REGEN-HZBM 的紧凑 PyTorch 实现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.layers(value)


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, count: int) -> torch.Tensor:
    total = values.new_zeros((count, values.shape[1]))
    denominator = values.new_zeros((count, 1))
    total.index_add_(0, index, values)
    denominator.index_add_(0, index, values.new_ones((len(index), 1)))
    return total / denominator.clamp_min(1.0)


class REGEN(nn.Module):
    """H（有向特征）+ Z（节点潜变量）+ B（Bernoulli 扩散）+ M（图消息）。"""

    def __init__(self, node_dim: int, pair_dim: int = 4, window_dim: int = 3, width: int = 96):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(node_dim, width), nn.SiLU(), ResidualBlock(width))
        self.pair_encoder = nn.Sequential(nn.Linear(pair_dim, width), nn.SiLU(), ResidualBlock(width))
        self.high_pair_encoder = nn.Sequential(nn.Linear(pair_dim, width), nn.SiLU(), ResidualBlock(width))
        latent = max(8, width // 4)
        self.latent_mu = nn.Linear(width, latent)
        self.latent_logvar = nn.Linear(width, latent)
        self.latent_node = nn.Linear(latent, width)
        self.latent_edge = nn.Sequential(nn.Linear(2 * latent, width), nn.SiLU(), ResidualBlock(width))
        self.edge_input = nn.Linear(3 * width + 2, width)
        self.edge_blocks = nn.ModuleList([ResidualBlock(width) for _ in range(3)])
        self.context_edge = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU(), ResidualBlock(width))
        self.context_update = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, width))
        self.context_scale = nn.Parameter(torch.full((width,), 0.1))
        self.window_encoder = nn.Sequential(nn.Linear(window_dim, width), nn.SiLU(), ResidualBlock(width))
        self.time_encoder = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        self.window_head = nn.Sequential(ResidualBlock(width), nn.Linear(width, 1))
        self.last_kl: torch.Tensor | None = None

    def forward(self, features: dict[str, torch.Tensor], state: torch.Tensor, diffusion_time: torch.Tensor):
        base_node = self.node_encoder(features["node"])
        mu, logvar = self.latent_mu(base_node), self.latent_logvar(base_node).clamp(-8.0, 8.0)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        self.last_kl = -0.5 * torch.mean(1.0 + logvar - mu.square() - logvar.exp())
        node = base_node + self.latent_node(latent)
        sources, targets = features["sources"], features["targets"]
        pair = self.pair_encoder(features["pair"]) + self.high_pair_encoder(features["pair"])
        pair = pair + self.latent_edge(torch.cat((latent[sources], latent[targets]), dim=1))
        edge = F.silu(self.edge_input(torch.cat((node[sources], node[targets], pair, state[:, None], diffusion_time[:, None]), dim=1)))
        for block in self.edge_blocks:
            edge = block(edge)
        context = self.context_edge(torch.cat((node[sources], node[targets], pair), dim=1))
        outgoing = _scatter_mean(context, sources, len(node))
        incoming = _scatter_mean(context, targets, len(node))
        edge = edge + self.context_scale * self.context_update(torch.cat((edge, outgoing[sources], incoming[targets]), dim=1))
        windows = features["window_pair"].shape[1]
        time = torch.linspace(0, 1, windows, device=edge.device)
        time_features = torch.stack((time, features["durations"]), dim=1)
        local = edge[:, None] + self.window_encoder(features["window_pair"]) + self.time_encoder(time_features)[None]
        rate = F.softplus(self.window_head(local).squeeze(-1)).clamp_max(50.0)
        dynamic = -torch.expm1(-rate * features["durations"][None])
        global_probability = 1.0 - torch.prod(1.0 - dynamic, dim=1)
        return torch.logit(global_probability.clamp(1e-6, 1 - 1e-6)), dynamic

