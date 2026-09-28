from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class NystromAttention(nn.Module):
    def __init__(self, dim=512, dim_head=64, heads=8, landmarks=256, dropout=0.1):
        super().__init__()
        self.heads, self.dim_head, self.landmarks = heads, dim_head, landmarks
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim), nn.Dropout(dropout))
        self.res_conv = nn.Conv2d(heads, heads, (33, 1), padding=(16, 0), groups=heads, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, original_n, _ = x.shape
        padding = (-original_n) % self.landmarks
        if padding:
            x = F.pad(x, (0, 0, padding, 0))
        n = x.shape[1]
        q, k, v = (t.view(batch, n, self.heads, self.dim_head).transpose(1, 2) for t in self.to_qkv(x).chunk(3, dim=-1))
        q = q * self.scale
        group = n // self.landmarks
        q_landmarks = q.reshape(batch, self.heads, self.landmarks, group, self.dim_head).mean(3)
        k_landmarks = k.reshape(batch, self.heads, self.landmarks, group, self.dim_head).mean(3)
        a1 = (q @ k_landmarks.transpose(-1, -2)).softmax(-1)
        a2 = (q_landmarks @ k_landmarks.transpose(-1, -2)).softmax(-1)
        a3 = (q_landmarks @ k.transpose(-1, -2)).softmax(-1)
        identity = torch.eye(self.landmarks, device=x.device, dtype=x.dtype).expand(batch * self.heads, -1, -1)
        a2_flat = a2.flatten(0, 1)
        z = a2_flat.transpose(-1, -2) / (a2_flat.abs().sum(-1).amax(-1, keepdim=True).unsqueeze(-1) * a2_flat.abs().sum(-2).amax(-1, keepdim=True).unsqueeze(-1)).clamp_min(1e-6)
        for _ in range(6):
            az = a2_flat @ z
            z = 0.25 * z @ (13 * identity - az @ (15 * identity - az @ (7 * identity - az)))
        out = a1 @ z.unflatten(0, (batch, self.heads)) @ (a3 @ v)
        out = out + self.res_conv(v)
        out = out.transpose(1, 2).reshape(batch, n, -1)
        return self.to_out(out[:, -original_n:])


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim) for k in (7, 5, 3)])

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        cls, patches = x[:, :1], x[:, 1:]
        grid = patches.transpose(1, 2).reshape(x.shape[0], -1, height, width)
        for layer in self.layers:
            grid = grid + layer(grid)
        return torch.cat([cls, grid.flatten(2).transpose(1, 2)], dim=1)


class LGSEAdapter(nn.Module):
    def __init__(self, dim=512, window_size=64, heads=8, dropout=0.25, max_scale=0.05):
        super().__init__()
        self.window_size, self.max_scale = window_size, max_scale
        self.local_norm, self.global_norm, self.output_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.local = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.global_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.gate, self.dropout = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid()), nn.Dropout(dropout)
        self.scale_raw = nn.Parameter(torch.zeros(()))

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        n, dim = patches.shape
        windows = math.ceil(n / self.window_size)
        total = windows * self.window_size
        padded = F.pad(patches, (0, 0, 0, total - n))
        local = padded.reshape(windows, self.window_size, dim)
        valid = torch.arange(total, device=patches.device).reshape(windows, self.window_size) < n
        local = local + self.dropout(self.local(self.local_norm(local), self.local_norm(local), self.local_norm(local), key_padding_mask=~valid, need_weights=False)[0])
        weights = valid.to(local.dtype).unsqueeze(-1)
        summaries = (local * weights).sum(1) / weights.sum(1).clamp_min(1)
        global_tokens = torch.cat([cls.unsqueeze(0), summaries.unsqueeze(0)], dim=1)
        global_tokens = global_tokens + self.dropout(self.global_attn(self.global_norm(global_tokens), self.global_norm(global_tokens), self.global_norm(global_tokens), need_weights=False)[0])
        context = global_tokens[:, 1:].squeeze(0).unsqueeze(1).expand_as(local)
        mixed = self.output_norm(local + self.gate(torch.cat([local, context], dim=-1)) * self.dropout(context)).reshape(total, dim)[:n]
        return patches + self.max_scale * torch.tanh(self.scale_raw) * (mixed - patches)


class GatedAttention(nn.Module):
    def __init__(self, dim=512, hidden=128, classes=2):
        super().__init__()
        self.v, self.u, self.w = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(hidden, 1)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.w(torch.tanh(self.v(features)) * torch.sigmoid(self.u(features))).transpose(0, 1), dim=1)
        return self.classifier(attention @ features)


class GLADMIL(nn.Module):
    def __init__(self, input_dim, num_classes=2, window_size=64, adapter_dropout=0.25, adapter_max_scale=0.05):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1, self.layer2 = NystromAttention(), NystromAttention()
        self.norm1, self.norm2, self.final_norm = nn.LayerNorm(512), nn.LayerNorm(512), nn.LayerNorm(512)
        self.ppeg = PPEG()
        self.adapter = LGSEAdapter(512, window_size, 8, adapter_dropout, adapter_max_scale)
        self.classifier, self.tier2 = nn.Linear(512, num_classes), GatedAttention(512, 128, num_classes)
        self.apply(_init)
        nn.init.normal_(self.cls_token, std=1e-6)

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = features.shape[0]
        h = self.encoder(features).unsqueeze(0)
        height = width = math.ceil(math.sqrt(n))
        if h.shape[1] < height * width:
            h = torch.cat([h, h[:, :height * width - h.shape[1]]], dim=1)
        h = torch.cat([self.cls_token.expand(1, -1, -1), h], dim=1)
        h = h + self.layer1(self.norm1(h))
        h = self.ppeg(h, height, width)
        real = self.adapter(h[:, 0], h[:, 1:1+n].squeeze(0))
        h = torch.cat([h[:, :1], real.unsqueeze(0), h[:, 1+n:]], dim=1)
        h = self.final_norm(h + self.layer2(self.norm2(h)))
        return self.classifier(h[:, 0]), h[:, 0]

    @staticmethod
    def _groups(n: int, groups: int, rng: np.random.Generator, random_partition: bool) -> list[np.ndarray]:
        indices = rng.permutation(n) if random_partition else np.arange(n)
        return [part for part in np.array_split(indices, min(max(1, groups), n)) if len(part)]

    def forward(self, features: torch.Tensor, num_groups=4, rng: np.random.Generator | None = None, random_partition=True) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng() if rng is None else rng
        full_logits, _ = self.encode(features)
        sub_logits, sub_features = [], []
        for indices in self._groups(features.shape[0], num_groups, rng, random_partition):
            logits, descriptor = self.encode(features.index_select(0, torch.as_tensor(indices, device=features.device)))
            sub_logits.append(logits)
            sub_features.append(descriptor)
        sub_logits = torch.cat(sub_logits, dim=0)
        tier2_logits = self.tier2(torch.cat(sub_features, dim=0))
        return {"full_logits": full_logits, "sub_logits": sub_logits, "tier2_logits": tier2_logits}
