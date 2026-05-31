from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization over temporal dimension.

    Input/Output shape: [B, T, N, C]
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, num_features))

    def normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True, unbiased=False)
        x_norm = self.gamma * (x - mu) / (std + self.eps) + self.beta
        return x_norm, mu, std

    def denormalize(self, y_norm: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return ((y_norm - self.beta) / (self.gamma + self.eps)) * (std + self.eps) + mu


class TemporalSSMBlock(nn.Module):
    """A lightweight selective temporal block (Mamba-style placeholder).

    This is a practical linear-time temporal mixer for each node sequence.
    Input/Output: [B, T, N, D]
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        v = torch.tanh(self.in_proj(h))
        g = torch.sigmoid(self.gate_proj(h))
        y = self.out_proj(v * g)
        y = self.dropout(y)
        return residual + y


class SpatialGraphBlock(nn.Module):
    """Spatial aggregation using fused multi-view graph.

    Eq-style mapping:
    - A_cong from normalized congestion vector
    - A_fused = softmax(w) weighted sum of geo/func/cong
    - row-normalize then aggregate
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.w_sp = nn.Linear(d_model, d_model, bias=True)
        self.graph_logits = nn.Parameter(torch.zeros(3))
        self.tau_cong = nn.Parameter(torch.tensor(0.2))

    @staticmethod
    def _row_normalize(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        denom = a.sum(dim=-1, keepdim=True).clamp_min(eps)
        return a / denom

    def _build_congestion_graph(self, x_t: torch.Tensor, vmax: float = 1.0) -> torch.Tensor:
        # x_t: [B, N, D], use channel-mean as proxy speed-like signal
        v = x_t.mean(dim=-1)  # [B, N]
        c = (1.0 - v / max(vmax, 1e-6)).clamp(min=0.0)
        c = F.normalize(c, p=2, dim=-1)
        outer = c.unsqueeze(-1) * c.unsqueeze(-2)  # [B, N, N]
        return F.relu(outer - self.tau_cong)

    def forward(
        self,
        h_temp: torch.Tensor,
        a_geo: torch.Tensor,
        a_func: torch.Tensor,
    ) -> torch.Tensor:
        # h_temp: [B,T,N,D], a_geo/a_func: [N,N] or [B,N,N]
        b, t, n, d = h_temp.shape

        if a_geo.dim() == 2:
            a_geo = a_geo.unsqueeze(0).expand(b, -1, -1)
        if a_func.dim() == 2:
            a_func = a_func.unsqueeze(0).expand(b, -1, -1)

        w = F.softmax(self.graph_logits, dim=0)  # [3]
        outs = []
        for i in range(t):
            h_i = h_temp[:, i]  # [B,N,D]
            a_cong = self._build_congestion_graph(h_i)
            a = w[0] * a_geo + w[1] * a_func + w[2] * a_cong
            a = self._row_normalize(a)
            m = torch.bmm(a, h_i)
            outs.append(F.relu(self.w_sp(m)))
        return torch.stack(outs, dim=1)


class CHPRRouter(nn.Module):
    """Calibrated Hierarchical Prototype Routing.

    Two-level routing:
      1) pattern-level within each source city
      2) city-level with uncertainty calibration + fallback + optional top-k
    """

    def __init__(
        self,
        d_model: int,
        num_sources: int,
        num_prototypes: int,
        temperature: float = 0.5,
        kappa_min: float = 0.4,
        topk: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_sources = num_sources
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.kappa_min = kappa_min
        self.topk = topk

        self.w_q = nn.Parameter(torch.randn(d_model) / math.sqrt(d_model))
        self.delta = nn.Parameter(torch.tensor(0.0))
        self.eta_s = nn.Parameter(torch.tensor(1.0))
        self.alpha_city_hat = nn.Parameter(torch.tensor(1.0))
        self.alpha_u_hat = nn.Parameter(torch.tensor(1.0))

    def _extract_query(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,T,N,D]
        h_t = h.mean(dim=2)  # [B,T,D]
        e = torch.einsum("btd,d->bt", h_t, self.w_q)
        a = F.softmax(e, dim=1)
        q = torch.einsum("bt,btd->bd", a, h_t)
        return q

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # a: [B,D], b: [S,M,D] -> [B,S,M]
        an = F.normalize(a, p=2, dim=-1)
        bn = F.normalize(b, p=2, dim=-1)
        return torch.einsum("bd,smd->bsm", an, bn)

    def forward(self, h: torch.Tensor, prototypes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # prototypes: [S,M,D]
        b = h.shape[0]
        q = self._extract_query(h)  # [B,D]

        s = self._cosine(q, prototypes)  # [B,S,M]
        beta = F.softmax(s / max(self.temperature, 1e-6), dim=-1)  # pattern-level

        r = (beta * s).sum(dim=-1)  # city score [B,S]
        rho = s.max(dim=-1).values  # [B,S]
        d_ent = -(beta * (beta.clamp_min(1e-8).log())).sum(dim=-1)  # [B,S]
        d_ent = d_ent / math.log(self.num_prototypes)

        u = torch.sigmoid(d_ent - self.eta_s.abs() * rho - self.delta)  # uncertainty [B,S]

        alpha_city = F.softplus(self.alpha_city_hat)
        alpha_u = F.softplus(self.alpha_u_hat)
        logits = alpha_city * r - alpha_u * u
        alpha = F.softmax(logits, dim=-1)  # [B,S]

        # fallback
        kappa = alpha.max(dim=-1, keepdim=True).values
        gamma_f = torch.clamp(1.0 - kappa / self.kappa_min, min=0.0)
        alpha_fb = (1.0 - gamma_f) * alpha + gamma_f / self.num_sources

        # top-k sparsification
        if self.topk is not None and self.topk < self.num_sources:
            top_vals, top_idx = torch.topk(alpha_fb, k=self.topk, dim=-1)
            mask = torch.zeros_like(alpha_fb).scatter_(1, top_idx, 1.0)
            alpha_star = alpha_fb * mask
            alpha_star = alpha_star / alpha_star.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        else:
            alpha_star = alpha_fb

        # city embedding pbar_k = sum_m beta_km p_km
        pbar = torch.einsum("bsm,smd->bsd", beta, prototypes)  # [B,S,D]
        context = torch.einsum("bs,bsd->bd", alpha_star, pbar)  # [B,D]

        return {
            "query": q,
            "pattern_weights": beta,
            "city_weights": alpha_star,
            "uncertainty": u,
            "context": context,
        }


class HEAAdapter(nn.Module):
    """Hypernetwork-based Efficient Adaptation.

    - Hypernetwork generates W_up from routed context c
    - Shared W_down is learned and can be frozen in adaptation protocol
    - Data-dependent gate blends adapted and frozen backbone features
    """

    def __init__(self, d_model: int, d_neck: int = 32, d_hidden: int = 128, eta: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_neck = d_neck
        self.eta = eta

        self.w_down = nn.Parameter(torch.randn(d_model, d_neck) * 0.02)
        self.hnet = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_neck * d_model),
            nn.Tanh(),
        )
        self.w_g = nn.Linear(d_model, d_model)

    def forward(self, h_backbone: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # h_backbone: [B,T,N,D], context: [B,D]
        b, t, n, d = h_backbone.shape
        w_up = self.eta * self.hnet(context).view(b, self.d_neck, self.d_model)  # [B,d_neck,D]

        down = torch.einsum("btnd,dk->btnk", h_backbone, self.w_down)  # [B,T,N,d_neck]
        down = F.relu(down)
        up = torch.einsum("btnk,bkd->btnd", down, w_up)  # [B,T,N,D]
        h_adapted = h_backbone + up

        g = torch.sigmoid(self.w_g(h_backbone))
        h_final = h_backbone + g * (h_adapted - h_backbone)
        return h_final


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, out_steps: int, out_dim: int):
        super().__init__()
        self.out_steps = out_steps
        self.out_dim = out_dim
        self.proj = nn.Linear(d_model, out_steps * out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B,T,N,D] -> use last-step token
        x = h[:, -1]  # [B,N,D]
        y = self.proj(x)  # [B,N,out_steps*out_dim]
        b, n, _ = y.shape
        return y.view(b, self.out_steps, n, self.out_dim)


class CharmMamba(nn.Module):
    """Paper-aligned technical skeleton for CHARM-Mamba.

    Modules:
      1) DC-Mamba backbone (temporal-spatial deep coupling)
      2) CHPR calibrated hierarchical routing
      3) HEA hypernetwork adapter
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        output_dim: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_sources: int = 3,
        num_prototypes: int = 10,
        pred_len: int = 12,
        d_neck: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.num_layers = num_layers

        self.revin = RevIN(num_features=input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_prompt = nn.Parameter(torch.zeros(1, 1, 1, d_model))

        self.temporal_blocks = nn.ModuleList([TemporalSSMBlock(d_model, dropout) for _ in range(num_layers)])
        self.spatial_blocks = nn.ModuleList([SpatialGraphBlock(d_model) for _ in range(num_layers)])
        self.gate_layers = nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(num_layers)])
        self.lambda_couple = nn.Parameter(torch.tensor(0.5))

        self.router = CHPRRouter(
            d_model=d_model,
            num_sources=num_sources,
            num_prototypes=num_prototypes,
            temperature=0.5,
            kappa_min=0.4,
            topk=2,
        )
        self.hea = HEAAdapter(d_model=d_model, d_neck=d_neck)
        self.head = PredictionHead(d_model=d_model, out_steps=pred_len, out_dim=output_dim)

    def forward(
        self,
        x_input: torch.Tensor,
        a_geo: torch.Tensor,
        a_func: torch.Tensor,
        prototypes: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x_input: [B,T,N,C_in]
            a_geo: [N,N] or [B,N,N]
            a_func: [N,N] or [B,N,N]
            prototypes: [S,M,D]
        """
        x_norm, mu, std = self.revin.normalize(x_input)
        h = self.input_proj(x_norm) + self.input_prompt

        # Deep-coupled temporal-spatial stack
        h_sp_prev = torch.zeros_like(h)
        for l in range(self.num_layers):
            h_temp = self.temporal_blocks[l](h + self.lambda_couple * h_sp_prev)
            h_sp = self.spatial_blocks[l](h_temp, a_geo=a_geo, a_func=a_func)

            z = torch.sigmoid(self.gate_layers[l](torch.cat([h_sp, h_temp], dim=-1)))
            h = z * h_sp + (1.0 - z) * h_temp
            h_sp_prev = h_sp

        routing = self.router(h, prototypes)
        h_final = self.hea(h, routing["context"])

        y_norm = self.head(h_final)
        y = self.revin.denormalize(y_norm, mu=mu[:, :1], std=std[:, :1])

        if return_aux:
            return y, routing
        return y
