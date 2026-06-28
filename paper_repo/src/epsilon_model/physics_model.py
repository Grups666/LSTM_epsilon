from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-6


class EpsilonStateResetModel(nn.Module):
    """Physics-informed epsilon-core LSTM adapted from Ara's LSTM-epsilon design."""

    def __init__(self, input_dim: int, hidden_size: int = 256, n_mul: int = 10, dropout_rate: float = 0.4) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_mul = n_mul
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.eps_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, n_mul),
        )
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, n_mul),
        )
        self.static_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 3 * n_mul),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_seq: torch.Tensor,
        pet_seq: torch.Tensor,
        sm_seq: torch.Tensor,
        rec_mask: torch.Tensor,
        start_mask: torch.Tensor,
        bounds: torch.Tensor,
        bufftime: int = 365,
    ) -> dict[str, torch.Tensor]:
        lstm_out, _ = self.lstm(z_seq)
        lstm_out = self.dropout(lstm_out)

        lstm_out = lstm_out[bufftime:, :, :]
        pet_seq = pet_seq[bufftime:, :, :]
        sm_seq = sm_seq[bufftime:, :, :]
        time, batch, _ = lstm_out.shape

        eps_t = F.softplus(self.eps_head(lstm_out))
        q_base_t = F.softplus(self.peak_head(lstm_out))

        h_final = lstm_out[-1, :, :]
        static_raw = self.static_head(h_final).view(batch, self.n_mul, 3)
        a_min, a_max = bounds[:, 0:1], bounds[:, 1:2]
        l_min, l_max = bounds[:, 2:3], bounds[:, 3:4]
        g_min, g_max = bounds[:, 4:5], bounds[:, 5:6]
        alpha = a_min + (a_max - a_min) * static_raw[:, :, 0]
        lp = l_min + (l_max - l_min) * static_raw[:, :, 1]
        gamma = g_min + (g_max - g_min) * static_raw[:, :, 2]

        sm_term = torch.clamp(sm_seq / (lp.unsqueeze(0) + EPS), min=EPS)
        aet_t = pet_seq * torch.pow(sm_term, gamma.unsqueeze(0))
        aet_t = torch.minimum(aet_t, pet_seq)

        q_out = []
        q_prev = q_base_t[0]
        for t in range(time):
            reset_val = q_base_t[t - 1] if t > 0 else q_base_t[0]
            q_curr = torch.where(start_mask[t] > 0.5, reset_val, q_prev)

            b_t = eps_t[t]
            a_t = b_t * (alpha * aet_t[t])
            denom = (b_t * q_curr + a_t) * torch.exp(a_t) - (b_t * q_curr)
            q_next_aet = (a_t * q_curr) / torch.clamp(denom, min=EPS)
            q_next_zero_aet = q_curr / (1.0 + b_t * q_curr)
            q_next = torch.where(a_t < 1e-6, q_next_zero_aet, q_next_aet)

            q_prev = torch.where(rec_mask[t] > 0.5, q_next, q_base_t[t])
            q_out.append(q_prev)

        q_components = torch.stack(q_out, dim=0)
        q_hat = torch.mean(q_components, dim=-1, keepdim=True)
        return {
            "q_hat": q_hat,
            "q_components": q_components,
            "q_base": q_base_t,
            "eps": eps_t,
            "aet": aet_t,
            "alpha": alpha,
            "lp": lp,
            "gamma": gamma,
        }


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
    err = torch.abs(pred - target)
    quad = torch.clamp(err, max=delta)
    lin = err - quad
    return 0.5 * quad**2 + delta * lin


class PhysicsInformedLoss(nn.Module):
    def __init__(
        self,
        lambda_path: float = 25.0,
        lambda_rhs: float = 10.0,
        lambda_smooth: float = 0.1,
        lambda_q0: float = 5.0,
        delta: float = 0.5,
    ) -> None:
        super().__init__()
        self.l_path = lambda_path
        self.l_rhs = lambda_rhs
        self.l_smooth = lambda_smooth
        self.l_q0 = lambda_q0
        self.delta = delta

    def forward(
        self,
        model_out: dict[str, torch.Tensor],
        obs_q: torch.Tensor,
        rec_mask: torch.Tensor,
        start_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        q_hat = model_out["q_hat"]
        q_base = model_out["q_base"]
        eps = model_out["eps"]
        aet = model_out["aet"]
        alpha = model_out["alpha"].unsqueeze(0)

        obs_q_safe = torch.nan_to_num(obs_q, nan=0.0)
        obs_q_log = torch.log(torch.clamp(obs_q_safe, min=EPS))
        q_hat_log = torch.log(torch.clamp(q_hat, min=EPS))

        rec_float = rec_mask.float()
        valid_rec = torch.sum(rec_float).clamp_min(1.0)
        loss_path = torch.sum(huber_loss(q_hat_log, obs_q_log, self.delta) * rec_float) / valid_rec

        obs_q_t = obs_q_safe[:-1, :, :]
        obs_q_tp1 = obs_q_safe[1:, :, :]
        d_q_obs = obs_q_tp1 - obs_q_t
        eps_t = eps[:-1, :, :]
        aet_t = aet[:-1, :, :]
        n_mul = eps_t.shape[-1]
        obs_q_comp = obs_q_t.repeat(1, 1, n_mul)
        d_q_pred_components = -eps_t * (obs_q_comp**2 + (alpha * aet_t) * obs_q_comp)
        d_q_pred = torch.mean(d_q_pred_components, dim=-1, keepdim=True)
        rhs_mask = ((rec_mask[:-1, :, :] > 0.5) & (rec_mask[1:, :, :] > 0.5)).float()
        valid_rhs = torch.sum(rhs_mask).clamp_min(1.0)
        loss_rhs = torch.sum(huber_loss(d_q_pred, d_q_obs, self.delta) * rhs_mask) / valid_rhs

        eps_mean = torch.mean(eps, dim=-1, keepdim=True)
        if eps_mean.shape[0] >= 3:
            second = eps_mean[:-2] - 2.0 * eps_mean[1:-1] + eps_mean[2:]
            smooth_mask = ((rec_mask[:-2] > 0.5) & (rec_mask[1:-1] > 0.5) & (rec_mask[2:] > 0.5)).float()
            loss_smooth = torch.sum((second**2) * smooth_mask) / torch.sum(smooth_mask).clamp_min(1.0)
        else:
            loss_smooth = torch.tensor(0.0, device=eps.device)

        q_base_mean = torch.mean(q_base, dim=-1, keepdim=True)
        q_base_log = torch.log(torch.clamp(q_base_mean, min=EPS))
        start_q0 = torch.zeros_like(start_mask)
        start_q0[:-1, :, :] = start_mask[1:, :, :]
        valid_starts = torch.sum(start_q0).clamp_min(1.0)
        loss_q0 = torch.sum(huber_loss(q_base_log, obs_q_log, self.delta) * start_q0) / valid_starts

        total = self.l_path * loss_path + self.l_rhs * loss_rhs + self.l_smooth * loss_smooth + self.l_q0 * loss_q0
        return {"total": total, "l_path": loss_path, "l_rhs": loss_rhs, "l_smooth": loss_smooth, "l_q0": loss_q0}
