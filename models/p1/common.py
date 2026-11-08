"""Shared P0 utilities: canonical simulator constants, features, MDN, metrics.

Canonical physics: `simulation/physics.py`.
  rho=1.20, m=0.430, R=0.110, C_D~U[0.15,0.30], C_L=min(1.4S,0.30), dt=0.002 s
  prior: vx[18,35], vy[-12,12], vz[0.5,12], omega[-60,60]^3
Noise: independent isotropic Gaussian on the two observed positions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

R0 = np.array([0.0, 0.0, 0.11])
DT = 0.05
MASS, RADIUS, RHO, GRAV = 0.43, 0.11, 1.20, 9.81
AREA = np.pi * RADIUS**2
LEVELS = (0.002, 0.005, 0.010, 0.015)
LEVEL_LABELS = {0.002: "2mm", 0.005: "5mm", 0.010: "10mm", 0.015: "15mm"}

ARMS = {
    "A": dict(input="raw", target="abs", label="raw6+sigma -> abs"),
    "B": dict(input="raw", target="res", label="raw6+sigma -> residual"),
    "C": dict(input="phys", target="abs", label="physics16 -> abs"),
    "D": dict(input="phys", target="res", label="physics16 -> residual"),
    "E": dict(input="rawbase", target="res", label="raw6+sigma+baseline -> residual"),
}


# ---------------------------------------------------------------- sim helpers
def time_to_goal(v, a):
    """Positive root of 0.5*a_x t^2 + v_x t - 11 = 0, constant acceleration."""
    v = np.asarray(v, float)
    a = np.asarray(a, float)
    vx, ax = v[..., 0], a[..., 0]
    n = len(vx)
    t = np.full(n, np.nan)
    small = np.abs(ax) < 1e-9
    t[small] = 11.0 / np.maximum(vx[small], 1e-9)
    disc = vx**2 + 22.0 * ax
    ok = (~small) & (disc >= 0)
    sq = np.sqrt(np.maximum(disc, 0.0))
    safe = np.where(small, 1.0, ax)
    t1 = (-vx + sq) / safe
    t2 = (-vx - sq) / safe
    cand = np.where((t1 > 0.1) & (~small), t1, np.inf)
    cand = np.minimum(cand, np.where((t2 > 0.1) & (~small), t2, np.inf))
    t[ok] = cand[ok]
    t[~np.isfinite(t)] = np.nan
    return t


def baseline_endpoint(obs):
    """Constant-acceleration (quadratic) endpoint from noisy observations."""
    obs = np.asarray(obs, float)
    r1, r2 = obs[..., 0, :], obs[..., 1, :]
    a = (r2 - 2.0 * r1 + R0) / DT**2
    v = (r1 - R0) / DT - 0.5 * a * DT  # velocity at t=0
    t = time_to_goal(v, a)
    y = v[..., 1] * t + 0.5 * a[..., 1] * t * t
    z = R0[2] + v[..., 2] * t + 0.5 * a[..., 2] * t * t
    out = np.stack([y, z], axis=-1)
    out[~np.isfinite(out).all(axis=-1)] = np.nan
    return out


def physics_features(obs, sigma):
    """15 physics features + sigma (reconstruction of the 16-dimensional reference set).

    v(0.05) central difference, total accel, aerodynamic accel, speed, tangential
    and perpendicular components, apparent C_D / C_L, constant-accel time-to-goal.
    """
    obs = np.asarray(obs, float)
    sigma = np.asarray(sigma, float).reshape(-1)
    r1, r2 = obs[:, 0, :], obs[:, 1, :]
    v = (r2 - R0) / (2.0 * DT)                      # velocity at t=0.05 s
    a = (r2 - 2.0 * r1 + R0) / DT**2                # total acceleration
    a_aero = a + np.array([0.0, 0.0, GRAV])         # subtract gravity
    speed = np.linalg.norm(v, axis=1)
    vhat = v / np.maximum(speed, 1e-9)[:, None]
    a_par = np.sum(a_aero * vhat, axis=1)
    a_perp = a_aero - a_par[:, None] * vhat
    a_perp_mag = np.linalg.norm(a_perp, axis=1)
    k = RHO * AREA / (2.0 * MASS)
    denom = np.maximum(k * speed**2, 1e-9)
    cd_app = -a_par / denom
    cl_app = a_perp_mag / denom
    v0 = v - a * DT                                  # velocity at t=0
    t_goal = time_to_goal(v0, a)
    return np.column_stack([
        v, a, a_aero, speed, a_par, a_perp_mag, cd_app, cl_app, t_goal, sigma,
    ])


def make_inputs(kind, obs, sigma):
    obs = np.asarray(obs, float)
    sigma = np.asarray(sigma, float).reshape(-1)
    if sigma.size == 1:
        sigma = np.full(len(obs), float(sigma[0]))
    flat = obs.reshape(len(obs), -1)
    if kind == "raw":
        return np.column_stack([flat, sigma])
    if kind == "phys":
        return physics_features(obs, sigma)
    if kind == "phys_reduced":
        # Drop the three gravity-shifted duplicates and apparent Cd/Cl.
        return physics_features(obs, sigma)[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 14, 15]]
    if kind == "phys_reduced13":
        # Pre-P1 13-column variant used by the P2 continuation/mirror experiments:
        # drops the gravity-shifted duplicates only (keeps apparent Cd/Cl columns).
        return physics_features(obs, sigma)[:, [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15]]
    if kind == "rawbase":
        return np.column_stack([flat, sigma, baseline_endpoint(obs)])
    raise ValueError(kind)


def make_target(kind, endpoint, obs):
    endpoint = np.asarray(endpoint, float)
    if kind == "abs":
        return endpoint
    if kind == "res":
        return endpoint - baseline_endpoint(obs)
    raise ValueError(kind)


def standardize_fit(X, Y):
    xm = X.mean(0)
    xs = np.maximum(X.std(0), 1e-6)
    ym = Y.mean(0)
    ys = np.maximum(Y.std(0), 1e-6)
    return dict(xm=xm, xs=xs, ym=ym, ys=ys)


# ------------------------------------------------------------------ the model
def build_mdn(input_dim, k=3, hidden=128, depth=2, layernorm=True, activation="SiLU",
              correlation=True):
    import torch
    from torch import nn

    layers = []
    activation_class = {"SiLU": nn.SiLU, "GELU": nn.GELU, "ReLU": nn.ReLU, "Tanh": nn.Tanh}[activation]
    d = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(d, hidden))
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        layers.append(activation_class())
        d = hidden
    layers.append(nn.Linear(d, 6 * k))
    net = nn.Sequential(*layers)

    class MDN(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.net = net

        def forward(self, x):
            raw = self.net(x).reshape(-1, self.k, 6)
            logits = raw[:, :, 0]
            mu = raw[:, :, 1:3]
            scale = torch.nn.functional.softplus(raw[:, :, 3:5]) + 1e-3
            corr = 0.995 * torch.tanh(raw[:, :, 5]) if correlation else torch.zeros_like(raw[:, :, 5])
            L = torch.zeros((*mu.shape, 2), device=x.device)
            L[:, :, 0, 0] = scale[:, :, 0]
            L[:, :, 1, 0] = corr * scale[:, :, 1]
            L[:, :, 1, 1] = scale[:, :, 1] * torch.sqrt(1 - corr**2)
            comp = torch.distributions.MultivariateNormal(mu, scale_tril=L)
            return torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=logits), comp)

    return MDN()


def load_legacy_mdn(checkpoint):
    """Load the earlier 7->64->64->18 baseline model."""
    import torch
    from torch import nn

    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    k = ck["k"]

    class Legacy(nn.Module):
        def __init__(self):
            super().__init__()
            self.k = k
            self.net = nn.Sequential(
                nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                nn.Linear(64, 6 * k))

        def forward(self, x):
            raw = self.net(x).reshape(-1, self.k, 6)
            logits = raw[:, :, 0]
            mu = raw[:, :, 1:3]
            scale = torch.nn.functional.softplus(raw[:, :, 3:5]) + 1e-3
            corr = 0.995 * torch.tanh(raw[:, :, 5])
            L = torch.zeros((*mu.shape, 2), device=x.device)
            L[:, :, 0, 0] = scale[:, :, 0]
            L[:, :, 1, 0] = corr * scale[:, :, 1]
            L[:, :, 1, 1] = scale[:, :, 1] * torch.sqrt(1 - corr**2)
            comp = torch.distributions.MultivariateNormal(mu, scale_tril=L)
            return torch.distributions.MixtureSameFamily(
                torch.distributions.Categorical(logits=logits), comp)

    model = Legacy()
    model.load_state_dict(ck["state"])
    model.eval()
    stats = {key: np.asarray(ck[key]) for key in ("xm", "xs", "ym", "ys")}
    return model, stats


# ---------------------------------------------------------------------- metrics
def _abs_from_dist(samples, stats, target_kind, base):
    """samples: [S,B,2] standardized; return absolute metres [S,B,2]."""
    out = samples * stats["ys"][None, None, :] + stats["ym"][None, None, :]
    if target_kind == "res":
        out = out + base[None, :, :]
    return out


def evaluate_distribution(dist, y_std, stats, target_kind, base_abs, goal_rect,
                           n_thr=512, n_area=1024, n_goal=512, n_energy=128,
                           hdr_masses=None):
    """Return per-sample metrics for a batch of predicted mixtures."""
    import torch

    y_std = torch.as_tensor(y_std, dtype=torch.float32)
    truth_lp = dist.log_prob(y_std)
    nll = (-truth_lp + float(np.log(stats["ys"]).sum())).detach().cpu().numpy()

    s = dist.sample((n_thr,))
    lp_s = dist.log_prob(s)
    if hdr_masses is None:
        hdr_masses = {0.5: 0.5, 0.95: 0.95}
    thr = {level: torch.quantile(lp_s, 1 - mass, dim=0)
           for level, mass in hdr_masses.items()}
    coverage = {a: (truth_lp >= thr[a]).cpu().numpy() for a in thr}

    s2 = dist.sample((n_area,))
    lp2 = dist.log_prob(s2)
    area = {}
    for a in hdr_masses:
        area[a] = (torch.where(lp2 >= thr[a], torch.exp(-lp2), torch.zeros_like(lp2))
                   .mean(0) * float(np.prod(stats["ys"]))).cpu().numpy()

    g = dist.sample((n_goal,))
    g_abs = _abs_from_dist(g.detach().cpu().numpy(), stats, target_kind, base_abs)
    inside = ((np.abs(g_abs[:, :, 0]) < goal_rect[0])
              & (g_abs[:, :, 1] > goal_rect[1]) & (g_abs[:, :, 1] < goal_rect[2]))
    p_goal = inside.mean(0)

    e = dist.sample((n_energy,)).detach().cpu().numpy()
    e_abs = _abs_from_dist(e, stats, target_kind, base_abs)
    truth_abs = y_std.detach().cpu().numpy() * stats["ys"] + stats["ym"]
    if target_kind == "res":
        truth_abs = truth_abs + base_abs
    h = n_energy // 2
    d_truth = np.linalg.norm(e_abs - truth_abs[None], axis=-1).mean(0)
    d_pair = np.linalg.norm(e_abs[:h] - e_abs[h:], axis=-1).mean(0)
    energy = d_truth - 0.5 * d_pair

    return dict(nll=nll, coverage=coverage, area=area, p_goal=p_goal,
                energy=energy, nll_mean=float(nll.mean()))


def ece(p, y, bins=10):
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= 1)
        if m.sum():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def summarize_errors(err, p_goal, goal_true):
    return dict(
        median=float(np.median(err)),
        mean=float(np.mean(err)),
        p90=float(np.quantile(err, 0.9)),
        p99=float(np.quantile(err, 0.99)),
        max=float(np.max(err)),
        goal_ece=ece(p_goal, goal_true),
        goal_brier=float(np.mean((p_goal - goal_true) ** 2)),
        goal_rate=float(np.mean(goal_true)),
    )
