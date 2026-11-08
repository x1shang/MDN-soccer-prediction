#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wiener-Regularized Residual MDN (Wiener-ResMDN) Training, Calibration & Evaluation Pipeline
兼容输入: data.npz, indices.npz (以及可选的 data_manifest.json)
标准输出: model.pt, config.json, curve.csv, test_scores.csv, case_predictions.json
"""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn as nn
import torch.nn.functional as F

G_ACCEL = 9.81
R_BALL = 0.11
GOAL_X = 11.0
GOAL_HALF_Y = 3.66
GOAL_Z = 2.44
K_AERO = 1.20 * np.pi * (R_BALL ** 2) / (2.0 * 0.43)  # 0.05304177 m^-1
CASE_OBS = np.array([[1.50, 0.45, 0.44, 2.96, 0.86, 0.74]], dtype=np.float64)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_wiener_features(obs_6d: np.ndarray, sigma_vec: np.ndarray):
    """
    提取 22 维物理特征与 2 维维纳滤波残差基线锚点 yz_anchor
    """
    obs_6d = np.asarray(obs_6d, dtype=np.float64).reshape(-1, 6)
    sigma_vec = np.asarray(sigma_vec, dtype=np.float64).reshape(-1, 1)

    r0 = np.array([[0.0, 0.0, R_BALL]], dtype=np.float64)
    r1 = obs_6d[:, 0:3]
    r2 = obs_6d[:, 3:6]
    dt = 0.05
    dt2 = 0.0025

    v_mid = (r2 - r0) / (2.0 * dt)
    a_est = (r2 - 2.0 * r1 + r0) / dt2
    v0_est = v_mid - a_est * dt

    g_vec = np.array([[0.0, 0.0, -G_ACCEL]], dtype=np.float64)
    a_aero = a_est - g_vec
    spd = np.linalg.norm(v_mid, axis=1, keepdims=True) + 1e-8
    v_hat = v_mid / spd

    a_par = np.sum(a_aero * v_hat, axis=1, keepdims=True)
    a_perp = a_aero - a_par * v_hat
    a_perp_mag = np.linalg.norm(a_perp, axis=1, keepdims=True)

    cd_app = np.clip(-a_par / (K_AERO * (spd ** 2)), -2.0, 3.0)
    cl_app = np.clip(a_perp_mag / (K_AERO * (spd ** 2)), 0.0, 2.0)

    # 低方差弹道基线 (仅用中心差分速度 + 重力)
    vx_mid = np.maximum(v_mid[:, 0:1], 6.0)
    t_lin = np.clip(GOAL_X / vx_mid, 0.20, 1.20)
    y_lin = v_mid[:, 1:2] * t_lin
    z_lin = R_BALL + v_mid[:, 2:3] * t_lin - 0.5 * G_ACCEL * (t_lin ** 2)

    # 二阶差分恒加速度基线
    vx0 = np.maximum(v0_est[:, 0:1], 6.0)
    ax = a_est[:, 0:1]
    disc = (vx0 ** 2) + 2.0 * ax * GOAL_X
    valid_d = disc > 4.0
    sqrt_d = np.sqrt(np.maximum(disc, 4.0))
    t_quad = (-vx0 + sqrt_d) / np.where(np.abs(ax) > 1e-4, ax, 1.0)
    t_c = np.where((np.abs(ax) > 1e-4) & valid_d & (t_quad > 0.18) & (t_quad < 1.15), t_quad, t_lin)

    y_quad = np.clip(v0_est[:, 1:2] * t_c + 0.5 * a_est[:, 1:2] * (t_c ** 2), -8.0, 8.0)
    z_quad = np.clip(R_BALL + v0_est[:, 2:3] * t_c + 0.5 * a_est[:, 2:3] * (t_c ** 2), -3.0, 8.0)

    # 噪声自适应维纳收缩锚点
    sig_a_obs = np.sqrt(5.0) * sigma_vec / dt2
    w_wiener = (7.5 ** 2) / ((7.5 ** 2) + (sig_a_obs ** 2))
    y_wiener = y_lin + w_wiener * (y_quad - y_lin)
    z_wiener = z_lin + w_wiener * (z_quad - z_lin)
    yz_anchor = np.hstack([y_wiener, z_wiener])

    feats = np.hstack([
        v_mid, a_est, spd, a_par, a_perp, a_perp_mag,
        cd_app, cl_app, t_lin, t_c, y_lin, z_lin, y_quad, z_quad, w_wiener, sigma_vec
    ])
    return feats.astype(np.float32), yz_anchor.astype(np.float32)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.ln(self.fc(x)))


class WienerResidualMDN(nn.Module):
    def __init__(self, in_dim: int = 22, hidden_dim: int = 192, depth: int = 3, k: int = 5):
        super().__init__()
        self.k = k
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth - 1)])
        self.head = nn.Linear(hidden_dim, 6 * k)

    def mixture_params(self, f_norm, yz_anchor, scale_sd=1.0):
        h = self.in_proj(f_norm)
        for blk in self.blocks:
            h = blk(h)
        raw = self.head(h).reshape(-1, self.k, 6)
        logits = raw[:, :, 0]
        mu = yz_anchor[:, None, :] + raw[:, :, 1:3]
        scale = (F.softplus(raw[:, :, 3:5]) + 1e-3) * scale_sd
        corr = 0.98 * torch.tanh(raw[:, :, 5])
        return logits, mu, scale, corr

    def nll(self, f_norm, yz_anchor, target, scale_sd=1.0):
        logits, mu, scale, corr = self.mixture_params(f_norm, yz_anchor, scale_sd)
        dy, dz = ((target[:, None, :] - mu) / scale).unbind(-1)
        conditional_scale = torch.sqrt(1.0 - corr.square() + 1e-6)
        dz = (dz - corr * dy) / conditional_scale
        log_density = (-math.log(2.0 * math.pi) - scale.log().sum(dim=-1)
                       - conditional_scale.log() - 0.5 * (dy.square() + dz.square()))
        return torch.logsumexp(logits, dim=-1) - torch.logsumexp(logits + log_density, dim=-1)

    def forward(self, f_norm: torch.Tensor, yz_anchor: torch.Tensor, scale_sd: float = 1.0):
        logits, mu, scale, corr = self.mixture_params(f_norm, yz_anchor, scale_sd)

        L = torch.zeros((*mu.shape, 2), device=f_norm.device, dtype=f_norm.dtype)
        L[:, :, 0, 0] = scale[:, :, 0]
        L[:, :, 1, 0] = corr * scale[:, :, 1]
        L[:, :, 1, 1] = scale[:, :, 1] * torch.sqrt(1.0 - corr ** 2 + 1e-6)
        comp = torch.distributions.MultivariateNormal(mu, scale_tril=L)
        return torch.distributions.MixtureSameFamily(torch.distributions.Categorical(logits=logits), comp)


@torch.no_grad()
def evaluate_hdr_and_metrics(
    model: WienerResidualMDN,
    obs_clean: np.ndarray,
    y_true: np.ndarray,
    sigma_val: float,
    xm: np.ndarray,
    xs: np.ndarray,
    device: torch.device,
    scale_sd: float = 1.0,
    hdr_mass_50: float = 0.50,
    hdr_mass_95: float = 0.95,
    n_mc: int = 256,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)
    s_vec = np.full((len(obs_clean), 1), sigma_val, dtype=np.float64)
    obs_noisy = obs_clean + rng.normal(size=obs_clean.shape) * s_vec

    feats, anc = extract_wiener_features(obs_noisy, s_vec)
    f_t = torch.tensor((feats - xm) / xs, device=device)
    anc_t = torch.tensor(anc, device=device)
    y_t = torch.tensor(y_true, dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    batches = []
    for f_b, anc_b, y_b in zip(f_t.split(2048), anc_t.split(2048), y_t.split(2048)):
        batches.append(score_batch(model(f_b, anc_b, scale_sd=scale_sd), y_b,
                                   hdr_mass_50, hdr_mass_95, n_mc))
    nll, err_mean, err_mode, tail_prob, area_50, area_95, p_goal = torch.cat(batches).cpu().numpy().T
    y_in_goal = ((np.abs(y_true[:, 0]) <= (GOAL_HALF_Y - R_BALL)) & (y_true[:, 1] >= R_BALL) & (y_true[:, 1] <= (GOAL_Z - R_BALL))).astype(np.float64)
    brier = float(np.mean((p_goal - y_in_goal) ** 2))

    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (p_goal >= b0) & (p_goal < b1 if b1 < 1.0 else p_goal <= b1)
        if np.any(m):
            ece += (np.sum(m) / len(p_goal)) * abs(float(np.mean(p_goal[m]) - np.mean(y_in_goal[m])))

    return {
        "n": int(len(obs_clean)),
        "median_m": float(np.median(err_mean)),
        "p90_m": float(np.percentile(err_mean, 90)),
        "mode_median_m": float(np.median(err_mode)),
        "mode_p90_m": float(np.percentile(err_mode, 90)),
        "nll": float(nll.mean(dtype=np.float64)),
        "cov50": float(np.mean(tail_prob >= 1.0 - hdr_mass_50)),
        "cov95": float(np.mean(tail_prob >= 1.0 - hdr_mass_95)),
        "area50_m2": float(np.median(area_50)),
        "area95_m2": float(np.median(area_95)),
        "goal_rate": float(np.mean(y_in_goal)),
        "goal_brier": brier,
        "goal_ece": float(ece),
        "tail_prob": tail_prob,
    }


def score_batch(dist, y_t, hdr_mass_50, hdr_mass_95, n_mc):
    samples = dist.sample((n_mc,))  # (M, B, 2)
    logp_samples = dist.log_prob(samples)  # (M, B)
    logp_true = dist.log_prob(y_t)         # (B,)

    # 最高密度点作为众数近似
    max_m = torch.argmax(logp_samples, dim=0)
    mode_pred = samples[max_m, torch.arange(len(y_t), device=y_t.device)]

    # 经验覆盖率: 真实点对数密度超过 (1 - hdr_mass) 比例的采样点
    tail_prob = torch.mean((logp_samples <= logp_true.unsqueeze(0)).float(), dim=0)

    # 重要性采样估计 HDR 面积: Area(alpha) = E_{q}[ I(q(y) >= c_alpha) / q(y) ]
    # 按对数密度排序求阈值索引
    sorted_logp, _ = torch.sort(logp_samples, dim=0, descending=True)
    k50 = min(max(int(round(hdr_mass_50 * n_mc)) - 1, 0), n_mc - 1)
    k95 = min(max(int(round(hdr_mass_95 * n_mc)) - 1, 0), n_mc - 1)
    thr_50 = sorted_logp[k50]
    thr_95 = sorted_logp[k95]

    inv_p = torch.exp(-logp_samples)
    area_50 = torch.where(logp_samples >= thr_50, inv_p, 0.).mean(dim=0)
    area_95 = torch.where(logp_samples >= thr_95, inv_p, 0.).mean(dim=0)

    # 进球概率与校准误差 ECE
    y, z = samples.unbind(-1)
    p_goal = ((y.abs() <= GOAL_HALF_Y - R_BALL) & (z >= R_BALL) &
              (z <= GOAL_Z - R_BALL)).float().mean(dim=0)
    return torch.stack((-logp_true, torch.linalg.vector_norm(dist.mean - y_t, dim=1),
                        torch.linalg.vector_norm(mode_pred - y_t, dim=1),
                        tail_prob, area_50, area_95, p_goal), dim=1)


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def replay_asai(directory, model, xm, xs, device, manifest):
    from openpyxl import load_workbook

    files = {"Fig3": "Fig3_Sample data 20261108.xlsx",
             "Sample05": "Sample 05 data 20261108.xlsx",
             "Sample12": "Sample 12 data 20261108.xlsx"}
    sources = {row["id"]: row["sources"] for row in manifest.get("anchors", [])
               if row["cohort"] == "asai" and row["used"]}
    sealed = {row["id"]: row["sources"] for row in manifest.get("sealed_asai", [])}
    if sources.keys() & sealed.keys():
        raise ValueError("An Asai trial is both a training prior and a holdout")
    sources.update(sealed)
    rows = []
    for name, filename in files.items():
        path = directory / filename
        if sha256_file(path) != sources.get(name, {}).get(filename):
            raise ValueError(f"{name}: workbook is missing or differs from the v2 fit source")
        with path.open("rb") as stream:
            book = load_workbook(stream, read_only=True, data_only=True)
            try:
                values = np.asarray(list(book.active.iter_rows(
                    min_row=3, min_col=2, max_col=5, values_only=True)), dtype=float)
            finally:
                book.close()
        values = values[np.isfinite(values).all(axis=1)]
        times, positions = values[:, 0], values[:, [2, 1, 3]].copy()
        if len(times) < 11 or not np.allclose(times, np.arange(len(times)) * .01):
            raise ValueError(f"{name}: expected a continuous 100 Hz position series")
        positions[:, :2] -= positions[0, :2].copy()
        x = positions[:, 0]
        crossing = np.flatnonzero((x[:-1] < GOAL_X) & (x[1:] >= GOAL_X))
        if not len(crossing):
            raise ValueError(f"{name}: observed track does not cross 11 m")
        i = int(crossing[0])
        fraction = (GOAL_X - x[i]) / (x[i + 1] - x[i])
        truth = positions[i, 1:] + fraction * (positions[i + 1, 1:] - positions[i, 1:])
        if truth[1] <= R_BALL or np.any(positions[:i + 1, 2] < R_BALL):
            raise ValueError(f"{name}: observed crossing is not airborne")
        shift = positions[0, 2] - R_BALL
        obs = positions[[5, 10]].copy()
        obs[:, 2] -= shift
        feats, anchor = extract_wiener_features(obs, [.010])
        dist = model(torch.as_tensor((feats - xm) / xs, device=device),
                     torch.as_tensor(anchor, device=device))
        prediction = dist.mean[0].cpu().numpy() + [0., shift]
        assessment = ("Source-held-out Asai benchmark" if name in sealed
                      else "Retrospective; trial contributed to the fitted simulation prior")
        rows.append(dict(case=name, model="fresh_wiener_res_mdn", sigma_mm=10,
                         error_cm=float(100 * np.linalg.norm(prediction - truth)),
                         assessment=assessment))
    return rows


class CapturedAdamStep:
    """Replay a minibatch update with the same buffers throughout the run."""

    def __init__(self, model, optimizer, features, anchors, targets, batch_size):
        self.indices = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        self.count = torch.full((), batch_size, device=features.device)
        self.positions = torch.arange(batch_size, device=features.device)

        def step():
            optimizer.zero_grad(set_to_none=True)
            losses = model.nll(features[self.indices], anchors[self.indices], targets[self.indices])
            loss = (losses * (self.positions < self.count)).sum() / self.count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=True)
            optimizer.step()
            return loss

        initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side_stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = step()
        model.load_state_dict(initial)
        for state in optimizer.state.values():
            for value in state.values():
                value.zero_()

    def __call__(self, indices):
        self.indices[:len(indices)].copy_(indices)
        self.count.fill_(len(indices))
        self.graph.replay()
        return self.loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mc", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11, help="Initialization, shuffling and training noise")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--asai-dir", type=Path, help="Optional directory with the three local workbooks")
    args = parser.parse_args()
    if args.epochs < 1 or args.mc < 32 or args.seed < 0 or args.batch_size < 1:
        parser.error("epochs >= 1, mc >= 32, seed >= 0 and batch-size >= 1 are required")
    torch.set_num_threads(2)
    t_start = time.perf_counter()
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print(f"[1/4] Training Wiener-ResMDN on {device}, seed {args.seed}")

    data_path = args.data_dir / "data.npz"
    idx_path = args.data_dir / "indices.npz"
    data = np.load(data_path)
    idx = np.load(idx_path)

    obs_all = data["observations"].reshape(-1, 6)
    end_all = data["endpoint"]
    tr_i, sel_i, cal_i, te_i = idx["train"], idx["selection"], idx["calibration"], idx["test"]

    manifest = {}
    manifest_path = args.data_dir / "data_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != sha256_file(data_path) or manifest["indices_sha256"] != sha256_file(idx_path):
            raise ValueError("Input file hash mismatch")
        quality = manifest["quality"]
        if not (quality["crossing_pass"] and quality["near_case_pass"]):
            print("Dataset quality gates did not pass; treat scores as exploratory.")
    all_indices = np.concatenate((tr_i, sel_i, cal_i, te_i))
    if not np.array_equal(np.sort(all_indices), np.flatnonzero(data["status"] == 1)):
        raise ValueError("Indices must be disjoint and cover exactly the airborne crossings")
    split_codes = data["split"]
    for group, code in zip((tr_i, sel_i, cal_i, te_i), (0, 1, 1, 2)):
        if not len(group) or not np.all(split_codes[group] == code):
            raise ValueError("Indices do not agree with data.npz split codes")

    obs_tr, y_tr = obs_all[tr_i], end_all[tr_i]
    obs_sel, y_sel = obs_all[sel_i], end_all[sel_i]
    obs_cal, y_cal = obs_all[cal_i], end_all[cal_i]
    obs_te, y_te = obs_all[te_i], end_all[te_i]

    fixed_rng = np.random.default_rng(11)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1. 计算训练集标准化统计量
    s_init = fixed_rng.uniform(0.001, 0.018, size=(len(obs_tr), 1))
    f_init, _ = extract_wiener_features(obs_tr + fixed_rng.normal(size=obs_tr.shape) * s_init, s_init)
    xm = f_init.mean(0, keepdims=True)
    xs = np.maximum(f_init.std(0, keepdims=True), 1e-5)

    # 2. 构建固定 Selection 验证集
    s_sel = fixed_rng.choice([0.002, 0.005, 0.010, 0.015], size=(len(obs_sel), 1))
    f_sel, anc_sel = extract_wiener_features(obs_sel + fixed_rng.normal(size=obs_sel.shape) * s_sel, s_sel)
    f_sel_t = torch.tensor((f_sel - xm) / xs, device=device)
    anc_sel_t = torch.tensor(anc_sel, device=device)
    y_sel_t = torch.tensor(y_sel, dtype=torch.float32, device=device)

    model = WienerResidualMDN(in_dim=22, hidden_dim=192, depth=3, k=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    use_graph = device.type == "cuda"
    lr = torch.tensor(2e-3, device=device) if use_graph else 2e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                           fused=use_graph, capturable=use_graph)
    epochs = args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    best_val = float("inf")
    selected_epoch = 1
    curve_rows = []
    captured_step = None
    fe_t = torch.empty((len(obs_tr), 22), device=device)
    anc_t = torch.empty_like(y_tr_t)

    for ep in range(1, epochs + 1):
        model.train()
        s_ep = rng.uniform(0.001, 0.018, size=(len(obs_tr), 1))
        nx = obs_tr + rng.normal(size=obs_tr.shape) * s_ep
        fe, anc = extract_wiener_features(nx, s_ep)
        fe_t.copy_(torch.from_numpy((fe - xm) / xs))
        anc_t.copy_(torch.from_numpy(anc))
        if use_graph and captured_step is None:
            captured_step = CapturedAdamStep(model, opt, fe_t, anc_t, y_tr_t, args.batch_size)

        perm = torch.randperm(len(fe_t), device=device)
        tr_loss_sum = torch.zeros((), device=device)
        for b_idx in perm.split(args.batch_size):
            if captured_step is not None:
                loss = captured_step(b_idx)
            else:
                opt.zero_grad(set_to_none=True)
                loss = model.nll(fe_t[b_idx], anc_t[b_idx], y_tr_t[b_idx]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            tr_loss_sum.add_(loss.detach(), alpha=len(b_idx))
        sched.step()

        tr_nll = tr_loss_sum.item() / len(fe_t)
        model.eval()
        with torch.no_grad():
            val_nll = model.nll(f_sel_t, anc_sel_t, y_sel_t).mean().item()
        if not (math.isfinite(tr_nll) and math.isfinite(val_nll)):
            raise RuntimeError(f"Nonfinite loss at epoch {ep}")
        curve_rows.append({"epoch": ep, "train_nll": tr_nll, "val_nll": val_nll})
        write_csv("curve.csv", curve_rows)

        if val_nll < best_val:
            best_val = val_nll
            selected_epoch = ep
            torch.save({
                "state": model.state_dict(),
                "stats": {"xm": xm, "xs": xs},
                "architecture": {"depth": 3, "hidden_dim": 192, "activation": "GELU", "residual": True},
                "features": "phys_wiener_22d",
                "k": 5,
            }, "model.pt")

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:03d}/{epochs} | Train NLL: {tr_nll:.4f} | Sel Val NLL: {val_nll:.4f} (Best: {best_val:.4f} @ Ep {selected_epoch})")

    ckpt = torch.load("model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()

    # 3. 在独立 Calibration 集上校准每个噪声水平的 scale_sd 与 hdr_mass
    print("[2/4] Calibrating scale_sd and HDR quantiles on independent calibration split...")
    calibration_cfg = {}
    test_rows = []

    for sig_mm in [2, 5, 10, 15]:
        sig_m = sig_mm / 1000.0
        rng_cal = np.random.default_rng(1000 + sig_mm)
        s_cal = np.full((len(obs_cal), 1), sig_m)
        f_cal, anc_cal = extract_wiener_features(obs_cal + rng_cal.normal(size=obs_cal.shape) * s_cal, s_cal)
        f_cal_t = torch.tensor((f_cal - xm) / xs, device=device)
        anc_cal_t = torch.tensor(anc_cal, device=device)
        y_cal_t = torch.tensor(y_cal, dtype=torch.float32, device=device)

        def cal_nll_obj(scale_try: float) -> float:
            with torch.no_grad():
                return model.nll(f_cal_t, anc_cal_t, y_cal_t, scale_sd=float(scale_try)).mean().item()

        opt_res = minimize_scalar(cal_nll_obj, bounds=(0.80, 1.35), method="bounded")
        if not opt_res.success:
            raise RuntimeError(opt_res.message)
        best_scale = float(opt_res.x)

        cal_eval = evaluate_hdr_and_metrics(model, obs_cal, y_cal, sig_m, xm, xs, device, scale_sd=best_scale, n_mc=args.mc, seed=1000 + sig_mm)
        tp = cal_eval["tail_prob"]
        # 找到使 calibration 集覆盖率恰好达到 50% 和 95% 的名义概率质量阈值
        hdr_50 = float(np.clip(1.0 - np.quantile(tp, 0.50), 0.35, 0.65))
        hdr_95 = float(np.clip(1.0 - np.quantile(tp, 0.05), 0.85, 0.985))
        calibration_cfg[str(sig_mm)] = {"scale_sd": best_scale, "hdr_mass": {"0.5": hdr_50, "0.95": hdr_95}}

        # 4. 在独立 Test 集上评估 raw / scaled / scaled_hdr 三种模式
        for cal_mode, sc_val, m50, m95 in [
            ("raw", 1.0, 0.50, 0.95),
            ("scaled", best_scale, 0.50, 0.95),
            ("scaled_hdr", best_scale, hdr_50, hdr_95),
        ]:
            te_m = evaluate_hdr_and_metrics(model, obs_te, y_te, sig_m, xm, xs, device, scale_sd=sc_val, hdr_mass_50=m50, hdr_mass_95=m95, n_mc=args.mc, seed=3000 + sig_mm)
            te_m.pop("tail_prob")
            row = {"model": "fresh_wiener_res_mdn", "calibration": cal_mode, "sigma_mm": float(sig_mm), "scale_sd": sc_val, **te_m}
            test_rows.append(row)
            if cal_mode == "scaled_hdr":
                print(f"  [Test σ={sig_mm:2d}mm] Median Err: {te_m['median_m']:.4f}m | NLL: {te_m['nll']:.4f} | Cov50/95: {te_m['cov50']*100:.1f}% / {te_m['cov95']*100:.1f}%")

    write_csv("test_scores.csv", test_rows)
    if manifest.get("sealed_asai"):
        common = te_i[data["anchor"][te_i] == -1]
        calibration = calibration_cfg["5"]
        summary = evaluate_hdr_and_metrics(
            model, obs_all[common], end_all[common], .005, xm, xs, device,
            scale_sd=calibration["scale_sd"], hdr_mass_50=calibration["hdr_mass"]["0.5"],
            hdr_mass_95=calibration["hdr_mass"]["0.95"], n_mc=args.mc, seed=3005)
        summary.pop("tail_prob")
        write_csv("common_canonical_scores.csv", [summary])

    # 5. 生成题目待测案例 (Case Study) 在各噪声水平下的预测结果
    print("[3/4] Generating competition Case Study predictions...")
    case_preds = {}
    with torch.no_grad():
        for sig_mm in [2, 5, 10, 15]:
            sig_m = sig_mm / 1000.0
            c_cfg = calibration_cfg[str(sig_mm)]
            f_c, anc_c = extract_wiener_features(CASE_OBS, np.array([[sig_m]]))
            dist_c = model(torch.tensor((f_c - xm) / xs, device=device), torch.tensor(anc_c, device=device), scale_sd=c_cfg["scale_sd"])
            samp_c = dist_c.sample((5000,)).squeeze(1)  # (5000, 2)
            logp_c = dist_c.log_prob(samp_c.unsqueeze(1)).squeeze(1)
            mode_c = samp_c[torch.argmax(logp_c)].cpu().numpy()
            mean_c = samp_c.mean(dim=0).cpu().numpy()
            case_preds[f"sigma_{sig_mm}mm"] = {
                "mode_yz_m": [round(float(mode_c[0]), 4), round(float(mode_c[1]), 4)],
                "mean_yz_m": [round(float(mean_c[0]), 4), round(float(mean_c[1]), 4)],
                "wiener_anchor_yz_m": [round(float(anc_c[0, 0]), 4), round(float(anc_c[0, 1]), 4)],
            }

    with open("case_predictions.json", "w", encoding="utf-8") as f:
        json.dump(case_preds, f, indent=2)
    if args.asai_dir:
        write_csv("asai_replay.csv", replay_asai(args.asai_dir, model, xm, xs, device, manifest))

    cfg_out = {
        "dataset_sha256": sha256_file(data_path),
        "indices_sha256": sha256_file(idx_path),
        "dataset_quality": manifest.get("quality"),
        "device": str(device),
        "mc": args.mc,
        "checkpoint_sha256": sha256_file(Path("model.pt")),
        "architecture": {"depth": 3, "hidden_dim": 192, "activation": "GELU", "residual": True},
        "features": "phys_wiener_22d",
        "k": 5,
        "epochs": epochs,
        "seed": args.seed,
        "normalization_and_selection_seed": 11,
        "batch_size": args.batch_size,
        "torch_version": torch.__version__,
        "training_script_sha256": sha256_file(Path(__file__)),
        "parameters": n_params,
        "seconds": time.perf_counter() - t_start,
        "sizes": {"train": int(len(tr_i)), "selection": int(len(sel_i)), "calibration": int(len(cal_i)), "test": int(len(te_i))},
        "selected_epoch": selected_epoch,
        "calibration": calibration_cfg,
    }
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    print("[4/4] All artifacts (model.pt, config.json, curve.csv, test_scores.csv, case_predictions.json) saved successfully!")


if __name__ == "__main__":
    main()
