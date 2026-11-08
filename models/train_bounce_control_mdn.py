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

    def forward(self, f_norm: torch.Tensor, yz_anchor: torch.Tensor, scale_sd: float = 1.0):
        h = self.in_proj(f_norm)
        for blk in self.blocks:
            h = blk(h)
        raw = self.head(h).reshape(-1, self.k, 6)
        logits = raw[:, :, 0]
        mu = yz_anchor[:, None, :] + raw[:, :, 1:3]
        scale = (F.softplus(raw[:, :, 3:5]) + 1e-3) * scale_sd
        corr = 0.98 * torch.tanh(raw[:, :, 5])

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
    noise=None,
):
    rng = np.random.default_rng(seed)
    s_vec = np.full((len(obs_clean), 1), sigma_val, dtype=np.float64)
    obs_noisy = obs_clean + (rng.normal(size=obs_clean.shape) if noise is None else noise) * s_vec

    feats, anc = extract_wiener_features(obs_noisy, s_vec)
    f_t = torch.tensor((feats - xm) / xs, device=device)
    anc_t = torch.tensor(anc, device=device)
    y_t = torch.tensor(y_true, dtype=torch.float32, device=device)

    dist = model(f_t, anc_t, scale_sd=scale_sd)
    nll = float(-dist.log_prob(y_t).mean().item())

    # 均值点预测误差
    pi = dist.mixture_distribution.probs.unsqueeze(-1)
    mean_pred = torch.sum(pi * dist.component_distribution.loc, dim=1).cpu().numpy()
    err_mean = np.linalg.norm(mean_pred - y_true, axis=1)
    projected_mean = mean_pred.copy()
    projected_mean[:, 1] = np.maximum(projected_mean[:, 1], R_BALL)
    projected_error = np.linalg.norm(projected_mean - y_true, axis=1)

    # Monte Carlo 采样估计众数 (Mode)、HDR 覆盖率、HDR 面积与进球概率
    torch.manual_seed(seed)
    samples = dist.sample((n_mc,))  # (M, B, 2)
    logp_samples = dist.log_prob(samples)  # (M, B)
    logp_true = dist.log_prob(y_t)         # (B,)

    # 最高密度点作为众数近似
    max_m = torch.argmax(logp_samples, dim=0)
    mode_pred = samples[max_m, torch.arange(len(obs_clean), device=device)].cpu().numpy()
    err_mode = np.linalg.norm(mode_pred - y_true, axis=1)

    # 经验覆盖率: 真实点对数密度超过 (1 - hdr_mass) 比例的采样点
    tail_prob = torch.mean((logp_samples <= logp_true.unsqueeze(0)).float(), dim=0).cpu().numpy()
    cov_50 = float(np.mean(tail_prob >= (1.0 - hdr_mass_50)))
    cov_95 = float(np.mean(tail_prob >= (1.0 - hdr_mass_95)))

    # 重要性采样估计 HDR 面积: Area(alpha) = E_{q}[ I(q(y) >= c_alpha) / q(y) ]
    # 按对数密度排序求阈值索引
    sorted_logp, _ = torch.sort(logp_samples, dim=0, descending=True)
    k50 = min(max(int(round(hdr_mass_50 * n_mc)) - 1, 0), n_mc - 1)
    k95 = min(max(int(round(hdr_mass_95 * n_mc)) - 1, 0), n_mc - 1)
    thr_50 = sorted_logp[k50]
    thr_95 = sorted_logp[k95]

    inv_p = torch.exp(-logp_samples)
    area_50 = float(torch.mean(torch.where(logp_samples >= thr_50.unsqueeze(0), inv_p, torch.zeros_like(inv_p)), dim=0).median().item())
    area_95 = float(torch.mean(torch.where(logp_samples >= thr_95.unsqueeze(0), inv_p, torch.zeros_like(inv_p)), dim=0).median().item())

    # 进球概率与校准误差 ECE
    samp_np = samples.cpu().numpy()
    in_goal_samp = (np.abs(samp_np[:, :, 0]) <= (GOAL_HALF_Y - R_BALL)) & (samp_np[:, :, 1] >= R_BALL) & (samp_np[:, :, 1] <= (GOAL_Z - R_BALL))
    p_goal = np.mean(in_goal_samp, axis=0)
    y_in_goal = ((np.abs(y_true[:, 0]) <= (GOAL_HALF_Y - R_BALL)) & (y_true[:, 1] >= R_BALL) & (y_true[:, 1] <= (GOAL_Z - R_BALL))).astype(np.float64)
    brier = float(np.mean((p_goal - y_in_goal) ** 2))

    ece = 0.0
    bins = np.linspace(0.0, 1.0, 11)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (p_goal >= b0) & (p_goal < b1 if b1 < 1.0 else p_goal <= b1)
        if np.any(m):
            ece += (np.sum(m) / len(p_goal)) * abs(float(np.mean(p_goal[m]) - np.mean(y_in_goal[m])))

    return {
        "projected_median_m": float(np.median(projected_error)),
        "sample_below_ground_fraction": float((samp_np[:, :, 1] < R_BALL).mean()),
        "n": int(len(obs_clean)),
        "median_m": float(np.median(err_mean)),
        "p90_m": float(np.percentile(err_mean, 90)),
        "mode_median_m": float(np.median(err_mode)),
        "mode_p90_m": float(np.percentile(err_mode, 90)),
        "nll": nll,
        "cov50": cov_50,
        "cov95": cov_95,
        "area50_m2": area_50,
        "area95_m2": area_95,
        "goal_rate": float(np.mean(y_in_goal)),
        "goal_brier": brier,
        "goal_ece": float(ece),
        "tail_prob": tail_prob,
    }


def write_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mc", type=int, default=256)
    parser.add_argument("source", type=Path, help="Directory with airborne data and the bounce/ subdirectory")
    parser.add_argument("output", type=Path, help="New directory for model and metrics")
    parser.add_argument("--arm", choices=("airborne", "bounce"), required=True)
    parser.add_argument("--bounce-dir", type=Path, help="Override the default source/bounce directory")
    args = parser.parse_args()
    if args.epochs < 1 or args.mc < 32:
        parser.error("epochs >= 1 and mc >= 32 are required")
    torch.set_num_threads(2)
    t_start = time.perf_counter()
    device_name = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[1/4] Training Wiener-ResMDN on device: {device}")

    source = args.source.resolve()
    bounce_dir = args.bounce_dir.resolve() if args.bounce_dir else source / "bounce"
    output = args.output.resolve()
    manifest = json.loads((bounce_dir / "bounce_manifest.json").read_text(encoding="utf-8"))
    air_path, air_idx = source / "data.npz", source / "indices.npz"
    full_path, full_idx = bounce_dir / "data_bounce.npz", bounce_dir / "indices_bounce.npz"
    for path, key in ((air_path, "baseline_data_sha256"), (air_idx, "baseline_indices_sha256"),
                      (full_path, "sha256"), (full_idx, "indices_sha256")):
        if sha256_file(path) != manifest[key]:
            raise ValueError(f"Input file hash mismatch: {path}")
    data_path, idx_path = (air_path, air_idx) if args.arm == "airborne" else (full_path, full_idx)
    data, idx = np.load(data_path, allow_pickle=False), np.load(idx_path, allow_pickle=False)
    with np.load(full_path, allow_pickle=False) as full, np.load(full_idx, allow_pickle=False) as full_indices:
        eval_ids = full_indices["test"]
        eval_obs = full["observations"][eval_ids].reshape(-1, 6)
        eval_target = full["endpoint"][eval_ids]
        eval_status = full["status"][eval_ids]
    obs_all = data["observations"].reshape(-1, 6)
    end_all = data["endpoint"]
    tr_i, sel_i, cal_i, te_i = (idx[key] for key in ("train", "selection", "calibration", "test"))
    allowed = [1] if args.arm == "airborne" else [1, 3]
    all_indices = np.concatenate((tr_i, sel_i, cal_i, te_i))
    if not np.array_equal(np.sort(all_indices), np.flatnonzero(np.isin(data["status"], allowed))):
        raise ValueError("Indices must partition the eligible crossings")
    for group, code in zip((tr_i, sel_i, cal_i, te_i), (0, 1, 1, 2)):
        if not len(group) or not np.all(data["split"][group] == code):
            raise ValueError("Indices do not agree with data split codes")
    output.mkdir(parents=True, exist_ok=False)

    obs_tr, y_tr = obs_all[tr_i], end_all[tr_i]
    obs_sel, y_sel = obs_all[sel_i], end_all[sel_i]
    obs_cal, y_cal = obs_all[cal_i], end_all[cal_i]

    rng = np.random.default_rng(11)
    torch.manual_seed(11)

    # 1. 计算训练集标准化统计量
    s_init = rng.uniform(0.001, 0.018, size=(len(obs_tr), 1))
    f_init, _ = extract_wiener_features(obs_tr + rng.normal(size=obs_tr.shape) * s_init, s_init)
    xm = f_init.mean(0, keepdims=True)
    xs = np.maximum(f_init.std(0, keepdims=True), 1e-5)

    # 2. 构建固定 Selection 验证集
    s_sel = rng.choice([0.002, 0.005, 0.010, 0.015], size=(len(obs_sel), 1))
    f_sel, anc_sel = extract_wiener_features(obs_sel + rng.normal(size=obs_sel.shape) * s_sel, s_sel)
    f_sel_t = torch.tensor((f_sel - xm) / xs, device=device)
    anc_sel_t = torch.tensor(anc_sel, device=device)
    y_sel_t = torch.tensor(y_sel, dtype=torch.float32, device=device)

    model = WienerResidualMDN(in_dim=22, hidden_dim=192, depth=3, k=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    best_val = float("inf")
    selected_epoch = 1
    curve_rows = []

    for ep in range(1, epochs + 1):
        model.train()
        s_ep = rng.uniform(0.001, 0.018, size=(len(obs_tr), 1))
        nx = obs_tr + rng.normal(size=obs_tr.shape) * s_ep
        fe, anc = extract_wiener_features(nx, s_ep)
        fe_t = torch.tensor((fe - xm) / xs, device=device)
        anc_t = torch.tensor(anc, device=device)

        perm = torch.randperm(len(fe_t), device=device)
        tr_loss_sum = 0.0
        for b_idx in perm.split(512):
            opt.zero_grad()
            loss = -model(fe_t[b_idx], anc_t[b_idx]).log_prob(y_tr_t[b_idx]).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {ep}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tr_loss_sum += loss.item() * len(b_idx)
        sched.step()

        tr_nll = tr_loss_sum / len(fe_t)
        model.eval()
        with torch.no_grad():
            val_nll = float(-model(f_sel_t, anc_sel_t).log_prob(y_sel_t).mean().item())
        curve_rows.append({"epoch": ep, "train_nll": tr_nll, "val_nll": val_nll})

        if val_nll < best_val:
            best_val = val_nll
            selected_epoch = ep
            torch.save({
                "state": model.state_dict(),
                "stats": {"xm": xm, "xs": xs},
                "architecture": {"depth": 3, "hidden_dim": 192, "activation": "GELU", "residual": True},
                "features": "phys_wiener_22d",
                "k": 5,
            }, output / "model.pt")

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:03d}/{epochs} | Train NLL: {tr_nll:.4f} | Sel Val NLL: {val_nll:.4f} (Best: {best_val:.4f} @ Ep {selected_epoch})")

    write_csv(output / "curve.csv", curve_rows)
    ckpt = torch.load(output / "model.pt", map_location=device, weights_only=False)
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
                return float(-model(f_cal_t, anc_cal_t, scale_sd=float(scale_try)).log_prob(y_cal_t).mean().item())

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

        # Generate once on the common test IDs, then slice for both arms and subgroups.
        noise = np.random.default_rng(3000 + sig_mm).normal(size=eval_obs.shape)
        masks = {"combined": np.ones(len(eval_ids), dtype=bool),
                 "airborne": eval_status == 1, "post_contact": eval_status == 3}
        for subset, mask in masks.items():
            if not mask.any():
                continue
            metrics = evaluate_hdr_and_metrics(
                model, eval_obs[mask], eval_target[mask], sig_m, xm, xs, device,
                scale_sd=best_scale, hdr_mass_50=hdr_50, hdr_mass_95=hdr_95,
                n_mc=args.mc, seed=3000 + sig_mm, noise=noise[mask])
            metrics.pop("tail_prob")
            test_rows.append(dict(model=args.arm, subset=subset, sigma_mm=sig_mm, **metrics))
    write_csv(output / "test_scores.csv", test_rows)

    # 5. 生成题目待测案例 (Case Study) 在各噪声水平下的预测结果
    print("[3/4] Generating competition Case Study predictions...")
    case_preds = {}
    with torch.no_grad():
        for sig_mm in [2, 5, 10, 15]:
            sig_m = sig_mm / 1000.0
            c_cfg = calibration_cfg[str(sig_mm)]
            f_c, anc_c = extract_wiener_features(CASE_OBS, np.array([[sig_m]]))
            dist_c = model(torch.tensor((f_c - xm) / xs, device=device), torch.tensor(anc_c, device=device), scale_sd=c_cfg["scale_sd"])
            torch.manual_seed(6000 + sig_mm)
            samp_c = dist_c.sample((5000,)).squeeze(1)  # (5000, 2)
            logp_c = dist_c.log_prob(samp_c.unsqueeze(1)).squeeze(1)
            mode_c = samp_c[torch.argmax(logp_c)].cpu().numpy()
            mean_c = samp_c.mean(dim=0).cpu().numpy()
            case_preds[f"sigma_{sig_mm}mm"] = {
                "goal_probability_given_crossing": float(
                    ((samp_c[:, 0].abs() <= GOAL_HALF_Y - R_BALL)
                     & (samp_c[:, 1] >= R_BALL) & (samp_c[:, 1] <= GOAL_Z - R_BALL)).float().mean()),
                "hdr_log_density_thresholds": {
                    name: float(torch.quantile(logp_c, 1 - mass))
                    for name, mass in c_cfg["hdr_mass"].items()
                },
                "mode_yz_m": [round(float(mode_c[0]), 4), round(float(mode_c[1]), 4)],
                "mean_yz_m": [round(float(mean_c[0]), 4), round(float(mean_c[1]), 4)],
                "wiener_anchor_yz_m": [round(float(anc_c[0, 0]), 4), round(float(anc_c[0, 1]), 4)],
            }

    with (output / "case_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(case_preds, f, indent=2)

    cfg_out = {
        "dataset_sha256": sha256_file(data_path),
        "indices_sha256": sha256_file(idx_path),
        "arm": args.arm,
        "evaluation_indices_sha256": manifest["indices_sha256"],
        "device": str(device),
        "mc": args.mc,
        "checkpoint_sha256": sha256_file(output / "model.pt"),
        "architecture": {"depth": 3, "hidden_dim": 192, "activation": "GELU", "residual": True},
        "features": "phys_wiener_22d",
        "k": 5,
        "epochs": epochs,
        "seed": 11,
        "parameters": n_params,
        "seconds": time.perf_counter() - t_start,
        "sizes": {"train": int(len(tr_i)), "selection": int(len(sel_i)), "calibration": int(len(cal_i)), "test": int(len(te_i))},
        "selected_epoch": selected_epoch,
        "calibration": calibration_cfg,
    }
    with (output / "config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)

    print("[4/4] All artifacts (model.pt, config.json, curve.csv, test_scores.csv, case_predictions.json) saved successfully!")


if __name__ == "__main__":
    main()
