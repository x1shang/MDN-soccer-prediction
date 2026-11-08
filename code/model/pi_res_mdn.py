#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UPC Problem B: Physics-Informed Residual Mixture Density Network (PI-ResMDN)
功能：
1. 极速向量化 RK4 三维空气动力学轨迹生成（含雷诺数阻力危机与旋转升力耦合）
2. 16 维物理特征工程（正交分解 a_parallel, a_perp + 恒加速度解析基线 Shortcut）
3. 全协方差残差混合密度网络（Full-Covariance Residual MDN，含相关系数 rho）
4. 动态噪声条件训练（Noise-Conditioned Training）与 50%/95% HPD 最高密度区域评测
5. 题目案例（Case Study）多噪声情景预测与 FIFA 标准球门热图生成
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =====================================================================
# 1. 全局物理常数与题目常量设置
# =====================================================================
G_ACCEL = 9.81          # 重力加速度 (m/s^2)
R_BALL = 0.11           # 球半径 (m)，初始位置 z0 = 0.11 m
M_BALL = 0.43           # 标准足球质量 (kg)
RHO_AIR = 1.225         # 空气密度 (kg/m^3)
AREA_BALL = np.pi * (R_BALL ** 2)
AERO_PREFACTOR = 0.5 * RHO_AIR * AREA_BALL / M_BALL  # ~ 0.0541 m^-1
X_GOAL = 11.0           # 点球点到球门线距离 (m)
GOAL_HALF_WIDTH = 3.66  # FIFA 标准球门半宽 (7.32m / 2)
GOAL_HEIGHT = 2.44      # FIFA 标准球门高度 (2.44m)


# =====================================================================
# 2. 极速向量化 RK4 物理仿真器（用于生成高质量跨体制训练/测试数据）
# =====================================================================
def generate_physics_dataset(
    n_samples: int = 50000,
    seed: int = 42,
    dt: float = 0.0025,
    max_steps: int = 280,
):
    """
    向量化生成三维点球轨迹（包含：大力抽射、强香蕉球、高低射门等物理体制）
    考虑：重力 + 二次空气阻力 + 有效马格努斯升力 + 旋转指数衰减
    仅保留有效空中穿越 x = 11.0 m 且未提前触地 (z > R_BALL) 的样本
    """
    rng = np.random.default_rng(seed)
    # 多采样 25% 以过滤提前触地或未到达球门的样本
    n_raw = int(n_samples * 1.25)

    # 采样初速度分量 (覆盖常见点球 + 题目案例 ~32.5 m/s 的高速区间)
    vx0 = rng.uniform(20.0, 33.5, size=n_raw)
    vy0 = rng.uniform(-10.5, 10.5, size=n_raw)
    vz0 = rng.uniform(1.5, 9.5, size=n_raw)

    # 采样空气阻力系数 Cd (风洞文献后危机超临界区 [0.14, 0.28] + 少量粗糙/退化球 [0.28, 0.38])
    cd = np.where(
        rng.random(n_raw) < 0.85,
        rng.uniform(0.15, 0.26, size=n_raw),
        rng.uniform(0.26, 0.38, size=n_raw),
    )

    # 采样旋转角速度向量 omega (rad/s): 侧旋 omega_z 为主，兼顾上旋/下旋 omega_y 和横滚 omega_x
    omega_x = rng.normal(0.0, 12.0, size=n_raw)
    omega_y = rng.uniform(-35.0, 35.0, size=n_raw)
    omega_z = rng.uniform(-75.0, 75.0, size=n_raw)
    omega0 = np.stack([omega_x, omega_y, omega_z], axis=-1)  # (N, 3)

    # 旋转衰减时间常数 tau (s) 与升力效率因子
    tau_spin = rng.uniform(0.8, 2.5, size=n_raw)
    cl_gain = rng.uniform(0.85, 1.15, size=n_raw)

    # 状态向量初始化: r = (x, y, z), v = (vx, vy, vz)
    r = np.zeros((n_raw, 3), dtype=np.float64)
    r[:, 2] = R_BALL
    v = np.stack([vx0, vy0, vz0], axis=-1).astype(np.float64)

    r_005 = np.zeros_like(r)
    r_010 = np.zeros_like(r)
    step_005 = int(round(0.05 / dt))
    step_010 = int(round(0.10 / dt))

    crossed = np.zeros(n_raw, dtype=bool)
    grounded_before_goal = np.zeros(n_raw, dtype=bool)
    t_cross = np.zeros(n_raw, dtype=np.float64)
    yz_cross = np.zeros((n_raw, 2), dtype=np.float64)

    g_vec = np.array([0.0, 0.0, -G_ACCEL], dtype=np.float64)

    def accel_fn(vel, t_val):
        speed = np.linalg.norm(vel, axis=-1, keepdims=True) + 1e-8
        # 旋转衰减
        decay = np.exp(-t_val / tau_spin)[:, None]
        omega_t = omega0 * decay
        omega_norm = np.linalg.norm(omega_t, axis=-1, keepdims=True) + 1e-8
        # 旋转参数 S = R * omega / v
        spin_param = (R_BALL * omega_norm) / speed
        # Robins-Magnus 经验饱和升力系数 Cl(S) ~ 0.38 * S / (0.18 + S)
        cl = cl_gain[:, None] * (0.38 * spin_param / (0.18 + spin_param))

        # 阻力加速度: - (0.5 * rho * A / m) * Cd * |v| * v
        a_drag = -AERO_PREFACTOR * cd[:, None] * speed * vel
        # 马格努斯加速度: (0.5 * rho * A / m) * Cl * |v|^2 * (omega x v) / (|omega||v|)
        cross_ov = np.cross(omega_t, vel)
        a_magnus = AERO_PREFACTOR * cl * speed * (cross_ov / omega_norm)
        return g_vec + a_drag + a_magnus

    # 向量化 RK4 积分
    for step in range(1, max_steps + 1):
        t_prev = (step - 1) * dt
        r_prev = r.copy()

        k1_r = v
        k1_v = accel_fn(v, t_prev)

        k2_r = v + 0.5 * dt * k1_v
        k2_v = accel_fn(k2_r, t_prev + 0.5 * dt)

        k3_r = v + 0.5 * dt * k2_v
        k3_v = accel_fn(k3_r, t_prev + 0.5 * dt)

        k4_r = v + dt * k3_v
        k4_v = accel_fn(k4_r, t_prev + dt)

        r = r + (dt / 6.0) * (k1_r + 2 * k2_r + 2 * k3_r + k4_r)
        v = v + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

        if step == step_005:
            r_005 = r.copy()
        elif step == step_010:
            r_010 = r.copy()

        # 检测未过线前的触地事件 (z <= R_BALL)
        newly_grounded = (~crossed) & (r[:, 2] <= R_BALL)
        grounded_before_goal |= newly_grounded

        # 检测首次正向穿越 x = 11.0 m 事件并线性插值
        newly_crossed = (~crossed) & (~grounded_before_goal) & (r_prev[:, 0] < X_GOAL) & (r[:, 0] >= X_GOAL)
        if np.any(newly_crossed):
            alpha = (X_GOAL - r_prev[newly_crossed, 0]) / (r[newly_crossed, 0] - r_prev[newly_crossed, 0] + 1e-12)
            r_interp = r_prev[newly_crossed] + alpha[:, None] * (r[newly_crossed] - r_prev[newly_crossed])
            # 确认插值点也在地面以上
            valid_air = r_interp[:, 2] > R_BALL
            idx_crossed = np.where(newly_crossed)[0]
            valid_idx = idx_crossed[valid_air]
            invalid_idx = idx_crossed[~valid_air]

            crossed[valid_idx] = True
            t_cross[valid_idx] = t_prev + alpha[valid_air] * dt
            yz_cross[valid_idx, 0] = r_interp[valid_air, 1]
            yz_cross[valid_idx, 1] = r_interp[valid_air, 2]
            grounded_before_goal[invalid_idx] = True

    valid_mask = crossed & (~grounded_before_goal)
    valid_indices = np.where(valid_mask)[0][:n_samples]

    x_clean = np.hstack([r_005[valid_indices], r_010[valid_indices]]).astype(np.float32)
    y_target = yz_cross[valid_indices].astype(np.float32)
    meta = {
        "total_simulated": int(n_raw),
        "valid_air_crossing": int(np.sum(valid_mask)),
        "excluded_ground_or_short_ratio": float(1.0 - np.mean(valid_mask)),
        "kept_samples": int(len(valid_indices)),
    }
    return x_clean, y_target, meta


# =====================================================================
# 3. 核心创新：16 维物理特征工程 + 恒加速度解析基线计算 (PyTorch 张量版)
# =====================================================================
def extract_physics_features(coords_6d: torch.Tensor, sigma_noise: torch.Tensor):
    """
    输入:
      coords_6d: (B, 6) -> [x_0.05, y_0.05, z_0.05, x_0.10, y_0.10, z_0.10]
      sigma_noise: (B, 1) -> 当前样本对应的位置测量噪声标准差 (m)
    输出:
      features_16d: (B, 16) 物理增强特征向量
      yz_const: (B, 2) 恒加速度外推解析落点 [Y_const, Z_const] (用于残差学习连接)
    """
    dt = 0.05
    dt2 = 0.0025  # dt^2

    r0 = torch.tensor([0.0, 0.0, R_BALL], device=coords_6d.device, dtype=coords_6d.dtype).unsqueeze(0)
    r1 = coords_6d[:, 0:3]  # r(0.05)
    r2 = coords_6d[:, 3:6]  # r(0.10)

    # 1. 中心差分求 t=0.05s 速度 v_mid 与常数加速度估计 a_est
    v_mid = (r2 - r0) / (2.0 * dt)                 # (B, 3)
    a_est = (r2 - 2.0 * r1 + r0) / dt2             # (B, 3)
    v0_est = v_mid - a_est * dt                    # (B, 3) t=0 初速度估计

    # 2. 正交物理分解: 扣除重力后的纯空气动力加速度 a_aero = a_est - g
    g_tensor = torch.tensor([0.0, 0.0, -G_ACCEL], device=coords_6d.device, dtype=coords_6d.dtype).unsqueeze(0)
    a_aero = a_est - g_tensor                      # (B, 3)

    speed_mid = torch.norm(v_mid, dim=-1, keepdim=True).clamp(min=1e-5)
    v_hat = v_mid / speed_mid                      # 单位速度向量 (B, 3)

    # 切向分量 (平行于 v，物理上由空气阻力主导)
    a_parallel = torch.sum(a_aero * v_hat, dim=-1, keepdim=True)  # (B, 1)
    # 法向分量 (垂直于 v，物理上由马格努斯升力主导)
    a_perp = a_aero - a_parallel * v_hat                          # (B, 3)
    a_perp_mag = torch.norm(a_perp, dim=-1, keepdim=True)         # (B, 1)

    # 表观阻力系数 Cd_apparent = - a_parallel / (AERO_PREFACTOR * |v|^2)
    # 用于让网络立刻识别出二阶差分是否被测量噪声污染（正常应在 0.15~0.35，噪声下可达 >1.0 或 <0）
    cd_apparent = (-a_parallel / (AERO_PREFACTOR * (speed_mid ** 2))).clamp(-2.0, 3.0)

    # 3. 解析求解恒加速度基线穿越 x = 11m 的时间 T_const 与落点 (Y_const, Z_const)
    # 0.5 * ax * T^2 + vx0 * T - X_GOAL = 0
    vx0 = v0_est[:, 0:1].clamp(min=5.0)
    ax = a_est[:, 0:1]
    discriminant = vx0 ** 2 + 2.0 * ax * X_GOAL
    # 若噪声导致判别式 <= 0，退化为匀速渡越时间估计并用平滑激活兜底
    valid_disc = discriminant > 1.0
    sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=1.0))

    t_quad = (-vx0 + sqrt_disc) / torch.where(torch.abs(ax) > 1e-4, ax, torch.ones_like(ax))
    t_lin = X_GOAL / vx0
    t_const = torch.where((torch.abs(ax) > 1e-4) & valid_disc & (t_quad > 0.15), t_quad, t_lin)
    t_const = t_const.clamp(0.20, 1.20)  # 合理飞行时间范围 (B, 1)

    y_const = v0_est[:, 1:2] * t_const + 0.5 * a_est[:, 1:2] * (t_const ** 2)
    z_const = R_BALL + v0_est[:, 2:3] * t_const + 0.5 * a_est[:, 2:3] * (t_const ** 2)
    yz_const = torch.cat([y_const, z_const], dim=-1)  # (B, 2)

    # 组装 16 维高信息量物理特征
    features_16d = torch.cat([
        r1,             # 3D: r(0.05)
        r2,             # 3D: r(0.10)
        v_mid / 30.0,   # 3D: 归一化速度向量
        a_parallel / 20.0,  # 1D: 切向阻力减速度
        a_perp / 20.0,      # 3D: 法向马格努斯升力加速度向量
        cd_apparent,        # 1D: 表观阻力系数诊断指标
        t_const,            # 1D: 基线渡越时间估计
        sigma_noise * 100.0 # 1D: 测量噪声水平 (以 cm 为单位)
    ], dim=-1)

    return features_16d, yz_const


# =====================================================================
# 4. 全协方差残差混合密度网络 (Full-Covariance Residual MDN)
# =====================================================================
class PhysicsResidualMDN(nn.Module):
    def __init__(self, in_features: int = 16, hidden_dim: int = 128, n_components: int = 5):
        super().__init__()
        self.n_components = n_components

        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        # 每个二维高斯分量包含 6 个参数: [logit_pi, d_mu_y, d_mu_z, log_sy, log_sz, atanh_rho]
        self.head = nn.Linear(hidden_dim, n_components * 6)

    def forward(self, coords_6d: torch.Tensor, sigma_noise: torch.Tensor, feat_mean: torch.Tensor, feat_std: torch.Tensor):
        feats, yz_const = extract_physics_features(coords_6d, sigma_noise)
        feats_norm = (feats - feat_mean) / feat_std

        h = self.backbone(feats_norm)
        raw = self.head(h).view(-1, self.n_components, 6)

        log_pi = F.log_softmax(raw[:, :, 0], dim=-1)            # (B, K)
        # 残差均值连接：真实预测均值 = 恒加速度基线落点 + 网络学习的空气动力与去噪修正残差
        mu_y = yz_const[:, 0:1] + raw[:, :, 1]                  # (B, K)
        mu_z = yz_const[:, 1:2] + raw[:, :, 2]                  # (B, K)
        mu = torch.stack([mu_y, mu_z], dim=-1)                  # (B, K, 2)

        # 标准差限制在 [1e-3, 5.0] m 保证数值绝对稳定
        sigma_y = F.softplus(raw[:, :, 3]) + 1e-3               # (B, K)
        sigma_z = F.softplus(raw[:, :, 4]) + 1e-3               # (B, K)
        sigma = torch.stack([sigma_y, sigma_z], dim=-1)         # (B, K, 2)

        # 相关系数 rho 限制在 (-0.95, 0.95) 防止协方差矩阵奇异
        rho = 0.95 * torch.tanh(raw[:, :, 5])                   # (B, K)

        return log_pi, mu, sigma, rho


def bivariate_gaussian_nll(
    target: torch.Tensor,
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """
    精确计算全协方差二维高斯混合分布的负对数似然损失 (Exact Bivariate GMM NLL)
    target: (B, 2)
    """
    y = target[:, 0:1]  # (B, 1)
    z = target[:, 1:2]  # (B, 1)

    mu_y, mu_z = mu[:, :, 0], mu[:, :, 1]
    sy, sz = sigma[:, :, 0], sigma[:, :, 1]

    zy = (y - mu_y) / sy
    zz = (z - mu_z) / sz

    one_minus_rho2 = 1.0 - (rho ** 2) + 1e-6
    quad = (zy ** 2 - 2.0 * rho * zy * zz + zz ** 2) / one_minus_rho2

    log_norm = np.log(2.0 * np.pi) + torch.log(sy) + torch.log(sz) + 0.5 * torch.log(one_minus_rho2)
    log_prob_k = log_pi - log_norm - 0.5 * quad  # (B, K)

    log_likelihood = torch.logsumexp(log_prob_k, dim=-1)  # (B,)
    return -torch.mean(log_likelihood)


# =====================================================================
# 5. 概率分布采样、众数估计与 50%/95% 最高密度区域 (HPD) 计算
# =====================================================================
@torch.no_grad()
def predict_distribution_and_hpd(
    model: PhysicsResidualMDN,
    coords_6d_np: np.ndarray,
    sigma_val: float,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    device: torch.device,
    grid_res: int = 120,
):
    """
    对单个观测样本计算二维网格上的精确概率密度、分布众数（Most Likely Point）、
    50%/95% 最高概率密度区域（HPD Region）面积及进球概率
    """
    model.eval()
    c_t = torch.tensor(coords_6d_np, dtype=torch.float32, device=device).reshape(1, 6)
    s_t = torch.tensor([[sigma_val]], dtype=torch.float32, device=device)

    log_pi, mu, sigma, rho = model(c_t, s_t, feat_mean, feat_std)
    pi_np = torch.exp(log_pi)[0].cpu().numpy()
    mu_np = mu[0].cpu().numpy()
    sig_np = sigma[0].cpu().numpy()
    rho_np = rho[0].cpu().numpy()

    # 构建自适应网格评估二维概率密度函数 PDF
    y_center = float(np.sum(pi_np * mu_np[:, 0]))
    z_center = float(np.sum(pi_np * mu_np[:, 1]))
    span_y = max(1.8, float(4.5 * np.max(sig_np[:, 0])))
    span_z = max(1.5, float(4.5 * np.max(sig_np[:, 1])))

    y_grid = np.linspace(y_center - span_y, y_center + span_y, grid_res)
    z_grid = np.linspace(max(0.0, z_center - span_z), z_center + span_z, grid_res)
    YY, ZZ = np.meshgrid(y_grid, z_grid)
    dy = float(y_grid[1] - y_grid[0])
    dz = float(z_grid[1] - z_grid[0])
    cell_area = dy * dz

    pdf = np.zeros_like(YY, dtype=np.float64)
    for k in range(len(pi_np)):
        sy, sz, r = sig_np[k, 0], sig_np[k, 1], rho_np[k]
        zy = (YY - mu_np[k, 0]) / sy
        zz = (ZZ - mu_np[k, 1]) / sz
        omr2 = 1.0 - r ** 2 + 1e-6
        quad = (zy ** 2 - 2.0 * r * zy * zz + zz ** 2) / omr2
        norm = 1.0 / (2.0 * np.pi * sy * sz * np.sqrt(omr2))
        pdf += pi_np[k] * norm * np.exp(-0.5 * quad)

    # 归一化网格概率质量
    prob_mass = pdf * cell_area
    prob_mass /= (np.sum(prob_mass) + 1e-12)

    # 数值寻找众数 (Mode: 最可能穿越点)
    max_idx = np.unravel_index(np.argmax(pdf), pdf.shape)
    mode_y, mode_z = float(YY[max_idx]), float(ZZ[max_idx])

    # 按密度降序排列求 50% 和 95% HPD 阈值与面积
    flat_pdf = pdf.ravel()
    flat_mass = prob_mass.ravel()
    sort_idx = np.argsort(flat_pdf)[::-1]
    cum_mass = np.cumsum(flat_mass[sort_idx])

    idx_50 = np.searchsorted(cum_mass, 0.50)
    idx_95 = np.searchsorted(cum_mass, 0.95)
    level_50 = float(flat_pdf[sort_idx[min(idx_50, len(flat_pdf) - 1)]])
    level_95 = float(flat_pdf[sort_idx[min(idx_95, len(flat_pdf) - 1)]])

    area_50 = float(idx_50 * cell_area)
    area_95 = float(idx_95 * cell_area)

    # 计算射入 FIFA 标准球门框内 (|Y| <= 3.66 - R_BALL, R_BALL <= Z <= 2.44 - R_BALL) 的进球概率
    in_goal_mask = (np.abs(YY) <= (GOAL_HALF_WIDTH - R_BALL)) & (ZZ >= R_BALL) & (ZZ <= (GOAL_HEIGHT - R_BALL))
    goal_prob = float(np.sum(prob_mass[in_goal_mask]))

    return {
        "mode_yz": [round(mode_y, 4), round(mode_z, 4)],
        "mean_yz": [round(y_center, 4), round(z_center, 4)],
        "area_50_m2": round(area_50, 4),
        "area_95_m2": round(area_95, 4),
        "goal_probability": round(goal_prob, 4),
        "grid": (YY, ZZ, pdf, level_50, level_95),
    }


# =====================================================================
# 6. 主训练、独立测试集评测与题目案例出图流水线
# =====================================================================
def main():
    out_dir = Path("runs_pi_res_mdn")
    out_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[1/5] 使用计算设备: {device}")

    # Step 1: 生成 50,000 条物理仿真数据并按 80/10/10 划分
    t0 = time.time()
    x_clean, y_target, sim_meta = generate_physics_dataset(n_samples=50000, seed=2026)
    print(f"[2/5] 向量化物理轨迹生成完成 ({time.time() - t0:.2f}s): {sim_meta}")

    n_total = len(x_clean)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    x_train, y_train = torch.tensor(x_clean[:n_train]), torch.tensor(y_target[:n_train])
    x_val, y_val = torch.tensor(x_clean[n_train:n_train + n_val]), torch.tensor(y_target[n_train:n_train + n_val])
    x_test, y_test = torch.tensor(x_clean[n_train + n_val:]), torch.tensor(y_target[n_train + n_val:])

    # 计算训练集物理特征均值与标准差（加入典型 5mm 噪声预估统计量）
    with torch.no_grad():
        sample_sigma = torch.full((n_train, 1), 0.005)
        sample_noisy_x = x_train + torch.randn_like(x_train) * sample_sigma
        train_feats, _ = extract_physics_features(sample_noisy_x, sample_sigma)
        feat_mean = train_feats.mean(dim=0, keepdim=True).to(device)
        feat_std = train_feats.std(dim=0, keepdim=True).clamp(min=1e-5).to(device)

    # Step 2: 训练 PI-ResMDN（动态噪声条件注入）
    model = PhysicsResidualMDN(in_features=16, hidden_dim=128, n_components=5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs = 60
    batch_size = 512
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    best_val_loss = float("inf")

    print(f"[3/5] 开始训练 Physics-Informed Residual MDN (共 {epochs} 轮)...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            # 动态采样每个样本的测量误差标准差 sigma ~ Uniform(0.001m, 0.020m)
            b_sigma = torch.empty((bx.size(0), 1), device=device).uniform_(0.001, 0.020)
            bx_noisy = bx + torch.randn_like(bx) * b_sigma

            optimizer.zero_grad()
            log_pi, mu, sigma, rho = model(bx_noisy, b_sigma, feat_mean, feat_std)
            loss = bivariate_gaussian_nll(by, log_pi, mu, sigma, rho)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            train_loss_sum += loss.item() * bx.size(0)

        scheduler.step()

        # 验证集评估 (固定在标准鹰眼噪声 sigma = 0.005m)
        model.eval()
        with torch.no_grad():
            vx, vy = x_val.to(device), y_val.to(device)
            v_sigma = torch.full((vx.size(0), 1), 0.005, device=device)
            torch.manual_seed(100)
            vx_noisy = vx + torch.randn_like(vx) * v_sigma
            v_log_pi, v_mu, v_sig, v_rho = model(vx_noisy, v_sigma, feat_mean, feat_std)
            val_loss = bivariate_gaussian_nll(vy, v_log_pi, v_mu, v_sig, v_rho).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feat_mean": feat_mean.cpu(),
                "feat_std": feat_std.cpu(),
            }, out_dir / "best_pi_res_mdn.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:02d}/{epochs} | Train NLL: {train_loss_sum / n_train:.4f} | Val NLL (5mm): {val_loss:.4f}")

    # 加载最优权重
    ckpt = torch.load(out_dir / "best_pi_res_mdn.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Step 3: 独立测试集对比评测 (恒加速度基线 vs PI-ResMDN 在不同噪声下)
    print("[4/5] 正在独立测试集上评测点预测精度与 50%/95% 概率覆盖率...")
    eval_results = {}
    for sigma_eval in [0.002, 0.005, 0.010]:
        with torch.no_grad():
            tx, ty = x_test.to(device), y_test.to(device)
            torch.manual_seed(2026)
            t_sigma = torch.full((tx.size(0), 1), sigma_eval, device=device)
            tx_noisy = tx + torch.randn_like(tx) * t_sigma

            # 恒加速度基线误差
            _, yz_const_test = extract_physics_features(tx_noisy, t_sigma)
            err_const = torch.norm(yz_const_test - ty, dim=-1).cpu().numpy()

            # PI-ResMDN 预测与马氏距离覆盖率验证
            t_start = time.perf_counter()
            log_pi, mu, sig, rho = model(tx_noisy, t_sigma, feat_mean, feat_std)
            infer_ms = (time.perf_counter() - t_start) * 1000.0 / tx.size(0)

            pi = torch.exp(log_pi).unsqueeze(-1)
            pred_mean = torch.sum(pi * mu, dim=1)
            err_mdn = torch.norm(pred_mean - ty, dim=-1).cpu().numpy()

            # 用混合分布对数似然蒙特卡洛分位数检验精确 50% / 95% 经验覆盖率
            # 从每个测试样本的预测 GMM 中采样 M=200 个点计算真实目标所在的密度分位
            B, K, _ = mu.shape
            M = 200
            comp_idx = torch.multinomial(torch.exp(log_pi), num_samples=M, replacement=True)  # (B, M)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M)

            sel_mu = mu[batch_idx, comp_idx]      # (B, M, 2)
            sel_sig = sig[batch_idx, comp_idx]    # (B, M, 2)
            sel_rho = rho[batch_idx, comp_idx]    # (B, M)

            eps1 = torch.randn((B, M), device=device)
            eps2 = torch.randn((B, M), device=device)
            samp_y = sel_mu[:, :, 0] + sel_sig[:, :, 0] * eps1
            samp_z = sel_mu[:, :, 1] + sel_sig[:, :, 1] * (
                sel_rho * eps1 + torch.sqrt(1.0 - sel_rho ** 2 + 1e-6) * eps2
            )

            # 评估真实点与采样点的对数密度
            def eval_gmm_logprob(pts_y, pts_z):
                # pts_y, pts_z: (B, N_pts)
                zy = (pts_y.unsqueeze(-1) - mu[:, :, 0].unsqueeze(1)) / sig[:, :, 0].unsqueeze(1)
                zz = (pts_z.unsqueeze(-1) - mu[:, :, 1].unsqueeze(1)) / sig[:, :, 1].unsqueeze(1)
                r = rho.unsqueeze(1)
                omr2 = 1.0 - r ** 2 + 1e-6
                quad = (zy ** 2 - 2.0 * r * zy * zz + zz ** 2) / omr2
                log_norm = np.log(2.0 * np.pi) + torch.log(sig[:, :, 0].unsqueeze(1)) + torch.log(sig[:, :, 1].unsqueeze(1)) + 0.5 * torch.log(omr2)
                return torch.logsumexp(log_pi.unsqueeze(1) - log_norm - 0.5 * quad, dim=-1)

            logp_true = eval_gmm_logprob(ty[:, 0:1], ty[:, 1:2]).squeeze(-1)  # (B,)
            logp_samp = eval_gmm_logprob(samp_y, samp_z)                      # (B, M)
            # 若真实点的密度高于 50% (或 5%) 的采样点，则落在 50% (或 95%) HPD 区域内
            frac_lower = torch.mean((logp_samp <= logp_true.unsqueeze(1)).float(), dim=-1).cpu().numpy()
            cov_50 = float(np.mean(frac_lower >= 0.50))
            cov_95 = float(np.mean(frac_lower >= 0.05))

        eval_results[f"sigma_{int(sigma_eval*1000)}mm"] = {
            "const_accel_median_err_m": round(float(np.median(err_const)), 4),
            "const_accel_p90_err_m": round(float(np.percentile(err_const, 90)), 4),
            "pi_res_mdn_median_err_m": round(float(np.median(err_mdn)), 4),
            "pi_res_mdn_p90_err_m": round(float(np.percentile(err_mdn, 90)), 4),
            "empirical_coverage_50": round(cov_50, 4),
            "empirical_coverage_95": round(cov_95, 4),
            "single_inference_time_ms": round(infer_ms, 4),
        }
        print(f"  [噪声 σ={sigma_eval*1000:.0f}mm] 恒加速中位误差: {np.median(err_const):.3f}m -> PI-ResMDN中位误差: {np.median(err_mdn):.3f}m | 50%/95%覆盖率: {cov_50*100:.1f}% / {cov_95*100:.1f}%")

    # Step 4: 题目案例 (Case Study) 预测与球门平面热图绘制
    print("[5/5] 正在生成题目案例结果卡与高清球门概率热图...")
    case_coords = np.array([1.50, 0.45, 0.44, 2.96, 0.86, 0.74], dtype=np.float32)
    case_summary = {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    noise_scenarios = [(0.002, "High Precision (σ = 2 mm)"), (0.005, "FIFA EPTS Standard (σ = 5 mm)"), (0.015, "Field Broadcast (σ = 15 mm)")]

    for ax_plot, (sig_val, title_str) in zip(axes, noise_scenarios):
        res = predict_distribution_and_hpd(model, case_coords, sig_val, feat_mean, feat_std, device, grid_res=160)
        YY, ZZ, pdf, lvl50, lvl95 = res.pop("grid")
        case_summary[f"sigma_{int(sig_val*1000)}mm"] = res

        # 绘制概率密度热图与 50% / 95% HPD 等高线
        ax_plot.contourf(YY, ZZ, pdf, levels=40, cmap="YlOrRd", alpha=0.85)
        ax_plot.contour(YY, ZZ, pdf, levels=[lvl95, lvl50], colors=["#1f77b4", "#08306b"], linewidths=[1.8, 2.2])

        # 绘制 FIFA 标准球门框 (Y in [-3.66, 3.66], Z in [0, 2.44])
        goal_rect = patches.Rectangle((-GOAL_HALF_WIDTH, 0.0), 2 * GOAL_HALF_WIDTH, GOAL_HEIGHT,
                                      linewidth=2.5, edgecolor="black", facecolor="none", linestyle="-", label="FIFA Goal Frame")
        ax_plot.add_patch(goal_rect)

        # 标注恒加速度基线点 vs PI-ResMDN 最可能点 (Mode)
        ax_plot.scatter([2.495], [1.921], c="cyan", edgecolors="black", marker="^", s=90, zorder=5, label="Const-Accel Baseline (2.50, 1.92)m")
        ax_plot.scatter([res["mode_yz"][0]], [res["mode_yz"][1]], c="lime", edgecolors="black", marker="*", s=140, zorder=6,
                        label=f"PI-ResMDN Mode ({res['mode_yz'][0]:.2f}, {res['mode_yz'][1]:.2f})m")

        ax_plot.set_title(f"{title_str}\n50%/95% Area: {res['area_50_m2']:.2f} / {res['area_95_m2']:.2f} m² | Goal Prob: {res['goal_probability']*100:.1f}%", fontsize=10.5)
        ax_plot.set_xlabel("Goal Line Horizontal Coordinate Y (m)")
        ax_plot.set_xlim(0.5, 4.2)
        ax_plot.set_ylim(0.0, 3.0)
        ax_plot.grid(True, linestyle="--", alpha=0.35)
        ax_plot.legend(loc="lower left", fontsize=8.5)

    axes[0].set_ylabel("Goal Line Vertical Coordinate Z (m)")
    plt.suptitle("UPC Problem B Case Study: Goal-Plane Crossing Probability & 50%/95% HPD Regions (x = 11 m)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "case_study_goal_heatmaps.png", dpi=300)
    plt.close()

    with open(out_dir / "pi_res_mdn_summary.json", "w", encoding="utf-8") as f:
        json.dump({"simulation_meta": sim_meta, "test_evaluation": eval_results, "case_study": case_summary}, f, indent=2, ensure_ascii=False)

    print(f"全部完成！结果 JSON 与 300 DPI 热图已保存至: {out_dir.resolve()}")


if __name__ == "__main__":
    main()