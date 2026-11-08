# todo.md — MDN 负责人实验清单

> **归档说明**：本文件写于 2026-11-07，记录当时的实验计划与判定标准。其中出现的路径
> （`student_b/`、`student-c/`、`论文模板/` 等）属于整理前的目录结构，已不适用于当前仓库；
> 现在的代码、数据与结果位置见仓库根目录的 `README.md`。文中数字保留原样，不作为论文引用来源。

**角色**：神经网络 / 训练 / 概率评测  
**当前时间**：2026-11-07 19:41（北京时间）  
**NN 实验冻结**：2026-11-08 18:00；之后只出最终表/图，不再改模型  
**提交截止**：2026-11-09 09:00  
**原则**：一个 simulator、一个 split、一个 test set；所有数字可追溯；先 P0 后 P1；**不做 NAS / 进化搜索**。

---

## 0. 现状锚点（先记住，避免拿错数）

| 项 | 现状 |
|---|---|
| 现有 MDN | `7 → 64 → 64 → 18`，K=3，SiLU，**5,842 参数**；输入 6 坐标 + σ |
| 现有 MDN 效果 | 2/5/15 mm 中位误差 **0.163 / 0.322 / 0.556 m**；50% 覆盖 **58.6 / 52.3 / 46.5%**；95% 覆盖 **97.6 / 96.3 / 96.8%** |
| 现有案例 | 众数 **(2.759, 1.899) m**；均值 (2.773, 1.924)；P(goal \| airborne) **0.9743** |
| `pi_res_mdn.py` | 16 维物理特征 → **128×3** → 30，K=5，SiLU+LayerNorm，残差学习，连续噪声 1–20 mm，cosine，60 epoch，约 **40k 参数** |
| ⚠️ 大坑 | `pi_res_mdn.py` 的 simulator 与 `student_b/physics.py` **不一致**：ρ=1.225 vs 1.20；Cd 先验 [0.15,0.26]/[0.26,0.38] vs [0.15,0.30]；Cl=0.38S/(0.18+S) vs min(1.4S,0.30)；有自旋衰减 vs 无；初值范围也不同。**不要直接把两套数字放在同一张表里。** |

**唯一推荐**：以 `student_b/physics.py` + `params.json` 为 canonical simulator（它有 14 tests、现有 50k 结果、物理 Bayesian 同源）；把 `pi_res_mdn.py` 的 feature/residual 思路移植过来。若坚持用 `pi_res_mdn.py` 的新 simulator，必须重跑 baseline MDN 和 physics Bayesian，否则论文数字不可比。

---

## 1. P0 — 先做，不做完不要跑大网格（目标 2 h）

- [ ] **1.1 跑通 `pi_res_mdn.py`**：记录 runtime、数据量、有效空中穿越数、NaN/异常、2/5/10 mm 的 median/coverage、案例图。
- [ ] **1.2 Simulator diff**：同一组 1,000 个 θ，分别用 `student_b/physics.py` 和 `pi_res_mdn.py` 的生成器，比较 `r(0.05)`、`r(0.10)`、endpoint 的 max/RMS 差；列出 law/参数差异。
- [ ] **1.3 决定 canonical simulator**（建议 `student_b/physics.py`），写进 `experiments/CONFIG.md`；之后所有实验只用它。
- [ ] **1.4 验证批评者的“<0.05 m”**：在同一 test set 上跑 PI-ResMDN 与现有 MDN，记录实际 median/p90。**不要预设 <0.05 m 成立**；如果只在案例成立，写“case-specific”，不要推广到 population。
- [ ] **1.5 残差分布**：统计 ΔY=Y_true−Y_const、ΔZ=Z_true−Z_const 的 mean/std/分位数/离群值；确认残差是否真的“小到可学”。
- [ ] **1.6 特征条件数**：16 维特征的相关系数矩阵、条件数；`cd_apparent`、`a_perp` 在 2/5/15 mm 噪声下的稳定性；标准化是否充分。
- [ ] **1.7 覆盖率估计器统一**：`pi_res_mdn.py` 用 200 样本 + 分位启发式；现有 `experiment.py` 用 512/1024 HDR。**统一成一个估计器**，否则覆盖率不可比。
- [ ] **1.8 混合权重诊断**：K=3/5 的 π_k 分布；是否有一个分量占 90%+（退化成 K=1）？是否有分量塌缩？

**Gate**：1.2 的 simulator 差异若大于 ~5 cm（r(0.05)/r(0.10)），先修 simulator/参数，不要进入第 2 阶段。

### 1.9 台式机独立复现（队友交接，可与 P1 并行）

- [x] **核验随包数据和代码**：`verify.py` 与 14 项单元测试均通过。SHA-256、数据划分、事件计数及 256 条轨迹的观测点、过门点和到达时间重算一致。

  ```powershell
  python -m pip install -r requirements.txt
  python verify.py
  python -m unittest discover -s student_b -v
  ```

- [x] **重建全量数据并审计覆盖**：用 seed 42 在 `data/rebuilt50k/` 另建 50,000 条。六个数据字段与整份 `data.npz` 的 SHA-256 均一致，覆盖报告只有输入路径不同。

  ```powershell
  python student_b/experiment.py generate --out data/rebuilt50k --n 50000 --seed 42
  python student_b/audit_coverage.py --data data/rebuilt50k/data.npz --out reports/rebuilt_coverage
  ```

- [x] **回传给队友**：实际为空中过门 41,733、提前触地 8,267、超时 0。空中过门样本的训练 / 验证 / 测试分别为 33,367 / 4,179 / 4,187。已把检查日志、覆盖审计和现有低速、零旋样本缺口上传团队仓库。

如安排后续网络比较，先固定随包数据和轨迹划分。验证集用于选模型和校准，4,187 条冻结集用于评分；实测轨迹按包内协议单列为外推对照。

---

## 2. P0 — Feature / target 消融（核心科学问题，目标 2–3 h）

固定架构：2×128，K=3，SiLU，连续噪声 1–20 mm，cosine，100 epoch；每个 arm 3 seeds。

| Arm | 输入 | 预测目标 | 回答的问题 |
|---|---|---|---|
| A | raw6 + σ | 绝对 (Y,Z) | 现有 MDN 基线 |
| B | raw6 + σ | 残差 (ΔY,ΔZ) | 残差本身值多少 |
| C | physics16 | 绝对 (Y,Z) | 物理特征本身值多少 |
| D | physics16 | 残差 (ΔY,ΔZ) | PI-ResMDN（批评者方案） |
| E | raw6 + σ + (Y_const,Z_const) | 残差 | 最小物理 shortcut 是否就够 |

**指标**：val/test NLL、median、p90、50%/95% 覆盖、HDR 面积、P(goal) ECE、训练/推理时间。  
**判定**：以 NLL 为主，覆盖率不能变差；报告所有 arm，不只报赢家。  
**关键**：如果 B 已经接近 D，说明“残差”是主因；如果 C/D 明显好于 A/B，说明“物理特征”是主因；如果都不明显，批评者的两个 claim 至少有一个不成立。

---

## 3. P1 — 网络结构筛选（1 seed，60 epoch，best arm）

**本轮实际执行**：采用三个种子、100 epoch。已比较 16/11 维、K=1/3、1–4 层、SiLU/GELU/ReLU/Tanh，并单项验证 `ρ=0`、固定学习率和固定训练噪声。最终配置为 11 维、3×128、GELU、LayerNorm、K=3，全协方差直接预测。下列旧排程留作范围记录。 [P1 实验报告](UPC155/student-c/P1/results.md)

按顺序做，不要一次铺满：

- [ ] **深度**：1 / 2 / 3 / 4 个 hidden layer，width=128，K=3，SiLU → 4 runs
- [ ] **宽度**：64 / 128 / 256，best depth → 3 runs
- [ ] **分量数 K**：1 / 3 / 5，best depth/width → 3 runs
- [ ] **激活**：SiLU / GELU / ReLU / Tanh，best depth/width/K → 4 runs
- [ ] **LayerNorm**：有 / 无，best config → 2 runs
- [ ] **最终**：top 2 configs × 3 seeds × 150 epoch → 6 runs

总计约 22 runs；每个 30–90 s，CPU 可跑。  
**注意**：K=1 就是高斯 MLP；如果 K=1 与 K=3 的 NLL/覆盖差在种子波动内，**选 K=1**，论文更简单、校准更稳。

---

## 4. P1 — 训练策略消融（1 seed，best 结构）

- [ ] 固定 lr vs cosine annealing
- [ ] 每 epoch 加噪 vs 每 batch 加噪
- [ ] 离散 σ={2,5,15} mm vs 连续 U(1,20) mm vs log-σ
- [ ] batch 256 vs 512
- [ ] early stop patience 12 vs 20
- [ ] weight decay 0 vs 1e-4
- [ ] grad clip 2 vs 10

每项 2 runs；只保留对 NLL/覆盖有稳定影响的 2–3 项，不要全上。  
批评者说的“cosine + dynamic noise”已经在 `pi_res_mdn.py` 里，重点验证它是否真的改善 **50% 覆盖随 σ 的漂移**。

---

## 5. P0/P1 — 校准（目标 2 h，必须做）

**本轮完成**：拆分验证集，按噪声档拟合尺度与 HDR 阈值。新测试集 5,036 条，校准后四档 50% 覆盖为 47.6–49.1%，95% 覆盖为 94.5–95.2%。 [校准与面积](UPC155/student-c/P1/results.md)

- [ ] 按 σ 报 50%/95% 覆盖；画 nominal vs empirical 可靠性图。
- [ ] 后校准：温度缩放（logits）/ 协方差按 σ 缩放 / conformal / P(goal) isotonic。
- [ ] 报告 pre/post 两套数；校准只能在 validation 上拟合，**不能碰 test**。
- [ ] 目标：50% 覆盖 45–55%；95% 覆盖 93–97%；面积不显著膨胀。
- [ ] 如果 50% 仍偏，论文写“95% 区域是主结果；50% 区域随噪声变化”，不要硬调 test。

---

## 6. P1 — 稳健性 / 分布偏移（目标 2 h）

- [x] 横风 3 m/s：训练无风，测试有风；报告配对覆盖变化。
- [x] 自旋衰减：训练无衰减，测试时间常数 0.5 s 的场景。
- [ ] C_D(v)：训练常数 Cd，测试速度依赖 Cd（仅模型形式对照）。
- [ ] 竖直净升力：ISSIA 提示的模型扩展，仅作 robustness，不进主模型。
- [ ] ISSIA 两段：feature extractor 一致性 + 外推；**只作 model-form check，不训练、不校准**。
- [ ] 事件模型：P(cross)/P(ground)；报告 `P(goal)=P(cross)×P(goal|cross)`。

---

## 7. P0 — 最终交付（2026-11-08 18:00 前）

- [ ] `results.md`：所有表（baseline、5-arm、结构、训练、校准、稳健性）。
- [ ] `best_config.json`：结构、K、激活、feature set、target、lr/noise、seed。
- [ ] `figures/`：learning curve、coverage vs σ、ablation bar、case heatmap、mixture weights。
- [ ] `run_all.ps1`：一条命令复现 final table。
- [ ] 环境/seed/git commit 记录。
- [ ] 给 C 的 LaTeX 三线表 + 300 dpi 图；数字与 JSON 一一对应。

---

## 8. 批评者的 claim — 逐条验证，不要照单全收

| Claim | 怎么验 | 预期/注意 |
|---|---|---|
| 残差学习“收敛快 10 倍、中位误差 <0.05 m” | Arm B/D vs A；同 test set 记录 median/p90 | **很可能只在 case 成立**；population 5 mm 下 <0.05 m 不太现实；未复现就不要写进论文 |
| “6 坐标数值病态，必须 16 维物理特征” | Arm A vs C；条件数/相关矩阵 | 标准化/LayerNorm 已缓解；残差可能是更大的 lever；分开测 |
| “输出层必须带 ρ” | 检查现有模型 | **现有 MDN 和 `pi_res_mdn.py` 都已有 ρ**；无需改 |
| “cosine + 动态噪声” | 第 4 节消融 | `pi_res_mdn.py` 已实现；重点看 50% 覆盖是否更平 |
| “σ 作为第 16 个特征” | 检查 | 现有 MDN 已是 7 维（6+σ）；`pi_res_mdn.py` 是 16 维含 σ；不是新问题 |
| “15 维特征” | 检查 | `pi_res_mdn.py` 实际 16 维；维数不是圣旨，避免共线/冗余；用 Arm E 测最小 shortcut |

---

## 9. 时间表（硬门）

| 时间 | 任务 | 产出 |
|---|---|---|
| 11-07 19:41–21:30 | P0：跑通、simulator diff、<0.05 m 验证、残差/特征诊断 | `diagnostics.md` |
| 11-07 21:30–23:30 | P0：5-arm 消融（3 seeds） | `ablation_features.csv` |
| 11-07 23:30–02:00 | P1：结构筛选（深度/宽度/K/激活/LayerNorm） | `architecture_screen.csv` |
| 11-08 08:00–11:00 | P1：训练策略 + top2 × 3 seeds | `training_ablation.csv` |
| 11-08 11:00–13:00 | P0：校准 pre/post | `calibration.csv` + 图 |
| 11-08 13:00–15:00 | P1：稳健性/事件模型/ISSIA | `robustness.csv` |
| 11-08 15:00–18:00 | P0：最终表/图/`best_config.json`/handoff | `results.md` |
| 11-08 18:00 | **冻结** | 不再改模型 |

**最低可交路径**（如果时间不够）：P0 的 1.1–1.5 + 5-arm A/B/D + K=1/3/5 + 校准 pre/post + 最终表。不要为了“完整”拖到 11-09。

---

## 10. Do-not-do

- ❌ 不做 NAS / 进化搜索 / 大规模超参搜索：验证集只有 4,187，容易过拟合；收益大概率小于校准。
- ❌ 不混 simulator：`student_b/physics.py` 与 `pi_res_mdn.py` 的数字不能放同一张表。
- ❌ 不用 test set 调参/早停/校准。
- ❌ 不在 ISSIA 上训练/校准（不是点球、25 fps、厘米级重建）。
- ❌ 不把“<0.05 m”“10× 收敛”写进论文，除非同 test set 复现并写清 scope。
- ❌ 不为了更大模型牺牲论文时间；11-08 18:00 必须冻结。

---

## 11. 文件/命令约定

```
student_b/
  experiment.py              # 现有 MDN 训练/评测
  physics.py                 # canonical simulator（建议）
  runs/literature50k/        # 现有 baseline
  runs_pi_res_mdn/           # pi_res_mdn.py 输出
  experiments/
    CONFIG.md                # canonical simulator + split + metrics
    ablation_features.csv
    architecture_screen.csv
    training_ablation.csv
    calibration.csv
    robustness.csv
    best_config.json
    figures/
```

复现命令示例（按实际路径调整）：

```powershell
# baseline
python student_b/experiment.py train --out student_b/runs/mdn_base --epochs 150
python student_b/experiment.py evaluate --out student_b/runs/mdn_base

# PI-ResMDN（先确认 simulator 口径）
python pi_res_mdn.py

# 5-arm / 结构筛选建议写成 config-driven runner：
python student_b/experiments/run_grid.py --config experiments/grid_p0.yaml
```

---

## 12. 一句话

**先证明“残差/物理特征”到底值多少（5-arm），再决定结构（深度/宽度/K/激活）；当前 5,842 参数的小 MDN 不是容量瓶颈，50% 覆盖的漂移是噪声条件化/校准问题。** 最完整的测试不是把网格铺满，而是让每个 arm 回答一个明确问题，并在 11-08 18:00 前把可复现的表交给写论文的人。
