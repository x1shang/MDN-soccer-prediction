# MDN-soccer-prediction

Materials for the UPC 2026 Problem B paper
**“Two Points to the Goal: A Noise-Conditioned Mixture Density Network for Probabilistic
Penalty Kick Prediction”** (Team 155).

Given the penalty spot at the origin, a fixed ball start and two tracked positions at
0.05 s and 0.10 s, the paper predicts where the ball crosses the goal plane at `x = 11 m`
and quantifies the uncertainty of that prediction. Three ingredients are compared under one
set of flight dynamics: a quadratic (constant-acceleration) extrapolation, a Bayesian
reference posterior over launch conditions, and a noise-conditioned Wiener-residual mixture
density network trained on 1,225,000 simulated launches.

Every number in the paper is reproduced by code or stored artefact in this repository.
The five large `.npz` launch banks live in a GitHub Release — see [data/README.md](data/README.md).

---

## 中文速览

- **论文**：`paper/main.tex`（pdfLaTeX 编译两次；图在 `paper/figures/`，靠 `\graphicspath` 找到）。
- **主结果**：`experiments/main_v3_all/`（v3 全锚点百万数据，3 个训练种子）、
  `experiments/asai_holdout/`（Asai 来源留出）、`experiments/ground_contact/`（反弹/滚动配对对照）。
- **数据**：`data/anchors/` 是实测锚点（已入库）；百万级 `.npz` 在 Release `data-v1.0`，
  用 `code\scripts\fetch_data.ps1` 一键下载到正确位置并校验 SHA-256。
- **代码**：`code/simulator`（仿真）、`code/model`（MDN 与共享物理）、
  `code/analysis`（模型筛选与诊断）、`code/validation`（实测轨迹核验）。
- **旧版**：早先基于 5 万条 canonical 数据的 P1 / v1-fresh 结果归档在 `experiments/legacy_pre_v3/`，
  仅作历史留存，论文不再引用其中的数字。

---

## Repository layout

```
paper/                  LaTeX source, compiled PDF, figures, archived earlier draft
  main.tex              the current paper
  main.pdf              compiled output
  figures/*_v3.png      the five figures referenced by main.tex
  archive/              earlier P1-era draft, kept for provenance
code/
  simulator/            frozen v3 generator (RK4 physics, prior sampling, aiming)
  model/                shared physics, MDN trainer, inference, paired-Bayes helper
  analysis/             screen.py (model comparisons), diagnostics.py (case figures)
  validation/           measured-trajectory checks: DFL prefixes, ISSIA arcs, C_D prior
  scripts/              run_experiment.ps1, fetch_data.ps1
data/
  anchors/              Asai workbooks, DFL/ISSIA/Barber trajectories (committed)
  datasets/             per-dataset manifest + split indices (data.npz from the Release)
  manifests/            frozen design of the v3 simulation
experiments/
  main_v3_all/          seed_{11,42,155} fits on the full v3 mixture
  asai_holdout/         v2 / v3 fits with Asai draws replaced
  diagnostics/          screen.json, diagnostics.json, calibration_recheck.json, paired_bayes.json
  validation/           C_D-prior, ISSIA, DFL and spin-check results
  ground_contact/       self-contained rebound/rolling experiment (code + results)
  legacy_pre_v3/        pre-v3 (50k-era) P1 and v1-fresh artefacts, archived only
docs/
  todo.md               the team's working checklist (frozen 2026-11-08)
  summary_million_sim.md  auto-generated summary of the three arms
  logs/                 simulator and numerical-check logs
  environment-freeze.txt  pinned package versions
```

`requirements.txt` at the root lists the Python dependencies.

## Quick start

```powershell
git clone https://github.com/x1shang/MDN-soccer-prediction
cd MDN-soccer-prediction
powershell -ExecutionPolicy Bypass -File code\scripts\fetch_data.ps1   # ~465 MB, verifies SHA-256
cd code\model; python -m unittest discover -s tests -v; cd ..\..        # 14 unit tests
```

Train the main model (needs a CUDA GPU for the timings reported in the paper; CPU works, slower):

```powershell
powershell -ExecutionPolicy Bypass -File code\scripts\run_experiment.ps1 -Python python
python code\model\summarize.py            # rewrites docs/summary_million_sim.md
python code\analysis\screen.py            # model comparison -> experiments/diagnostics/screen.json
python code\analysis\diagnostics.py       # case figures + calibration -> paper/figures, experiments/diagnostics
```

Compile the paper:

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` also compiles under XeLaTeX; that path is only needed if the Chinese AI-record
section is enabled (`% \input{ai-report.tex}` is commented out until the verbatim records are
pasted — see `paper/ai-report.tex`).

## Which artefact backs which result

| Paper element | Artefact |
| --- | --- |
| Table “Airborne crossings by split” | `data/datasets/*/data_manifest.json` → `index_counts`, `quality` |
| Case tables and Figs. 1–4 | `experiments/main_v3_all/seed_*/case_predictions.json`, `experiments/diagnostics/diagnostics.json`, `paper/figures/*_v3.png` |
| Table “Full-mixture v3 test results” | `experiments/main_v3_all/seed_*/test_scores.csv` |
| Table “Calibration stages”, Fig. 5 | `experiments/diagnostics/calibration_recheck.json` |
| Table “Model comparison”, Fig. 6 | `experiments/diagnostics/screen.json` |
| Table “Asai source holdout” | `experiments/asai_holdout/*/seed_*/asai_replay.csv` |
| Table “Paired rebound experiment” | `experiments/ground_contact/results/` (`report_tables.md`, `paired_bootstrap.json`) |
| Table “Sensitivity to the drag prior” | `experiments/validation/case_cd_prior/cd_prior_sensitivity.json` |
| Table “Model-form check on tracked arcs” | `experiments/validation/issia_A/cd_lift_fit.json` |
| Table “Two DFL penalties” | `experiments/validation/real_penalties/scores.json` |
| ISSIA 2972–2991 and Barber Fig. 5 spin check | `experiments/validation/real_spin_final/scores.json` |
| Appendix A.1–A.2 | analytic; no artefact needed |

Appendix A.3 (local crossing spread) and the distribution-shift paragraph were computed from the
v3 test split by session-local scripts that are not archived here; the seeds and the estimator are
stated in the paper (`default_rng(155107)`, local Jacobian covariance with a Gaussian reference prior).

## Environments

| Track | Interpreter | Notes |
| --- | --- | --- |
| million-scale training, screen, diagnostics | Python 3.12, `numpy` 2.5.4, `scipy` 1.18.1, `torch` (CUDA 12.x) | `requirements.txt`; frozen versions in `docs/environment-freeze.txt` |
| ground-contact experiment | same stack, separate venv | `experiments/ground_contact/run_all.ps1` |

Local interpreter environments (`env/runtime`, `experiments/ground_contact/.venv`) are git-ignored.

## Conventions worth knowing

- **One simulator.** `code/model/physics.py` + `params.json` is the canonical simulator; the
  frozen v3 generator in `code/simulator/` shares its equations. Numbers from the two are never
  mixed inside one table.
- **One split.** `indices.npz` fixes train / selection / calibration / test per dataset. Test is
  scored once; calibration is fitted on its own fold; NLL on the selection fold picks the epoch.
- **Seeds.** Training seeds 11, 42, 155; evaluation perturbations `default_rng(3000 + sigma_mm)`;
  paired Bayesian subset `default_rng(155108)`.
- **Provenance.** Every dataset manifest carries SHA-256 hashes for the data, the split, the
  simulator sources and every anchor file; every `config.json` repeats the data and split hashes
  plus the training-script hash.
