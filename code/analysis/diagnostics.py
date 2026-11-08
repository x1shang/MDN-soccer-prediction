"""Recompute the case and paired Bayes diagnostics with the frozen v3 MDN."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import brentq, minimize
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, multivariate_normal, norm

HERE = Path(__file__).resolve().parent          # code/analysis
ROOT = HERE.parents[1]                          # repository root
MODEL = ROOT / "code" / "model"
DATA = ROOT / "data" / "datasets" / "v3_all"
PAPER = ROOT / "paper" / "figures"
OUT = ROOT / "experiments" / "diagnostics"
sys.path[:0] = [str(MODEL)]

from common import baseline_endpoint
from inference import posterior
from paired_bayes import bayesian_forecast, metrics
from physics import CASE, Config, rk4, simulate
from train_wiener_mdn import WienerResidualMDN, evaluate_hdr_and_metrics, extract_wiener_features

SEEDS = (11, 42, 155)
NOISE_MM = (2, 5, 10, 15)
FRAME = (-3.55, 3.55, .11, 2.33)
FIXED_Y = (2.62, 2.67)
PAPER_BG = "white"
TEXT = "#191919"
CYAN = "#00d9e2"
GREEN = "#2ca02c"
COLORS = ("#1f77b4", "#ff7f0e", GREEN, "#9467bd", "#8c564b")


def load_model(seed):
    run = ROOT / "experiments" / "main_v3_all" / f"seed_{seed}"
    cfg = json.loads((run / "config.json").read_text())
    checkpoint = run / "model.pt"
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != cfg["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash mismatch for seed {seed}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = WienerResidualMDN().eval()
    model.load_state_dict(ckpt["state"])
    return model, ckpt["stats"], cfg["calibration"]


@torch.no_grad()
def forecast(bundle, obs, mm):
    model, stats, cal = bundle
    x, anchor = extract_wiener_features(obs.reshape(-1, 6), np.full(len(obs), mm / 1000))
    features = torch.from_numpy((x - stats["xm"]) / stats["xs"])
    return model(features, torch.from_numpy(anchor), cal[str(mm)]["scale_sd"])


class CaseDensity:
    def __init__(self, bundle, mm):
        dist = forecast(bundle, CASE[None], mm)
        self.weights = dist.mixture_distribution.probs[0].numpy().astype(float)
        self.weights /= self.weights.sum()
        comp = dist.component_distribution
        self.means = comp.mean[0].numpy()
        self.cov = comp.covariance_matrix[0].numpy()
        self.sd = np.sqrt(np.diagonal(self.cov, axis1=1, axis2=2))
        self.gaussians = [multivariate_normal(mean, cov) for mean, cov in zip(self.means, self.cov)]
        cfg = bundle[2][str(mm)]["hdr_mass"]
        self.masses = [cfg[key] for key in ("0.5", "0.95")]

    def component_pdf(self, xy):
        return np.stack([w * component.pdf(xy) for w, component in zip(self.weights, self.gaussians)])

    def density(self, xy):
        return self.component_pdf(xy).sum(axis=0)

    def marginal(self, values, axis, kind="pdf"):
        return np.stack([w * getattr(norm, kind)(values, loc=mu[axis], scale=sd[axis])
                         for w, mu, sd in zip(self.weights, self.means, self.sd)])

    def in_frame(self):
        lo, hi = [FRAME[0], FRAME[2]], [FRAME[1], FRAME[3]]
        return float(sum(w * g.cdf(hi, lower_limit=lo, rng=np.random.default_rng(0))
                         for w, g in zip(self.weights, self.gaussians)))

    def grid(self):
        lo, hi = (np.min(self.means - 8 * self.sd, axis=0), np.max(self.means + 8 * self.sd, axis=0))
        y, z = [np.linspace(a, b, 1201) for a, b in zip(lo, hi)]
        yy, zz = np.meshgrid(y, z)
        pdf = self.density(np.stack((yy, zz), axis=-1))
        cell = (y[1] - y[0]) * (z[1] - z[0])
        ordered = np.sort(pdf.ravel())[::-1]
        cum = cell * np.cumsum(ordered)
        ranks = np.searchsorted(cum, self.masses)
        is_peak = (pdf == maximum_filter(pdf, size=3)) & (pdf >= .002 * pdf.max())
        maxima = []
        for i, j in np.argwhere(is_peak):
            result = minimize(lambda q: -float(self.density(q)), (y[j], z[i]), method="BFGS")
            if all(np.linalg.norm(result.x - pt) > .05 for pt in maxima):
                maxima.append(result.x)
        maxima.sort(key=self.density, reverse=True)
        return y, z, pdf, ordered[ranks], (ranks + 1) * cell, maxima

    def profile(self, z, y=None):
        if y is None:
            return self.marginal(z, 1)
        return self.component_pdf(np.column_stack((np.full(len(z), y), z)))

    def describe(self, z, grid):
        _, _, _, thresholds, area, maxima = grid
        quantiles = [[brentq(lambda x: self.marginal(x, axis, "cdf").sum() - q, -20, 20)
                      for q in (.05, .5, .95)] for axis in (0, 1)]
        return dict(mode=maxima[0].tolist(), joint_modes=[pt.tolist() for pt in maxima],
                    mean=(self.weights @ self.means).tolist(), area=area.tolist(),
                    hdr_thresholds=thresholds.tolist(), in_frame=self.in_frame(),
                    above_bar=float(1 - self.marginal(FRAME[3], 1, "cdf").sum()),
                    lateral_out=float(self.marginal(FRAME[0], 0, "cdf").sum()
                                      + 1 - self.marginal(FRAME[1], 0, "cdf").sum()),
                    quantiles=quantiles, marginal_peaks=peak_positions(z, self.profile(z).sum(0)),
                    slice_peaks={str(y): peak_positions(z, self.profile(z, y).sum(0)) for y in FIXED_Y},
                    weights=self.weights.tolist())


def peak_positions(axis, pdf):
    ids, _ = find_peaks(pdf, distance=.05 / (axis[1] - axis[0]), prominence=.002 * pdf.max())
    return axis[ids].tolist()


def weighted_quantiles(values, weights):
    order = np.argsort(values)
    return np.interp([.05, .5, .95], np.cumsum(weights[order]), values[order]).tolist()


def magnus(spin, velocity, cfg):
    cross = np.cross(spin, velocity)
    cross_norm = np.linalg.norm(cross, axis=-1)
    speed2 = np.sum(velocity**2, axis=-1)
    lift = np.minimum(1.4 * cfg.radius * cross_norm / speed2, .30)
    k = cfg.density * np.pi * cfg.radius**2 / (2 * cfg.mass)
    return (k * lift * speed2 / np.maximum(cross_norm, 1e-24))[..., None] * cross


def trajectories(theta, density):
    cfg = Config()
    out = []
    for label, index in zip("ABC", (6953, 2827, 18017)):
        shot = theta[[index]]
        sim = simulate(shot, cfg)
        endpoint = sim["endpoint"][0]
        state = np.array([[0., 0., cfg.radius, *shot[0, :3]]])
        path = [state[0, :3].copy()]
        for step in range(round(cfg.horizon / cfg.dt)):
            t = step * cfg.dt
            state = rk4(state, shot, t, cfg.dt, cfg)
            if (step + 1) * cfg.dt == .05:
                lift = magnus(shot[0, 3:6], state[0, 3:], cfg)
            if state[0, 0] >= cfg.goal:
                path.append(np.r_[cfg.goal, endpoint])
                break
            path.append(state[0, :3].copy())
        components = density.component_pdf(endpoint)
        out.append(dict(label=label, index=index, cd=float(shot[0, 6]), yz=endpoint.tolist(),
                        rmse_mm=float(np.sqrt(np.mean((sim["observations"][0] - CASE)**2)) * 1000),
                        lift=lift.tolist(), responsibility=(components / components.sum()).tolist(),
                        path=np.asarray(path)))
    return out


def save_figure(fig, name):
    fig.savefig(PAPER / name, dpi=280, facecolor=PAPER_BG)
    plt.close(fig)


def paper_style():
    plt.rcParams.update({"figure.facecolor": PAPER_BG, "axes.facecolor": PAPER_BG, "savefig.facecolor": PAPER_BG,
                         "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": TEXT,
                         "xtick.color": TEXT, "ytick.color": TEXT, "font.size": 10,
                         "axes.titlesize": 10.5, "legend.fontsize": 8.5})


def frame(ax):
    ax.axhline(FRAME[3], color=GREEN, ls="--", lw=1.0, alpha=.8)
    ax.axvline(FRAME[1], color=GREEN, ls="--", lw=1.0, alpha=.8)


def density_panel(ax, grid, *, heatmap=True):
    y, z, pdf, thresholds, _, peaks = grid
    if heatmap:
        ax.pcolormesh(y[::3], z[::3], pdf[::3, ::3], cmap="magma", shading="auto",
                      norm=matplotlib.colors.PowerNorm(.6, vmin=0, vmax=pdf.max()), rasterized=True)
        ax.plot(*peaks[0], marker="+", color="white", ms=8, mew=1.4)
    ax.contour(y[::2], z[::2], pdf[::2, ::2], thresholds[::-1], colors=[CYAN if heatmap else "#008991", "white" if heatmap else "#565656"], linewidths=[1, 1.15])
    frame(ax)


def case_figures(grids, densities, z, physics, cloud, paths):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3), sharex=True, sharey=True, layout="constrained")
    for ax, mm in zip(axes, (2, 5, 15)):
        density_panel(ax, grids[mm])
        ax.set(xlim=(1.65, 4.48), ylim=(.75, 2.95), title=f"Noise SD: {mm} mm", xlabel="$Y$ (m)")
    axes[0].set_ylabel("$Z$ (m)")
    save_figure(fig, "case_regions_v3.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.9, 6.1), layout="constrained")
    density_panel(axes[0, 0], grids[5])
    ax = axes[0, 1]
    scatter = ax.scatter(*cloud[:2], c=cloud[2], s=2.7, cmap="coolwarm", alpha=.58, linewidths=0, vmin=-8, vmax=8)
    density_panel(ax, grids[5], heatmap=False)
    cb = fig.colorbar(scatter, ax=ax, shrink=.91)
    cb.set_label("Vertical Magnus acceleration (m/s²)")
    for i, draw in enumerate(paths):
        label, endpoint, path = draw["label"], draw["yz"], draw["path"]
        color = COLORS[i]
        ax.scatter(*endpoint, color=color, edgecolor="white", s=32, zorder=5)
        ax.annotate(label, endpoint, xytext=(7, 3), textcoords="offset points", color=color,
                    fontweight="bold", zorder=6)
        for axis, k in zip(axes[1], (2, 1)):
            axis.plot(path[:, 0], path[:, k], color=color, label=label, lw=1.7)
            axis.plot(*path[-1, [0, k]], "o", color=color, ms=3)
    for axis, title in zip(axes[0], ("(a) Calibrated MDN density", "(b) Weighted physics posterior")):
        axis.set(xlim=(2.15, 3.7), ylim=(1.28, 2.72), title=title, xlabel="$Y$ (m)", ylabel="$Z$ (m)")
    for axis, coord, title in zip(axes[1], ("Z", "Y"), ("(c) Height along each trajectory", "(d) Lateral motion")):
        for x in CASE[:, 0]:
            axis.axvline(x, color="#a2a0aa", ls=":", lw=.7)
        axis.axvline(11, color=GREEN, ls="--", lw=.8)
        axis.set(xlim=(0, 11.45), xlabel="$x$ (m)", ylabel=f"${coord}$ (m)", title=title)
        axis.legend(loc="upper left", ncols=3, frameon=False)
    save_figure(fig, "case_decomposition_v3.png")

    fig, axes = plt.subplots(2, 2, figsize=(8.9, 5.9), sharex=True, layout="constrained")
    for r, mm in enumerate((5, 15)):
        for c, fixed_y in enumerate((2.67, None)):
            ax = axes[r, c]
            components = densities[mm].profile(z, fixed_y)
            if fixed_y is not None:
                components /= np.trapezoid(components.sum(0), z)
            for k, values in enumerate(components):
                ax.plot(z, values, ls="--", lw=.85, color=COLORS[k], alpha=.8)
            ax.plot(z, components.sum(0), lw=1.65, color=TEXT, label="MDN total")
            if mm == 5:
                key = str(fixed_y) if fixed_y is not None else "marginal"
                ax.plot(z, physics[key], lw=1.3, color=GREEN, label="Physics KDE")
            name = f"Slice at $Y={fixed_y}$ m" if fixed_y else "Marginal over $Y$"
            ax.set(xlim=(1.05, 2.98), title=f"{name} · {mm} mm", ylabel="Density (1/m)")
            if r == 1:
                ax.set_xlabel("$Z$ (m)")
            if r == 0 and c == 0:
                ax.legend(frameon=False)
    save_figure(fig, "case_profiles_v3.png")


def calibration_figure():
    records = []
    for seed in SEEDS:
        with (ROOT / "experiments" / "main_v3_all" / f"seed_{seed}/test_scores.csv").open(newline="") as file:
            records.extend(csv.DictReader(file))
    rows = []
    for mm in NOISE_MM:
        for stage in ("raw", "scaled", "scaled_hdr"):
            group = [row for row in records if float(row["sigma_mm"]) == mm and row["calibration"] == stage]
            assert len(group) == 3 and {int(row["n"]) for row in group} == {109108}
            keys = ("nll", "cov50", "cov95", "area50_m2", "area95_m2", "goal_ece", "goal_brier", "goal_rate")
            rows.append(dict(mm=mm, stage=stage, **{key: float(np.mean([float(row[key]) for row in group]))
                                                    for key in keys}))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.15), layout="constrained")
    for ax, nominal in zip(axes, (50, 95)):
        ax.axhline(nominal, color="#666666", ls=":", lw=1.1, label="Nominal")
        for stage, title, color, marker, line in zip(("raw", "scaled", "scaled_hdr"),
            ("Raw", "Scale only", "Scale + HDR"), COLORS, ("o", "s", "D"), ("-", "--", "-.")):
            group = [row for row in rows if row["stage"] == stage]
            ax.plot(NOISE_MM, [100*row[f"cov{nominal}"] for row in group], label=title,
                    color=color, marker=marker, ls=line, ms=4)
        ax.set(title=f"{nominal}% nominal region", xticks=NOISE_MM, xlabel="Position noise SD (mm)",
               ylabel="Observed coverage (%)")
        ax.grid(alpha=.17)
    fig.legend(loc="outside lower center", ncols=4, frameon=False)
    save_figure(fig, "coverage_v3.png")
    return rows


def case_diagnostics():
    paper_style()
    z = np.linspace(.15, 3.8, 2801)
    report = {"mdn": {}}
    figures = {}
    for seed in SEEDS:
        bundle = load_model(seed)
        all_mm = {}
        for mm in NOISE_MM:
            density = CaseDensity(bundle, mm)
            grid = density.grid()
            all_mm[mm] = density.describe(z, grid)
            if seed == 11:
                figures[mm] = (density, grid)
        report["mdn"][seed] = all_mm
    post = posterior(n=40000, sigma=.005, seed=8047)
    sim = simulate(post["theta"], Config())
    theta = post["theta"][sim["status"] == 1]
    endpoints, w = post["endpoints"], post["weights"]
    np.testing.assert_array_equal(endpoints, sim["endpoint"][sim["status"] == 1])
    qy, qz = (weighted_quantiles(axis, w) for axis in endpoints.T)
    in_frame = (np.abs(endpoints[:, 0]) <= FRAME[1]) & (endpoints[:, 1] >= FRAME[2]) & (endpoints[:, 1] <= FRAME[3])
    kde = gaussian_kde(endpoints.T, weights=w)
    width = kde.factor
    profiles = {"marginal": gaussian_kde(endpoints[:, 1], weights=w)(z)}
    peak_sets = {}
    for y in FIXED_Y:
        peak_sets[str(y)] = {}
        for factor in (.75, 1.0, 1.5):
            kde.set_bandwidth(width * factor)
            values = kde(np.vstack((np.full_like(z, y), z)))
            peak_sets[str(y)][str(factor)] = peak_positions(z, values)
            if factor == 1.0:
                profiles[str(y)] = values / np.trapezoid(values, z)
    # Weighted correlations use the same posterior weights as the case density.
    cov = np.cov(np.column_stack((theta[:, 4:7], endpoints)), rowvar=False, aweights=w)
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))
    report["physics"] = dict(ess=post["conditional_ess"], n=len(w), quantiles=[qy, qz],
                             in_frame=float(w[in_frame].sum()), above_bar=float(w[endpoints[:, 1] > FRAME[3]].sum()),
                             lateral_out=float(w[np.abs(endpoints[:, 0]) > FRAME[1]].sum()),
                             slice_peaks=peak_sets, marginal_peaks=peak_positions(z, profiles["marginal"]),
                             corr_wy_z=float(corr[0, 4]), corr_wz_y=float(corr[1, 3]), corr_cd_y=float(corr[2, 3]))
    idx = np.random.default_rng(7).choice(len(w), 9000, p=w)
    midpoint_velocity = (CASE[1] - [0, 0, .11]) / .1
    cloud = (*endpoints[idx].T, magnus(theta[idx, 3:6], midpoint_velocity, Config())[:, 2])
    models, grids = ({mm: pair[i] for mm, pair in figures.items()} for i in range(2))
    paths = trajectories(theta, models[5])
    case_figures(grids, models, z, profiles, cloud, paths)
    report["trajectories"] = [{key: value for key, value in path.items() if key != "path"} for path in paths]
    report["calibration"] = calibration_figure()
    write_json("diagnostics.json", report)
    print("Case, decomposition, profile and coverage diagnostics complete", flush=True)


def write_json(name, report):
    (OUT / name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def paired_comparison():
    seed = 155108
    rng = np.random.default_rng(seed)
    with np.load(DATA / "indices.npz") as indices, np.load(DATA / "data.npz") as data:
        test = indices["test"]
        subset = np.sort(rng.choice(len(test), 32, replace=False))
        rows = test[subset]
        perturbation = np.random.default_rng(3005).normal(size=(len(test), 2, 3))[subset] * .005
        obs, truth = data["observations"][rows] + perturbation, data["endpoint"][rows]
    baselines = baseline_endpoint(obs)
    bundle = load_model(11)
    torch.manual_seed(85155)
    forecast(bundle, obs[:1], 5).sample((4096,))
    output = []
    for row, x, y, base, bayes_seed in zip(rows, obs, truth, baselines, rng.integers(2**30, size=len(rows))):
        bayes = bayesian_forecast((x, y, int(bayes_seed)))
        start = perf_counter()
        dist = forecast(bundle, x[None], 5)
        draws = dist.sample((4096,))[:, 0].numpy()
        mdn = metrics(draws, y)
        mdn.update(error=float(np.linalg.norm(dist.mean[0].numpy() - y)), seconds=perf_counter() - start)
        quad = float(np.linalg.norm(base - y)) if np.isfinite(base).all() else None
        output.append(dict(row=int(row), quadratic=quad, bayes=bayes, mdn=mdn))
        print(f"Paired flight {len(output)}/{len(rows)}: ESS {bayes.get('ess', 0):.1f}", flush=True)
    paired = [row for row in output if row["bayes"].get("ess", 0) >= 100 and row["quadratic"] is not None]
    summary = {key: {metric: float(np.median([row[key][metric] for row in paired]))
                     for metric in ("error", "energy", "seconds")} for key in ("bayes", "mdn")}
    summary["quadratic"] = {"error": float(np.median([row["quadratic"] for row in paired]))}
    write_json("paired_bayes.json", dict(subset_seed=seed, n=len(output), paired=len(paired),
                                        summary=summary, rows=output))


def screen_diagnostics():
    paper_style()
    screen = json.loads((OUT / "screen.json").read_text())
    models, protocol = screen["models"], screen["protocol"]
    chosen_depth = screen["depth_at_activation_screen"]
    depths = ["depth_1", "features_wiener", "depth_3", "depth_4"]
    activations = [depths[chosen_depth - 1], "Wiener" if chosen_depth == 3 else "activation_GELU",
                   "activation_ReLU", "activation_Tanh"]
    nll = {}
    for name, model in models.items():
        rows = model["rows"]
        assert [row["seed"] for row in rows] == list(SEEDS), name
        values = [np.mean([x["nll"] for x in row["selection"]]) for row in rows]
        nll[name] = dict(mean=float(np.mean(values)), sd=float(np.std(values, ddof=1)))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.15), layout="constrained")
    panels = ((depths, (1, 2, 3, 4), "Depth with SiLU"),
              (activations, ("SiLU", "GELU", "ReLU", "Tanh"), f"Activation at depth {chosen_depth}"))
    for ax, (names, labels, title) in zip(axes, panels):
        x = np.arange(len(names))
        ax.errorbar(x, [nll[name]["mean"] for name in names],
                    yerr=[nll[name]["sd"] for name in names], capsize=4, fmt="o", color=COLORS[0])
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel="Selection NLL (lower is better)")
        ax.grid(alpha=.17, axis="y")
    save_figure(fig, "architecture_v3.png")

    # Use the same rows and perturbations as the retraining comparison.
    rng = np.random.default_rng(protocol["subset_seed"])
    with np.load(DATA / "indices.npz") as indices, np.load(DATA / "data.npz") as data:
        for key in ("train", "selection", "test"):
            ids = np.sort(rng.choice(indices[key], protocol["sizes"][key], replace=False))
        obs, y = data["observations"][ids], data["endpoint"][ids]
    bundles = {seed: load_model(seed) for seed in SEEDS}
    frozen, invalid = [], {}
    for mm in NOISE_MM:
        noisy = obs + np.random.default_rng(3000 + mm).normal(size=obs.shape) * mm / 1000
        invalid[mm] = int((~np.isfinite(baseline_endpoint(noisy)).all(axis=1)).sum())
        for seed, bundle in bundles.items():
            dist = forecast(bundle, noisy, mm)
            nll_mean = float(-dist.log_prob(torch.from_numpy(y.astype(np.float32))).mean())
            err = np.linalg.norm(dist.mean.numpy() - y, axis=1)
            frozen.append(dict(seed=seed, sigma_mm=mm, median=float(np.median(err)),
                               p90=float(np.quantile(err, .90)), nll=nll_mean))
    write_json("screen_summary.json", dict(selection_nll=nll, invalid=invalid, frozen=frozen))


def verify_calibration():
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    with np.load(DATA / "indices.npz") as indices, np.load(DATA / "data.npz") as data:
        ids = indices["test"]
        obs, truth = data["observations"][ids].reshape(-1, 6), data["endpoint"][ids]
    rows = []
    for seed in SEEDS:
        model, stats, cal = load_model(seed)
        model.to(device)
        with (ROOT / "experiments" / "main_v3_all" / f"seed_{seed}/test_scores.csv").open(newline="") as file:
            reference = {(int(float(row["sigma_mm"])), row["calibration"]): row
                         for row in csv.DictReader(file)}
        for mm in NOISE_MM:
            spec = cal[str(mm)]
            adjusted = spec["hdr_mass"]
            for stage, scale, m50, m95 in (("raw", 1., .5, .95),
                                          ("scaled", spec["scale_sd"], .5, .95),
                                          ("scaled_hdr", spec["scale_sd"], adjusted["0.5"], adjusted["0.95"])):
                scores = evaluate_hdr_and_metrics(model, obs, truth, mm / 1000, stats["xm"], stats["xs"],
                                                  device, scale, m50, m95, 256, 3000 + mm)
                scores.pop("tail_prob")
                max_delta = max(abs(scores[key] - float(reference[mm, stage][key])) for key in scores)
                rows.append(dict(seed=seed, mm=mm, stage=stage, max_abs_csv_delta=max_delta, **scores))
            print(f"Rechecked all calibration stages: seed {seed}, {mm} mm", flush=True)
    write_json("calibration_recheck.json", dict(n=len(ids), torch=torch.__version__, rows=rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("case", "paired", "screen", "verify-calibration"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"case": case_diagnostics, "paired": paired_comparison, "screen": screen_diagnostics,
     "verify-calibration": verify_calibration}[args.stage]()
