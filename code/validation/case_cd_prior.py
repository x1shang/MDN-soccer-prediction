"""Case-study sensitivity to the C_D prior upper bound (A's request, handoff v2 §8).

Main model: C_D ~ U[0.15, 0.30]. Sensitivity: C_D ~ U[0.15, 0.45].
For sigma = 5 mm, report posterior mean, grid mode, joint 50% / 95% HDR areas
and the percentage area change. Several independent seeds quantify Monte-Carlo
variability of the area estimate.

Usage (from repo root):
    python student_b/case_cd_prior.py --n 40000 --seeds 8047 20261107 1234
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde

from inference import posterior


def hdr_report(result, seed, upper):
    """Weighted-KDE grid HDR, same estimator as validate.case_regions."""
    rng = np.random.default_rng(seed)
    points, weights = result["endpoints"], result["weights"]
    keep = weights > weights.max() * 1e-10
    kde = gaussian_kde(points[keep].T, weights=weights[keep])
    draws = kde.resample(6000, seed=rng)
    lo, hi = np.quantile(draws, [.0005, .9995], axis=1)
    lo -= .2 * (hi - lo)
    hi += .2 * (hi - lo)
    y, z = np.meshgrid(np.linspace(lo[0], hi[0], 160), np.linspace(lo[1], hi[1], 160))
    cell = (hi[0] - lo[0]) * (hi[1] - lo[1]) / 159**2
    grid = np.stack((y.ravel(), z.ravel()))
    density = kde(grid)
    order = np.argsort(density)[::-1]
    masses = np.cumsum(density[order]) * cell
    thresholds = [float(density[order[min(np.searchsorted(masses, p), len(order) - 1)]])
                  for p in (.5, .95)]
    mode = grid[:, order[0]]
    return dict(cd_upper=upper, seed=seed,
                mean_yz_m=result["conditional_mean"],
                mode_yz_m=mode.tolist(),
                ess=result["conditional_ess"],
                airborne_probability=result["airborne_probability"],
                grid_mass=float(masses[-1]),
                hdr_area_m2=[float(np.sum(density >= q) * cell) for q in thresholds],
                grid_y_m=[float(lo[0]), float(hi[0])], grid_z_m=[float(lo[1]), float(hi[1])],
                density=density.reshape(y.shape), y=y, z=z, thresholds=thresholds)


def main(n, seeds, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports, grids = [], {}
    for upper in (.30, .45):
        for seed in seeds:
            result = posterior(n=n, sigma=.005, seed=seed, cd_upper=upper)
            report = hdr_report(result, seed, upper)
            grids[(upper, seed)] = dict(density=report.pop("density"),
                                        y=report.pop("y"), z=report.pop("z"),
                                        thresholds=report.pop("thresholds"))
            reports.append(report)
            print(json.dumps({k: v for k, v in report.items()
                              if k in ("cd_upper", "seed", "mean_yz_m", "ess", "hdr_area_m2")},
                             ensure_ascii=False), flush=True)
    summary = {}
    for upper in (.30, .45):
        group = [r for r in reports if r["cd_upper"] == upper]
        area50 = np.array([r["hdr_area_m2"][0] for r in group])
        area95 = np.array([r["hdr_area_m2"][1] for r in group])
        summary[f"{upper:.2f}"] = dict(
            n_reports=len(group),
            area50_mean_m2=float(area50.mean()), area50_range_m2=[float(area50.min()), float(area50.max())],
            area95_mean_m2=float(area95.mean()), area95_range_m2=[float(area95.min()), float(area95.max())],
            mean_yz_m=np.mean([r["mean_yz_m"] for r in group], axis=0).tolist())
    base, wide = summary["0.30"], summary["0.45"]
    comparison = dict(
        area50_change_pct=float(100 * (wide["area50_mean_m2"] / base["area50_mean_m2"] - 1)),
        area95_change_pct=float(100 * (wide["area95_mean_m2"] / base["area95_mean_m2"] - 1)),
        mean_shift_m=float(np.linalg.norm(np.array(wide["mean_yz_m"]) - np.array(base["mean_yz_m"]))))
    payload = dict(sigma_m=.005, samples_per_stage=n, seeds=seeds,
                   main_model="C_D ~ U[0.15, 0.30] (params.json)",
                   sensitivity="C_D ~ U[0.15, 0.45]", summary=summary,
                   comparison=comparison, reports=reports,
                   note="Weighted KDE + grid HDR, conditional on airborne crossing; "
                        "area differences include Monte-Carlo variability across seeds.")
    (output / "cd_prior_sensitivity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(grids, reports, seeds[0], output / "cd_prior_contours.png")
    print("\nSUMMARY", json.dumps(summary, ensure_ascii=False))
    print("COMPARISON", json.dumps(comparison, ensure_ascii=False))


def _plot(grids, reports, seed, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, upper in zip(axes, (.30, .45)):
        report = next(r for r in reports if r["cd_upper"] == upper and r["seed"] == seed)
        grid_data = grids[(upper, seed)]
        y, z = grid_data["y"], grid_data["z"]
        density = grid_data["density"]
        ax.pcolormesh(y, z, density, shading="auto")
        thresholds = grid_data["thresholds"][::-1]
        contours = ax.contour(y, z, density, levels=thresholds, colors=["cyan", "white"])
        ax.clabel(contours, fmt=dict(zip(thresholds, ["95%", "50%"])))
        ax.plot(*report["mode_yz_m"], "r+", label="mode")
        ax.plot(*report["mean_yz_m"], "kx", label="mean")
        a50, a95 = report["hdr_area_m2"]
        ax.set(xlabel="Y (m)", ylabel="Z (m)",
               title=f"C_D ~ U[0.15, {upper:.2f}]\n50%: {a50:.3f} m², 95%: {a95:.3f} m²")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Case study, sigma = 5 mm: C_D prior sensitivity")
    fig.savefig(path, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[8047, 20261107, 1234])
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "experiments" / "validation" / "case_cd_prior")
    args = parser.parse_args()
    main(args.n, args.seeds, args.output)
