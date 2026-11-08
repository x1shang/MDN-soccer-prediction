import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "simulation"))
from physics import CASE
from common import build_mdn, make_inputs

GOAL = (3.66, 0., 2.44)
HIDDEN = 128
DEPTH = 2


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def get_checkpoint(out, config, seed):
    kind, k = config
    return out / "checkpoints" / f"{kind}_k{k}_{seed}.pt"


def load_checkpoint(path, config):
    # P0 checkpoints are locally produced and store NumPy normalization arrays.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    kind, k = config
    stats = {key: np.asarray(value) for key, value in checkpoint["stats"].items()}
    model = build_mdn(len(stats["xm"]), k=k, hidden=HIDDEN, **checkpoint.get("architecture", {"depth": DEPTH}))
    model.load_state_dict(checkpoint["state"])
    return model.eval(), stats


def predict(model, stats, kind, noisy, sigma):
    features = make_inputs(kind, noisy, sigma)
    x = np.nan_to_num((features - stats["xm"]) / stats["xs"], nan=0., posinf=0., neginf=0.)
    return model(torch.as_tensor(x, dtype=torch.float32))


def rescale(dist, scale):
    component = dist.component_distribution
    normal = torch.distributions.MultivariateNormal(component.loc, scale_tril=component.scale_tril * scale)
    return torch.distributions.MixtureSameFamily(dist.mixture_distribution, normal)


@torch.no_grad()
def plot_case(out, config, frozen, assets):
    kind, k = config
    seed = frozen["case_seed"]
    checkpoint = get_checkpoint(assets, config, seed)
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != frozen["checkpoint_sha256"][checkpoint.stem]:
        raise ValueError("P1 checkpoint hash mismatch")
    model, stats = load_checkpoint(checkpoint, config)
    calibration = frozen["calibration"][f"{kind}_k{k}_{seed}"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), layout="constrained", sharex=True, sharey=True)
    records = []
    for mm, axis in zip((2, 5, 15), axes):
        spec = calibration[str(mm)]
        dist = rescale(predict(model, stats, kind, CASE[None], mm / 1000), spec["scale_sd"])
        torch.manual_seed(75000 + mm)
        samples = dist.sample((65536,))
        points = samples[:, 0].numpy() * stats["ys"] + stats["ym"]
        log_threshold = {level: float(torch.quantile(dist.log_prob(samples), 1 - mass))
                         for level, mass in spec["hdr_mass"].items()}
        lower, upper = np.quantile(points, [0.0005, 0.9995], axis=0)
        lower -= .15 * (upper - lower)
        upper += .15 * (upper - lower)
        y, z = np.meshgrid(np.linspace(lower[0], upper[0], 300), np.linspace(lower[1], upper[1], 300))
        grid = np.stack((y, z), axis=-1)
        standardized = torch.as_tensor((grid - stats["ym"]) / stats["ys"], dtype=torch.float32)
        log_density = dist.log_prob(standardized).numpy()
        density = np.exp(log_density) / np.prod(stats["ys"])
        mode = grid[np.unravel_index(np.argmax(density), density.shape)]
        cell_area = float((y[0, 1] - y[0, 0]) * (z[1, 0] - z[0, 0]))
        goal = (np.abs(points[:, 0]) < GOAL[0]) & (points[:, 1] > GOAL[1]) & (points[:, 1] < GOAL[2])
        records.append(dict(sigma_mm=mm, mode_m=mode.tolist(),
                            mean_m=(dist.mean[0].numpy() * stats["ys"] + stats["ym"]).tolist(),
                            goal_probability_given_airborne=float(goal.mean()),
                            grid_integrated_mass=float(density.sum() * cell_area),
                            hdr_area_m2={level: float((log_density >= threshold).sum() * cell_area)
                                         for level, threshold in log_threshold.items()}))
        axis.set_facecolor("#000004")
        axis.pcolormesh(y, z, density, cmap="magma", shading="auto")
        axis.contour(y, z, log_density, [log_threshold["0.95"], log_threshold["0.5"]],
                     colors=["cyan", "white"])
        axis.add_patch(Rectangle((-GOAL[0], GOAL[1]), 2 * GOAL[0], GOAL[2] - GOAL[1],
                                 fill=False, edgecolor="limegreen", linestyle="--"))
        axis.plot(*mode, marker="+", color="white")
        axis.set(xlabel="Y (m)", title=f"{mm} mm noise SD", aspect="equal")
    axes[0].set(ylabel="Z (m)")
    fig.legend(handles=[
        Line2D([], [], color="0.35", label="50% HDR (white on plot)"),
        Line2D([], [], color="darkturquoise", label="95% HDR"),
        Line2D([], [], color="limegreen", linestyle="--", label="Goal frame"),
        Line2D([], [], color="0.35", linestyle="none", marker="+", label="Mode"),
    ], loc="outside lower center", ncols=4, frameon=False)
    fig.savefig(out / "figures" / "case_regions.png", dpi=300)
    plt.close(fig)
    save_json(out / "case.json", dict(features=kind, k=k, seed=seed, rows=records))

def main():
    parser = argparse.ArgumentParser(description="Regenerate the published P1 case heatmap")
    parser.add_argument("output", type=Path, help="New output directory")
    args = parser.parse_args()
    config = json.loads((HERE / "best_config.json").read_text(encoding="utf-8"))
    (args.output / "figures").mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    plot_case(args.output, tuple(config["winner"]), config, HERE)


if __name__ == "__main__":
    main()
