"""Retrain the archived model comparisons on fixed subsets of the v3 splits."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import softplus

HERE = Path(__file__).resolve().parent          # code/analysis
ROOT = HERE.parents[1]                          # repository root
MODEL = ROOT / "code" / "model"
DATA = ROOT / "data" / "datasets" / "v3_all"
OUT = ROOT / "experiments" / "diagnostics"
sys.path[:0] = [str(MODEL)]

from common import ARMS, baseline_endpoint, make_inputs
from pi_res_mdn import PhysicsResidualMDN, extract_physics_features
from train_wiener_mdn import CapturedAdamStep, WienerResidualMDN, extract_wiener_features, score_batch

SEEDS = (11, 42, 155)
NOISE_MM = (2, 5, 10, 15)
SIZES = {"train": 65536, "selection": 4096, "test": 8192}
BATCH = 512
ACT = {"SiLU": nn.SiLU, "GELU": nn.GELU, "ReLU": nn.ReLU, "Tanh": nn.Tanh}


class ScreenMDN(WienerResidualMDN):
    def __init__(self, dim, depth, activation, k, family, correlated):
        super().__init__(dim, 192, depth, k)
        self.corr_limit = 0.98 if correlated else 0.0
        act = ACT[activation]
        self.in_proj[-1] = act()
        for block in self.blocks:
            block.act = act()
        if family == "legacy":
            self.in_proj = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU())
            self.blocks = nn.ModuleList()
            self.head = nn.Linear(64, 6 * k)
            self.corr_limit = 0.995
        if family == "pi":
            pi = PhysicsResidualMDN(16, 128, k)
            self.in_proj, self.head = pi.backbone, pi.head
            self.blocks = nn.ModuleList()
            self.corr_limit = 0.95

    def mixture_params(self, features, anchor, scale_sd=1.0):
        hidden = self.in_proj(features)
        for block in self.blocks:
            hidden = block(hidden)
        raw = self.head(hidden).reshape(-1, self.k, 6)
        return (raw[..., 0], anchor[:, None, :] + raw[..., 1:3],
                (softplus(raw[..., 3:5]) + 0.001) * scale_sd,
                self.corr_limit * torch.tanh(raw[..., 5]))


def design(name, features="wiener", target="res", depth=3, activation="GELU", k=5,
           family="residual", intervention="reference"):
    return dict(name=name, features=features, target=target, depth=depth, activation=activation,
                k=k, family=family, intervention=intervention)


def features_and_anchor(spec, obs, sigma):
    kind = spec["features"]
    if kind == "wiener":
        features, anchor = extract_wiener_features(obs, sigma)
    elif kind == "pi":
        features, anchor = extract_physics_features(torch.from_numpy(obs.reshape(-1, 6)).float(),
                                                    torch.from_numpy(sigma[:, None]).float())
        features, anchor = features.numpy(), anchor.numpy()
    else:
        features = make_inputs(kind, obs, sigma)
        anchor = baseline_endpoint(obs) if spec["target"] == "res" else np.zeros((len(obs), 2))
    return features.astype(np.float32), anchor.astype(np.float32)


def noisy(part, rng, sigma):
    obs, target = part
    return obs + rng.normal(size=obs.shape) * sigma[:, None, None], target


def inputs(spec, part, rng, sigma):
    obs, target = noisy(part, rng, sigma)
    features, anchor = features_and_anchor(spec, obs, sigma)
    return features, anchor, target


def tensor_inputs(parts, xm, xs, *, training=False):
    features, anchor, target = parts
    valid = np.isfinite(anchor).all(axis=1) if training else slice(None)
    scaled = np.nan_to_num((features[valid] - xm) / xs, nan=0, posinf=0, neginf=0)
    arrays = (scaled, np.nan_to_num(anchor[valid], nan=0), target[valid])
    return tuple(torch.from_numpy(np.asarray(item, dtype=np.float32)).cuda() for item in arrays)


@torch.no_grad()
def evaluate(model, eval_sets):
    result = []
    for mm, (features, anchor, target) in zip(NOISE_MM, eval_sets):
        torch.manual_seed(7000 + mm)
        scored = torch.cat([score_batch(model(f, a), y, 0.50, 0.95, 128)
                            for f, a, y in zip(features.split(1024), anchor.split(1024), target.split(1024))])
        nll, error, _, rank, area50, area95, goal_prob = scored.cpu().numpy().T
        truth = target.cpu().numpy()
        goal = (np.abs(truth[:, 0]) <= 3.55) & (truth[:, 1] >= .11) & (truth[:, 1] <= 2.33)
        result.append(dict(sigma_mm=mm, nll=float(nll.mean()), median=float(np.median(error)),
                           p90=float(np.quantile(error, .90)), cov50=float((rank >= .50).mean()),
                           cov95=float((rank >= .05).mean()), area50=float(np.median(area50)),
                           area95=float(np.median(area95)), brier=float(np.mean((goal_prob-goal)**2))))
    return result


def train(spec, seed, parts, epochs, score_test):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ref_rng = np.random.default_rng(11)
    n = len(parts["train"][0])
    ref, _, _ = inputs(spec, parts["train"], ref_rng, ref_rng.uniform(.001, .018, n))
    xm, xs = np.nanmean(ref, axis=0), np.maximum(np.nanstd(ref, axis=0), 1e-5)

    def evaluation_sets(key, offset):
        return [tensor_inputs(inputs(spec, parts[key], np.random.default_rng(offset + mm),
                                     np.full(len(parts[key][0]), mm / 1000)), xm, xs)
                for mm in NOISE_MM]

    selection = evaluation_sets("selection", 1000)
    model = ScreenMDN(len(xm), spec["depth"], spec["activation"], spec["k"], spec["family"],
                      spec["intervention"] != "rho_zero").cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=torch.tensor(.002, device="cuda"),
                            weight_decay=1e-4, fused=True, capturable=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    buffer = [torch.empty((n, width), device="cuda") for width in (len(xm), 2, 2)]
    step = None
    best = float("inf")
    perm_rng = torch.Generator(device="cuda").manual_seed(seed)
    for epoch in range(1, epochs + 1):
        if epoch == 1 or spec["intervention"] != "fixed_noise":
            batch_data = inputs(spec, parts["train"], rng, rng.uniform(.001, .018, n))
            arrays = tensor_inputs(batch_data, xm, xs, training=True)
            count = len(arrays[0])
            for dest, src in zip(buffer, arrays):
                dest[:count].copy_(src)
        if step is None:
            step = CapturedAdamStep(model, opt, *buffer, BATCH)
        for ids in torch.randperm(count, device="cuda", generator=perm_rng).split(BATCH):
            loss = step(ids)
        if spec["intervention"] != "fixed_lr":
            schedule.step()
        with torch.no_grad():
            val = torch.stack([model.nll(*fold).mean() for fold in selection]).mean().item()
        if not np.isfinite(val) or not np.isfinite(loss.item()):
            raise RuntimeError(f"Nonfinite loss: {spec['name']}, seed {seed}, epoch {epoch}")
        if val < best:
            best, best_epoch = val, epoch
            state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    model.load_state_dict(state)
    row = dict(seed=seed, selected_epoch=best_epoch, selection=evaluate(model, selection))
    if score_test:
        row["test"] = evaluate(model, evaluation_sets("test", 3000))
    return row


def selection_nll(rows):
    return np.asarray([np.mean([score["nll"] for score in row["selection"]]) for row in rows])


def coverage_gap(rows):
    return np.mean([abs(score[key] - goal) for row in rows for score in row["selection"]
                    for key, goal in (("cov50", .50), ("cov95", .95))])


def prefer_simpler(simple, reference):
    difference = selection_nll(simple) - selection_nll(reference)
    return (difference.mean() <= difference.std(ddof=1)
            and coverage_gap(simple) <= coverage_gap(reference) + .015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=OUT / "screen.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    torch.distributions.Distribution.set_default_validate_args(False)
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf-8"))
    for filename, key in (("data.npz", "sha256"), ("indices.npz", "indices_sha256")):
        with (DATA / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != manifest[key]:
                raise ValueError(f"v3 hash mismatch: {filename}")
    rng = np.random.default_rng(20261108)
    with np.load(DATA / "indices.npz") as indices, np.load(DATA / "data.npz") as data:
        rows = {key: np.sort(rng.choice(indices[key], size, replace=False)) for key, size in SIZES.items()}
        obs, target = data["observations"], data["endpoint"]
        parts = {key: (obs[ids], target[ids]) for key, ids in rows.items()}
    protocol = dict(dataset_sha256=manifest["sha256"], index_sha256=manifest["indices_sha256"],
                    subset_seed=20261108, sizes=SIZES, epochs=args.epochs, seeds=SEEDS,
                    batch=BATCH, mc=128, torch=torch.__version__)
    report = dict(protocol=protocol, models={})
    if args.out.exists():
        report = json.loads(args.out.read_text(encoding="utf-8"))
        if json.dumps(report["protocol"], sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise ValueError("Existing screen output uses a different protocol")

    def save():
        temp = args.out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(args.out)

    def run(spec, *, score_test=False):
        name = spec["name"]
        entry = report["models"].setdefault(name, dict(spec=spec, rows=[]))
        for seed in SEEDS[len(entry["rows"]):]:
            print(f"Training {name}, seed {seed}", flush=True)
            row = train(spec, seed, parts, args.epochs, score_test)
            entry["rows"].append(row)
            save()
            print(f"{name} seed {seed}: selection NLL {selection_nll([row])[0]:.4f}", flush=True)
        return entry["rows"]

    run(design("Wiener"), score_test=True)
    for name, arm in ARMS.items():
        run(design(name, arm["input"], arm["target"]), score_test=True)
    run(design("Legacy", "raw", "abs", k=3, family="legacy"), score_test=True)
    run(design("PI-ResMDN", "pi", family="pi"), score_test=True)

    for kind in ("phys_reduced", "phys", "wiener"):
        run(design(f"features_{kind}", kind, "abs" if kind != "wiener" else "res",
                   depth=2, activation="SiLU", k=5))
    # The 11/16-feature comparison keeps the direct target fixed; the v3 K
    # and architecture screens hold the full Wiener input and residual fixed.
    for k in (1, 3):
        run(design(f"mixture_{k}", depth=2, activation="SiLU", k=k))
    depths = {depth: run(design(f"depth_{depth}" if depth != 2 else "features_wiener",
                               depth=depth, activation="SiLU")) for depth in range(1, 5)}
    best_depth = min(depths, key=lambda depth: selection_nll(depths[depth]).mean())
    chosen_depth = next((depth for depth in depths if depth < best_depth
                         and prefer_simpler(depths[depth], depths[best_depth])), best_depth)
    report["depth_at_activation_screen"] = chosen_depth
    save()
    for activation in ("GELU", "ReLU", "Tanh"):
        if activation == "GELU" and chosen_depth == 3:
            continue
        run(design(f"activation_{activation}", depth=chosen_depth, activation=activation))
    for option in ("rho_zero", "fixed_lr", "fixed_noise"):
        run(design(option, intervention=option))
    print("Completed all v3 screens", flush=True)


if __name__ == "__main__":
    main()
