"""Extend paired airborne data with a ground-contact model and sensitivity audit."""
import argparse
import json
from pathlib import Path

import numpy as np

from aiming import sha256
from ground_contact import continue_ground
from physics import CASE, Config

GROUPS = ("train", "selection", "calibration", "test")
BASE = (.60, .40, .08)
SWEEP = {
    "restitution_low": (.56, .40, .08), "restitution_high": (.67, .40, .08),
    "friction_low": (.60, .25, .08), "friction_high": (.60, .55, .08),
    "rolling_low": (.60, .40, .046), "rolling_high": (.60, .40, .153),
}


def in_frame(endpoint):
    y, z = endpoint.T
    return (np.abs(y) <= 3.55) & (z >= .11) & (z <= 2.33)


def case_audit(data):
    obs = data["observations"]
    keep = np.isfinite(obs).all(axis=(1, 2))
    err2 = np.sum((obs[keep] - CASE)**2, axis=(1, 2))
    status = data["status"][keep]
    goal = in_frame(data["endpoint"][keep])
    results = {}
    for mm in (2, 5, 10, 15):
        logw = -err2 / (2 * (mm / 1000)**2)
        weights = np.exp(logw - logw.max())
        weights /= weights.sum()
        air, contact = status == 1, status == 3
        results[str(mm)] = dict(
            ess=float(1 / (weights @ weights)),
            direct_crossing=float(weights[air].sum()),
            contact_crossing=float(weights[contact].sum()),
            direct_in_frame=float(weights[air & goal].sum()),
            contact_in_frame=float(weights[contact & goal].sum()),
            total_in_frame=float(weights[goal].sum()),
        )
    return dict(method="Finite v3 bank weighted by Gaussian prefix likelihood; low ESS limits inference. "
                "A numerical zero cannot establish a probability bound.",
                rows_with_finite_prefix=int(keep.sum()), by_noise_mm=results)


def extend_indices(base, data, recovered):
    if not np.array_equal(np.sort(np.concatenate([base[k] for k in GROUPS])),
                          np.flatnonzero(data["status"] == 1)):
        raise ValueError("Baseline indices do not exactly partition direct crossings")
    rng = np.random.default_rng(155211)
    split = data["split"]
    anchor = data["anchor"]
    validation = recovered[split[recovered] == 1]
    sel, cal = [], []
    for j, a in enumerate(np.unique(anchor[validation])):
        ids = rng.permutation(validation[anchor[validation] == a])
        cut = (len(ids) + j % 2) // 2
        sel.extend(ids[:cut])
        cal.extend(ids[cut:])
    extra = [recovered[split[recovered] == 0], sel, cal, recovered[split[recovered] == 2]]
    return {k: np.sort(np.concatenate((base[k], np.asarray(ids, dtype=int))))
            for k, ids in zip(GROUPS, extra)}


def simulate(theta, params, cfg):
    # Keep transient memory bounded when a desktop regenerates a larger prior.
    pieces = [continue_ground(theta[i:i+10000], *params, cfg)
              for i in range(0, len(theta), 10000)]
    if not pieces:
        return continue_ground(theta, *params, cfg)
    return tuple(np.concatenate(items) for items in zip(*pieces))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing data.npz and indices.npz")
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    source, out = args.source, args.output
    data_path, index_path = source / "data.npz", source / "indices.npz"
    with np.load(data_path, allow_pickle=False) as f:
        data = dict(f)
    with np.load(index_path, allow_pickle=False) as f:
        base_indices = {k: f[k] for k in GROUPS}
    ground = np.flatnonzero(data["status"] == 2)
    air = data["status"] == 1
    theta = data["theta"][ground]
    cfg = Config(horizon=3.)
    kind, endpoint, contacts, obs, arrival = simulate(theta, BASE, cfg)
    hit = kind > 0
    recovered = ground[hit]
    indices = extend_indices(base_indices, data, recovered)
    if not np.isfinite(obs).all() or not np.isfinite(endpoint[hit]).all() or not np.isfinite(arrival[hit]).all():
        raise RuntimeError("Nonfinite ground trajectory output")
    if np.any(endpoint[hit, 1] < cfg.radius - 1e-9) or not np.all(contacts[hit] > 0):
        raise RuntimeError("Invalid ground-first endpoint")
    original = data["observations"][ground]
    np.testing.assert_allclose(obs[np.isfinite(original)], original[np.isfinite(original)], atol=1e-10, rtol=0)
    data["observations"][ground] = np.where(np.isfinite(original), original, obs)
    data["endpoint"][recovered] = endpoint[hit]
    data["arrival"][recovered] = arrival[hit]
    data["status"][recovered] = 3
    data["contact_kind"] = np.zeros(len(air), dtype=np.int8)
    data["contact_kind"][ground] = kind
    data["contact_count"] = np.zeros(len(air), dtype=np.int16)
    data["contact_count"][ground] = contacts
    np.testing.assert_array_equal(np.sort(np.concatenate(list(indices.values()))),
                                  np.flatnonzero(np.isin(data["status"], [1, 3])))

    out.mkdir(parents=True, exist_ok=False)
    dataset, index = out / "data_bounce.npz", out / "indices_bounce.npz"
    np.savez_compressed(dataset, **data)
    np.savez_compressed(index, **indices)
    manifest = dict(
        n=len(air), sha256=sha256(dataset), indices_sha256=sha256(index),
        baseline_data_sha256=sha256(data_path), baseline_indices_sha256=sha256(index_path),
        counts={str(s): int(np.sum(data["status"] == s)) for s in range(4)},
        split_counts={k: dict(total=len(ids), direct=int(air[ids].sum()),
                              contact=int((data["status"][ids] == 3).sum()))
                      for k, ids in indices.items()},
        restitution=BASE[0], impact_friction=BASE[1], rolling_resistance=BASE[2],
        horizon_s=cfg.horizon, step_s=cfg.dt,
        references=["https://store.astm.org/stp25348s.html",
                    "https://doi.org/10.1080/02640414.2011.587443"],
    )
    (out / "bounce_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(manifest["counts"], manifest["split_counts"], flush=True)
    baseline_goals = int(in_frame(data["endpoint"][air]).sum())
    rows = {}
    configs = {} if args.skip_sensitivity else SWEEP
    for name, params in {"baseline": BASE, **configs}.items():
        branch, end = (kind, endpoint) if name == "baseline" else simulate(theta, params, cfg)[:2]
        common = hit & (branch > 0)
        delta = end[common] - endpoint[common]
        goal = in_frame(end)
        rows[name] = dict(
            params=params, bounce_crossings=int((branch == 1).sum()),
            rolling_crossings=int((branch == 2).sum()), additional_in_frame=int(goal.sum()),
            total_in_frame=baseline_goals + int(goal.sum()), common_crossings=int(common.sum()),
            median_shift_cm=float(100*np.median(np.linalg.norm(delta, axis=1))) if len(delta) else None,
            median_abs_yz_shift_cm=(100*np.median(np.abs(delta), axis=0)).tolist() if len(delta) else None,
        )
    report = dict(n=len(air), baseline_airborne_in_frame=baseline_goals, scenarios=rows,
                  case_finite_bank=case_audit(data))
    (out / "bounce_sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
