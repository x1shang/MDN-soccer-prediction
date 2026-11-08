"""Paired per-row analysis on the shared 10,000 test rows (both arms, identical noise)."""
import json
from pathlib import Path

import numpy as np
import torch

import train_bounce_control_mdn as T

R = Path("results")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full = np.load("bounce/data_bounce.npz", allow_pickle=False)
fidx = np.load("bounce/indices_bounce.npz", allow_pickle=False)
eval_ids = fidx["test"]
eval_obs = full["observations"][eval_ids].reshape(-1, 6)
eval_target = full["endpoint"][eval_ids]
eval_status = full["status"][eval_ids]

models = {}
for arm in ("airborne", "bounce"):
    ck = torch.load(R / arm / "model.pt", map_location=device, weights_only=False)
    m = T.WienerResidualMDN(in_dim=22, hidden_dim=192, depth=3, k=5).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    cfg = json.loads((R / arm / "config.json").read_text(encoding="utf-8"))
    models[arm] = (m, ck["stats"], cfg)

masks = {
    "combined": np.ones(len(eval_ids), bool),
    "airborne": eval_status == 1,
    "post_contact": eval_status == 3,
}

per_row = {arm: {"err": {}, "err_proj": {}, "nll": {}} for arm in models}
replication = {}

for sig_mm in (2, 5, 10, 15):
    sig = sig_mm / 1000.0
    noise = np.random.default_rng(3000 + sig_mm).normal(size=eval_obs.shape)
    obs_noisy = eval_obs + noise * sig
    s_vec = np.full((len(eval_obs), 1), sig)
    feats, anc = T.extract_wiener_features(obs_noisy, s_vec)
    y_t = torch.tensor(eval_target, dtype=torch.float32, device=device)
    for arm, (m, stats, cfg) in models.items():
        f_t = torch.tensor((feats - stats["xm"]) / stats["xs"], device=device)
        anc_t = torch.tensor(anc, device=device)
        scale = cfg["calibration"][str(sig_mm)]["scale_sd"]
        with torch.no_grad():
            dist = m(f_t, anc_t, scale_sd=scale)
            nll = -dist.log_prob(y_t).cpu().numpy()
            pi = dist.mixture_distribution.probs.unsqueeze(-1)
            mean_pred = torch.sum(pi * dist.component_distribution.loc, dim=1).cpu().numpy()
        err_raw = np.linalg.norm(mean_pred - eval_target, axis=1)
        projected = mean_pred.copy()
        projected[:, 1] = np.maximum(projected[:, 1], T.R_BALL)
        err_proj = np.linalg.norm(projected - eval_target, axis=1)
        per_row[arm]["err"][sig_mm] = err_raw
        per_row[arm]["err_proj"][sig_mm] = err_proj
        per_row[arm]["nll"][sig_mm] = nll
        # Replication check against the pipeline's own test_scores.csv
        rows = __import__("csv").DictReader((R / arm / "test_scores.csv").open(encoding="utf-8-sig"))
        ref = {(r["subset"], int(r["sigma_mm"])): r for r in rows}
        for name, mask in masks.items():
            replication[(arm, name, sig_mm)] = (
                float(np.median(err_raw[mask])), float(ref[(name, sig_mm)]["median_m"]),
                float(np.percentile(err_raw[mask], 90)), float(ref[(name, sig_mm)]["p90_m"]),
                float(np.mean(nll[mask])), float(ref[(name, sig_mm)]["nll"]),
                float(np.median(err_proj[mask])), float(ref[(name, sig_mm)]["projected_median_m"]))

max_dev_full = 0.0
worst = None
for key, tup in replication.items():
    for a, b in ((tup[0], tup[1]), (tup[2], tup[3]), (tup[4], tup[5]), (tup[6], tup[7])):
        if abs(a - b) > max_dev_full:
            max_dev_full, worst = abs(a - b), (key, a, b)
print(f"[replication] per-row recomputation vs test_scores.csv, max abs deviation = {max_dev_full:.3e} {worst}")

rng = np.random.default_rng(20261108)
n_boot = 4000
lines = []
print("\nPaired bootstrap (4000 resamples) on the shared test rows; delta = bounce - airborne")
for name, mask in masks.items():
    idx_mask = np.flatnonzero(mask)
    n = len(idx_mask)
    boot = rng.integers(0, n, size=(n_boot, n))
    for sig_mm in (2, 5, 10, 15):
        ea, eb = per_row["airborne"]["err"][sig_mm][mask], per_row["bounce"]["err"][sig_mm][mask]
        pa, pb = per_row["airborne"]["err_proj"][sig_mm][mask], per_row["bounce"]["err_proj"][sig_mm][mask]
        na, nb = per_row["airborne"]["nll"][sig_mm][mask], per_row["bounce"]["nll"][sig_mm][mask]
        d_med = np.median(eb[boot], axis=1) - np.median(ea[boot], axis=1)
        d_proj = np.median(pb[boot], axis=1) - np.median(pa[boot], axis=1)
        d_nll = nb[boot].mean(axis=1) - na[boot].mean(axis=1)
        better = float(np.mean(eb < ea))
        better_proj = float(np.mean(pb < pa))
        lines.append(dict(subset=name, sigma_mm=sig_mm, n=n,
                          median_air=float(np.median(ea)), median_bounce=float(np.median(eb)),
                          d_median=float(np.median(eb) - np.median(ea)),
                          d_median_lo=float(np.percentile(d_med, 2.5)),
                          d_median_hi=float(np.percentile(d_med, 97.5)),
                          projected_median_air=float(np.median(pa)),
                          projected_median_bounce=float(np.median(pb)),
                          d_projected=float(np.median(pb) - np.median(pa)),
                          d_projected_lo=float(np.percentile(d_proj, 2.5)),
                          d_projected_hi=float(np.percentile(d_proj, 97.5)),
                          d_nll=float(nb.mean() - na.mean()),
                          d_nll_lo=float(np.percentile(d_nll, 2.5)),
                          d_nll_hi=float(np.percentile(d_nll, 97.5)),
                          frac_rows_bounce_better=better,
                          frac_rows_bounce_better_projected=better_proj))
        print(f"{name:13s} sigma={sig_mm:2d} n={n:5d}  median {np.median(ea):.3f} -> {np.median(eb):.3f} "
              f"delta={lines[-1]['d_median']:+.3f} [{lines[-1]['d_median_lo']:+.3f},{lines[-1]['d_median_hi']:+.3f}]  "
              f"proj {np.median(pa):.3f} -> {np.median(pb):.3f} delta={lines[-1]['d_projected']:+.3f} "
              f"[{lines[-1]['d_projected_lo']:+.3f},{lines[-1]['d_projected_hi']:+.3f}]  "
              f"dNLL={lines[-1]['d_nll']:+.3f}  rows_better={better:.3f}/{better_proj:.3f}")

(R / "paired_bootstrap.json").write_text(json.dumps({
    "note": "delta = bounce arm minus airborne arm; interval = 2.5/97.5 percentile of the paired bootstrap.",
    "replication_max_abs_deviation": max_dev_full,
    "rows": lines}, indent=2), encoding="utf-8")
print(f"\nwrote {R / 'paired_bootstrap.json'}")
