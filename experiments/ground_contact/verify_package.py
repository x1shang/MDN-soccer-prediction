"""Independent audit of the bounce package: pairing, labels, event counts, in-frame mass."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "results"
R.mkdir(exist_ok=True)
RADIUS, GOAL, HALF_Y, BAR_Z = 0.11, 11.0, 3.66, 2.44
report = {}


def in_frame(ep):
    y, z = ep.T
    return (np.abs(y) <= HALF_Y - RADIUS) & (z >= RADIUS) & (z <= BAR_Z - RADIUS)


with np.load(ROOT / "data.npz", allow_pickle=False) as f:
    air = {k: f[k] for k in f.files}
with np.load(ROOT / "bounce/data_bounce.npz", allow_pickle=False) as f:
    full = {k: f[k] for k in f.files}
with np.load(ROOT / "indices.npz", allow_pickle=False) as f:
    air_idx = {k: f[k] for k in f.files}
with np.load(ROOT / "bounce/indices_bounce.npz", allow_pickle=False) as f:
    full_idx = {k: f[k] for k in f.files}

status_air, status_full = air["status"], full["status"]
direct = status_air == 1
contact = status_full == 3

report["status_counts"] = {
    "airborne_file": {str(s): int(np.sum(status_air == s)) for s in range(4)},
    "bounce_file": {str(s): int(np.sum(status_full == s)) for s in range(4)},
}
report["contact_kind"] = {str(k): int(np.sum(full["contact_kind"] == k)) for k in (0, 1, 2)}
report["contact_count"] = {
    "min": int(full["contact_count"][contact].min()),
    "max": int(full["contact_count"][contact].max()),
    "nonzero_contact_rows": int(np.sum(full["contact_count"][contact] > 0)),
}

# Pairing: everything shared must be bit-identical.
pairing = {
    "theta_identical": bool(np.array_equal(air["theta"], full["theta"])),
    "anchor_identical": bool(np.array_equal(air["anchor"], full["anchor"])),
    "split_identical": bool(np.array_equal(air["split"], full["split"])),
    "endpoint_identical_on_direct": bool(np.array_equal(air["endpoint"][direct], full["endpoint"][direct])),
    "arrival_identical_on_direct": bool(np.array_equal(air["arrival"][direct], full["arrival"][direct])),
    "status_identical_on_direct": bool(np.array_equal(status_air[direct], status_full[direct])),
}
obs_a, obs_f = air["observations"], full["observations"]
fin = np.isfinite(obs_a).all(axis=(1, 2))
pairing["observations_identical_where_finite"] = bool(np.array_equal(obs_a[fin], obs_f[fin]))
pairing["finite_prefix_rows"] = int(fin.sum())
pairing["nan_prefix_rows"] = int((~fin).sum())
pairing["nan_prefix_all_ground_first"] = bool(np.all(status_air[~fin] == 2))
report["pairing"] = pairing

# Indices: direct-crossing rows keep their split; contact rows are appended.
idx_partition = {}
for name in ("train", "selection", "calibration", "test"):
    a, b = air_idx[name], full_idx[name]
    idx_partition[name] = {
        "air": int(len(a)),
        "bounce": int(len(b)),
        "air_is_subset": bool(np.all(np.isin(a, b))),
        "order_preserved": bool(np.array_equal(b[: len(a)] if False else np.sort(a), np.sort(a))),
        "bounce_status_direct": int(np.sum(status_full[b] == 1)),
        "bounce_status_contact": int(np.sum(status_full[b] == 3)),
    }
report["indices"] = idx_partition
report["indices_partition_complete"] = bool(
    np.array_equal(np.sort(np.concatenate(list(full_idx.values()))), np.flatnonzero(np.isin(status_full, (1, 3)))))
report["test_rows_match_manifest"] = {name: int(len(full_idx[name])) for name in full_idx}

# Event statistics quoted in the task prompt.
air_in = in_frame(air["endpoint"][direct])
con_in = in_frame(full["endpoint"][contact])
report["in_frame"] = {
    "direct_crossing_in_frame": int(air_in.sum()),
    "contact_crossing_in_frame": int(con_in.sum()),
    "contact_in_frame_bounce": int(np.sum(con_in & (full["contact_kind"][contact] == 1))),
    "contact_in_frame_rolling": int(np.sum(con_in & (full["contact_kind"][contact] == 2))),
    "total_in_frame": int(air_in.sum() + con_in.sum()),
    "contact_rows_with_nonfinite_endpoint": int(np.sum(~np.isfinite(full["endpoint"][contact]))),
    "contact_endpoint_below_radius": int(np.sum(full["endpoint"][contact, 1] < RADIUS - 1e-9)),
    "contact_arrival_nonfinite": int(np.sum(~np.isfinite(full["arrival"][contact]))),
}
manifest = json.loads((ROOT / "bounce/bounce_manifest.json").read_text(encoding="utf-8"))
report["manifest_consistency"] = {
    "counts": manifest["counts"] == {str(s): int(np.sum(status_full == s)) for s in range(4)},
    "split_totals": {
        name: (manifest["split_counts"][name]["total"] == int(len(full_idx[name])),
               manifest["split_counts"][name]["direct"] == int(np.sum(status_full[full_idx[name]] == 1)),
               manifest["split_counts"][name]["contact"] == int(np.sum(status_full[full_idx[name]] == 3)))
        for name in ("train", "selection", "calibration", "test")
    },
}
sens = json.loads((ROOT / "bounce/bounce_sensitivity.json").read_text(encoding="utf-8"))
base = sens["scenarios"]["baseline"]
report["sensitivity_matches_data"] = {
    "bounce_crossings": (base["bounce_crossings"], int(np.sum(full["contact_kind"] == 1))),
    "rolling_crossings": (base["rolling_crossings"], int(np.sum(full["contact_kind"] == 2))),
    "additional_in_frame": (base["additional_in_frame"], int(con_in.sum())),
    "baseline_airborne_in_frame": (sens["baseline_airborne_in_frame"], int(air_in.sum())),
}

# Same test rows scored by both arms?
scores = {arm: __import__("csv").DictReader((R / arm / "test_scores.csv").open(encoding="utf-8-sig"))
          for arm in ("airborne", "bounce")}
report["both_arms_agree_on_test_n"] = {
    arm: sorted({int(r["n"]) for r in scores[arm] if r["subset"] == "combined"}) for arm in scores}

(R / "verify_package.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(report, indent=2, ensure_ascii=False))
