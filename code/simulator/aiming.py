import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import numpy as np
from physics import CASE, Config, HIGH, LOW, simulate
from sampling import partition

SEED = 155313
AIM_FRACTION = .65
AIM_Y = (-3.85, 3.85)
AIM_Z = (.55, 2.65)
MIN_SPEED = 24.
GOAL = 11.
CFG = Config()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def transform(theta, anchor, eligible, rng, mode):
    result = theta.copy()
    candidates = np.flatnonzero(np.isin(anchor, eligible) & (theta[:, 0] >= MIN_SPEED))
    ids = rng.permutation(candidates)[:round(AIM_FRACTION * len(candidates))]
    if mode == "v2":
        return result, np.empty(0, dtype=int)
    draws = result[ids]
    original = draws[:, :3].copy()
    time = GOAL / original[:, 0]
    draws[:, 1] = rng.uniform(*AIM_Y, size=len(ids)) / time
    draws[:, 2] = (rng.uniform(*AIM_Z, size=len(ids)) - CFG.radius) / time + .5 * CFG.gravity * time
    draws[:, :3] = np.clip(draws[:, :3], LOW[:3], HIGH[:3])
    # Transport axial spin along the changed horizontal heading.
    yaw = np.arctan2(draws[:, 1], draws[:, 0]) - np.arctan2(original[:, 1], original[:, 0])
    wx, wy = draws[:, 3:5].T.copy()
    draws[:, 3] = np.cos(yaw) * wx - np.sin(yaw) * wy
    draws[:, 4] = np.sin(yaw) * wx + np.cos(yaw) * wy
    if mode == "aim-spin":
        velocity = draws[:, :3]
        spin = draws[:, 3:6]
        unit = velocity / np.linalg.norm(velocity, axis=1, keepdims=True)
        spin -= np.sum(spin * unit, axis=1, keepdims=True) * unit
    result[ids] = np.clip(draws, LOW, HIGH)
    return result, ids


def integrate(theta, pool=None):
    chunks = (theta[i:i+2500] for i in range(0, len(theta), 2500))
    pieces = ([simulate(chunk, CFG) for chunk in chunks] if pool is None else
              list(pool.map(simulate, chunks, repeat(CFG), chunksize=2)))
    return {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}


def generic_metrics(theta, data):
    valid = data["status"] == 1
    yz = data["endpoint"]
    fast = valid & (theta[:, 0] >= MIN_SPEED)
    cells, _, _ = np.histogram2d(yz[fast, 0], yz[fast, 1],
                                 bins=[np.linspace(-3.66, 3.66, 7), np.linspace(.11, 2.44, 4)])
    return dict(airborne_rate=float(valid.mean()), fast_goal_count=int(cells.sum()),
                fast_goal_min_cell=int(cells.min()), fast_goal_grid=cells.astype(int).tolist())


def write_dataset(path, theta, anchor, records, data, meta):
    path.mkdir(parents=True, exist_ok=False)
    valid = data["status"] == 1
    if len(np.unique(theta, axis=0)) != len(theta):
        raise ValueError("Invalid or duplicate synthetic rows")
    if not all(np.isfinite(data[key][valid]).all() for key in ("observations", "endpoint", "arrival")):
        raise ValueError("Nonfinite airborne observations or labels")
    if np.any(data["endpoint"][valid, 1] < CFG.radius) or np.any(data["arrival"][valid] <= .1):
        raise ValueError("Airborne endpoints must be above ground and after both observations")
    split, groups = partition(anchor, valid, np.random.default_rng(SEED + 2))
    dpath, ipath = path / "data.npz", path / "indices.npz"
    np.savez_compressed(dpath, theta=theta, anchor=anchor, split=split, **data)
    np.savez_compressed(ipath, **groups)
    distance = np.linalg.norm((data["observations"][valid] - CASE).reshape(-1, 6), axis=1)
    near10 = int((distance < .10).sum())
    manifest = dict(meta, anchors=records, n=len(theta), sha256=sha256(dpath),
                    indices_sha256=sha256(ipath),
                    index_counts={name: len(idx) for name, idx in groups.items()},
                    generic=generic_metrics(theta, data),
                    quality=dict(crossing_rate=float(valid.mean()), crossing_pass=float(valid.mean()) > .85,
                                 near_case_10cm=near10, near_case_pass=near10 >= 150,
                                 near_case_5cm=int((distance < .05).sum())))
    (path / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path.name, manifest["quality"], flush=True)
