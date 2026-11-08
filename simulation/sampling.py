from collections import Counter
import numpy as np
from scipy.stats import truncnorm
from physics import HIGH, LOW, sample_prior

LAUNCH_COHORTS = {"asai", "dfl_penalty"}
PERTURB = np.array([1.5, 2.0, 1.2, 6.0, 6.0, 6.0, .025])
CORE = np.array([1.8, 2.4, 1.5, 7.5, 7.5, 7.5, .025])
WIDE = np.array([3., 3.8, 2.5, 12., 12., 12., .035])
NON_LAUNCH = [(28.5, 3.2), (0., 6.5), (5.5, 2.8)]
NAMES = ("train", "selection", "calibration", "test")
WEIGHTS = np.array([80, 5, 5, 10])
COHORT_PERCENT = {"canonical": 40, "dfl_penalty": 24, "asai": 12, "aero_only": 24}


def sample(n, rng, records):
    counts = Counter(row["cohort"] for row in records if row["used"])
    if counts != Counter(asai=3, dfl_penalty=2, dfl_extra=11, issia=3, barber=1):
        raise ValueError("The exact v2 quotas require all 20 eligible anchors")
    allocation = [(-1, n * 40 // 100)]
    for i, row in enumerate(records):
        cohort = row["cohort"]
        if cohort in LAUNCH_COHORTS:
            size = n * COHORT_PERCENT[cohort] // (100 * counts[cohort])
        else:
            size = n * 24 // 1500
        allocation.append((i, size))
    anchor = np.concatenate([np.full(size, i, dtype=np.int16) for i, size in allocation])
    theta = sample_prior(n, rng)
    for i, size in allocation[1:]:
        rows = np.flatnonzero(anchor == i)
        record = records[i]
        fitted = np.asarray(record["theta"])
        draws = np.broadcast_to(fitted, (size, 7)).copy()
        if record["use_launch"]:
            wide = rng.permutation(size) < size * 3 // 10
            sd = np.where(wide[:, None], WIDE, CORE)
            theta[rows] = draws + rng.normal(size=draws.shape) * sd
            continue
        for axis, (mu, sigma) in enumerate(NON_LAUNCH):
            lo, hi = LOW[axis], HIGH[axis]
            indices = rows if axis else rows[rng.permutation(size)[:size // 2]]
            theta[indices, axis] = truncnorm.rvs(
                (lo - mu) / sigma, (hi - mu) / sigma, loc=mu, scale=sigma,
                size=len(indices), random_state=rng)
        yaw = np.arctan2(theta[rows, 1], theta[rows, 0]) - np.arctan2(fitted[1], fitted[0])
        draws[:, 3] = np.cos(yaw) * fitted[3] - np.sin(yaw) * fitted[4]
        draws[:, 4] = np.sin(yaw) * fitted[3] + np.cos(yaw) * fitted[4]
        theta[rows, 3:] = draws[:, 3:] + rng.normal(size=(size, 4)) * PERTURB[3:]
    np.clip(theta, LOW, HIGH, out=theta)
    theta[:, [1, 3, 5]] *= rng.choice((-1., 1.), size=n)[:, None]
    order = rng.permutation(n)
    return theta[order], anchor[order]


def partition(anchor, valid, rng):
    split = np.full(len(anchor), -1, dtype=np.int8)
    groups = {name: [] for name in NAMES}
    for a in np.unique(anchor):
        ids = np.flatnonzero(anchor == a)
        good = rng.permutation(ids[valid[ids]])
        other = rng.permutation(ids[~valid[ids]])
        raw_sizes = len(ids) * WEIGHTS // 100
        counts, remainder = np.divmod(len(good) * WEIGHTS, 100)
        counts[np.argsort(-remainder, kind="stable")[:len(good) - counts.sum()]] += 1
        remaining = raw_sizes - counts
        if np.any(remaining < 0) or remaining.sum() != len(other):
            raise RuntimeError(f"Cannot satisfy raw split quotas for anchor {a}")
        for j, (g, o) in enumerate(zip(np.split(good, np.cumsum(counts)[:-1]),
                                       np.split(other, np.cumsum(remaining)[:-1]))):
            groups[NAMES[j]].append(g)
            split[g] = split[o] = (0, 1, 1, 2)[j]
    indices = {name: np.sort(np.concatenate(groups[name])) for name in NAMES}
    np.testing.assert_array_equal(np.sort(np.concatenate(list(indices.values()))), np.flatnonzero(valid))
    if np.any(split < 0):
        raise RuntimeError("Unassigned simulation split")
    return split, indices
