"""Regenerate or scale the frozen UPC155 v3 simulation without source workbooks."""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy

from aiming import SEED, integrate, sha256, transform, write_dataset
from physics import sample_prior
from sampling import sample

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--n", type=int, default=100_000, help="Rows per arm, a multiple of 2500")
    parser.add_argument("--arm", choices=("all", "v3_all", "paired"), default="all")
    args = parser.parse_args()
    if args.n < 2500 or args.n % 2500:
        parser.error("--n must be a positive multiple of 2500 for exact anchor and split quotas")
    prior_path = HERE / "frozen_prior.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    design_path = HERE.parent / "frozen_design.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if prior["sha256"] != design["source_v2_sha256"] or sha256(HERE / "physics.py") != design["simulator_sha256"]:
        raise ValueError("Packaged simulator or source prior differs from the frozen design")
    records = prior["anchors"]
    sealed_ids = [i for i, row in enumerate(records) if row["cohort"] == "asai"]
    eligible = [i for i, row in enumerate(records) if row["cohort"] != "asai"]
    mode = design["frozen_policy"]
    args.output.mkdir(parents=True, exist_ok=False)

    prior_seed = np.random.SeedSequence(prior["seed"]).spawn(2)[0]
    theta, anchor = sample(args.n, np.random.default_rng(prior_seed), records)
    production, aimed = transform(theta, anchor, eligible, np.random.default_rng(SEED + 1), mode)
    meta = dict(
        design, frozen_prior_sha256=sha256(prior_path), frozen_design_sha256=sha256(design_path),
        v2_sampling_seed=prior["seed"], aimed_rows=len(aimed),
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        generator_sha256=sha256(Path(__file__)), sampling_sha256=sha256(HERE / "sampling.py"),
        aiming_sha256=sha256(HERE / "aiming.py"))
    if args.arm in ("all", "v3_all"):
        write_dataset(args.output / "v3_all", production, anchor, records, integrate(production),
                      dict(meta, policy=mode, sealed_asai=[]))
    if args.arm in ("all", "paired"):
        replacement = np.isin(anchor, sealed_ids)
        theta[replacement] = sample_prior(int(replacement.sum()), np.random.default_rng(SEED + 3))
        mapping = {-1: -1, **dict(zip(eligible, range(len(eligible))))}
        reduced_anchor = np.array([mapping.get(int(i), -1) for i in anchor], dtype=np.int16)
        reduced = [records[i] for i in eligible]
        sealed = [dict(id=records[i]["id"], sources=records[i]["sources"]) for i in sealed_ids]
        no_asai = dict(meta, sealed_asai=sealed, asai_replacement="Independent canonical draws",
                       prior_anchor_count=len(reduced))
        write_dataset(args.output / "v2_no_asai", theta, reduced_anchor, reduced, integrate(theta),
                      dict(no_asai, policy="v2", aimed_rows=0))
        theta[aimed] = production[aimed]
        write_dataset(args.output / "v3_no_asai", theta, reduced_anchor, reduced, integrate(theta),
                      dict(no_asai, policy=mode))


if __name__ == "__main__":
    main()
