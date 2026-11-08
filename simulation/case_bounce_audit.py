"""Estimate the extra in-frame probability using adaptive importance sampling."""
import argparse
import json
from pathlib import Path

import numpy as np

from generate_bounce_dataset import BASE, SWEEP, in_frame, simulate
from inference import posterior
from physics import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples", type=int, default=30_000, help="Proposals per stage per seed")
    args = parser.parse_args()
    if args.samples < 1000:
        parser.error("At least 1000 samples are required")
    rows = []
    for mm in (2, 5, 10, 15):
        for seed in (155313, 155314):
            post = posterior(n=args.samples, sigma=mm / 1000, seed=seed)
            air = post["simulation"]
            w = post["parameter_weights"]
            direct = air["status"] == 1
            contact = air["status"] == 2
            positive = contact & (w > 0)
            p_direct = float(w[direct & in_frame(air["endpoint"])].sum())
            extras = {}
            for name, params in {"baseline": BASE, **SWEEP}.items():
                _, endpoints, *_ = simulate(post["theta"][positive], params, Config(horizon=3.))
                extra = float(w[positive][in_frame(endpoints)].sum())
                extras[name] = dict(additional_in_frame=extra, total_in_frame=p_direct + extra)
            row = dict(sigma_mm=mm, seed=seed, ess=post["diagnostics"][-1]["ess"],
                       direct_crossing=float(w[direct].sum()),
                       ground_first_mass=float(w[contact].sum()), positive_ground_rows=int(positive.sum()),
                       direct_in_frame=p_direct, scenarios=extras)
            rows.append(row)
            print(mm, seed, "ESS", round(row["ess"]), "ground mass", row["ground_first_mass"], flush=True)
    report = dict(prior="Canonical uniform launch prior; different from the v3 mixture",
                  likelihood="Independent Gaussian coordinate errors. The two frames are modelled before first contact.",
                  scope="Finite importance sample. Numerical zero does not establish a probability bound.",
                  proposals_per_stage=args.samples, rows=rows)
    with args.output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
