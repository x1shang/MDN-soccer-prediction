"""Paired Bayesian reference forecasts for the case comparison.

Extracted verbatim from student_b/experiments/p1_model_comparison.py so that
diagnostics.py can run without the legacy 50k experiment package.
"""

import time

import numpy as np

from inference import posterior

BAYES_SUBSET_SEED = 155108
BAYES_SUBSET_N = 32
BAYES_ESS = 100
SIGMA = .005


def metrics(draws, truth):
    half = len(draws) // 2
    return dict(error=float(np.linalg.norm(draws.mean(0) - truth)),
                energy=float(np.linalg.norm(draws - truth, axis=1).mean()
                             - .5 * np.linalg.norm(draws[:half] - draws[half:], axis=1).mean()))


def bayesian_forecast(task):
    obs, truth, seed = task
    start = time.perf_counter()
    try:
        result = posterior(n=6000, sigma=SIGMA, seed=seed, observations=obs)
        if result["conditional_ess"] < BAYES_ESS:
            result = posterior(n=18000, sigma=SIGMA, seed=seed + 1, observations=obs)
    except RuntimeError as error:
        return {"failure": str(error), "seconds": time.perf_counter() - start}
    rng = np.random.default_rng(seed)
    draws = result["endpoints"][rng.choice(len(result["weights"]), 4096, p=result["weights"])]
    scored = metrics(draws, truth)
    scored.update(error=float(np.linalg.norm(np.asarray(result["conditional_mean"]) - truth)),
                  ess=result["conditional_ess"], p_cross=result["airborne_probability"],
                  seconds=time.perf_counter() - start)
    return scored
