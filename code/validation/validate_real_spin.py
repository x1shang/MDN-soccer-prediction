"""Prefix-only importance sampling for measured football flight segments."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from physics import Config, rk4
from validate_real_penalties import read_csv, save_json


def advance(state, theta, start, stop, cfg):
    count = max(1, int(np.ceil((stop-start) / cfg.dt - 1e-10)))
    h = (stop-start) / count
    for step in range(count):
        state = rk4(state, theta, start+step*h, h, cfg)
    return state


def sample_prefix(times, obs, axes, sigma, count, seed, rotating, cfg, *, aero_draws=None):
    """Draw a Gaussian proposal for state conditional on each prior aero draw."""
    rng = np.random.default_rng(seed)
    if aero_draws is None:
        theta = np.zeros((count, 7))
        theta[:, 6] = rng.uniform(.15, .30, count)
        if rotating:
            # Planar Barber model uses spin perpendicular to the observed x-z plane.
            spin_axes = [1] if len(axes) == 2 else [0, 1, 2]
            theta[:, 3+np.array(spin_axes)] = rng.uniform(-60, 60, (count, len(spin_axes)))
    else:
        theta = np.asarray(aero_draws, dtype=float)
        if theta.shape != (count, 7) or not np.isfinite(theta).all():
            raise ValueError('Expected finite aerodynamic draws of shape (count, 7)')
    design = np.column_stack((np.ones(len(times)), times))
    inverse = np.linalg.pinv(design)
    initial = np.zeros((count, 6))
    initial[:, np.r_[axes, axes+3]] = (inverse @ obs).ravel()
    nominal = initial.copy()
    offsets = []
    previous = 0.
    for t in times:
        nominal = advance(nominal, theta, previous, t, cfg)
        offsets.append(nominal[:, axes] - initial[:, axes] - t*initial[:, axes+3])
        previous = t
    mean = np.einsum('ij,njk->nik', inverse, obs-np.stack(offsets, axis=1))
    covariance = sigma**2 * np.linalg.inv(design.T @ design)
    standard = rng.normal(size=mean.shape)
    deviation = np.einsum('ij,njk->nik', np.linalg.cholesky(covariance), standard)
    proposed = mean+deviation
    initial[:, axes] = proposed[:, 0]
    initial[:, axes+3] = proposed[:, 1]
    state = initial.copy()
    rss = np.zeros(count)
    previous = 0.
    valid = (initial[:, 2] > cfg.radius)
    for t, position in zip(times, obs):
        state = advance(state, theta, previous, t, cfg)
        rss += np.sum((state[:, axes]-position)**2, axis=1)
        valid &= state[:, 2] > cfg.radius
        previous = t
    valid &= (np.abs(proposed[:, 0]-obs[0]) <= 1.).all(axis=1)
    valid &= (np.abs(proposed[:, 1]) <= 60.).all(axis=1)
    # Aero parameters were proposed from their priors. Uniform state prior
    # within the stated bounds leaves the likelihood divided by Gaussian q.
    log_weight = -.5*rss/sigma**2 + .5*np.sum(standard**2, axis=(1, 2))
    if not valid.any():
        raise ValueError('No accepted proposal')
    weights = softmax(np.where(valid, log_weight, -np.inf))
    return state, theta, weights


def forecast(state, theta, weights, start, times, target, cfg):
    alive = weights > 0
    goal = np.full((len(weights), 2), np.nan)
    predictions = []
    t = start
    for end in times:
        while t < end-1e-12:
            h = min(cfg.dt, end-t)
            indices = np.flatnonzero(alive)
            old = state[indices]
            new = rk4(old, theta[indices], t, h, cfg)
            ground = (new[:, 2] <= cfg.radius) & (new[:, 2] < old[:, 2])
            if target is not None:
                cross = (old[:, 0] < target) & (new[:, 0] >= target)
                fraction = np.full(len(indices), np.inf)
                fraction[cross] = (target-old[cross, 0])/(new[cross, 0]-old[cross, 0])
                ground_fraction = np.full(len(indices), np.inf)
                ground_fraction[ground] = (cfg.radius-old[ground, 2])/(new[ground, 2]-old[ground, 2])
                wins = cross & (fraction < ground_fraction) & np.isnan(goal[indices, 0])
                goal[indices[wins]] = old[wins, 1:3]+fraction[wins, None]*(new[wins, 1:3]-old[wins, 1:3])
            alive[indices[ground]] = False
            state[indices] = new
            t += h
        mass = weights[alive].sum()
        mean = np.average(state[alive, :3], axis=0, weights=weights[alive]).tolist() if mass > 0 else None
        predictions.append({'time_s': float(end), 'airborne_mass': float(mass), 'mean_xyz_m': mean})
    return predictions, goal


def arrays(rows, axes):
    keys = ['xyz'[i]+'_m' for i in axes]
    return np.array([float(r['t_s']) for r in rows]), np.array([[float(r[k]) for k in keys] for r in rows])


def marginal_intervals(points, weights):
    intervals = []
    for coordinate in points.T:
        order = np.argsort(coordinate)
        cumulative = np.cumsum(weights[order])
        intervals.append(np.interp([.025, .975], cumulative/cumulative[-1], coordinate[order]).tolist())
    return intervals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument('--data', type=Path)
    sources.add_argument('--dfl-positions', type=Path, help='Screened DFL positions.csv with input/held_out roles')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--samples', type=int, default=16000)
    parser.add_argument('--step', type=float, default=.005)
    args = parser.parse_args()
    if args.samples < 2 or args.step <= 0:
        parser.error('samples must be >=2 and step positive')
    out = args.out or root/'experiments/validation'/('dfl_screened_spin' if args.dfl_positions else 'real_spin_final')
    out.mkdir(parents=True, exist_ok=True)
    if args.dfl_positions:
        # The provider table contains both roles. Only the input rows enter the fit.
        prefix = [r for r in read_csv(args.dfl_positions) if r['role'] == 'input']
        config = {case: {'score_axes': list('xyz'), 'target_local_x_m': None}
                  for case in dict.fromkeys(r['case_id'] for r in prefix)}
        source_files = [args.dfl_positions]
        if not config:
            raise ValueError('No DFL input segments found')
    else:
        data = args.data or root/'data/anchors/dfl/flight_trajectory_validation_2026-11-07'
        config = json.loads((data/'case_config.json').read_text(encoding='utf-8-sig'))
        prefix = read_csv(data/'inputs_only.csv')
        source_files = [data/name for name in ('inputs_only.csv', 'held_out.csv', 'case_config.json')]
    cfg = Config(dt=args.step)
    fitted = {}
    for case, spec in config.items():
        axes = np.array(['xyz'.index(a) for a in spec['score_axes']])
        times, obs = arrays([r for r in prefix if r['case_id'] == case], axes)
        if len(times) != 3 or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f'{case}: expected three chronological prefix points')
        sigmas = (.067, .100) if len(axes) == 2 else (.015, .030)
        for sigma in sigmas:
            for seed in (8047, 20261107):
                for rotating in (False, True):
                    fitted[case, sigma, seed, rotating] = sample_prefix(times, obs, axes, sigma, args.samples, seed, rotating, cfg)
    # Only score after all state and aerodynamic posterior draws are fixed.
    if args.dfl_positions:
        held = [r for r in read_csv(args.dfl_positions) if r['role'] == 'held_out']
    else:
        held = read_csv(data/'held_out.csv')
    if set(config) != {row['case_id'] for row in held}:
        raise ValueError('Each input segment must have matching held-out rows')
    records = []
    for (case, sigma, seed, rotating), (state, theta, weights) in fitted.items():
        spec = config[case]
        axes = np.array(['xyz'.index(a) for a in spec['score_axes']])
        last = max(float(r['t_s']) for r in prefix if r['case_id'] == case)
        times, truth = arrays([r for r in held if r['case_id'] == case], axes)
        if not np.all(np.diff(times) > 0) or times[0] <= last or not np.isfinite(truth).all():
            raise ValueError(f'{case}: invalid held-out times or positions')
        target = spec['target_local_x_m']
        reference = None
        if target is not None:
            bracket, = np.flatnonzero((truth[:-1, 0] < target) & (truth[1:, 0] >= target))
            fraction = (target-truth[bracket, 0])/(truth[bracket+1, 0]-truth[bracket, 0])
            reference = (truth[bracket]+fraction*(truth[bracket+1]-truth[bracket]))[1:]
            keep = times < times[bracket]+fraction*(times[bracket+1]-times[bracket])
            times, truth = times[keep], truth[keep]
        query = np.r_[times, 1.5] if target is not None else times
        forecasts, goals = forecast(state, theta, weights, last, query, target, cfg)
        forecasts = forecasts[:len(times)]
        available = np.array([p['mean_xyz_m'] is not None for p in forecasts])
        predicted = np.array([p['mean_xyz_m'] for p in forecasts if p['mean_xyz_m'] is not None]).reshape(-1, 3)
        residual = predicted[:, axes]-truth[available]
        crossing = np.isfinite(goals[:, 0]) & (weights > 0)
        goal_mass = weights[crossing].sum()
        mean_goal = np.average(goals[crossing], axis=0, weights=weights[crossing]) if goal_mass > 0 else None
        goal_ess = goal_mass**2/np.sum(weights[crossing]**2) if goal_mass > 0 else None
        records.append(dict(case=case, sigma_m=sigma, seed=seed, spin=rotating, axes=spec['score_axes'],
            ess=float(1/np.sum(weights**2)), max_weight=float(weights.max()),
            goal_ess=float(goal_ess) if goal_ess is not None else None,
            frames_scored=int(available.sum()), frames_total=len(times),
            rmse_m=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))) if available.any() else None,
            goal_probability=float(goal_mass) if target is not None else None,
            goal_mean_local_yz_m=mean_goal.tolist() if mean_goal is not None else None,
            goal_marginal_95_local_yz_m=marginal_intervals(goals[crossing], weights[crossing]) if goal_mass > 0 else None,
            goal_reference_local_yz_m=reference.tolist() if reference is not None else None,
            goal_error_m=float(np.linalg.norm(mean_goal-reference)) if mean_goal is not None else None,
            frame_predictions=forecasts))
    save_json(out/'scores.json', {'samples_per_fit': args.samples, 'dt_max_s': args.step,
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        'prior': 'Cd U[.15,.30]; spin U[-60,60] per active axis or zero; initial position within 1 m per coordinate of observed origin and above ground; velocity +/-60 m/s per coordinate',
        'records': records})
    for case in config:
        for row in (r for r in records if r['case'] == case):
            print(case, row['sigma_m'], row['seed'], row['spin'],
                  *(f'{row[k]:.4f}' if row[k] is not None else '-' for k in ('ess','rmse_m','goal_error_m')),
                  min(p['airborne_mass'] for p in row['frame_predictions']))


if __name__ == '__main__':
    main()
