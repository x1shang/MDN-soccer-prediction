"""Vectorized flight simulator. SI units; theta = vx,vy,vz,wx,wy,wz,Cd."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Config:
    mass: float = .43
    radius: float = .11
    density: float = 1.20
    gravity: float = 9.81
    goal: float = 11.
    dt: float = .002
    horizon: float = 1.5
    wind_y: float = 0.
    spin_decay: float = float('inf')
    lift_scale: float = 1.
    fixed_cd: float | None = None
    # Validation-only override used by A's self-test case (constant C_L model form).
    # Never set in training/evaluation experiments.
    fixed_cl: float | None = None


LOW = np.array([18., -12., .5, -60., -60., -60., .15])
HIGH = np.array([35., 12., 12., 60., 60., 60., .30])
NOMINAL_CD = .22


def sample_prior(n, rng):
    """Independent uniform sensitivity prior; launch-state bounds remain provisional."""
    return rng.uniform(LOW, HIGH, (n, 7))


def acceleration(v, theta, t, cfg):
    u = v - np.array([0., cfg.wind_y, 0.])
    speed = np.linalg.norm(u, axis=1)
    omega = theta[:, 3:6] * np.exp(-t / cfg.spin_decay)
    cross = np.cross(omega, u)
    norm = np.linalg.norm(cross, axis=1)
    # Rotational speed product: |omega| |u|. Used to detect the zero-cross
    # case (omega parallel to u, or zero spin/velocity): the lift direction is
    # undefined there, and the lift must be exactly zero (no 0/0, no blow-up).
    scale = np.linalg.norm(omega, axis=1) * speed
    valid = norm > 1e-9 * scale
    # |omega x u| / |u| is the transverse spin magnitude.
    S = cfg.radius * np.where(valid, norm, 0.) / np.maximum(speed**2, 1e-24)
    cl = (cfg.fixed_cl if cfg.fixed_cl is not None
          else cfg.lift_scale * np.minimum(1.4 * S, .30))
    direction = np.where(valid[:, None], cross / np.maximum(norm[:, None], 1e-24), 0.)
    k = cfg.density * np.pi * cfg.radius**2 / (2 * cfg.mass)
    cd = theta[:, 6] if cfg.fixed_cd is None else np.full(len(theta), cfg.fixed_cd)
    return (-k * cd[:, None] * speed[:, None] * u
            + k * (cl * speed**2)[:, None] * direction
            + np.array([0., 0., -cfg.gravity]))


def derivative(state, theta, t, cfg):
    return np.column_stack((state[:, 3:], acceleration(state[:, 3:], theta, t, cfg)))


def rk4(state, theta, t, h, cfg):
    a = derivative(state, theta, t, cfg)
    b = derivative(state + h*a/2, theta, t+h/2, cfg)
    c = derivative(state + h*b/2, theta, t+h/2, cfg)
    d = derivative(state + h*c, theta, t+h, cfg)
    return state + h*(a+2*b+2*c+d)/6


def simulate(theta, cfg=None, *, observation_times=(.05, .1), initial_z=None, goal_x=None):
    """status: 0 timeout, 1 airborne goal-plane crossing, 2 ground.

    Observation times are hit exactly. Events use linear interpolation (not
    exact roots); assess event accuracy by step refinement. Invalid labels NaN.
    """
    cfg = cfg or Config()
    theta = np.atleast_2d(np.asarray(theta, float))
    t1, t2 = observation_times
    if not np.isfinite((t1, t2, cfg.dt, cfg.horizon)).all() or not (
            cfg.dt > 0 and 0 < t1 < t2 <= cfg.horizon):
        raise ValueError('Require positive dt and 0 < t1 < t2 <= horizon')
    n = len(theta)
    height = np.broadcast_to(cfg.radius if initial_z is None else initial_z, (n,))
    plane = np.broadcast_to(cfg.goal if goal_x is None else goal_x, (n,))
    if not np.isfinite(height).all() or not np.isfinite(plane).all() or np.any(height < cfg.radius) or np.any(plane <= 0):
        raise ValueError('Require finite start heights at or above contact and positive goal distances')
    state = np.zeros((n, 6)); state[:, 2] = height; state[:, 3:] = theta[:, :3]
    observations = np.full((n, 2, 3), np.nan)
    endpoint = np.full((n, 2), np.nan); arrival = np.full(n, np.nan)
    status = np.zeros(n, np.int8); active = np.ones(n, bool)
    t = 0.; targets = [t1, t2, cfg.horizon]
    for j, target in enumerate(targets):
        while t < target - 1e-12:
            h = min(cfg.dt, target-t)
            idx = np.flatnonzero(active)
            if len(idx):
                old = state[idx]; new = rk4(old, theta[idx], t, h, cfg)
                local_plane = plane[idx]
                goal = (old[:, 0] < local_plane) & (new[:, 0] >= local_plane)
                ground = (new[:, 2] <= cfg.radius) & (new[:, 2] < old[:, 2])
                ag = np.full(len(idx), np.inf); az = ag.copy()
                ag[goal] = (local_plane[goal]-old[goal, 0])/(new[goal, 0]-old[goal, 0])
                az[ground] = (cfg.radius-old[ground, 2])/(new[ground, 2]-old[ground, 2])
                wins = goal & (ag < az)
                hits = ground & ~wins
                endpoint[idx[wins]] = (old[wins, 1:3]
                    + ag[wins, None]*(new[wins, 1:3]-old[wins, 1:3]))
                arrival[idx[wins]] = t + h*ag[wins]
                status[idx[wins]] = 1; status[idx[hits]] = 2
                active[idx[wins | hits]] = False
                state[idx] = new
            t += h
        if j < 2:
            observations[active, j] = state[active, :3]
    return dict(observations=observations, endpoint=endpoint, arrival=arrival, status=status)


CASE = np.array([[1.5, .45, .44], [2.96, .86, .74]])


def quadratic_baseline(obs):
    obs = np.asarray(obs)
    r0 = np.array([0., 0., .11])
    a = (obs[..., 1, :] - 2*obs[..., 0, :] + r0)/.05**2
    v = (obs[..., 0, :]-r0)/.05 - .025*a
    roots = np.roots([.5*a[0], v[0], -11.])
    roots = [r.real for r in roots if abs(r.imag)<1e-10 and r.real>.1]
    t = min(roots) if roots else np.nan
    return dict(time=float(t), endpoint=(r0+v*t+.5*a*t*t)[1:].tolist())
