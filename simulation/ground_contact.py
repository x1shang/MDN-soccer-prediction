"""Paired sensitivity analysis of bounce and roll after first ground contact."""
import numpy as np

from physics import Config, rk4

SHELL_INERTIA = 2 / 3  # I / (m R^2)
ROLL_TRANSITION_SPEED = .50  # Incoming normal speed, m/s


def impact(state, theta, restitution, friction, radius):
    """Resolve normal restitution and a Coulomb-limited tangential impulse."""
    normal = -(1 + restitution) * state[:, 5]
    slip = state[:, 3:5] + radius * theta[:, [4, 3]] * [-1, 1]
    speed = np.linalg.norm(slip, axis=1)
    impulse = -slip * np.minimum(1 / (1 + 1 / SHELL_INERTIA),
                                friction * normal / np.maximum(speed, 1e-15))[:, None]
    state[:, 3:5] += impulse
    state[:, 5] *= -restitution
    theta[:, 3:5] += impulse[:, ::-1] * [1, -1] / (SHELL_INERTIA * radius)


def roll(state, duration, resistance, drag):
    speed = np.linalg.norm(state[:, 3:5], axis=1)
    deceleration = resistance + drag * speed**2
    time = np.minimum(duration, speed / deceleration)
    direction = state[:, 3:5] / np.maximum(speed, 1e-15)[:, None]
    distance = (speed - .5 * deceleration * time) * time
    state[:, :2] += direction * distance[:, None]
    state[:, 3:5] = direction * np.maximum(speed - deceleration * time, 0.)[:, None]
    return state


def continue_ground(theta, restitution, friction, rolling_resistance, cfg):
    """Return crossings of previously ground-first shots: 1 bouncing, 2 rolling."""
    theta = theta.copy()
    n = len(theta)
    state = np.zeros((n, 6))
    state[:, 2] = cfg.radius
    state[:, 3:] = theta[:, :3]
    endpoint = np.full((n, 2), np.nan)
    kind = np.zeros(n, dtype=np.int8)
    observations = np.full((n, 2, 3), np.nan)
    arrival = np.full(n, np.nan)
    contacts = np.zeros(n, dtype=np.int16)
    active = np.ones(n, dtype=bool)
    rolling = np.zeros(n, dtype=bool)
    deceleration = rolling_resistance * cfg.gravity

    drag = cfg.density * np.pi * cfg.radius**2 / (2 * cfg.mass) * theta[:, 6]

    def record(ids, old, new, crossed, is_rolling, start_time, duration):
        sel = ids[crossed]
        start, end = old[crossed], new[crossed]
        fraction = (cfg.goal - start[:, 0]) / (end[:, 0] - start[:, 0])
        endpoint[sel] = start[:, 1:3] + fraction[:, None] * (end[:, 1:3] - start[:, 1:3])
        arrival[sel] = np.broadcast_to(start_time, len(ids))[crossed] + fraction * np.broadcast_to(duration, len(ids))[crossed]
        kind[sel] = np.where(is_rolling[crossed], 2, 1)
        active[sel] = False

    observation_steps = {round(time / cfg.dt): j for j, time in enumerate((.05, .10))}
    for step in range(int(np.ceil(cfg.horizon / cfg.dt))):
        t = step * cfg.dt
        h = min(cfg.dt, cfg.horizon - t)
        if step in observation_steps:
            observations[:, observation_steps[step]] = state[:, :3]
        if not active.any() and step >= max(observation_steps):
            break
        ids = np.flatnonzero(active & rolling)
        old = state[ids]
        new = roll(old.copy(), h, deceleration, drag[ids])
        record(ids, old, new, (old[:, 0] <= cfg.goal) & (new[:, 0] >= cfg.goal)
               & (new[:, 0] > old[:, 0]), np.ones(len(ids), dtype=bool), t, h)
        state[ids] = new
        active[ids[np.linalg.norm(new[:, 3:5], axis=1) <= 1e-12]] = False

        ids = np.flatnonzero(active & ~rolling)
        old = state[ids]
        new = rk4(old, theta[ids], t, h, cfg)
        hit = (new[:, 2] <= cfg.radius) & (new[:, 2] < old[:, 2])
        goal = (old[:, 0] <= cfg.goal) & (new[:, 0] >= cfg.goal) & (new[:, 0] > old[:, 0])
        ag = np.full(len(ids), np.inf)
        az = np.full(len(ids), np.inf)
        ag[goal] = (cfg.goal - old[goal, 0]) / (new[goal, 0] - old[goal, 0])
        az[hit] = (cfg.radius - old[hit, 2]) / (new[hit, 2] - old[hit, 2])
        direct = goal & (ag < az)
        if np.any(contacts[ids[direct]] == 0):
            raise ValueError("Ground-first input reached the plane before its first contact")
        record(ids, old, new, direct, np.zeros(len(ids), dtype=bool), t, h)
        state[ids] = new
        hit &= ~direct
        impacted = ids[hit]
        if not len(impacted):
            continue
        remaining = (1 - az[hit]) * h
        contact = old[hit] + az[hit, None] * (new[hit] - old[hit])
        contact[:, 2] = cfg.radius
        spin = theta[impacted]
        becomes_rolling = (contact[:, 5] > -ROLL_TRANSITION_SPEED) | (restitution == 0)
        impact(contact, spin, np.where(becomes_rolling, 0., restitution), friction, cfg.radius)
        theta[impacted] = spin
        contacts[impacted] += 1
        rolling[impacted] = becomes_rolling
        contact[becomes_rolling, 5] = 0.
        after = contact.copy()
        after[becomes_rolling] = roll(after[becomes_rolling], remaining[becomes_rolling], deceleration, drag[impacted[becomes_rolling]])
        flying = ~becomes_rolling
        after[flying] = rk4(contact[flying], spin[flying],
                            (t + h - remaining[flying])[:, None], remaining[flying, None], cfg)
        record(impacted, contact, after, (contact[:, 0] <= cfg.goal) & (after[:, 0] >= cfg.goal)
               & (after[:, 0] > contact[:, 0]), becomes_rolling, t + h - remaining, remaining)
        state[impacted] = after
    return kind, endpoint, contacts, observations, arrival

