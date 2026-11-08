"""Adaptive importance sampling with explicit prior / proposal correction."""
import numpy as np
from dataclasses import replace
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from physics import LOW, HIGH, NOMINAL_CD, CASE, Config, simulate


def posterior(n=20000, sigma=.005, seed=42, cfg=None, observations=CASE, cd_upper=None):
    if sigma <= 0:
        raise ValueError('sigma must be positive; exact observations need separate treatment')
    rng = np.random.default_rng(seed); cfg = cfg or Config()
    dim = len(LOW) if cfg.fixed_cd is None else len(LOW)-1
    low, high = LOW[:dim], HIGH[:dim]
    if cd_upper is not None:
        if cfg.fixed_cd is not None or cd_upper <= LOW[-1]:
            raise ValueError("cd_upper requires a variable drag coefficient and a valid range")
        high = high.copy()
        high[-1] = cd_upper
    eps = .2

    def physical_parameters(z):
        parameters = low + z * (high-low)
        if cfg.fixed_cd is not None:
            parameters = np.column_stack([parameters, np.full(len(z), cfg.fixed_cd)])
        return parameters
    # Uniform prior in normalized coordinates z in [0,1]^7.
    obs = np.asarray(observations)
    a = (obs[1]-2*obs[0]+[0,0,.11])/.05**2
    v = (obs[0]-[0,0,.11])/.05-.025*a
    mean = np.clip((np.r_[v, 0, 0, 0, NOMINAL_CD][:dim]-low)/(high-low), .01, .99)
    cov = np.diag([.04, .04, .08, .3, .3, .3, .3][:dim])**2
    diagnostics=[]
    for stage in range(4):
        # Each iteration uses a fresh batch; no incorrect pooling of proposals.
        choose = rng.random(n)<eps
        z = rng.multivariate_normal(mean, cov, size=n)
        z[choose] = rng.random((choose.sum(), dim))
        inside = np.all((z>=0)&(z<=1),axis=1)
        logq = np.logaddexp(np.where(inside,np.log(eps),-np.inf),
            np.log1p(-eps)+multivariate_normal.logpdf(z,mean,cov))
        ids = np.flatnonzero(inside)
        theta = physical_parameters(z[inside])
        stage_cfg = cfg if stage == 3 else replace(cfg, horizon=.1)
        sim = simulate(theta,stage_cfg)
        residual = sim['observations']-obs
        logw = np.full(n,-np.inf)
        # Widen only the pilot likelihoods at low noise to locate the posterior.
        # The final importance weights always use the requested observation SD.
        stage_sigma = max(sigma, (.005, .005, .003, 0.)[stage])
        ll = -.5*np.sum(residual**2,axis=(1,2))/stage_sigma**2
        ll[~np.isfinite(ll)] = -np.inf
        logw[ids] = ll-logq[ids]
        if not np.isfinite(logw).any():
            raise RuntimeError('No finite weights. Increase sample count or improve proposal.')
        weights=np.exp(logw-logsumexp(logw))
        ess=1/np.sum(weights**2)
        diagnostics.append(dict(stage=stage,ess=float(ess),max_weight=float(weights.max()),
            best_sample_observation_rmse_m=float(stage_sigma*np.sqrt(-ll.max()/3))))
        if stage < 3:
            mean = weights@z
            centered=z-mean
            cov=2*(centered.T@(weights[:,None]*centered))+np.eye(dim)*1e-5
    w=weights[inside]; event_prob=float(w[sim['status']==1].sum())
    valid=sim['status']==1; endpoints=sim['endpoint'][valid]
    ew=w[valid]
    if not len(ew) or ew.sum()==0: raise RuntimeError('No weighted airborne crossings')
    ew=ew/ew.sum()
    return dict(endpoints=endpoints,weights=ew,theta=theta,simulation=sim,
        parameter_weights=w, diagnostics=diagnostics,
        airborne_probability=event_prob,conditional_ess=float(1/(ew@ew)),
        conditional_mean=(ew@endpoints).tolist(),
        warning='Literature-informed aerodynamics with provisional launch prior; finite-sample approximation. '
                'Low ESS requires more samples or improved proposal; compare independent seeds.')
