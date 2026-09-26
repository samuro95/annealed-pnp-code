"""The hyper-parameters of one run.

A run is either ANNEALED,

    sigma_k = eps + (sigma_0 - eps) r^k          ->  Params(K, sigma_0=5, r=0.98, eps=0.01, ...)

or CONSTANT,

    sigma_k = sigma_c                             ->  Params(K, sigma_c=0.5, ...)

and the other fields are the parameters of the algorithm:

    lam        weight of the regularisation (RED-GD, SNORE, ERED)
    gamma_rel  step of the RED algorithms, gamma = gamma_rel / (L_f + lam L_g)
    c_f        step of the PnP algorithms, gamma = c_f / L_f
    q          Robbins-Monro exponent of the decreasing step (SNORE, ERED) or injected noise (SNOPnP)
    warm_mult  the Robbins-Monro decay starts once sigma_k - eps < warm_mult * eps (None = from k = 0)
    tau        SNOPnP's injected noise: 'track' (square-summable) or 'coupled' (tau_k = sigma_k)
    group      symmetry group of ERED: 'hflip' (faces) or 'dihedral' (natural images)

The values used in the paper are written out in the notebook, one line per run.
"""
from dataclasses import dataclass


@dataclass
class Params:
    K: int                      # number of iterations
    sigma_0: float = None       # annealed schedule ...
    r: float = None
    eps: float = None           # ... and its floor
    sigma_c: float = None       # constant noise level (set it instead of sigma_0, r, eps)
    lam: float = None
    gamma_rel: float = 1.0
    c_f: float = None
    q: float = None
    warm_mult: float = None
    tau: str = "track"
    group: str = "hflip"

    @property
    def constant(self):
        return self.sigma_c is not None
