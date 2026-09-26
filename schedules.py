"""Noise levels sigma_k and step sizes gamma_k.

The paper's schedule is geometric with a positive floor eps:

    sigma_k = eps + (sigma_0 - eps) * r**k        (0 < r < 1)

It decreases strictly, converges to eps and never reaches it.  The constant-sigma
baseline is simply sigma_k = sigma_c for every k.
"""
import numpy as np


def annealed_sigmas(K, sigma_0, eps, r):
    """sigma_k = eps + (sigma_0 - eps) r^k, k = 0 ... K-1."""
    k = np.arange(K)
    return eps + (sigma_0 - eps) * r ** k


def constant_sigmas(K, sigma_c):
    """sigma_k = sigma_c for every k (the baseline we compare against)."""
    return np.full(K, float(sigma_c))


def warm_start_length(sigmas, eps, warm_mult):
    """Number of first iterations where sigma_k - eps > warm_mult * eps.

    The stochastic algorithms keep a constant step while the noise level is still
    moving appreciably, and start their Robbins-Monro decay after that.
    warm_mult = None means "decay from k = 0".
    """
    if warm_mult is None:
        return 0
    return int(((sigmas - eps) > warm_mult * eps).sum())


def robbins_monro_steps(K, gamma_0, q, k_warm=0):
    """gamma_k = gamma_0 for k < k_warm, then gamma_0 (1 + (k - k_warm) / k_0)^(-q) with k_0 = K / 4.

    Square-summable (and not summable) as soon as 1/2 < q <= 1, which is what the
    convergence proofs of SNORE / ERED / SNOPnP ask for.
    """
    k = np.arange(K)
    k_0 = K / 4
    decay = gamma_0 * (1.0 + np.maximum(k - k_warm, 0) / k_0) ** (-q)
    return np.where(k < k_warm, gamma_0, decay)


def snopnp_noise_levels(sigmas, eps, K, q, k_warm=0):
    """Noise tau_k injected by SNOPnP: tau_k = (sigma_k - eps) + Robbins-Monro tail.

    tau_k follows the schedule while it is decreasing, then vanishes like k^(-q) so
    that sum_k tau_k^2 < infinity.
    """
    return (sigmas - eps) + robbins_monro_steps(K, eps, q, k_warm)
