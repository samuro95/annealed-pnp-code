"""The function each algorithm decreases, evaluated at the terminal noise level eps.

The theory says the iterates approach a stationary point of a FIXED function, obtained
by freezing the noise level at the floor eps:

    RED-GD    F(x) = f(x) + lam g_eps(x)
    SNORE     F(x) = f(x) + lam E_Z g_eps(x + eps Z)          (Monte-Carlo, M draws)
    ERED      F(x) = f(x) + lam E_Q g_eps(Q x)                (average over the group)
    PnP-PGD   F(x) = f(x) + phi_eps(x) / gamma
    SNOPnP    idem

where phi_eps is the function whose proximal operator is D_eps:

    phi_eps(x) = g_eps(z) - 1/2 ||z - x||^2,     z = D_eps^{-1}(x).

z is computed by a fixed point (denoisers.GradientStepDenoiser.inverse), started from the
denoiser input u_k that produced x = D_sigma_k(u_k): the preimage the iterates follow.

Each algorithm has its own eps, lam and gamma, so only the DECREASE of a curve is
meaningful, not the level at which it sits.
"""
import torch

MONTE_CARLO_DRAWS = 4


def red(x, y, problem, denoiser, eps, lam):
    return problem.f(x, y) + lam * denoiser.potential(x, eps)


def smoothed_red(x, y, problem, denoiser, eps, lam, draws=MONTE_CARLO_DRAWS):
    """SNORE's objective: the potential is smoothed by the noise the algorithm injects."""
    values = [denoiser.potential(x + float(eps) * torch.randn_like(x), eps) for _ in range(draws)]
    return problem.f(x, y) + lam * torch.stack(values).mean(0)


def equivariant_red(x, y, problem, denoiser, eps, lam, group="hflip"):
    """ERED's objective: the potential is averaged over the symmetry group."""
    if group == "hflip":
        ops = [lambda z: z, lambda z: z.flip(-1)]
    else:
        ops = [(lambda z, k=k, f=f: torch.rot90(z.flip(-1) if f else z, k, dims=(-2, -1)))
               for k in range(4) for f in (0, 1)]
    values = [denoiser.potential(Q(x), eps) for Q in ops]
    return problem.f(x, y) + lam * torch.stack(values).mean(0)


def pnp(x, y, problem, denoiser, eps, gamma, u=None):
    """f(x) + phi_eps(x) / gamma; `u` is the denoiser input that produced x (start of the inversion)."""
    z = denoiser.inverse(x, eps, x if u is None else u)
    phi = denoiser.potential(z, eps) - 0.5 * ((z - x) ** 2).flatten(1).sum(1)
    return problem.f(x, y) + phi / gamma
