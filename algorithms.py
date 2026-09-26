"""The five algorithms of the paper, plus the DPIR baseline, as plain loops.

All of them alternate a gradient (or proximal) step on the data-fidelity term
f(x) = 1/2 ||A x - y||^2 and a denoising step at the noise level sigma_k.  The whole
point of the paper is that sigma_k may DECREASE along the iterations, down to a small
positive floor eps, instead of being held at a constant sigma_c.

`sigmas` is the array of noise levels (one per iteration); its length is the number of
iterations.  `on_step(k, x, sigma)` is called after every update (used to record PSNR,
objective and residual); it can be left to None.  The PnP algorithms also pass the
denoiser input u_k (x_{k+1} = D_sigma_k(u_k)), which the objective curves need.
"""
import torch


def red_gd(y, problem, denoiser, x_0, sigmas, lam, gamma, on_step=None):
    """RED gradient descent:  x_{k+1} = x_k - gamma [ grad f(x_k) + lam (x_k - D_sigma_k(x_k)) ]."""
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        x = x - gamma * (problem.grad_f(x, y) + lam * (x - denoiser(x, sigma)))
        if on_step:
            on_step(k, x, sigma)
    return x


def snore(y, problem, denoiser, x_0, sigmas, lam, steps, on_step=None):
    """SNORE: RED gradient descent with fresh noise in the denoiser input and a decreasing step,

        x_{k+1} = x_k - gamma_k [ grad f(x_k) + lam (x_k - D_sigma_k(x_k + sigma_k Z_k)) ].
    """
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        noisy = x + float(sigma) * torch.randn_like(x)
        x = x - steps[k] * (problem.grad_f(x, y) + lam * (x - denoiser(noisy, sigma)))
        if on_step:
            on_step(k, x, sigma)
    return x


def random_symmetry(group):
    """Draw Q in the group; returns (Q, Q^-1) as functions. 'dihedral' = 90 deg rotations x flip."""
    if group == "hflip":
        flip = bool(torch.randint(2, (1,), device="cpu"))
        op = (lambda x: x.flip(-1)) if flip else (lambda x: x)
        return op, op
    rot = int(torch.randint(4, (1,), device="cpu"))
    flip = bool(torch.randint(2, (1,), device="cpu"))

    def forward(x):
        return torch.rot90(x.flip(-1) if flip else x, rot, dims=(-2, -1))

    def inverse(x):
        x = torch.rot90(x, -rot, dims=(-2, -1))
        return x.flip(-1) if flip else x

    return forward, inverse


def ered(y, problem, denoiser, x_0, sigmas, lam, steps, group="hflip", on_step=None):
    """Equivariant RED: the denoiser is randomly symmetrised,

        x_{k+1} = x_k - gamma_k [ grad f(x_k) + lam (x_k - Q_k^-1 D_sigma_k(Q_k x_k)) ].
    """
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        Q, Q_inv = random_symmetry(group)
        x = x - steps[k] * (problem.grad_f(x, y) + lam * (x - Q_inv(denoiser(Q(x), sigma))))
        if on_step:
            on_step(k, x, sigma)
    return x


def pnp_pgd(y, problem, denoiser, x_0, sigmas, gamma, on_step=None):
    """PnP proximal gradient descent:  x_{k+1} = D_sigma_k( x_k - gamma grad f(x_k) ),  gamma L_f < 1."""
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        u = x - gamma * problem.grad_f(x, y)
        x = denoiser(u, sigma)
        if on_step:
            on_step(k, x, sigma, u)
    return x


def snopnp(y, problem, denoiser, x_0, sigmas, gamma, taus, on_step=None):
    """Stochastic PnP (decoupled PnP-Flow):  x_{k+1} = D_sigma_k( x_k - gamma grad f(x_k) + tau_k Z_k )."""
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        u = x - gamma * problem.grad_f(x, y)
        if taus[k] > 0:
            u = u + float(taus[k]) * torch.randn_like(u)
        x = denoiser(u, sigma)
        if on_step:
            on_step(k, x, sigma, u)
    return x


def dpir(y, problem, denoiser, x_0, sigmas, stepsizes, on_step=None):
    """DPIR baseline (half-quadratic splitting, 8 iterations, decreasing sigma_k):

        z_k     = prox_{gamma_k f}(x_k)          (solved exactly by deepinv)
        x_{k+1} = D_sigma_k(z_k).
    """
    x = x_0.clone()
    for k, sigma in enumerate(sigmas):
        z = problem.physics.prox_l2(x, y, float(stepsizes[k]))
        x = denoiser(z, sigma)
        if on_step:
            on_step(k, x, sigma)
    return x


def dpir_parameters(noise_level, n_iter=8, sigma_start=49 / 255):
    """The usual DPIR schedule: sigma_k log-spaced from 49/255 to the noise level, step (sigma_k/sigma_n)^2 / 0.23."""
    import numpy as np
    sigmas = np.logspace(np.log10(sigma_start), np.log10(max(noise_level, 1e-3)), n_iter)
    stepsizes = (sigmas / max(0.01, noise_level)) ** 2 / 0.23
    return sigmas, stepsizes
