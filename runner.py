"""Run one algorithm on one problem and record its curves.

This is the only place where the pieces are put together: the schedule, the step size,
the algorithm and what is recorded along the iterations.  A run reads as

    p = Params(K=500, sigma_0=5, r=0.98, eps=0.014, lam=0.3)
    result = runner.run("red_gd", problem, y, x_true, denoiser, p, L_g=4.2)
"""
from dataclasses import dataclass, field

import numpy as np
import torch

import algorithms
import objectives
import schedules
from utils import psnr

ALGORITHM_NAMES = ["red_gd", "pnp_pgd", "snore", "snopnp", "ered"]
PNP_ALGORITHMS = ("pnp_pgd", "snopnp")          # they use the proximal denoiser and the step c_f / L_f
LABELS = {"red_gd": "RED-GD", "pnp_pgd": "PnP-PGD", "snore": "SNORE", "snopnp": "SNOPnP",
          "ered": "ERED", "dpir": "DPIR"}


@dataclass
class Result:
    """What a run returns.

    The reported reconstruction is the last iterate x_K, except for the ANNEALED RED
    algorithms, which report D_eps(x_K) (a last denoising step at the floor, worth ~0.01 dB).
    A constant run at a large sigma_c is not denoised again: D_sigma_c would undo the fill.
    `estimate` and `psnr` already follow that rule.
    """
    name: str
    x: torch.Tensor                 # the last iterate x_K
    estimate: torch.Tensor          # the reported reconstruction (see above)
    sigmas: np.ndarray
    gamma: float
    psnr: np.ndarray                # PSNR of `estimate`, one value per image
    psnr_curve: np.ndarray = field(default=None)        # mean PSNR of x_1 ... x_K
    residual_curve: np.ndarray = field(default=None)    # mean ||x_{k+1} - x_k|| / ||x_0||
    objective_curve: np.ndarray = field(default=None)   # mean F_eps(x_k), k = 0 ... K


class Recorder:
    """Called after every iteration: keeps the mean PSNR, residual and objective."""

    def __init__(self, x_0, x_true, objective=None):
        self.x_prev, self.x_true, self.objective = x_0, x_true, objective
        self.norm_x_0 = x_0.flatten(1).norm(dim=1).clamp_min(1e-12)
        self.psnr, self.residual = [], []
        self.values = [float(objective(x_0, None).mean())] if objective else []

    def __call__(self, k, x, sigma, u=None):
        self.psnr.append(float(psnr(x, self.x_true).mean()))
        self.residual.append(float(((x - self.x_prev).flatten(1).norm(dim=1) / self.norm_x_0).mean()))
        if self.objective is not None:
            self.values.append(float(self.objective(x, u).mean()))
        self.x_prev = x

    def curves(self):
        return (np.array(self.psnr), np.array(self.residual),
                np.array(self.values) if self.values else None)


def _objective_function(algorithm, y, problem, denoiser, eps, p, gamma):
    """F_eps(x); the second argument is the denoiser input u_k, used by the PnP ones only."""
    if algorithm == "red_gd":
        return lambda x, u: objectives.red(x, y, problem, denoiser, eps, p.lam)
    if algorithm == "snore":
        return lambda x, u: objectives.smoothed_red(x, y, problem, denoiser, eps, p.lam)
    if algorithm == "ered":
        return lambda x, u: objectives.equivariant_red(x, y, problem, denoiser, eps, p.lam, p.group)
    return lambda x, u: objectives.pnp(x, y, problem, denoiser, eps, gamma, u)


@torch.no_grad()
def run(algorithm, problem, y, x_true, denoiser, p, L_g, seed=0, record_objective=False):
    """Run `algorithm` for p.K iterations. `p` is a params.Params, `L_g` the Lipschitz constant of grad g."""
    torch.manual_seed(seed)
    K = p.K
    if p.constant:
        sigmas, eps = schedules.constant_sigmas(K, p.sigma_c), p.sigma_c
    else:
        sigmas, eps = schedules.annealed_sigmas(K, p.sigma_0, p.eps, p.r), p.eps
    k_warm = 0 if p.constant else schedules.warm_start_length(sigmas, eps, p.warm_mult)

    if algorithm in PNP_ALGORITHMS:
        gamma = p.c_f / problem.lipschitz                       # gamma L_f < 1
    else:
        gamma = p.gamma_rel / (problem.lipschitz + p.lam * L_g)  # gamma (L_f + lam L_g) < 2

    x_0 = problem.init(y)
    objective = _objective_function(algorithm, y, problem, denoiser, eps, p, gamma) if record_objective else None
    record = Recorder(x_0, x_true, objective)

    if algorithm == "red_gd":
        x = algorithms.red_gd(y, problem, denoiser, x_0, sigmas, p.lam, gamma, record)
    elif algorithm == "snore":
        steps = schedules.robbins_monro_steps(K, gamma, p.q, k_warm)
        x = algorithms.snore(y, problem, denoiser, x_0, sigmas, p.lam, steps, record)
    elif algorithm == "ered":
        steps = schedules.robbins_monro_steps(K, gamma, p.q, k_warm)
        x = algorithms.ered(y, problem, denoiser, x_0, sigmas, p.lam, steps, p.group, record)
    elif algorithm == "pnp_pgd":
        x = algorithms.pnp_pgd(y, problem, denoiser, x_0, sigmas, gamma, record)
    elif algorithm == "snopnp":
        taus = sigmas.copy() if p.tau == "coupled" else schedules.snopnp_noise_levels(sigmas, eps, K, p.q, k_warm)
        x = algorithms.snopnp(y, problem, denoiser, x_0, sigmas, gamma, taus, record)
    else:
        raise ValueError(f"unknown algorithm {algorithm}")

    psnr_curve, residual_curve, objective_curve = record.curves()
    final_denoising = algorithm not in PNP_ALGORITHMS and not p.constant
    estimate = denoiser(x, eps) if final_denoising else x
    return Result(name=algorithm, x=x, estimate=estimate, sigmas=sigmas, gamma=gamma,
                  psnr=psnr(estimate, x_true).cpu().numpy(), psnr_curve=psnr_curve,
                  residual_curve=residual_curve, objective_curve=objective_curve)


@torch.no_grad()
def run_dpir(problem, y, x_true, denoiser):
    """The DPIR baseline: 8 half-quadratic-splitting iterations with its own decreasing sigma_k."""
    sigmas, stepsizes = algorithms.dpir_parameters(problem.noise_level)
    x_0 = problem.init(y)
    record = Recorder(x_0, x_true)
    x = algorithms.dpir(y, problem, denoiser, x_0, sigmas, stepsizes, record)
    psnr_curve, residual_curve, _ = record.curves()
    return Result(name="dpir", x=x, estimate=x, sigmas=sigmas, gamma=float("nan"),
                  psnr=psnr(x, x_true).cpu().numpy(), psnr_curve=psnr_curve, residual_curve=residual_curve)
