"""Plotting: the restoration strips, the convergence curves and the PSNR tables of the paper.

  restoration_strip(...)   ground truth, observation, DPIR, then each algorithm with a
                           constant and an annealed noise level (optionally with a zoom)
  plot_curves(...)         PSNR, objective F_eps and residual along the iterations
  psnr_table(...)          mean PSNR, constant / annealed, one row per algorithm
"""
import matplotlib.pyplot as plt
import numpy as np

from runner import LABELS
from utils import psnr

COLORS = {"red_gd": "C0", "pnp_pgd": "C1", "snore": "C2", "snopnp": "C3", "ered": "C4", "dpir": "C7"}


def to_numpy(image):
    """A (3, H, W) tensor in [0, 1] as an image array."""
    return image.detach().float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()


def restoration_strip(x_true, observation, columns, zoom=None, save=None):
    """One row of images: ground truth, observation, then `columns` = [(title, image), ...].

    The PSNR of every reconstruction is written under it.  zoom = (x, y, size) adds a
    second row with that crop (the yellow square of the first row).
    """
    cols = [("Ground\ntruth", x_true, None), ("Observation", observation,
            psnr(observation[None], x_true[None]).item() if observation.shape == x_true.shape else None)]
    cols += [(title, image, psnr(image[None], x_true[None]).item()) for title, image in columns]
    rows = 1 if zoom is None else 2
    fig, axes = plt.subplots(rows, len(cols), figsize=(1.5 * len(cols), 1.9 * rows), squeeze=False)
    for j, (title, image, value) in enumerate(cols):
        axes[0, j].imshow(to_numpy(image), interpolation="nearest")
        axes[0, j].set_title(title, fontsize=7, pad=2)
        if zoom is not None:
            x0, y0, z = zoom
            axes[0, j].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), z, z, fill=False, edgecolor="yellow", lw=0.9))
            axes[1, j].imshow(to_numpy(image[:, y0:y0 + z, x0:x0 + z]), interpolation="nearest")
        if value is not None:
            axes[-1, j].text(0.5, -0.05, f"{value:.2f} dB", fontsize=7, ha="center", va="top",
                             transform=axes[-1, j].transAxes)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.subplots_adjust(left=0.003, right=0.997, top=0.85, bottom=0.08, wspace=0.03, hspace=0.04)
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def worst_window(image, x_true, size):
    """Crop (x, y, size) centred where the colour error of `image` is largest (used for demosaicing)."""
    import torch
    error = image - x_true
    colour = (error - error.mean(0, keepdim=True)).abs().mean(0)[None, None]
    pooled = torch.nn.functional.avg_pool2d(colour, size, stride=1)[0, 0]
    y, x = divmod(int(pooled.argmax()), pooled.shape[1])
    return x, y, size


def plot_curves(annealed, constant, k_min=25, psnr_floor=None, save=None):
    """Three panels along the iterations k:

        PSNR           annealed (solid) and constant sigma_c (dotted)
        F_eps(x_k)     the objective the annealed run is proved to decrease
        residual       ||x_{k+1} - x_k|| / ||x_0|| of the annealed run

    `annealed` and `constant` are dictionaries {algorithm name: runner.Result}.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.7))
    for name, result in annealed.items():
        style = dict(color=COLORS[name], lw=1.1)
        k = np.arange(1, len(result.psnr_curve) + 1)
        axes[0].plot(k[k >= k_min], result.psnr_curve[k >= k_min], label=LABELS[name], **style)
        if result.objective_curve is not None:
            k = np.arange(len(result.objective_curve))
            axes[1].semilogy(k[k >= k_min], result.objective_curve[k >= k_min], **style)
        k = np.arange(1, len(result.residual_curve) + 1)
        axes[2].semilogy(k[k >= k_min], result.residual_curve[k >= k_min], **style)
    for name, result in constant.items():
        k = np.arange(1, len(result.psnr_curve) + 1)
        axes[0].plot(k[k >= k_min], result.psnr_curve[k >= k_min], ":", color=COLORS[name], lw=1.1)
    axes[0].plot([], [], "k-", label="annealed")
    axes[0].plot([], [], "k:", label=r"constant $\sigma_c$")
    if psnr_floor is not None:
        axes[0].set_ylim(bottom=psnr_floor)
    axes[0].legend(fontsize=7, ncol=2, frameon=False)
    for ax, ylabel in zip(axes, ["PSNR (dB)", r"$F_\varepsilon(x_k)$", r"$\|x_{k+1}-x_k\|/\|x_0\|$"]):
        ax.set_xlabel("iteration $k$"); ax.set_ylabel(ylabel); ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_schedule(sigmas, eps, sigma_c=None, save=None):
    """The noise level along the iterations, with its floor (and a constant level for comparison)."""
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.semilogy(sigmas, label=r"annealed $\sigma_k$")
    ax.axhline(eps, color="k", linestyle=":", label=r"floor $\varepsilon$")
    if sigma_c is not None:
        ax.axhline(sigma_c, color="C3", linestyle="--", label=r"constant $\sigma_c$")
    ax.set_xlabel("iteration $k$"); ax.set_ylabel(r"$\sigma_k$"); ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def psnr_table(results_by_problem, algorithms=("red_gd", "pnp_pgd", "snore", "snopnp", "ered")):
    """Mean PSNR (dB) as text: one row per algorithm, two columns per problem.

    `results_by_problem` is {problem name: {(algorithm, 'constant' | 'annealed'): runner.Result,
    'dpir': runner.Result}}, i.e. what the notebook collects.
    """
    problems = list(results_by_problem)
    lines = [f"{'':10s}" + "".join(f"{p:^24s}" for p in problems),
             f"{'':10s}" + "".join(f"{'constant':>12s}{'annealed':>12s}" for _ in problems)]
    for algorithm in algorithms:
        row = f"{LABELS[algorithm]:10s}"
        for problem in problems:
            for version in ("constant", "annealed"):
                result = results_by_problem[problem].get((algorithm, version))
                row += f"{np.mean(result.psnr):12.2f}" if result else f"{'--':>12s}"
        lines.append(row)
    row = f"{LABELS['dpir']:10s}"
    for problem in problems:
        result = results_by_problem[problem].get("dpir")
        row += (f"{np.mean(result.psnr):18.2f}" if result else f"{'--':>18s}") + " " * 6
    lines.append(row)
    return "\n".join(lines)
