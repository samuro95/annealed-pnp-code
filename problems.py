"""The inverse problems of the paper, as deepinv physics + an initialisation.

Every problem gives  y = A(x) + noise,  and the algorithms minimise

    f(x) = 1/2 || A(x) - y ||^2   +   regularisation given by the denoiser.

Each builder returns a Problem: the physics, the initial point x_0, the noise level
of the measurement, and the Lipschitz constant L_f = || A ||^2 of grad f.
"""
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
import deepinv as dinv


@dataclass
class Problem:
    name: str
    physics: object
    init: Callable          # y -> x_0
    noise_level: float
    lipschitz: float        # L_f = ||A||^2

    def measure(self, x_true, seed=300):
        """y = A(x_true) + noise, with a fixed seed so that every algorithm sees the same y."""
        torch.manual_seed(seed)
        return self.physics(x_true)

    def grad_f(self, x, y):
        return self.physics.A_adjoint(self.physics.A(x) - y)

    def f(self, x, y):
        """1/2 ||A x - y||^2, one value per image."""
        return 0.5 * ((self.physics.A(x) - y) ** 2).flatten(1).sum(1)


def _lipschitz(physics, size, device):
    x = torch.rand(1, 3, size, size, device=device)
    return float(physics.compute_norm(x, max_iter=200, tol=1e-5, verbose=False))


def _fill_holes(y, physics, blur_sigma):
    """Initialisation for inpainting: keep the observed pixels, fill the rest by normalised convolution."""
    mask = physics.mask.expand_as(y).float()
    kernel = dinv.physics.functional.gaussian_blur(sigma=(blur_sigma, blur_sigma), device=y.device)
    num = dinv.physics.functional.conv2d(mask * y, kernel, padding="circular")
    den = dinv.physics.functional.conv2d(mask, kernel, padding="circular")
    return mask * y + (1 - mask) * num / den.clamp_min(1e-3)


def _noise(sigma_noise):
    return dinv.physics.GaussianNoise(sigma=sigma_noise)


def random_inpainting(size, device, missing_ratio=0.5, sigma_noise=0.01):
    """50% of the pixels removed at random (the same mask for every image)."""
    physics = dinv.physics.Inpainting(img_size=(3, size, size), mask=1 - missing_ratio, pixelwise=True,
                                      device=device, rng=torch.Generator(device=device).manual_seed(0),
                                      noise_model=_noise(sigma_noise))
    return Problem("inpainting", physics, lambda y: _fill_holes(y, physics, 3.0),
                   sigma_noise, _lipschitz(physics, size, device))


def hole_inpainting(size, device, hole=64, sigma_noise=0.01):
    """A centred square hole of `hole` pixels (64 px in a 128 px image for the main figure)."""
    mask = torch.ones(1, size, size, device=device)
    a = (size - hole) // 2
    mask[:, a:a + hole, a:a + hole] = 0
    physics = dinv.physics.Inpainting(img_size=(3, size, size), mask=mask, device=device,
                                      noise_model=_noise(sigma_noise))
    blur = 6.0 * hole / 32                     # the wider the hole, the smoother the initial fill
    return Problem(f"hole{hole}", physics, lambda y: _fill_holes(y, physics, blur),
                   sigma_noise, _lipschitz(physics, size, device))


def super_resolution(size, device, factor=4, sigma_noise=0.01):
    """Bicubic downsampling by `factor`; the initial point is the bicubic upsampling of y."""
    physics = dinv.physics.Downsampling(img_size=(3, size, size), filter="bicubic", factor=factor,
                                        padding="circular", device=device, noise_model=_noise(sigma_noise))
    init = lambda y: F.interpolate(y, scale_factor=factor, mode="bicubic", align_corners=False).clamp(0, 1)
    return Problem(f"srx{factor}", physics, init, sigma_noise, _lipschitz(physics, size, device))


def demosaicing(size, device, sigma_noise=0.01):
    """Bayer colour filter array: one colour channel observed per pixel."""
    physics = dinv.physics.Demosaicing(img_size=(3, size, size), pattern="bayer", device=device,
                                       noise_model=_noise(sigma_noise))
    return Problem("demosaic", physics, lambda y: _fill_holes(y, physics, 3.0),
                   sigma_noise, _lipschitz(physics, size, device))


def tomography(size, device, angles=60, sigma_noise=0.01):
    """Sparse-view parallel-beam CT; the initial point is the filtered back-projection."""
    physics = dinv.physics.Tomography(angles=angles, img_width=size, circle=False, normalize=True,
                                      device=device, noise_model=_noise(sigma_noise))
    return Problem(f"tomo{angles}", physics, lambda y: physics.A_dagger(y).clamp(0, 1),
                   sigma_noise, _lipschitz(physics, size, device))
