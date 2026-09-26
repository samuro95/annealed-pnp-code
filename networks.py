"""The two denoising networks N(x, sigma) used in the paper, with a common interface.

Both take an image batch in [0, 1] and a scalar noise level sigma, and return the
denoised batch.  They are plain networks: the gradient-step / proximal denoisers of
the paper are built on top of them in denoisers.py.
"""
import torch
import torch.nn.functional as F
from deepinv.models import DRUNet, DiffUNet


def drunet(device, act_mode="s"):
    """DRUNet backbone (softplus activations, as in the GS-DRUNet / Prox-DRUNet training)."""
    net = DRUNet(in_channels=3, out_channels=3, nb=2, nc=(64, 128, 256, 512),
                 act_mode=act_mode, pretrained=None)
    return net.to(device).eval().requires_grad_(False)


class DiffUNetDenoiser(torch.nn.Module):
    """deepinv's DiffUNet (the DiffPIR/ADM network) used as a denoiser N(x, sigma).

    The diffusion model is indexed by a time step t, not by a noise level, so we map
    sigma -> t by inverting the variance-preserving schedule analytically:

        alphabar(sigma) = 1 / (1 + 4 sigma^2),
        -log alphabar   = beta_min u + (beta_max - beta_min) u^2 / 2,   u = t / (T - 1).

    Using the exact inverse (instead of the nearest of the 1000 tabulated steps) makes
    sigma -> N(x, sigma) smooth, which matters when sigma decreases along the iterations.
    """
    beta_min, beta_max, T = 0.1, 20.0, 1000

    def __init__(self, large=False):
        super().__init__()
        self.net = DiffUNet(large_model=large, pretrained=None)

    def timestep(self, sigma):
        ell = torch.log1p(4 * sigma.float().square()).reshape(-1)
        d = self.beta_max - self.beta_min
        u = 2 * ell / (self.beta_min + (self.beta_min ** 2 + 2 * d * ell).sqrt())
        return (self.T - 1) * u

    def forward(self, x, sigma):
        if not torch.is_tensor(sigma):
            sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.reshape(-1).expand(x.shape[0]).view(-1, 1, 1, 1)
        pad = (-x.size(-1) % 32, 0, -x.size(-2) % 32, 0)          # the network needs multiples of 32
        xp = F.pad(x, pad, mode="circular") if any(pad) else x
        alphabar = 1 / (1 + 4 * sigma ** 2)
        x_t = alphabar.sqrt() * (2 * xp - 1)                       # the network lives in [-1, 1]
        eps = self.net.forward_diffusion(x_t, self.timestep(sigma).to(x.device))[:, :3]
        out = xp - sigma * eps                                     # back to [0, 1]
        return out[..., pad[-2]:, pad[-4]:] if any(pad) else out


def diffunet(device, large=False):
    return DiffUNetDenoiser(large=large).to(device).eval().requires_grad_(False)
