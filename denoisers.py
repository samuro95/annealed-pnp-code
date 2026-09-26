"""The denoisers of the paper, built on top of a plain network N(x, sigma).

Gradient-step denoiser (Hurault et al., 2022):

    g_sigma(x) = (delta / 2) * || x - N(x, sigma) ||^2        (the potential)
    D_sigma(x) = x - grad g_sigma(x)                          (the denoiser)

so the RED algorithms minimise an explicit function, and when ||Hess g_sigma|| < 1 the
same D_sigma is the proximal operator of an explicit (weakly convex) function phi_sigma,
which is what the PnP algorithms need.  delta = 1 for the RED algorithms; the PnP
algorithms use the "Prox-" checkpoints (trained with a Lipschitz penalty) relaxed by
delta = 0.7.
"""
import torch

import networks


class GradientStepDenoiser(torch.nn.Module):
    """D_sigma(x) = x - grad g_sigma(x) with g_sigma(x) = (delta/2) ||x - N(x, sigma)||^2."""

    def __init__(self, net, delta=1.0):
        super().__init__()
        self.net = net
        self.delta = float(delta)

    def potential(self, x, sigma):
        """g_sigma(x), one value per image of the batch."""
        return 0.5 * self.delta * ((x - self.net(x, sigma)) ** 2).flatten(1).sum(1)

    def forward(self, x, sigma):
        """D_sigma(x)."""
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            grad = torch.autograd.grad(self.potential(x, sigma).sum(), x)[0]
        return (x - grad).detach()

    def inverse(self, x, sigma, z_0, tol=1e-6, max_iter=1000, patience=30):
        """z = D_sigma^{-1}(x), by the damped fixed point  z <- z + theta (x - D_sigma(z)).

        theta = 1 is  z <- x + grad g(z), a contraction when ||Hess g|| < 1; theta is halved
        (per image) whenever a step would increase the residual ||x - D(z)|| / ||x||.
        Needed only to evaluate phi_sigma(x) = g_sigma(z) - 1/2 ||z - x||^2 for the PnP curves.
        """
        view = lambda t: t.view(-1, 1, 1, 1)
        norm_x = x.flatten(1).norm(dim=1).clamp_min(1e-12)
        z = z_0.clone()
        r = x - self(z, sigma)
        res = r.flatten(1).norm(dim=1) / norm_x
        theta, best, stall = torch.ones_like(res), res.clone(), 0
        for _ in range(max_iter):
            if bool((res <= tol).all()) or stall >= patience:
                break
            z_new = z + view(theta) * r
            r_new = x - self(z_new, sigma)
            res_new = r_new.flatten(1).norm(dim=1) / norm_x
            ok = (res_new <= res) | (res <= tol)
            z, r, res = torch.where(view(ok), z_new, z), torch.where(view(ok), r_new, r), torch.where(ok, res_new, res)
            theta = torch.where(ok, theta, theta / 2)
            stall = 0 if bool((res < 0.999 * best).any()) else stall + 1
            best = torch.minimum(best, res)
        return z


def _load_backbone_weights(net, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(state["ema"] if "ema" in state else state.get("state_dict", state), strict=True)
    return net


def gs_drunet(checkpoint, device, delta=1.0):
    """GS-DRUNet (CBSD68, RED algorithms): DRUNet fine-tuned as a gradient step for sigma in [0.002, 2]."""
    net = _load_backbone_weights(networks.drunet(device), checkpoint).to(device)
    return GradientStepDenoiser(net, delta).to(device).eval()


def prox_drunet(checkpoint, device, delta=0.7):
    """Prox-DRUNet (CBSD68, PnP algorithms): same, trained with the Lipschitz penalty."""
    return gs_drunet(checkpoint, device, delta)


def gs_diffunet(checkpoint, device, delta=1.0):
    """GS-DiffUNet (FFHQ 128, RED algorithms): DiffUNet fine-tuned as a gradient step."""
    net = _load_backbone_weights(networks.diffunet(device), checkpoint).to(device)
    return GradientStepDenoiser(net, delta).to(device).eval()


def prox_diffunet(checkpoint, device, delta=0.7):
    """Prox-DiffUNet (FFHQ 128, PnP algorithms): same, trained with the Lipschitz penalty."""
    return gs_diffunet(checkpoint, device, delta)


def pretrained_drunet(device):
    """The plain pretrained DRUNet, used only by the DPIR baseline."""
    from deepinv.models import DRUNet
    net = DRUNet(pretrained="download", device=device).eval().requires_grad_(False)
    return lambda x, sigma: net(x, sigma)
