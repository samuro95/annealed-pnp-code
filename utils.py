"""Small helpers: device and PSNR."""
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def psnr(x, x_true):
    """PSNR in dB of each image of the batch (images in [0, 1]). Returns a tensor of shape (B,)."""
    mse = ((x - x_true) ** 2).flatten(1).mean(1)
    return -10 * torch.log10(mse)
