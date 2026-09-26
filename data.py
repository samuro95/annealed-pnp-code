"""The two test sets: CBSD68 (natural images) and FFHQ (faces), both at 128 x 128.

CBSD68: the short side is resized to 128 (bicubic, antialiased) and the long side is
centre-cropped, so the whole scene stays in the image.
FFHQ: the 16 faces listed in FFHQ_VALIDATION are used to tune the hyper-parameters and
never for the reported numbers; the test set is the first `n` of the remaining ones.
"""
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FFHQ_VALIDATION = list(range(69000, 69004)) + list(range(69200, 69212))


def _load(path):
    x = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(x).permute(2, 0, 1).float() / 255


def _resize_and_crop(x, size):
    h, w = x.shape[-2:]
    scale = size / min(h, w)
    new = (max(size, round(h * scale)), max(size, round(w * scale)))
    x = F.interpolate(x[None], size=new, mode="bicubic", align_corners=False, antialias=True)[0].clamp(0, 1)
    top, left = (new[0] - size) // 2, (new[1] - size) // 2
    return x[:, top:top + size, left:left + size]


def cbsd68(folder, device, n=68, size=128, split="test"):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNG in {folder} (export deepinv's CBSD68 there)")
    files = files[:n] if split == "test" else files[50:50 + n]
    return torch.stack([_resize_and_crop(_load(f), size) for f in files]).to(device)


def ffhq(folder, device, n=100, size=128, split="test"):
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not files:
        raise FileNotFoundError(f"no PNG in {folder}: run `python get_ffhq128.py` to build it "
                                f"(the FFHQ faces 69000+ at {size} x {size})")
    validation = [f for f in files if int(os.path.basename(f)[:-4]) in FFHQ_VALIDATION]
    files = validation[:n] if split == "val" else [f for f in files if f not in set(validation)][:n]
    images = torch.stack([_load(f) for f in files]).to(device)
    if images.shape[-1] != size:
        raise ValueError(f"{folder} holds {images.shape[-1]} px images, {size} px expected")
    return images
