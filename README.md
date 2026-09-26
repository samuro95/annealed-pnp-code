# Plug-and-play with a decreasing noise level — code

Code, denoiser checkpoints and a notebook that reproduces the figures and tables of the paper.
Five PnP / RED algorithms (RED-GD, SNORE, ERED, PnP-PGD, SNOPnP) are run with a **constant**
noise level `sigma_c` and with the **annealed** schedule

    sigma_k = eps + (sigma_0 - eps) * r**k        (0 < r < 1, eps > 0)

Everything is plain PyTorch on top of [deepinv](https://deepinv.github.io): each algorithm is
one explicit `for` loop, and every parameter is written in the notebook cell that uses it.

## Contents

| file | what is inside |
|---|---|
| `paper_figures.ipynb` | **the notebook**: reproduces Figures 1–4, the two appendix figures and the PSNR tables |
| `algorithms.py` | RED-GD, SNORE, ERED, PnP-PGD, SNOPnP and the DPIR baseline, one loop each |
| `schedules.py` | annealed and constant `sigma_k`, Robbins-Monro steps |
| `params.py` | `Params`: the hyper-parameters of one run |
| `runner.py` | one run = schedule + step size + algorithm + recorded curves |
| `denoisers.py` | gradient-step / proximal denoiser `D = Id - grad g`, checkpoint loading |
| `networks.py` | the two backbones `N(x, sigma)`: DRUNet and DiffUNet |
| `problems.py` | inpainting (hole, random), demosaicing, super-resolution, tomography |
| `objectives.py` | the objective `F_eps` each algorithm decreases |
| `data.py` | the FFHQ and CBSD68 test images |
| `figures.py` | restoration strips, convergence curves, PSNR tables |
| `get_ffhq128.py`, `get_cbsd68.py` | download the test images into `data/` |
| `checkpoints/` | the four denoisers (below) |

## Checkpoints

| file | denoiser | used for |
|---|---|---|
| `checkpoints/gs_diffunet128.pt` | GS-DiffUNet, faces 128 px | FFHQ, RED algorithms |
| `checkpoints/prox_diffunet128.pt` | Prox-DiffUNet, faces 128 px | FFHQ, PnP algorithms |
| `checkpoints/gs_drunet.pt` | GS-DRUNet, natural images, `sigma` in [0.002, 2] | CBSD68, RED algorithms |
| `checkpoints/prox_drunet.pt` | Prox-DRUNet, natural images | CBSD68, PnP algorithms |

Each file holds the EMA weights of the network, `{"ema": state_dict}`.  The training recipes are
in the appendix of the paper.

## Installation

Python >= 3.10, then

```
pip install -r requirements.txt
```

A GPU is strongly recommended (CUDA, or Apple `mps`; the code picks the device itself).

## Data

```
python get_cbsd68.py      # data/CBSD68_png/00.png ... 67.png  (deepinv's copy of CBSD68)
python get_ffhq128.py     # data/ffhq128/69000.png ...           (116 FFHQ faces, 128 x 128)
```

The FFHQ faces are among the last 1000 of the dataset, which the denoisers never saw in training.
16 of them are the validation faces used to tune the parameters (`data.FFHQ_VALIDATION`); the
other 100 are the test faces.

## Running

```
jupyter notebook paper_figures.ipynb
```

and run the cells in order.  The notebook is organised as the paper:

1. the schedule;
2. **Figure 1**: RED-GD on a 64 x 64 hole, constant `sigma` against the annealed schedule;
3. **Figures 2–4** (FFHQ: hole inpainting, demosaicing, super-resolution x4): the restoration strip
   and the three convergence panels (PSNR, `F_eps(x_k)`, `||x_{k+1} - x_k|| / ||x_0||`);
4. **Table 1** (FFHQ);
5. **Appendix**: CBSD68 random inpainting and sparse-view tomography figures, **Table 2** (CBSD68).

Each problem has one parameter cell, one line per run, e.g.

```python
K = 500
sigma_0 = sigma_0_ffhq
r = 0.98

hole_annealed = {
    "red_gd":  Params(K, sigma_0=sigma_0, r=r, eps=0.014, lam=0.3),
    ...
}
hole_constant = {
    "red_gd":  Params(K, sigma_c=2, lam=0.05),
    ...
}
```

The figures are computed on the image the paper shows, with the paper's random seed.  The tables
average over the first `N_IMAGES` test images (first parameter cell); set `N_IMAGES = 100` for FFHQ
and `68` for CBSD68 to get the paper's averages.

## One run, without the notebook

```python
import data, denoisers, problems, runner
from params import Params
from utils import get_device

device = get_device()
images = data.cbsd68("data/CBSD68_png", device, n=4)
problem = problems.random_inpainting(128, device, missing_ratio=0.5, sigma_noise=0.01)
y = problem.measure(images)
denoiser = denoisers.gs_drunet("checkpoints/gs_drunet.pt", device)

p = Params(K=500, sigma_0=1.0, r=0.98, eps=0.01, lam=0.3)      # annealed: 1 -> 0.01
result = runner.run("red_gd", problem, y, images, denoiser, p, L_g=2.2)
print(result.psnr.mean())
```

or, writing the iteration out,

```python
import algorithms, schedules

sigmas = schedules.annealed_sigmas(500, sigma_0=1.0, eps=0.01, r=0.98)
gamma = 1.0 / (problem.lipschitz + 0.3 * 2.2)                  # gamma (L_f + lam L_g) = 1
x = algorithms.red_gd(y, problem, denoiser, problem.init(y), sigmas, lam=0.3, gamma=gamma)
```
