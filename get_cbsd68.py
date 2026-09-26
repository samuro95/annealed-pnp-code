"""Build data/CBSD68_png: the 68 CBSD68 images as PNG files 00.png ... 67.png (deepinv's copy of the dataset).

    python get_cbsd68.py
"""
import argparse
import os

from deepinv.datasets import CBSD68

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/CBSD68", help="where deepinv downloads the dataset")
p.add_argument("--out", default="data/CBSD68_png")
args = p.parse_args()

dataset = CBSD68(root=args.root, download=not os.path.exists(os.path.join(args.root, "dataset_info.json")))
os.makedirs(args.out, exist_ok=True)
for i in range(len(dataset)):
    dataset[i].convert("RGB").save(os.path.join(args.out, f"{i:02d}.png"))   # a PIL image
print(len(dataset), "images in", args.out)
