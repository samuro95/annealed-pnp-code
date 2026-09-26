"""Build data/ffhq128: the 116 FFHQ faces used in the paper, at 128 x 128.

They are taken among the last 1000 faces of FFHQ (index 69000 and above), which the
denoisers were never trained on, from the 128 px thumbnails of FFHQ mirrored on the
Hugging Face hub.  16 of them are the validation faces (see data.FFHQ_VALIDATION), the
100 others the test faces.

    python get_ffhq128.py                  # writes data/ffhq128/69000.png, ...
"""
import argparse
import io
import os
import zipfile

from huggingface_hub import hf_hub_download
from PIL import Image

FACES = (list(range(69000, 69027)) + list(range(69100, 69127))
         + list(range(69200, 69212)) + list(range(69300, 69350)))      # 116 faces

p = argparse.ArgumentParser()
p.add_argument("--out", default="data/ffhq128")
p.add_argument("--repo", default="nuwandaa/ffhq128")
p.add_argument("--file", default="thumbnails128x128.zip")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
path = hf_hub_download(repo_id=args.repo, filename=args.file, repo_type="dataset")
wanted = set(FACES)
with zipfile.ZipFile(path) as z:
    for name in z.namelist():
        stem = os.path.splitext(os.path.basename(name))[0]
        if stem.isdigit() and int(stem) in wanted:
            image = Image.open(io.BytesIO(z.read(name))).convert("RGB")
            if image.size != (128, 128):
                image = image.resize((128, 128), Image.LANCZOS)
            image.save(os.path.join(args.out, f"{int(stem)}.png"))
print(len(os.listdir(args.out)), "faces in", args.out)
