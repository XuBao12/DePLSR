import os
import random
import sys
import json

import numpy as np
import pandas as pd

import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm


def generate_caption(meta, ci, dataroot, mode):

    future_df = {"filepath": [], "title": []}

    with open(meta, "r") as f:
        data = json.load(f)

    for item in tqdm(data):
        path = item["path"]
        img_path = os.path.join(dataroot, path)
        image = Image.open(img_path).convert("RGB")
        caption = ci.generate_caption(image)
        title = f"{caption}"

        future_df["filepath"].append(img_path)
        future_df["title"].append(title)

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot, f"srclip_{mode}.csv"), index=False, sep="\t"
    )


def generate_val_json(dataroot):
    import glob
    directory = "datasets/LSDIR/data/HR/val"
    file_paths = []

    for path in sorted(glob.glob(os.path.join(directory, "**", "*.png"), recursive=True)):
        file_paths.append({"path": path})

    output_json_path = os.path.join(dataroot, "val.json")

    with open(output_json_path, "w") as json_file:
        json.dump(file_paths, json_file, indent=4)


if __name__ == "__main__":
    dataroot = "datasets/LSDIR/data"
    generate_val_json(dataroot)
    
    meta_val = "datasets/LSDIR/data/val.json"
    meta_train = "datasets/LSDIR/data/train.json"

    ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

    generate_caption(meta_val, ci, dataroot, "val")
    generate_caption(meta_train, ci, dataroot, mode="train")
