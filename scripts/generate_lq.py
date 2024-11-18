import os
import random
import sys
import json
import yaml

import numpy as np
import pandas as pd

import torch
from PIL import Image
from clip_interrogator import Config, Interrogator
from tqdm import tqdm
from open_clip import get_tokenizer
from open_clip_train.srclip_dataset import SRClipDataset
from torchvision.utils import save_image
import time
from tqdm import tqdm


def generate_lq(dataroot, opt_path, mode):

    tokenizer = get_tokenizer("srclip_ViT-B-32")
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)
    data = SRClipDataset(opt["datasets"][mode], tokenizer)
    data.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    future_df = {"filepath": [], "title": [], "title2": []}

    for i in tqdm(range(len(data))):
        _, _, x = data[i]
        gt_path = x["gt_path"]
        lq = x["lq"]
        lq_path = os.path.dirname(gt_path) + "/"
        lq_path = lq_path.replace("HR", "LQ")
        os.makedirs(lq_path, exist_ok=True)
        lq_path = os.path.join(lq_path, os.path.basename(gt_path))
        future_df["filepath"].append(lq_path)
        future_df["title"].append(x["semantic_caption"])
        future_df["title2"].append(",".join(x["degradation_caption"])) #TODO 可以尝试换种方式组合
        save_image(lq, lq_path)
        if i % 1000 == 0:
            pd.DataFrame.from_dict(future_df).to_csv(
                os.path.join(dataroot, f"srclip_{mode}_lq.csv"), index=False, sep="\t"
            )

    pd.DataFrame.from_dict(future_df).to_csv(
        os.path.join(dataroot, f"srclip_{mode}_lq.csv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    dataroot = "datasets/LSDIR/data"
    opt_path = "options/train_srclip.yml"

    generate_lq(dataroot, opt_path, "val")
    # generate_lq(dataroot, opt_path, mode="train")
