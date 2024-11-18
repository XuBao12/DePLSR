import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv
from time import time
import argparse

import torch

metrics = [
    "PSNR",
    "SSIM",
    "LPIPS",
    "FID",
    "CLIPIQA",
    "CNNIQA",
    "MUSIQ",
    "DISTS",
]


def cal_single_metric(gt, sr, metric_name="PSNR", metric_mode="FR", device="cuda"):
    """calculate image quality metrics

    Args:
        sr (str): sr of model's outputs, could be a folder or a single image path
        gt (str): gt of test images, could be a folder or a single image path
    """

    metric_name = metric_name.lower()

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode=metric_mode, device=device)
    metric_mode = iqa_model.metric_mode

    if os.path.isfile(gt):
        input_paths = [gt]
        if sr is not None:
            ref_paths = [sr]
    else:
        input_paths = sorted(glob.glob(os.path.join(gt, "*")))
        if sr is not None:
            ref_paths = sorted(glob.glob(os.path.join(sr, "*")))

    avg_score = 0
    test_img_num = len(input_paths)
    best_score = float('-inf')
    best_img = None
    if metric_name != "fid":
        pbar = tqdm(total=test_img_num, unit="image")
        for idx, img_path in enumerate(input_paths):
            img_name = os.path.basename(img_path)
            if metric_mode == "FR":
                ref_img_path = ref_paths[idx]
            else:
                ref_img_path = None

            start_time = time()
            score = iqa_model(img_path, ref_img_path).cpu().item()
            end_time = time()
            avg_score += score
            if score > best_score:
                best_score = score
                best_img = img_name
            pbar.update(1)
            pbar.set_description(f"{metric_name} of {img_name}: {score}")
            pbar.write(
                f"{metric_name} of {img_name}: {score}\tTime: {end_time - start_time:.2f}s"
            )

        pbar.close()
        avg_score /= test_img_num
    else:
        assert os.path.isdir(gt), "input path must be a folder for FID."
        avg_score = iqa_model(gt, sr)

    msg = f"Average {metric_name} score of {gt} with {test_img_num} images is: {avg_score}"
    print(msg)

    return {f"{metric_name}": avg_score, f"best_{metric_name}_img": best_img}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate image quality metrics.")
    parser.add_argument("--sr_dir", type=str, required=True, help="Path to SR images.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor of SR images.")
    parser.add_argument("--output", type=str, default="metrics.csv", help="Output file to save metrics.")
    args = parser.parse_args()

    output_path = os.path.join(args.sr_dir, args.output)
    DATASETS = ["DIV2K", "LSDIR" , "RealSR_Cannon", "RealSR_Nikon", "Set5", "Set14"] # NOTE 每个数据集的超分结果分别存放在不同的文件夹中，命名为DIV2K, LSDIR, RealSR_Cannon, RealSR_Nikon, Set5, Set14
    SR_PATH = []
    for dataset in DATASETS:
        sr = os.path.join(args.sr_dir, dataset)
        SR_PATH.append(sr)

    if args.scale == 4: # NOTE 修改x4超分和x2超分的GT为自己的路径
        GT_PATH = ["datasets/DIV2K/DIV2K_valid_HR",
                   "datasets/LSDIR/data/HR/val",
                   "datasets/RealSRv3/Canon/Test/HR/X4",
                   "datasets/RealSRv3/Nikon/Test/HR/X4",
                   "datasets/Set5/GTmod12",
                   "datasets/Set14/GTmod12",]
    elif args.scale == 2:
        GT_PATH = ["datasets/DIV2K/DIV2K_valid_HR",
                   "datasets/LSDIR/data/HR/val",
                   "datasets/RealSRv3/Canon/Test/HR/X2",
                   "datasets/RealSRv3/Nikon/Test/HR/X2",
                   "datasets/Set5/GTmod12",
                   "datasets/Set14/GTmod12",]

    first_write = True
    for sr, gt in zip(SR_PATH, GT_PATH):
        print(sr, gt)
        rst = {}
        for metric in metrics:
            rst.update(cal_single_metric(sr, gt, metric_name=metric))

        print(rst)

        with open(output_path, mode='w' if first_write else 'a', newline='') as file:
            writer = csv.writer(file)
            if first_write:
                writer.writerow(["Dataset", "Metric", "Score", "Best Image"])
                first_write = False
            for key, value in rst.items():
                if "best_" in key:
                    continue
                writer.writerow([os.path.basename(sr), key, value, rst.get(f"best_{key}_img", "N/A")])
