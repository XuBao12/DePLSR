from basicsr.data.realesrgan_dataset import RealESRGANDataset, RealESRGANDataset_LSDIR
from basicsr.models.realesrgan_model import RealESRGANModel
import yaml
import torch
from tqdm import tqdm
import os
from torchvision.utils import save_image
from copy import deepcopy


def generate_LSDIR_lq(opt_path="options/train_realesrgan_x2plus.yml"):
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)
    data = RealESRGANDataset_LSDIR(opt["datasets"]["train"])
    model = RealESRGANModel(opt)

    for i in tqdm(range(len(data))):
        x = data[i]
        x["gt"].unsqueeze_(0)
        model.feed_data(x)
        lq = model.lq.squeeze(0)
        gt_path = x["gt_path"]
        lq_path = os.path.dirname(gt_path) + "/"
        lq_path = lq_path.replace("HR", "RealLQ")
        os.makedirs(lq_path, exist_ok=True)
        lq_path = os.path.join(lq_path, os.path.basename(gt_path))
        save_image(lq, lq_path)


def generate_pairdataset_lq(opt_path="options/train_realesrgan_x2plus_DIV2K.yml", crop=False):
    with open(opt_path, "r") as f:
        opt = yaml.safe_load(f)
    data = RealESRGANDataset(opt["datasets"]["train"])
    model = RealESRGANModel(opt)

    for i in tqdm(range(len(data))):
        x = data[i]
        x["gt"].unsqueeze_(0)
        if crop:
            for crop_idx in range(1, 41):
                x_crop = deepcopy(x)
                x_crop["gt"] = random_crop(x["gt"].squeeze(0)).unsqueeze(0)
                gt_path = x["gt_path"]
                img_name, ext = os.path.splitext(os.path.basename(gt_path))
                gt_path = os.path.join("datasets/DIV2K/DIV2K_valid_HR_sub", f"{img_name}_{crop_idx:03d}"+ext)
                save_image(x_crop["gt"], gt_path)

                model.feed_data(x_crop)
                lq = model.lq.squeeze(0)
                lq_path = os.path.join(opt["datasets"]["train"]["dataroot_lq"], f'X{opt["scale"]}')
                os.makedirs(lq_path, exist_ok=True)
                lq_path = os.path.join(lq_path, os.path.basename(gt_path))
                save_image(lq, lq_path)
        else:
            model.feed_data(x)
            lq = model.lq.squeeze(0)
            gt_path = x["gt_path"]
            lq_path = os.path.join(opt["datasets"]["train"]["dataroot_lq"], f'X{opt["scale"]}')
            os.makedirs(lq_path, exist_ok=True)
            lq_path = os.path.join(lq_path, os.path.basename(gt_path))
            save_image(lq, lq_path)

def random_crop(img):
    h, w = img.shape[-2:]
    crop_h, crop_w = 512, 512
    i = torch.randint(0, h - crop_h + 1, (1,))
    j = torch.randint(0, w - crop_w + 1, (1,))
    return img[..., i:i + crop_h, j:j + crop_w]


if __name__ == "__main__":
    opt_path = "options/train_realesrgan_x2plus_DIV2K.yml"
    # generate_LSDIR_lq(opt_path)
    generate_pairdataset_lq(opt_path, crop=False)
