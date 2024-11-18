import open_clip
from open_clip_train.srclip_dataset import SRClipPairDatasetV2
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch.utils
import torch.utils.data
import os
from evaluate import Degra, parse_args
from tqdm import tqdm


def plot_tsne(features, labels, output):
    """
    features:(N*m) N*m大小特征，其中N代表有N个数据，每个数据m维
    label:(N) 有N个标签
    """
    import pandas as pd

    tsne = TSNE(n_components=2, init="pca", random_state=0)
    import seaborn as sns

    class_num = len(np.unique(labels))  # 要分类的种类个数  eg:[0, 1, 2, 3]这个就是为4
    latent = features
    tsne_features = tsne.fit_transform(features)  # 将特征使用PCA降维至2维
    print("tsne_features的shape:", tsne_features.shape)

    df = pd.DataFrame()
    df["y"] = labels
    df["comp-1"] = tsne_features[:, 0]
    df["comp-2"] = tsne_features[:, 1]

    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", class_num),
        data=df,
    )
    plt.savefig(f"results/tsne/{output}")


def get_feat_target(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    model, preprocess = open_clip.create_model_from_pretrained(
        args.model_name, pretrained=checkpoint, is_train=False
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    val_data = args.val_data

    data = SRClipPairDatasetV2(val_data, preprocess, tokenizer)

    # templates = lambda d: f"A degraded photo with {d}"
    templates = lambda d: f"A photo with {d}"
    # templates = lambda d: d

    degra = Degra().gaussian_noise  # NOTE: change different degradation type here
    prompts = [templates(d) for d in degra.prompts]
    text = tokenizer(prompts).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    degra.text_features = text_features

    feat = []
    target = []

    for idx in tqdm(range(len(data))):
        image = data.transforms(Image.open(str(data.images[idx]))).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            image_features, degra_features, fine_degra_features = model.encode_image(
                image, control=True
            )
            fine_degra_features /= fine_degra_features.norm(dim=-1, keepdim=True)

            text_probs = (
                100.0 * fine_degra_features[:, degra.idx, :] @ degra.text_features.T
            ).softmax(dim=-1)

            feat.append(fine_degra_features[:, degra.idx, :].detach().cpu().numpy())
            target.append(torch.argmax(text_probs[0]).detach().cpu().numpy())

    return np.concatenate(feat, axis=0), np.array(target)


class Set14_DE(torch.utils.data.Dataset):
    def __init__(self, transforms, num, path="datasets/Set14/LR_DE"):
        """
        path
        ├── deg_0
        │   ├── 1.png
        │   ├── 2.png
        │   └── ...
        ├── deg_1
        │   ├── 1.png
        │   ├── 2.png
        │   └── ...
        ├── ...
        """
        super().__init__()
        self.transforms = transforms
        self.deg_type = []
        self.deg_img = []
        for deg_i in sorted(os.listdir(path)):
            deg_path = os.path.join(path, deg_i)
            images = [os.path.join(deg_path, img_i) for img_i in os.listdir(deg_path)]
            self.deg_img.append(images)
            self.deg_type.append(deg_i.split("_")[1])
            if len(self.deg_img) == num:
                break

    def __len__(self):
        return len(self.deg_img)

    def __getitem__(self, idx):
        imgs = []
        for img in self.deg_img[idx]:
            imgs.append(self.transforms(Image.open(img)).unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)  # (N, C, H, W)
        return imgs, self.deg_type[idx]


def get_feat_target_v2(args, num=5):
    """Use Set14 De Sets. Include 100 different degradations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = args.checkpoint
    model, preprocess = open_clip.create_model_from_pretrained(
        args.model_name, pretrained=checkpoint, is_train=False
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    val_data = "datasets/Set14/LR_DE"

    data = Set14_DE(preprocess, num, val_data)

    feat = []
    target = []

    for idx in tqdm(range(len(data))):
        image, deg_type = data[idx]
        image = image.to(device)
        with torch.no_grad():
            image_features, degra_features, fine_degra_features = model.encode_image(
                image, control=True
            )
            fine_degra_features /= fine_degra_features.norm(dim=-1, keepdim=True)
            degra_features /= degra_features.norm(dim=-1, keepdim=True)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            feat.append(fine_degra_features[:, 1, :].detach())
            # feat.append(degra_features.detach())
            # feat.append(image_features.detach())
        target.extend([deg_type] * image.size(0))

    feat = torch.cat(feat, dim=0).cpu().numpy()

    return feat, np.array(target)


if __name__ == "__main__":

    args = parse_args()
    features, labels = get_feat_target_v2(args)
    print(features.shape)
    print(labels.shape)
    plot_tsne(features, labels, args.output)
