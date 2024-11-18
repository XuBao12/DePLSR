from visualizer import get_local

get_local.activate()

import os
import open_clip
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch
import cv2
import torchvision.utils as vutils
from sklearn.decomposition import PCA
from einops import rearrange


def visualize_filter(log_dir, model):

    kernel_idx = -1

    for name, layer in model.named_modules():
        if "conv" not in name:
            continue
        kernel_idx += 1
        kernels = layer.weight.detach()
        c_out, c_int, k_h, k_w = tuple(kernels.shape)  # 768 3 32 32
        print(kernel_idx, kernels.shape)
        # 将后3个维度合并，使用主成分分析工具要求输入维度为2
        kernels = rearrange(kernels, "n c h w -> n (c h w)")
        # 与ViT论文一致，做28个主成分; pca components: (28, 3072)
        pca = PCA(n_components=28)
        pca.fit(kernels)
        filters = torch.tensor(pca.components_.reshape((-1, 3, 32, 32)))
        # 再将主成分转换成tensorboard 能够展示的RGB的形式
        # filters: (28, 3, 32, 32)
        plt.figure(figsize=(20, 17))
        for i in range(1, 29):
            plt.subplot(4, 7, i)
            plt.axis("off")
            # 从filters中取出第i个主成分，合并RGB通道
            X = filters[i - 1]
            X = torch.sum(X, dim=0) / X.shape[0]  # 32 32
            plt.imshow(X, cmap="gray")
        plt.savefig(os.path.join(log_dir, f"{name}.png"))


def visualize_feature_map(log_dir, model, image):
    # Only used in Convolution-based models
    output_features = model.encode_image(image, control=True, normalize=True)

    def visualize_map(features, plt):
        print(features.shape)
        features = features.squeeze(0)
        features = torch.sum(features, dim=0) / features.shape[0]
        print(features.shape)
        plt.imshow(features.detach().cpu().numpy(), cmap="gray")

    plt.figure(figsize=(20, 17))
    for i in range(len(output_features)):
        plt.subplot(1, 3, i + 1)
        plt.axis("off")
        visualize_map(output_features[i], plt)
    plt.savefig(os.path.join(log_dir, "feature_map.png"))


def visualize_attention_map(log_dir, model, image):
    def visulize_attention_ratio(
        img_path, attention_mask, ratio=0.5, cmap="jet", save_path=None
    ):
        """
        img_path: 读取图片的位置
        attention_mask: 2-D 的numpy矩阵
        ratio:  放大或缩小图片的比例，可选
        cmap:   attention map的style，可选
        """
        # load the image
        img = Image.open(img_path, mode="r")
        img_h, img_w = img.size[0], img.size[1]
        plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

        # scale the image
        img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
        img = img.resize((img_h, img_w))
        plt.imshow(img, alpha=1)
        plt.axis("off")

        # normalize the attention mask
        mask = cv2.resize(attention_mask, (img_h, img_w))
        normed_mask = mask / mask.max()
        normed_mask = (normed_mask * 255).astype("uint8")
        plt.imshow(normed_mask, alpha=0.5, interpolation="nearest", cmap=cmap)
        plt.savefig(save_path)

    with torch.no_grad():
        # image_features, attention_list = model.visual(image, output_hiddens=True) # For CLIP
        _, _, _ = model.encode_image(image, control=True, normalize=True)  # For SrCLIP
        cache = get_local.cache
        attn_maps = cache["ResidualAttentionBlock.attention"]

        for i in tqdm(range(len(attn_maps))):
            attn = torch.from_numpy(attn_maps[i]).unsqueeze(1)
            attn = attn[:, :, 0, 1:].reshape(1, 7, 7)
            visulize_attention_ratio(
                img_path,
                attn.squeeze().cpu().numpy(),
                ratio=0.5,
                cmap="jet",
                save_path=os.path.join(log_dir, f"attention_map_{i}.png"),
            )


if __name__ == "__main__":
    log_dir = "results"

    checkpoint = "logs/srclipv2_ViT-B-32_siglip_b512x1_lr2e-5_e100_fix_exp1/checkpoints/epoch_30.pt"
    model, preprocess = open_clip.create_model_from_pretrained(
        "srclipv2_ViT-B-32", pretrained=checkpoint, is_train=False
    )

    visualize_filter(os.path.join(log_dir, "filter"), model.visual)

    img_path = "datasets/LSDIR/data/LQ/val/0000002.png"
    image = preprocess(Image.open(img_path)).unsqueeze(0)
    # visualize_feature_map(os.path.join(log_dir, "feature_map"), model, image)
    visualize_attention_map(os.path.join(log_dir, "attention_map"), model, image)
