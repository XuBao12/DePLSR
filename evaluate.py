import torch
from PIL import Image
import argparse
import open_clip
from tqdm import tqdm
from open_clip_train.srclip_dataset import SRClipPairDatasetV2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SRCLIP")
    parser.add_argument(
        "--output",
        type=str,
        help="name of the output file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="logs/srclipv2_ViT-B-32_siglip_b512x1_lr2e-5_e100_fix_exp1/checkpoints/epoch_30.pt",
        help="Path to the checkpoint file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="datasets/LSDIR/data/srclip_val_lq_v2.csv",
        help="Path to the validation data CSV file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SRClipPairDatasetV2",
        help="name of the dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="srclipv2_ViT-B-32",
        help="name of the model",
    )
    return parser.parse_args()


class Degra_base:
    def __init__(self, name, idx):
        self.name = name
        self.idx = idx
        self.acc = 0
        self.prompts = []

    def update_acc(self, label, pred, bias):
        label = float(label.split(":")[-1])
        pred = float(pred.split(":")[-1])
        if abs(label - pred) < bias:
            self.acc += 1


class Degra_blur(Degra_base):
    def __init__(self, name="blur", idx=0):
        super().__init__(name, idx)
        self.prompts = [
            "blur with sigma:0.0",
            # "blur with sigma:0.5",
            "blur with sigma:1.0",
            # "blur with sigma:1.5",
            "blur with sigma:2.0",
            # "blur with sigma:2.5",
            "blur with sigma:3.0",
        ]

    def update_acc(self, label, pred, bias=0.5):
        return super().update_acc(label, pred, bias)


class Degra_gaussian_noise(Degra_base):
    def __init__(self, name="gaussian_noise", idx=1):
        super().__init__(name, idx)
        self.prompts = [
            "gaussian noise with sigma:0",
            "gaussian noise with sigma:15",
            "gaussian noise with sigma:30",
        ]

    def update_acc(self, label, pred, bias=10):
        return super().update_acc(label, pred, bias)


class Degra_poisson_noise(Degra_base):
    def __init__(self, name="poisson_noise", idx=2):
        super().__init__(name, idx)
        self.prompts = [
            "poisson noise with scale:0.0",
            # "poisson noise with scale:0.5",
            "poisson noise with scale:1.0",
            # "poisson noise with scale:1.5",
            "poisson noise with scale:2.0",
            # "poisson noise with scale:2.5",
            "poisson noise with scale:3.0",
        ]

    def update_acc(self, label, pred, bias=0.5):
        return super().update_acc(label, pred, bias)


class Degra_jpeg(Degra_base):
    def __init__(self, name="jpeg", idx=3):
        super().__init__(name, idx)
        self.prompts = [
            "jpeg with quality factor:30",
            # "jpeg with quality factor:40",
            "jpeg with quality factor:50",
            # "jpeg with quality factor:60",
            "jpeg with quality factor:70",
            # "jpeg with quality factor:80",
            "jpeg with quality factor:90",
        ]

    def update_acc(self, label, pred, bias=10):
        return super().update_acc(label, pred, bias)


class Degra_ringing(Degra_base):
    def __init__(self, name="ringing", idx=4):
        super().__init__(name, idx)
        self.prompts = [
            "no ringing and overshoot artifacts",
            "ringing and overshoot artifacts",
        ]

    def update_acc(self, label, pred, bias=0.2):
        if label == pred:
            self.acc += 1


class Degra:
    def __init__(self):
        self.blur = Degra_blur()
        self.gaussian_noise = Degra_gaussian_noise()
        self.poisson_noise = Degra_poisson_noise()
        self.jpeg = Degra_jpeg()
        self.ringing = Degra_ringing()

    def __iter__(self):
        return iter([self.blur, self.gaussian_noise, self.poisson_noise, self.jpeg, self.ringing])


if __name__ == "__main__":
    args = parse_args()
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

    degra = Degra()
    for de in degra:
        prompts = [templates(d) for d in de.prompts]
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        de.text_features = text_features

    for idx in tqdm(range(len(data))):
        image = data.transforms(Image.open(str(data.images[idx]))).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            image_features, degra_features, fine_degra_features = model.encode_image(
                image, control=True
            )
            fine_degra_features /= fine_degra_features.norm(dim=-1, keepdim=True)

            for de in degra:
                text_probs = (
                    100.0 * fine_degra_features[:, de.idx, :] @ de.text_features.T
                ).softmax(dim=-1)
                index = torch.argmax(text_probs[0])
                # print(f"Task: {de.name} - {de.prompts[index]}, prob: {text_probs[0][index]}")
                de.update_acc(
                    label=data.fine_degradations[de.idx][idx], pred=de.prompts[index]
                )

    for de in degra:
        print(f"Task: {de.name}, acc: {de.acc/len(data)}")
