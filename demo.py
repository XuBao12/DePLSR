import torch
from PIL import Image
import open_clip

def demo_srclip():
    checkpoint = "logs/srclip_ViT-B-32_b784x1_lr2e-5_e50_exp1/checkpoints/epoch_5.pt"
    model, preprocess = open_clip.create_model_from_pretrained(
        "srclip_ViT-B-32", pretrained=checkpoint
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    image = preprocess(Image.open("datasets/LSDIR/data/LQ/val/0000015.png")).unsqueeze(0)
      #   degradations = [
      #       "blur with sigma:0.5",
      #       "blur with sigma:2.5",
      #       "gaussian noise with sigma:5",
      #       "gaussian noise with sigma:25",
      #       "poisson noise with sigma:0.5",
      #       "poisson noise with sigma:2.5",
      #       "jpeg with quality factor:40",
      #       "jpeg with quality factor:80",
      #       "ringing and overshoot artifacts"
      #   ]
    degradations = [
        "gaussian noise",
        "poisson noise",
    ]
    text = tokenizer(degradations)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        image_features, degra_features = model.encode_image(image, control=True)
        degra_features /= degra_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
        index = torch.argmax(text_probs[0])

        print(f"Task: {degradations[index]} - {text_probs[0][index]}")
        print(text_probs)

def demo_srclipv2():
    checkpoint = "logs/srclipv2_ViT-B-32_b64x1_lr2e-5_e100_exp1/checkpoints/epoch_50.pt"
    model, preprocess = open_clip.create_model_from_pretrained(
        "srclipv2_ViT-B-32", pretrained=checkpoint
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    image = preprocess(Image.open("datasets/LSDIR/data/LQ/val/0000009.png")).unsqueeze(0)
    # degradations = [
    #     "blur with sigma:0.0",
    #     "blur with sigma:0.5",
    #     "blur with sigma:1.0",
    #     "blur with sigma:1.5",
    #     "blur with sigma:2.0",
    #     "blur with sigma:2.5",
    #     "blur with sigma:3.0",
    # ]
    # degradations = [
    #     "no ringing and overshoot artifacts",
    #     "ringing and overshoot artifacts",
    # ]
    degradations = [
        "gaussian noise with sigma:0",
        "gaussian noise with sigma:15",
        "gaussian noise with sigma:30",
    ]
    text = tokenizer(degradations)

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text)
        image_features, degra_features, fine_degra_features = model.encode_image(image, control=True)
        fine_degra_features /= fine_degra_features.norm(dim=-1, keepdim=True)
        degra_features /= degra_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * fine_degra_features[:,1,:] @ text_features.T).softmax(dim=-1)
        # text_probs = (100.0 * degra_features @ text_features.T).softmax(dim=-1)
        index = torch.argmax(text_probs[0])

        print(f"Task: {degradations[index]} - {text_probs[0][index]}")
        print(text_probs)


if __name__ == "__main__":
    demo_srclipv2()
