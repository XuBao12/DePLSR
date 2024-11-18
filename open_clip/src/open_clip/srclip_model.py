from typing import Optional

import logging
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import copy


from .transformer import (
    ControlTransformer
)
from .model import CLIP, CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


class SrCLIP(nn.Module):
    def __init__(self, clip_model: CLIP):
        super().__init__()
        self.clip = clip_model
        self.visual = clip_model.visual
        self.visual_control = copy.deepcopy(clip_model.visual)
        self.visual_control.transformer = ControlTransformer(self.visual_control.transformer)
        self.logit_scale = copy.deepcopy(clip_model.logit_scale)

    def initial_controller(self):
        for (kv, param_v), (kc, param_c) in zip(self.clip.visual.named_parameters(), self.visual_control.named_parameters()):
            if 'transformer' not in kv:
                param_c.data.copy_(param_v.data)
        for param_v, param_c in zip(self.clip.visual.transformer.parameters(), self.visual_control.transformer.parameters()):
            param_c.data.copy_(param_v.data)

        self.logit_scale.data.copy_(self.clip.logit_scale.data)

    def lock_clip(self):
        for param in self.clip.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.clip.visual.set_grad_checkpointing(enable)
        self.clip.transformer.grad_checkpointing = enable
        self.visual_control.set_grad_checkpointing(enable)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            image_features = self.clip.visual(image, control=hiddens)

            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            return image_features, degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def encode_text(self, text, normalize: bool = False):
        return self.clip.encode_text(text, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        (caption, degradation) = text.chunk(2, dim=-1) if text is not None else (None, None)
        image_features, image_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features,
            "logit_scale": self.logit_scale.exp()
        }


class SrCLIPV2(SrCLIP):
    def __init__(self, clip_model: CLIP, scale_degra_dim=5):
        super().__init__(clip_model)
        self.scale_degra_dim = scale_degra_dim
        self.last_dim = self.visual_control.output_dim # 512
        self.out_linear = nn.Linear(self.last_dim, scale_degra_dim * self.last_dim)

    def encode_image(self, image, control=False, normalize: bool = False):
        if control:
            degra_features, hiddens = self.visual_control(image, output_hiddens=True)
            image_features = self.clip.visual(image, control=hiddens)
            fine_degra_features = self.out_linear(degra_features).view(degra_features.size(0), self.scale_degra_dim, self.last_dim) # b n 512

            image_features = F.normalize(image_features, dim=-1) if normalize else image_features
            degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
            fine_degra_features = F.normalize(fine_degra_features, dim=-1) if normalize else fine_degra_features
            return image_features, degra_features, fine_degra_features
        else:
            return self.clip.encode_image(image, normalize)

    def forward(
            self,
            image: Optional[torch.Tensor] = None, # b * 3 * 224 * 224
            text: Optional[torch.Tensor] = None, # b * (1+1+n) * 77
    ):
        (caption, degradation, fine_degradation) = text[:, 0, :], text[:, 1, :], text[:, 2:, :] if text is not None else (None, None, None)
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None
        text_fine_degra_features = []
        for i in range(fine_degradation.size(1)):
            text_fine_degra_feature = self.encode_text(fine_degradation[:, i, :], normalize=True) if fine_degradation[i] is not None else None
            text_fine_degra_features.append(text_fine_degra_feature)
        text_fine_degra_features = torch.stack(text_fine_degra_features, dim=1)

        image_features, image_degra_features, image_fine_degra_features = self.encode_image(image, control=True, normalize=True) if image is not None else None

        output_dict = {
            "image_features": image_features,
            "text_features": text_features, # b 512
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features, # b 512
            "image_fine_degra_features": image_fine_degra_features,
            "text_fine_degra_features": text_fine_degra_features, # b n 512
            "logit_scale": self.logit_scale.exp()
        }
    
        if self.logit_bias is not None:
            output_dict["logit_bias"] = self.logit_bias

        return output_dict

class FinetuneCLIP(CLIP):
    def __init__(self, scale_degra_dim=5, **kwargs):
        super().__init__(**kwargs)                                   
        self.scale_degra_dim = scale_degra_dim
        self.last_dim = 512 # TODO 改成clip的输出维度
        self.out_linear = nn.Linear(self.last_dim, scale_degra_dim * self.last_dim)


        self.fine_linear = nn.Linear(self.last_dim, (scale_degra_dim+1) * self.last_dim)  #512 * 6
        self.fine1_linear = nn.Linear((scale_degra_dim+1) * self.last_dim, ((scale_degra_dim+1) * self.last_dim) // 2) #256 *6
        self.fine2_linear = nn.Linear(((scale_degra_dim+1) * self.last_dim) // 2, (scale_degra_dim+1) * self.last_dim) # 512 *6
        self.fine3_linear = nn.Linear(((scale_degra_dim+1) * self.last_dim), self.last_dim) # 512 *5
       
    def encode_image(self, image, normalize: bool = False):
        image_features = self.visual(image)
        degra_features = self.fine_linear(image_features)
        degra_features = torch.relu(degra_features)  # 使用ReLU激活函数
        degra_features = self.fine1_linear(degra_features)
        degra_features = torch.relu(degra_features)
        degra_features = self.fine2_linear(degra_features) 
        degra_features = torch.relu(degra_features)
        degra_features = self.fine3_linear(degra_features)

        fine_degra_features = self.out_linear(degra_features).view(degra_features.size(0), self.scale_degra_dim, self.last_dim) # b n 512
        
        image_features = F.normalize(image_features, dim=-1) if normalize else image_features
        degra_features = F.normalize(degra_features, dim=-1) if normalize else degra_features
        fine_degra_features = F.normalize(fine_degra_features, dim=-1) if normalize else fine_degra_features
        return image_features, degra_features, fine_degra_features
        
    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        (caption, degradation, fine_degradation) = text[:, 0, :], text[:, 1, :], text[:, 2:, :] if text is not None else (None, None, None)
        text_features = self.encode_text(caption, normalize=True) if text is not None else None
        text_degra_features = self.encode_text(degradation, normalize=True) if degradation is not None else None
        text_fine_degra_features = []
        for i in range(fine_degradation.size(1)):
            text_fine_degra_feature = self.encode_text(fine_degradation[:, i, :], normalize=True) if fine_degradation[i] is not None else None
            text_fine_degra_features.append(text_fine_degra_feature)
        text_fine_degra_features = torch.stack(text_fine_degra_features, dim=1)

        image_features, image_degra_features, image_fine_degra_features = self.encode_image(image, normalize=True) if image is not None else None

        output_dict = {
            "image_features": image_features,
            "text_features": text_features, # b 512
            "image_degra_features": image_degra_features,
            "text_degra_features": text_degra_features, # b 512
            "image_fine_degra_features": image_fine_degra_features,
            "text_fine_degra_features": text_fine_degra_features, # b n 512
            "logit_scale": self.logit_scale.exp()
        }
    
        if self.logit_bias is not None:
            output_dict["logit_bias"] = self.logit_bias

        return output_dict
