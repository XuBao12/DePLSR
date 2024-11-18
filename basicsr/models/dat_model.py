import open_clip
import torch
import torch.nn as nn
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from collections import OrderedDict


@MODEL_REGISTRY.register()
class DATModel(SRModel):

    def test(self):
        self.use_chop = self.opt['val']['use_chop'] if 'use_chop' in self.opt['val'] else False
        if not self.use_chop:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(self.lq)
                self.net_g.train()

        # test by partitioning
        else:
            _, C, h, w = self.lq.size()
            split_token_h = h // 200 + 1  # number of horizontal cut sections
            split_token_w = w // 200 + 1  # number of vertical cut sections

            patch_size_tmp_h = split_token_h
            patch_size_tmp_w = split_token_w

            # padding
            mod_pad_h, mod_pad_w = 0, 0
            if h % patch_size_tmp_h != 0:
                mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
            if w % patch_size_tmp_w != 0:
                mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w

            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]

            _, _, H, W = img.size()
            split_h = H // split_token_h  # height of each partition
            split_w = W // split_token_w  # width of each partition

            # overlapping
            shave_h = 16
            shave_w = 16
            scale = self.opt.get('scale', 1)
            ral = H // split_h
            row = W // split_w
            slices = []  # list of partition borders
            for i in range(ral):
                for j in range(row):
                    if i == 0 and i == ral - 1:
                        top = slice(i * split_h, (i + 1) * split_h)
                    elif i == 0:
                        top = slice(i*split_h, (i+1)*split_h+shave_h)
                    elif i == ral - 1:
                        top = slice(i*split_h-shave_h, (i+1)*split_h)
                    else:
                        top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                    if j == 0 and j == row - 1:
                        left = slice(j*split_w, (j+1)*split_w)
                    elif j == 0:
                        left = slice(j*split_w, (j+1)*split_w+shave_w)
                    elif j == row - 1:
                        left = slice(j*split_w-shave_w, (j+1)*split_w)
                    else:
                        left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                    temp = (top, left)
                    slices.append(temp)
            img_chops = []  # list of partitions
            for temp in slices:
                top, left = temp
                img_chops.append(img[..., top, left])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g_ema(chop)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                            if j == 0:
                                _left = slice(0, split_w*scale)
                            else:
                                _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
            else:
                self.net_g.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g(chop)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                            if j == 0:
                                _left = slice(0, split_w * scale)
                            else:
                                _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
                self.net_g.train()
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

@MODEL_REGISTRY.register()
class RealDATModel(SRModel):
    def __init__(self, opt):
        super(RealDATModel, self).__init__(opt)

        self.clip_type = opt['network_clip']['type']
        if opt['path']['srclip_pretrain_network'] is not None:
            clip_model, _ = open_clip.create_model_from_pretrained(
                self.clip_type, pretrained=opt['path']['srclip_pretrain_network'], is_train=False)
        else:
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.clip_model = clip_model.to(self.device)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        img4clip = nn.functional.interpolate(self.lq, size=(224,224), mode='bilinear', align_corners=False)
        with torch.no_grad(), torch.cuda.amp.autocast():
            if "srclipv2_" in self.clip_type:
                image_features, degra_features, fine_degra_features = self.clip_model.encode_image(
                    img4clip, control=True
                )
            else:
                image_features = self.clip_model.encode_image(img4clip)
                fine_degra_features = image_features.unsqueeze(1).repeat(1, 5, 1)
            image_features /= image_features.norm(dim=-1, keepdim=True) # (N, 512)
            fine_degra_features /= fine_degra_features.norm(dim=-1, keepdim=True) # (N, 5, 512)

        self.xi = image_features.unsqueeze(1).float().to(self.device)
        self.xd = fine_degra_features.float().to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.xi, self.xd)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.use_chop = self.opt['val']['use_chop'] if 'use_chop' in self.opt['val'] else False
        if not self.use_chop:
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    self.output = self.net_g_ema(self.lq, self.xi, self.xd)
            else:
                self.net_g.eval()
                with torch.no_grad():
                    self.output = self.net_g(self.lq, self.xi, self.xd)
                self.net_g.train()

        # test by partitioning
        else:
            _, C, h, w = self.lq.size()
            split_token_h = h // 200 + 1  # number of horizontal cut sections
            split_token_w = w // 200 + 1  # number of vertical cut sections

            patch_size_tmp_h = split_token_h
            patch_size_tmp_w = split_token_w

            # padding
            mod_pad_h, mod_pad_w = 0, 0
            if h % patch_size_tmp_h != 0:
                mod_pad_h = patch_size_tmp_h - h % patch_size_tmp_h
            if w % patch_size_tmp_w != 0:
                mod_pad_w = patch_size_tmp_w - w % patch_size_tmp_w

            img = self.lq
            img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h+mod_pad_h, :]
            img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w+mod_pad_w]

            _, _, H, W = img.size()
            split_h = H // split_token_h  # height of each partition
            split_w = W // split_token_w  # width of each partition

            # overlapping
            shave_h = 16
            shave_w = 16
            scale = self.opt.get('scale', 1)
            ral = H // split_h
            row = W // split_w
            slices = []  # list of partition borders
            for i in range(ral):
                for j in range(row):
                    if i == 0 and i == ral - 1:
                        top = slice(i * split_h, (i + 1) * split_h)
                    elif i == 0:
                        top = slice(i*split_h, (i+1)*split_h+shave_h)
                    elif i == ral - 1:
                        top = slice(i*split_h-shave_h, (i+1)*split_h)
                    else:
                        top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                    if j == 0 and j == row - 1:
                        left = slice(j*split_w, (j+1)*split_w)
                    elif j == 0:
                        left = slice(j*split_w, (j+1)*split_w+shave_w)
                    elif j == row - 1:
                        left = slice(j*split_w-shave_w, (j+1)*split_w)
                    else:
                        left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                    temp = (top, left)
                    slices.append(temp)
            img_chops = []  # list of partitions
            for temp in slices:
                top, left = temp
                img_chops.append(img[..., top, left])
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g_ema(chop, self.xi, self.xd)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                            if j == 0:
                                _left = slice(0, split_w*scale)
                            else:
                                _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
            else:
                self.net_g.eval()
                with torch.no_grad():
                    outputs = []
                    for chop in img_chops:
                        out = self.net_g(chop, self.xi, self.xd)  # image processing of each partition
                        outputs.append(out)
                    _img = torch.zeros(1, C, H * scale, W * scale)
                    # merge
                    for i in range(ral):
                        for j in range(row):
                            top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                            left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                            if i == 0:
                                _top = slice(0, split_h * scale)
                            else:
                                _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                            if j == 0:
                                _left = slice(0, split_w * scale)
                            else:
                                _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                            _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                    self.output = _img
                self.net_g.train()
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]