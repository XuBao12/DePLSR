import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import math
import pandas as pd
from PIL import Image
import torch
from torch.utils import data as data
from torch.nn import functional as F

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels, add_gaussian_noise_pt, add_poisson_noise_pt
from basicsr.data.transforms import augment
from basicsr.utils.diffjpeg import DiffJPEG
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SRClipDataset(data.Dataset):
    """Dataset used for Real-ESRGAN model:
    Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    It loads gt (Ground-Truth) images, and augments them.
    It also generates blur kernels and sinc kernels for generating low-quality images.
    Note that the low-quality images are processed in tensors on GPUS for faster processing.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            meta_info (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).
            Please see more options in the codes.
    """

    def __init__(self, opt, tokenizer):
        super(SRClipDataset, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.lq_size = opt['lq_size'] if type(opt['lq_size']) is tuple else (opt['lq_size'], opt['lq_size'])
        self.tokenizer = tokenizer

        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            # disk backend with meta_info
            # Each line in the meta_info describes the relative path to an image and its caption.
            df = pd.read_csv(self.opt['meta_info'], sep='\t')
            self.paths = df['filepath'].tolist()
            self.captions = df['title'].tolist()

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        # TODO: kernel range is now hard-coded, should be in the configure file
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.jpeger = DiffJPEG(differentiable=False).to(self.device)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------- Get semantic caption -------------------- #
        caption = str(self.captions[index])
        caption = self.tokenizer([caption])[0]

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]

        # -------------------- Get degradation captions -------------------- #
        kernels = self.get_kernels()
        img_lq, degradation_caption = self.degradation(img_gt, kernels)
        img_lq = img_lq.detach()

        #TODO cat degradation的方式 1. 231维，2.77维
        degradation = self.tokenizer("".join(degradation_caption))[0]

        return_d = {'gt': img_gt, 'gt_path': gt_path, 'lq': img_lq, 'caption': caption, 'degradation':degradation, 'degradation_caption': degradation_caption, 'semantic_caption': str(self.captions[index])}

        return img_lq, torch.cat([caption, degradation], dim=0), return_d

    def __len__(self):
        return len(self.paths)

    def get_kernels(self):
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        blur_sigma = random.choice(np.arange(self.blur_sigma[0], self.blur_sigma[1], self.blur_sigma[2]))
        kernel_size = 2 * math.ceil(3 * blur_sigma) + 1
        sigma = (blur_sigma, blur_sigma)
        if np.random.uniform() < self.opt['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                sigma,
                sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        # 和1中的参数保持一致
        blur_sigma2 = blur_sigma
        kernel_size2 = 2 * math.ceil(3 * blur_sigma2) + 1
        sigma2 = (blur_sigma2, blur_sigma2)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size2 < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size2, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size2,
                sigma2,
                sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size2) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
            final_sinc = True
        else:
            sinc_kernel = self.pulse_tensor
            final_sinc = False

        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'blur_sigma': blur_sigma, 'final_sinc': final_sinc}

    def degradation(self, img_gt, kernels):
        #TODO batch并行计算
        img_gt = img_gt.unsqueeze(0).to(self.device)
        kernel1 = kernels['kernel1'].to(self.device)
        kernel2 = kernels['kernel2'].to(self.device)
        sinc_kernel = kernels['sinc_kernel'].to(self.device)
        blur_sigma = kernels['blur_sigma']
        final_sinc = kernels['final_sinc']
        degradation_caption = []

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(img_gt, kernel1)
        degradation_caption.append(f'blur with sigma:{blur_sigma:.1f}')
        # add noise
        if np.random.uniform() < self.opt['gaussian_noise_prob']:
            noise_sigma = np.random.uniform(*self.opt['noise_range'])
            out = add_gaussian_noise_pt(out, sigma=noise_sigma, clip=True, rounds=False)
            degradation_caption.append(f'gaussian noise with sigma:{noise_sigma:.1f}')
            gaussian_flag = True
        else:
            noise_scale = np.random.uniform(*self.opt['poisson_scale_range'])
            out = add_poisson_noise_pt(out, scale=noise_scale, clip=True, rounds=False)
            degradation_caption.append(f'poisson noise with scale:{noise_scale:.1f}')
            gaussian_flag = False

        # ----------------------- The second degradation process ----------------------- #
        # same as the first degradation process and parameters
        out = filter2D(img_gt, kernel2)
        # add noise
        if gaussian_flag:
            out = add_gaussian_noise_pt(out, sigma=noise_sigma, clip=True, rounds=False)
        else:
            out = add_poisson_noise_pt(out, scale=noise_scale, clip=True, rounds=False)
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
        degradation_caption.append(f'jpeg with quality factor:{jpeg_p.item():.1f}')
        # jpeg_p = torch.round(jpeg_p * 10) / 10
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = self.jpeger(out, quality=jpeg_p)

        # the final sinc filter
        if final_sinc:
            out = filter2D(out, sinc_kernel)
            degradation_caption.append('ringing and overshoot artifacts')

        # downsample to the lq size
        out = F.interpolate(out, size=self.lq_size, mode='bicubic')
        out = out.squeeze(0)

        return out, degradation_caption

class SRClipPairDataset(data.Dataset):
    def __init__(self, input_filename, transforms, tokenizer=None, crop=False):
        self.df = pd.read_csv(input_filename, sep='\t')

        self.images = self.df['filepath'].tolist()
        self.captions = self.df['title'].tolist()
        self.degradations = self.df['title2'].tolist()
        self.transforms = transforms

        self.tokenize = tokenizer
        self.crop = crop

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        caption = str(self.captions[idx])
        degradation = str(self.degradations[idx])

        caption = self.tokenize([caption])[0]
        degradation = self.tokenize([degradation])[0]
        texts = torch.cat([caption, degradation], dim=0)

        return images, texts

class SRClipPairDatasetV2(SRClipPairDataset):
    def __init__(self, input_filename, transforms, tokenizer=None, crop=False, num_degradations=5):
        super().__init__(input_filename, transforms, tokenizer, crop)
        self.fine_degradations = [] # num_degradations * [list]
        for i in range(num_degradations):
            self.fine_degradations.append(self.df['title'+str(i+3)].tolist())

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        caption = str(self.captions[idx])
        degradation = [str(self.degradations[idx])]
        for fine_de in self.fine_degradations:
            degradation.append(str(fine_de[idx])) # num_degradations + 1

        caption = self.tokenize([caption])[0].unsqueeze(0)
        degradation = [self.tokenize(i)[0] for i in degradation]
        degradation =  torch.stack([i for i in degradation], dim=0) # 6(1+num) * 77
        texts = torch.cat([caption, degradation], dim=0) # 7 * 77

        return images, texts


if __name__ == "__main__":
    import yaml
    from open_clip import get_tokenizer
    from open_clip_train.srclip_dataset import SRClipDataset
    import time
    from tqdm import tqdm
    opt_path = "options/train_srclip.yml"
    tokenizer = get_tokenizer("srclip_ViT-B-32")
    with open(opt_path, 'r') as f:
        opt = yaml.safe_load(f)
    data = SRClipDataset(opt['datasets']['train'], tokenizer)

    for i in tqdm(range(len(data))):
        x = data[i]
        gt = x['gt'].cuda()
        lq = x['lq'].cuda()
        caption = x['caption'].cuda()
        pass