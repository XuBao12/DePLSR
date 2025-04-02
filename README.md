# DePLSR: Degradation Prior Learning for real-world Super-Resolution

---

> **Abstract:** In real-world image super-resolution (Real-SR), complex unknown degradations frequently induce ambiguous local structures, fundamentally constraining reconstruction fidelity. While accurate estimation of degradation processes from low-resolution observations enables SR models to mitigate the domain shift between synthetic training data and real-world imaging conditions, this critical estimation paradigm remains an open research challenge. Inspired by advances in multimodal approaches, we propose DePLSR, a method that leverages large-scale pre-trained vision-language models to learn degradation features. Specifically, we introduce Degradation Adaptor to predict degradation features from LR images while preserving clear content representations. To train DePLSR, we design Semantic-driven Degradation Pipeline and construct a mixed degradation dataset containing LR images with corresponding captions. Additionally, we propose Cross-modal Fusion Module to enable downstream SR models to utilize prior degradation information. Our method has achieved impressive performance, with the highest improvement of 0.98 dB on PSNR. Through extensive experiments on both synthetic and real-world datasets, we demonstrate the superior ability for DePLSR to extract real image degradation and improve super-resolution performance. Visualization results indicate that our model yields better restoration for heavily degraded LR images and showcases the capability in removing complex degradations.

![image-20241118194900229](figs/image-20241118194900229.png)

![compare](figs/compare.png)



## Results

We achieved state-of-the-art performance. Detailed results can be found in the paper. All of the results are reproducible, using the [pre-trained  models](./pretrained_models) we provide.

- Comparision with SOTA in Table 2 of the main paper.

<p align="center">
  <img width="900" src="figs/image-20241118183423824.png">
</p>

<details>
<summary>Click to expand</summary>


- Visual comparison (x4) in Figure 4 of the main paper.

<p align="center">
  <img width="900" src="figs/image-20241118183135926.png">
</p>


- Visual comparison (x4) in Figure 3 of the supplementary material.

<p align="center">
  <img width="900" src="figs/image-20241118183202916.png">
</p>




- Visual comparison (x4) in Figure 4 of the supplementary material.

<p align="center">
  <img width="900" src="figs/image-20241118183214366.png">
</p>
</details>

## How to run the code?

> We will introduce more details about how to run the code after our paper is accepted.

### Datasets

Download training and test datasets and put them into the corresponding folders of `datasets/`. Datasets can be downloaded as follows:

-  Training SRCLIP

Use `scripts/generate_lq.py` to generate LSDIR-DE using our proposed degradation pipeline.
| Training Set                                                 |                         Test Set                          | 
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| LSDIR-DE|  LSDIR-DE|
- Training DAT+SRCLIP

| Training Set                                                 |                         Test Set                          | 
| :----------------------------------------------------------- | :----------------------------------------------------------: |
| [LSDIR](https://data.vision.ee.ethz.ch/yawli/index.html)|  [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)  + LSDIR + [RealSR](https://drive.google.com/file/d/17ZMjo-zwFouxnm_aFM6CUHBwgRrLZqIM/view)|






### Training

#### SRCLIP

```shell
python main.py
```

#### DAT+SRCLIP

- Run the following scripts. The training configuration is in `options/train/`.

```shell
python basicsr/train.py
```



### Test

- Evaluate the accuracies of SRCLIP in degradation classification.

```shell
python evaluate.py
```

- Test super-resolution model with SRCLIP and calculate the metrics.

```shell
python basicsr/test.py
python scripts/iqa.py
```






## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR), [open_clip](https://github.com/mlfoundations/open_clip) and [DAT](https://github.com/zhengchen1999/DAT). [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) is used to calculate metrics. Thanks for their excellent work.
