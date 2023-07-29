# FFCLIP
Official implementation of "One Model to Edit them all:  Free-Form Text-Driven Image Manipulation with Semantic Modulations" NeurIPS 2022 (Spotlight)

### GUI DISPLAY:
<p align="center">
  <a href='https://github.com/kristen-rang/FFCLIP/blob/main/demo.gif'>
  <img src='https://github.com/kristen-rang/FFCLIP/blob/main/demo.gif' width=600 ></a>

</p>

>**Abstract:** Free-form text prompts allow users to describe their intentions during image manipulation conveniently. Based on the visual latent space of StyleGAN[21] and text embedding space of CLIP[34], studies focus on how to map these two latent spaces for text-driven attribute manipulations. Currently, the latent mapping between these two spaces is empirically designed and confines that each manipulation model can only handle one fixed text prompt. In this paper, we propose a method named Free-Form CLIP (FFCLIP), aiming to establish an automatic latent mapping so that one manipulation model handles free-form text prompts. Our FFCLIP has a cross-modality semantic modulation module containing semantic alignment and injection. The semantic alignment performs the automatic latent mapping via linear transformations with a cross attention mechanism. After alignment, we inject semantics from text prompt embeddings to the StyleGAN latent space. For one type of image (e.g., 'human portrait'), one FFCLIP model can be learned to handle free-form text prompts. Meanwhile, we observe that although each training text prompt only contains a single semantic meaning, FFCLIP can leverage text prompts with multiple semantic meanings for image manipulation. In the experiments, we evaluate FFCLIP on three types of images (i.e., 'human portraits', 'cars', and 'churches'). Both visual and numerical results show that FFCLIP effectively produces semantically accurate and visually realistic images.


### Requirements:
 ```shell script
pip install pytorch=1.7.1 torchvision cudatoolkit 
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git
```

### Data:
We use the FFHQ, LSUN cars, LSUN churches that were inverted by e4e: train set, test set, validation set. https://github.com/omertov/encoder4editing

### Usage:
#### Training
The main training script is placed in Coach.py
Training arguments can be found at train_configs.yaml.
Attributes are set in the data/dataset.py 
Intermediate training results are saved to opts.base_dir. This includes checkpoints, train outputs, and test outputs. 

In addition, we provide various pretrained models needed for training.
| Path | Description
| :--- | :----------
|[FFHQ StyleGAN](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing) | StyleGAN model pretrained on FFHQ taken from [rosinality](https://github.com/rosinality/stylegan2-pytorch) with 1024x1024 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during training.
|[MOCOv2 Model](https://drive.google.com/file/d/18rLcNGdteX5LwT7sv_F7HWr12HpVEzVe/view?usp=sharing) | Pretrained ResNet-50 model trained using MOCOv2 for use in our simmilarity loss for domains other then human faces during training.
|[Parsenet Model](https://github.com/stoneyang-detection/caffe_ssd) | Pretrained Parsenet model to segment background during training.

### Pretrained Models for Inference:
We release our checkpoints on human face, car, church dataset. During Training, we found the batch size and GPU mermory is essential to a good performance. We encourge readers to train and finetune the model by youself if you have better experiment enviorments.
#### FFHQ Dataset:
Download link: https://drive.google.com/file/d/1oYMi5jsrUI_WyQ-xGf-F2T6myVGBx3od/view?usp=drive_link
#### LUSN Car Dataset:
Download link: https://drive.google.com/file/d/11Zsx_bjMbIzzTjLxWDhOEgoXmnbNwH7t/view?usp=drive_link
#### LUSN Church Dataset:
Download link: https://drive.google.com/file/d/1ihmvppjLXh71ARx4NLnIMzJRAFLdQnmn/view?usp=drive_link
### Citation
If you use this code for your research, please cite our paper: One Model to Edit them all:  Free-Form Text-Driven Image Manipulation with Semantic Modulations
https://arxiv.org/abs/2210.07883
```
@inproceedings{zhuone,
  title={One Model to Edit Them All: Free-Form Text-Driven Image Manipulation with Semantic Modulations},
  author={Zhu, Yiming and Liu, Hongyu and Song, Yibing and Yuan, Ziyang and Han, Xintong and Yuan, Chun and Chen, Qifeng and Wang, Jue},
  booktitle={Advances in Neural Information Processing Systems}
}
```
