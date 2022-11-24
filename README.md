# Equipotential Learning for Semantic Segmentation

This is an offical implementation of [Contour-Aware Equipotential Learning for Semantic Segmentation](https://arxiv.org/abs/2210.00223) (accepted by IEEE TMM 2022)
You can download all models in our paper at [Google drive](https://drive.google.com/drive/folders/1KJmzhPK1aFe-BWU5pz2Alv9B-N9RBG3C?usp=sharing)

# Acknowlegements 
Our work is an add-on approach and can be pluged into existing deep semantic segmentation models.

We achieved below PyTorch implementationsb to test our method:
1. [PSPNet & PSANet](https://github.com/hszhao/semseg) (including Pascal Voc and Cityscapes)
2. (Pascal Voc only) [DeepLab V3+](https://github.com/jfzhang95/pytorch-deeplab-xception)
3. (Cityscapes only) [DeepLab V3+](https://github.com/NVIDIA/semantic-segmentation)

The first repository is released by the author of [PSPNet](https://arxiv.org/abs/1612.01105) and [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf), and we achieved powerful mIoU performances on both datasets with DeepLab V3+. 

 
