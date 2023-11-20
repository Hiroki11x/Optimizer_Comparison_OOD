# Empirical Study on Optimizer Selection for Out-of-Distribution Generalization

![Screenshot 2022-07-14 at 2 51 49 PM](https://user-images.githubusercontent.com/8721858/201168624-cdc92939-25e4-465c-978d-8d9c41fc07d4.png)


## Abstract
Modern deep learning systems are fragile and do not generalize well under distribution shifts. While much promising work has been accomplished to address these concerns, a systematic study of the role of optimizers and their out-of-distribution generalization performance has not been undertaken. In this study, we examine the performance of popular first-order optimizers for different classes of distributional shift under empirical risk minimization and invariant risk minimization. We address the problem settings for image and text classification using DomainBed, WILDS, and Backgrounds Challenge as out-of-distribution datasets for the exhaustive study. We search over a wide range of hyperparameters and examine the classification accuracy (in-distribution and out-of-distribution) for over 20,000 models. We arrive at the following findings:  i) contrary to conventional wisdom, adaptive optimizers (e.g., Adam) perform worse than non-adaptive optimizers (e.g., SGD, momentum-based SGD),  ii) in-distribution performance and out-of-distribution performance exhibit three types of behavior depending on the dataset – linear returns, increasing returns, and diminishing returns.  We believe these findings can help practitioners choose the right optimizer and know what behavior to expect. 

## Prerequisites
- Python >= 3.6.5
- Pytorch >= 1.6.0
- cuDNN >= 7.6.2
- CUDA >= 10.0

## Downloads (10 Datasets)
1. [ColoredMNIST](https://github.com/facebookresearch/DomainBed)
2. [RotatedMNIST](https://github.com/facebookresearch/DomainBed)
3. [VLCS](https://github.com/facebookresearch/DomainBed)
4. [PACS](https://github.com/facebookresearch/DomainBed)
5. [OfficeHome](https://github.com/facebookresearch/DomainBed)
6. [DomainNet](https://github.com/facebookresearch/DomainBed)
7. [TerraIncognita](https://github.com/facebookresearch/DomainBed)
8. [Background Challenge](https://github.com/MadryLab/backgrounds_challenge)
9. [WILDS Amazon](https://github.com/p-lambda/wilds)
10. [WILDS CivilComment](https://github.com/p-lambda/wilds)

## Implementation
As for the DomainBed, WILDS and Background Challenge implementations, we follow the official implementations shown in the links below.

- [DomainBed](https://github.com/facebookresearch/DomainBed)
- [Background Challenge](https://github.com/MadryLab/backgrounds_challenge)
- [WILDS](https://github.com/p-lambda/wilds)

## Citation
TMLR (2023) / Paper [Link / OpenReview](https://openreview.net/forum?id=ipe0IMglFF)

```
@article{
naganuma2023empirical,
title={Empirical Study on Optimizer Selection for Out-of-Distribution Generalization},
author={Hiroki Naganuma and Kartik Ahuja and Shiro Takagi and Tetsuya Motokawa and Rio Yokota and Kohta Ishikawa and Ikuro Sato and Ioannis Mitliagkas},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=ipe0IMglFF},
note={}
}
```

## Paper authors
- [Hiroki Naganuma]()
- [Kartik Ahuja](https://ahujak.github.io/)
- [Shiro Takagi](https://t46.github.io/)
- [Tetsuya Motokawa](https://github.com/mtkwT)
- [Rio Yokota](https://www.rio.gsic.titech.ac.jp/en/index.html)
- [Kohta Ishikawa](https://dblp.org/pid/157/8482.html)
- [Ikuro Sato](https://scholar.google.com/citations?user=WGKTs8sAAAAJ&hl=ja)
- [Ioannis Mitliagkas](http://mitliagkas.github.io/)


