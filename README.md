# SMAR-Net

This repository contains the source code for paper "Self-supervised Multi-scale Adversarial Regression Network for Stereo Disparity Estimation"

## Abstract

Deep learning approaches have significantly contributed to recent progress in stereo matching. These deep stereo matching methods are usually based on supervised training, which requires a large amount of high- quality ground truth depth map annotation data. However, collecting large amounts of ground truth depth data can be very expensive. Further, only a limited quantity of stereo vision training data is currently available, obtained either by active sensors (Lidar, ToF cameras) or through computer graphics simulations and not meeting requirements for deep supervised training. Here, we propose a novel deep stereo approach called ``Self-supervised Multi-scale Adversarial Regression Network (SMAR-Net)", which relaxes the need for ground truth depth maps for training. Specifically, we design a two-stage network: the first stage is a disparity regressor, in which the regression network estimates disparity values from stacked stereo image pairs; stereo image stacking method is a novel contribution as it not only contains the spatial appearances of stereo images but also implies matching correspondences with different disparity values. In the second stage, a synthetic left image is generated based on the left-right consistency assumption. Our network is trained by minimizing a hybrid loss function composed of a content-loss and an adversarial-loss. The content loss minimizes the average warping error between the synthetic images and the real ones. In contrast to the conventional generative adversarial loss, our proposed adversarial loss penalizes mismatches using multi-scale features. This constrains the synthetic image and real image as being pixel-wise identical instead of just belonging to the same distribution. Further, the combined utilization of multi-scale feature extraction in both the content loss and adversarial loss further improves the adaptability of SMAR-Net in ill-posed regions. Experiments on multiple benchmark datasets show that SMAR-Net outperforms the current state-of-the-art self-supervised methods and achieves comparable outcomes to supervised methods.

![Architecture](https://github.com/Dawnstar8411/SMAR-Net/blob/master/Images/SMAR-Net.png)

### Dependencies

- [Python3.5](https://www.python.org/downloads/)
- [PyTorch (1.0.1)](http://pytorch.org)
- [torchvision (0.2.0)](http://pytorch.org)
- [KITTI Stereo](http://www.cvlibs.net/datasets/kitti/eval_object.php)
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### Results on KITTI2015 leaderboard

[leaderboard website](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

### Compared with self-supervised methods
| Method | >2px | >3px| >5px |Mean error|
|---|---|---|---|---}
| PSMNet | 4.85 % | 3.56 % | 2.43 |0.74 px
| [Zhou et al](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhou_Unsupervised_Learning_of_ICCV_2017_paper.pdf) | 4.01 % | 2.73 % | 2.05% | 0.67px
| [Tonioni et al](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tonioni_Unsupervised_Adaptation_for_ICCV_2017_paper.pdf) | 2.87 % | 2.61 % | 0.90 |
| [Lar et al](https://papers.nips.cc/paper/6639-semi-supervised-learning-for-optical-flow-with-generative-adversarial-networks.pdf)| 3.91 % | 2.58% | 1.83 | 0.63px|
|SMAR-Net|3.71%|2.42%|1.75%|0.61px|

### Visualized results on KITTI VO dataset

![KITTI VO](https://github.com/Dawnstar8411/SMAR-Net/blob/master/Images/KITTI_VO.gif)

### Visualized results on Beihang Dataset

![Beihang](https://github.com/Dawnstar8411/SMAR-Net/blob/master/Images/Beihang.gif)