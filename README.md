
# AUC-CL: A Batchsize-Robust Framework for Self-Supervised Contrastive Representation Learning
PyTorch implementation for [AUC-CL](https://openreview.net/forum?id=YgMdDQB09U&referrer=%5Bthe%20profile%20of%20Kaiyi%20Ji%5D(%2Fprofile%3Fid%3D~Kaiyi_Ji1).

## Abstract
Self-supervised learning through contrastive representations is an emergent and promising avenue, aiming at alleviating the availability of labeled data. Recent research in the field also demonstrates its viability for several downstream tasks, henceforth leading to works that implement the contrastive principle through innovative loss functions and methods. However, despite achieving impressive progress, most methods depend on prohibitively large batch sizes and compute requirements for good performance. In this work, we propose the AUC-Contrastive Learning, a new approach to contrastive learning that demonstrates robust and competitive performance in compute-limited regimes. We propose to incorporate the contrastive objective within the AUC-maximization framework, by noting that the AUC metric is maximized upon enhancing the probability of the network's binary prediction difference between positive and negative samples which inspires adequate embedding space arrangements in representation learning. Unlike standard contrastive methods, when performing stochastic optimization, our method maintains unbiased stochastic gradients and thus is more robust to batchsizes as opposed to standard stochastic optimization problems. Remarkably, our method with a batch size of 256, outperforms several state-of-the-art methods that may need much larger batch sizes (e.g., 4096), on ImageNet and other standard datasets. Experiments on transfer learning, few-shot learning, and other downstream tasks also demonstrate the viability of our method.

## Training Cifar
To train a ResNet-50 model on the Cifar-10 dataset, run the following

```
cd cifar
python main.py
```
This generates the k-NN accuracy on the validation set at every epoch.

## Training ImageNet
```
python -m torch.distributed.launch --nproc_per_node=8 main.py --arch vit_small --data_path /path/to/imagenet/train --output_dir /path/to/saving_dir
```

## Evaluation: k-NN classification on ImageNet
To evaluate a simple k-NN classifier with a single GPU on a pre-trained model, run:
```
python -m torch.distributed.launch --nproc_per_node=1 eval_knn.py --pretrained_weights /path/to/checkpoint.pth --checkpoint_key teacher --data_path /path/to/imagenet 
```

## Evaluation: Linear classification on ImageNet
To train a supervised linear classifier on frozen weights on a single node with 8 gpus, run:
```
python -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --data_path /path/to/imagenet
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
The code for this repo is based on the [Dino code base](https://github.com/facebookresearch/dino).

## Citation

```
@inproceedings{
sharma2024auccl,
title={{AUC}-{CL}: A Batchsize-Robust Framework for Self-Supervised Contrastive Representation Learning},
author={Rohan Sharma and Kaiyi Ji and zhiqiang xu and Changyou Chen},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=YgMdDQB09U}
}
```
