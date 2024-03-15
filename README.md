
# Self-Supervised Vision: AUC-CL

PyTorch implementation and pretrained models for AUC-CL: A Batchsize-Robust Framework for Self-Supervised Contrastive Representation Learning

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
The code for this repo is based on the [Dino code base](https://github.com/facebookresearch/dino)

