# Dominance in an Overall Sight: Attentive Channel Pruning via Filter Importance Learning (DOMINOS)
Quantization, Efficient Inference, Neural Networks

## Requirements

* python3
* pytorch==1.7.1
* cudatoolkit==11.0.221 
* numpy==1.19.2
* tensorboardx==1.4
* ptflops ([Github](https://github.com/sovrasov/flops-counter.pytorch))


## Pretrained weights download

* CIFAR-10
  - [VGG-16](https://drive.google.com/file/d/1Jk3gxPIQMZMw42ihyN5YNq3--12P654V/view?usp=sharing)
  - [ResNet-56](https://drive.google.com/file/d/1z1oz-Set-iTnpOU2si-bqySzgcWzQjZN/view?usp=sharing)
  - [ResNet-110](https://drive.google.com/file/d/1jOApBG_cvvSaAna8Jz4eXZP426ncX-6T/view?usp=sharing)
  - [DenseNet-40](https://drive.google.com/file/d/1xeFqb2FU7d3Dq4T7tcRx_Gu6oFAUKOku/view?usp=sharing)

* CIFAR-100
  - [ResNet-56](https://drive.google.com/file/d/12NTMSaEx1Th3HNIFIxk68j3Hr-VF5UXq/view?usp=sharing)
  - [ResNet-110](https://drive.google.com/file/d/1sIAsteFdPDAKN9KnYB6oCYbAx1dXnA4H/view?usp=sharing)

* SVHN
  - [VGG-16](https://drive.google.com/file/d/1H6I1s0wAfl1avuaiI61xdmD99xzJ82h8/view?usp=sharing)
  - [MobileNet-v2](https://drive.google.com/file/d/1vX7I7xTyrrmYlkTaLOdu0cPck-3knAMM/view?usp=sharing)

* ImageNet
  - [ResNet-50](https://drive.google.com/file/d/1Zg7lxT-X7nmvEkL6fEik2gQdWuP_cGey/view?usp=sharing)


## Experiment

### e.g. ResNet-56 on CIFAR-10.

### Preparation of pretrained weights

```shell
cd dominos/resnet-56-cifar-10
mkdir pretrained/
```
Download the pretrained weights under the path dominos/resnet-56-cifar-10/pretrained/.

### Dominant Structure Search (DSS) stage

```shell
python main.py --job_dir <pruning_results_dir> --teacher_dir <pretrain_weights_dir> --teacher_file <pretrain_weights_file> --refine None --arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse --num_epochs 100 --train_batch_size 128 --eval_batch_size 100 --lr 0.01 --momentum 0.9 --miu 1 --sigma 0.2 --mask 0.3 --sparse_lambda 0.001 --sparse_lambda2 0.01 --lr_decay_step 30 --mask_step 200 --weight_decay 0.0002 --t 2 --thres 0.2
```

### Fine-tuning stage

```shell
python ft.py --job_dir <finetuning_results_dir> --refine <pruning_results_dir> --num_epochs 100 --lr 0.05
```

## Results

Dataset              |Model                | Top-1 err (%)     | Acc loss (%)   | FLOPs (pruned ratio)  | #Param (pruned ratio)
---                  |---                  |---                |---             |---                    |---                    
CIFAR-10             |VGG-16               | 6.59              |0.55            |67.7%                  |71.0%    
CIFAR-10             |VGG-16               | 6.77              |0.73            |71.0%                  |83.0%  
---                  |---                  |---                |---             |---                    |---   
CIFAR-10             |VGG-16               | 6.77              |0.73            |71.0%                  |83.0% 

