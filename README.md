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

### 1. ResNet-56 on CIFAR-10.

#### Training & structure pruning stage

```shell
python main.py --job_dir <experiment_results_dir> --teacher_dir <pretrain_weights_dir> --teacher_file <pretrain_weights_file> --refine None --arch resnet --teacher_model resnet_56 --student_model resnet_56_sparse --num_epochs 100 --train_batch_size 128 --eval_batch_size 100 --lr 0.01 --momentum 0.9 --miu 1 --sparse_lambda 0.6 --lr_decay_step 30 --mask_step 200 --weight_decay 0.0002
```

#### Fine-tuning stage

```shell
python finetune.py --job_dir <finetuning_results_dir> --refine <experiment_results_dir> --num_epochs 30 --lr 0.01
```

### Results

Model                | Stage               | #Sructures (blocks)   | FLOPs (pruned ratio)  | #Param (pruned ratio) | Top-1 accuracy
---                  |---                  |---                                    |---                    |---                         |---     
Resnet-56 (Original) |Pretrained           | 27                                    |125.49M (0%)           |0.85M (0%)                  | 93.26  
Resnet-56 (Sparse)   |Training & Pruning   | 27                                    |125.49M (0%)           |0.85M (0%)                  | 91.72      
Resnet-56 (Pruned)   |Pruned & Fine-tuning | 17                                    |79.24M (37.7%)         |0.67M (21.7%)               | 92.22  


