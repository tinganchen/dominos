import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Channel Pruning')

MIU = 1

MASK = 0.1
LAMBDA = 0.0005 ##  0.0005 sparsity
LAMBDA2 = 0.01 ## 0.012, 0.01, 0.01 sparsity
KD = 0.8 ## knowledge distillation

SIGMA = 0.2 #  ## cascade
T = 2

THRES = 0.375

## rand_0, 0, 1, 2
# 0.125, 0.17, 0.12, 0.18 ## 20PR
# 0.145, 0.42, 0.135, 0.32 ## 30PR
# 0.2, 0.4445, 0.15, 0.374 ## 40PR
# 0.3, 0.449, 0.388, 0.3816 ## 50PR
# 0.335, 0.45, 0.3963, 0.39 ## 60PR
# 0.345, 0.463, 0.398, 0.396 ## 70PR
# 0.3478, 0.5, 0.42, 0.4 ## 80PR

INDEX = '0'
PRETRAINED = True
PRUNED = False


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset', type = str, default = 'cifar100', help = 'Dataset to train')

parser.add_argument('--data_dir', type = str, default = os.getcwd() + '/data/cifar100/', help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = f'experiment/resnet/rere2_ft66_t_{T}_mask_{MASK}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{KD}_{INDEX}/', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_thres_{THRES}_t_{T}_sigma_{SIGMA}_lambda_{LAMBDA}_{INDEX}/'

parser.add_argument('--pretrained', action = 'store_true', default = PRETRAINED, help = 'Load pruned model')
parser.add_argument('--teacher_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--teacher_file', type = str, default = 'model_best.pt', help = 'The file the teacher model weights saved as.')

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')

parser.add_argument('--refine', type = str, default = f'experiment/resnet/t_{T}_mask_{0.7}_sigma_{SIGMA}_lambda_{LAMBDA}_{LAMBDA2}_kd_{0.8}_{INDEX}/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None
#parser.add_argument('--refine', type = str, default = None, help = 'Path to the model to be fine tuned.') # None


## Training
parser.add_argument('--arch', type = str, default = 'resnet', help = 'Architecture of teacher and student')
parser.add_argument('--target_model', type = str, default = '9-block', help = 'The target model.')
parser.add_argument('--student_model', type = str, default = 'resnet_110_sparse', help = 'The model of student.')

parser.add_argument('--teacher_model', type = str, default = 'resnet_110', help = 'The model of teacher.')
parser.add_argument('--num_epochs', type = int, default = 100, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 128, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 100, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 5e-2)
# 1. train: default = 0.1
# 2. fine_tuned: default = 5e-2

parser.add_argument('--lr_decay_step',type = int, default = 30)
# 1. train: default = 30
# 2. fine_tuned: default = 30

parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update') # 800
parser.add_argument('--weight_decay', type = float, default = 2e-4, help = 'The weight decay of loss.')

parser.add_argument('--miu', type = float, default = MIU, help = 'The miu of data loss.')
parser.add_argument('--sigma', type = float, default = SIGMA, help = 'The sigma of survival loss.')
parser.add_argument('--mask', type = float, default = MASK, help = 'The penalty for importance scores imitation.')
parser.add_argument('--lambda', dest = 'sparse_lambda', type = float, default = LAMBDA, help = 'The sparse lambda for l1 loss') # 0.6
parser.add_argument('--lambda2', dest = 'sparse_lambda2', type = float, default = LAMBDA2, help = 'The sparse lambda for l1 loss') # 0.6
parser.add_argument('--kd', dest = 'kd', type = float, default = KD, help = 'The knowledge distillation multiplier.')
parser.add_argument('--t', type = int, default = T, help = 'The num of layers that filters affect.')

parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')
parser.add_argument('--pruned', action = 'store_true', default = PRUNED, help = 'Load pruned model')

#parser.add_argument('--seed', type = int, default = SEED, help = 'Random seed.')

parser.add_argument('--thres', type = float, default = THRES, help = 'Threshold of zeo-masked channels that the block will be pruned.')
#parser.add_argument('--grow', type = float, default = GROW, help = 'Threshold of zeo-masked channels that the block will be pruned.')
parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')

## Status
parser.add_argument('--print_freq', type = int, default = 200, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 


args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

