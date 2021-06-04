import os
import re
import time
import utils.common as utils
from importlib import import_module
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.options import args
import pdb
from model import *
from model.resnet import pruned_resnet_110_sparse

from fista import FISTA

from ptflops import get_model_complexity_info # from thop import profile


device = torch.device(f'cuda:{args.gpus[0]}') # f"cuda:{args.gpus[0]}"
ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')

def main():
    start_epoch = 0
    best_prec1, best_prec5 = 0.0, 0.0


    # Data loading
    print('=> Preparing data..')
    loader = import_module('data.' + args.dataset).Data(args)

    # Create model
    print('=> Building model...')
    criterion = nn.CrossEntropyLoss()

    # original model
    model_t = import_module(f'model.{args.arch}').__dict__[args.teacher_model]().to(device)
    
    checkpoint = torch.load(args.teacher_dir + args.teacher_file, map_location = device)
    state_dict = checkpoint[list(checkpoint.keys())[0]]
    state_dict = dict((k.replace('linear', 'fc'), v) for (k, v) in state_dict.items())
    
    model_t.load_state_dict(state_dict)
    model_t = model_t.to(device)
    
    # Fine tune from a checkpoint
    refine = args.refine
    assert refine is not None, 'refine is required'
    checkpoint = torch.load(refine, map_location = device)
    
    # pruned model
    model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, checkpoint['state_dict_s'], args.t, args.thres)
    #model = torch.load_state_dict(checkpoint['state_dict_s'])
    
    
    flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings = False, print_per_layer_stat = False)
    compressionInfo(flops, params)
    
    models = [model_t, model]
        
    if args.test_only:
        return 

    if args.keep_grad:
        for name, weight in model.named_parameters():
            if 'mask' in name: 
                weight.requires_grad = False

    train_param = [param for name, param in model.named_parameters() if 'surv' not in name and 'mask.' not in name]
    
    optimizer = optim.SGD(train_param, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    scheduler = StepLR(optimizer, step_size = args.lr_decay_step, gamma = 0.1)

    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))


    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)

        train(args, loader.loader_train, models, criterion, optimizer, writer_train, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model, criterion, writer_test, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        state = {
            'state_dict_s': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }

        ckpt.save_model(state, epoch + 1, is_best)

    print_logger.info(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

def compressionInfo(flops, params, org_gflops = 0.25604, org_params = 1.7401079956118397):
    GFLOPs = flops / 10 ** 9
    params_num = params
    params_mem = params / 1000 ** 2
    pruned_FLOPs_ratio = (org_gflops - GFLOPs) / org_gflops
    pruned_param_ratio = (org_params - params_mem) / org_params
    
    print(f'Model FLOPs: {round(GFLOPs*1000, 2)} (-{round(pruned_FLOPs_ratio, 4) * 100} %)')
    print(f'Model params: {round(params_mem, 2)} (-{round(pruned_param_ratio, 4) * 100} %) MB')
    print(f'Model num of params: {round(params_num)}')
    
    if not os.path.isdir(args.job_dir + '/run/plot'):
        os.makedirs(args.job_dir + '/run/plot')
    
    with open(args.job_dir + 'run/plot/compressInfo.txt', 'w') as f:
        f.write('Model FLOPs: {0} (-{1} %)\n'.format(round(GFLOPs*1000, 2), round(pruned_FLOPs_ratio, 4) * 100))
        f.write('Model params: {0} (-{1} %) MB\n'.format(round(params_mem, 2), round(pruned_param_ratio, 4) * 100))
        f.write('Model num of params: {}\n'.format(round(params_num)))
        
def train(args, loader_train, models, criterion, optimizer, writer_train, epoch):
    losses = utils.AverageMeter()
    losses_kd = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    model_t = models[0]
    model = models[1]
    
    for param in list(model_t.parameters())[:-2]:
        param.requires_grad = False
   
    for name, param in model.named_parameters():
        param.requires_grad = True
  
    model_t.train()
    model.train()
    num_iterations = len(loader_train)

    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i
        
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # performance
        logits = model(inputs).to(device)
        loss = criterion(logits, targets)
        loss = args.miu * criterion(logits, targets) 
        loss.backward(retain_graph = True) # retain_graph = True
        losses.update(loss.item(), inputs.size(0))
        
        
        writer_train.add_scalar('Train_loss (fine-tuned)', loss.item(), num_iters)
        
        optimizer.step()
        
        
        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))     
        
        writer_train.add_scalar('Prec@1', top1.avg, num_iters)
        writer_train.add_scalar('Prec@5', top5.avg, num_iters)
        

       
        # knowledge distillation
        features_t = model_t(inputs).to(device)
        
        #error_data = args.miu * F.mse_loss(features_t, features.to(device))
        error_data = args.kd * (-1) * torch.mean(F.softmax(features_t, -1) * torch.log(F.softmax(logits, -1)))
        
        losses_kd.update(error_data.item(), inputs.size(0))
        error_data.backward()
        
        writer_train.add_scalar('KD_loss', error_data.item(), num_iters)
      

        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                'KD_loss: {kd_loss.val:.4f} ({kd_loss.avg:.4f})\n'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                epoch, i, num_iterations, 
                train_loss = losses, 
                kd_loss = losses_kd,
                top1 = top1, top5 = top5))


def test(args, loader_test, model, criterion, writer_test, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs).to(device)
            loss = criterion(logits, targets)
            
            writer_test.add_scalar('Test_loss (fine-tuned)', loss.item(), num_iters)
            
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Prec@1', top1.avg, num_iters)
            writer_test.add_scalar('Prec@5', top5.avg, num_iters)

    print_logger.info(f'[Epoch {epoch}] * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===========================================================\n')
  
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
