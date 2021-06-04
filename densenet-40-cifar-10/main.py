import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from fista import FISTA
# from model import Discriminator

from data import cifar10

from ptflops import get_model_complexity_info # from thop import profile


# torch.backends.cudnn.benchmark = False
device = torch.device(f"cuda:{args.gpus[0]}")

checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    loader = cifar10(args)

    # Create model
    print('=> Building model...')
    
    model_t = import_module(f'model.{args.arch}').__dict__[args.teacher_model]().to(device)
    
    model_s = import_module(f'model.{args.arch}').__dict__[args.student_model](T = args.t).to(device)
    
    if args.pretrained:
        # Load pretrained weights
        ckpt = torch.load(args.teacher_dir + args.teacher_file, map_location = device)
        state_dict = ckpt[list(ckpt.keys())[0]]
        state_dict = dict((k[7:].replace('linear', 'fc'), v) for (k, v) in state_dict.items())
        
        model_dict_s = model_s.state_dict()
        model_dict_s.update(state_dict)
        model_s.load_state_dict(model_dict_s)
        model_s = model_s.to(device)
    
    model_t.load_state_dict(state_dict)
    model_t = model_t.to(device)
    
    models = [model_t, model_s]
    
    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]   

    optimizer_s = optim.SGD(param_s, lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    optimizer_m = FISTA(param_m, lr = args.lr, gamma = args.sparse_lambda)

    scheduler_s = StepLR(optimizer_s, step_size = args.lr_decay_step, gamma = 0.1)
    scheduler_m = StepLR(optimizer_m, step_size = args.lr_decay_step, gamma = 0.1)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location=device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_s.load_state_dict(ckpt['state_dict_s'])

        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])

        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        
        print('=> Continue from epoch {}...'.format(start_epoch))

    '''
    if args.test_only:
        test_prec1, test_prec5 = test(args, loader.loader_test, model_t)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return
    '''

    optimizers = [optimizer_s, optimizer_m]
    schedulers = [scheduler_s, scheduler_m]
    
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)

        train(args, loader.loader_train, models, optimizers, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)
        
        '''
        model_state_dict = model_t.module.state_dict() if len(args.gpus) > 1 else model_t.state_dict()
        '''
        
        state = {
            'state_dict_s': model_s.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            
            'optimizer_s': optimizer_s.state_dict(),
            'optimizer_m': optimizer_m.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'scheduler_m': scheduler_m.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)
        
        
        model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, model_s.state_dict(), args.t)
        flops, params = get_model_complexity_info(model.to(device), (3, 32, 32), as_strings = False, print_per_layer_stat = False)
        compressionInfo(epoch, flops, params, test_prec1, test_prec5)

    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

    best_model = torch.load(f'{args.job_dir}checkpoint/model_best.pt', map_location = device)


def compressionInfo(epoch, flops, params, test_prec1, test_prec5, org_gflops = 0.2901354911, org_params = 1.0620989113000001):
    GFLOPs = flops / 10 ** 9
    params_num = params
    params_mem = params / 1000 ** 2
    pruned_FLOPs_ratio = (org_gflops - GFLOPs) / org_gflops
    pruned_param_ratio = (org_params - params_mem) / org_params
    
    test_prec1 = test_prec1.item()
    test_prec5 = test_prec5.item()
    
    print(f'Model FLOPs: {round(GFLOPs*1000, 2)} (-{round(pruned_FLOPs_ratio, 4) * 100} %)')
    print(f'Model params: {round(params_mem, 2)} (-{round(pruned_param_ratio, 4) * 100} %) MB')
    print(f'Model num of params: {round(params_num)}\n')
    
    if not os.path.isdir(args.job_dir + '/run/plot'):
        os.makedirs(args.job_dir + '/run/plot')        
        with open(args.job_dir + 'run/plot/compressInfo.txt', 'w') as f:
            f.write('epoch, top-1, top-5, flops, flops-pr, param_mb, param_mb-pr, num_param, \n')
    
    with open(args.job_dir + 'run/plot/compressInfo.txt', 'a') as f:
        f.write(f'{epoch}, {round(test_prec1, 4)}, {round(test_prec5, 4)}, {round(GFLOPs*1000, 2)}, {round(pruned_FLOPs_ratio, 4) * 100}, {round(params_mem, 2)}, {round(pruned_param_ratio, 4) * 100}, {round(params_num)}\n')
        
    with open(args.job_dir + 'run/plot/compressInfo_r.txt', 'a') as f:
        f.write('Epoch[{0}]\n'.format(epoch))
        f.write('Top-1: {0}\nTop-5: {1}\n'.format(round(test_prec1, 4), round(test_prec5, 4)))
        f.write('FLOPs: {0} (-{1} %)\n'.format(round(GFLOPs*1000, 2), round(pruned_FLOPs_ratio, 4) * 100))
        f.write('Params: {0} (-{1} %) MB\n'.format(round(params_mem, 2), round(pruned_param_ratio, 4) * 100))
        f.write('Num of params: {}\n'.format(round(params_num)))
        f.write('===========================\n')
        
        
def train(args, loader_train, models, optimizers, epoch):
    losses_s = utils.AverageMeter()
    losses_sparse = utils.AverageMeter()
    losses_redundant = utils.AverageMeter()
    losses_cascade = utils.AverageMeter()
    losses_kd = utils.AverageMeter()
    
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
    model_s = models[1]
    
    for param in list(model_t.parameters())[:-2]:
        param.requires_grad = False
        
    for name, param in model_s.named_parameters():
        param.requires_grad = True
        
    cross_entropy = nn.CrossEntropyLoss()
    
    optimizer_s = optimizers[0]
    optimizer_m = optimizers[1]
    
    # switch to train mode
    model_t.train()
    model_s.train()
        
    num_iterations = len(loader_train)
    
    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer_s.zero_grad()
        optimizer_m.zero_grad()

    
        ## train weights
        output_t = model_t(inputs).to(device)
        output_s = model_s(inputs).to(device)
        
        error_s = cross_entropy(output_s, targets)

        error_s.backward(retain_graph = True) # retain_graph = True
        
        losses_s.update(error_s.item(), inputs.size(0))
        
        writer_train.add_scalar('Performance_loss', error_s.item(), num_iters)
        
        if args.arch == 'densenet': 
            mask = []
            for j, layer in enumerate([model_s.dense1, model_s.trans1, model_s.dense2, model_s.trans2, model_s.dense3]):
                if j % 2 == 0:
                    for block in layer:                  
                        mask.append(block.mask.alpha.view(-1))
                else:
                    mask.append(layer.mask.alpha.view(-1))
            mask = torch.cat(mask)
            attention = model_s.att # [batch_size, total_num_channels]
            
            
            error_sparse = args.sparse_lambda * (torch.norm(mask, 1) / len(mask))
            error_sparse.backward(retain_graph = True)

            error_redundant_mimic = args.mask * torch.mean(1 - torch.sum(mask.view([1, -1]) * attention, dim = 1)/ torch.norm(mask, 2)) 
            error_redundant_mimic.backward(retain_graph = True)
            
            
            losses_sparse.update(error_sparse.item(), inputs.size(0))
            writer_train.add_scalar('Sparse_loss', error_sparse.item(), num_iters)
            
            losses_redundant.update(error_redundant_mimic.item(), inputs.size(0))
            writer_train.add_scalar('Redundancy_imitation_loss', error_redundant_mimic.item(), num_iters)

            if args.t > 0:
                surv = model_s.weibull_fs
                surv = torch.cat(surv)
                
                error_info_cascade = args.sigma * (-1) * torch.mean(torch.log(surv + 1e-5))
                error_info_cascade.backward()
                
                losses_cascade.update(error_info_cascade.item(), inputs.size(0))
                writer_train.add_scalar('Cascades_fit_loss', error_info_cascade.item(), num_iters)
            
            error_kd = args.kd * (-1) * torch.mean(F.softmax(output_t, -1) * torch.log(F.softmax(output_s, -1)))
            error_kd.backward()
            
            losses_kd.update(error_kd.item(), inputs.size(0))
            writer_train.add_scalar('KD_loss', error_kd.item(), num_iters)
        
            ## step forward
            optimizer_s.step()
            
            decay = (epoch % args.lr_decay_step == 0 and i == 1)
            if num_iters % args.mask_step == 0:
                optimizer_m.step(decay)
            

        ## evaluate
        prec1, prec5 = utils.accuracy(output_s, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        writer_train.add_scalar('Train-top-1', top1.avg, num_iters)
        writer_train.add_scalar('Train-top-5', top5.avg, num_iters)
        
        if i % args.print_freq == 0:
            if args.t > 0:
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Sparse_loss: {sparse_loss.val:.4f} ({sparse_loss.avg:.4f})\n'
                    'Redundant_loss: {redundant_loss.val:.4f} ({redundant_loss.avg:.4f})\n'
                    'Cascade_loss: {cascade_loss.val:.4f} ({cascade_loss.avg:.4f})\n'
                    'KD_loss: {kd_loss.val:.4f} ({kd_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_s, 
                    sparse_loss = losses_sparse,
                    redundant_loss = losses_redundant,
                    cascade_loss = losses_cascade,
                    kd_loss = losses_kd,
                    top1 = top1, top5 = top5))
            else:
                print_logger.info(
                    'Epoch[{0}]({1}/{2}): \n'
                    'Train_loss: {train_loss.val:.4f} ({train_loss.avg:.4f})\n'
                    'Sparse_loss: {sparse_loss.val:.4f} ({sparse_loss.avg:.4f})\n'
                    'Redundant_loss: {redundant_loss.val:.4f} ({redundant_loss.avg:.4f})\n'
                    'KD_loss: {kd_loss.val:.4f} ({kd_loss.avg:.4f})\n'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'.format(
                    epoch, i, num_iterations, 
                    train_loss = losses_s,
                    sparse_loss = losses_sparse,
                    redundant_loss = losses_redundant,
                    kd_loss = losses_kd,
                    top1 = top1, top5 = top5))
            
            pruned = torch.sum(mask == 0).detach().cpu()
            num = len(mask)
           
            print_logger.info("Pruned {} / {}\n".format(pruned, num))
 
            
def test(args, loader_test, model_s, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs).to(device)
            loss = cross_entropy(logits, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
        
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Test-top-1', top1.avg, num_iters)
            writer_test.add_scalar('Test-top-5', top5.avg, num_iters)
        
    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg
    

if __name__ == '__main__':
    main()

