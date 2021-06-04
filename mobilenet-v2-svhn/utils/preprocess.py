import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from importlib import import_module
from model.mobilenetV2 import pruned_mobile_v2_sparse, mobile_v2_sparse
import torch

from collections import OrderedDict


device = torch.device('cuda:5') 

'''
state_dict = torch.load('experiment/mobile/t_1_mask_0.3_sigma_0.2_lambda_0.001_0.01_kd_0.8_1/' + 'checkpoint/model_best.pt', map_location = device)
state_dict = state_dict['state_dict_s']

list(state_dict.keys())
list(pruned_model.state_dict().keys())

name1 = 'layers.15.shortcut.1.weight'
weight1 = state_dict[name1].shape

name2 =  'layers.15.shortcut.1.num_batches_tracked'
weight2 = state_dict[name2].shape

name3 = 'layers.4.shortcut.1.weight'
weight3 = state_dict[name3].shape



np.percentile([weight for name, weight in state_dict.items() if 'mask.' in name], 60)

model.state_dict()['layer3.11.bn1.running_var']
'''

def mobilenetV2(args, state_dict, thres = None): # thres
    default_cfg = [([[16]*3], 1, 1),
                   ([[24]*3]*2, 2, 1),  
                   ([[32]*3]*3, 3, 2),
                   ([[64]*3]*4, 4, 2),
                   ([[96]*3]*3, 3, 1),
                   ([[160]*3]*3, 3, 2),
                   ([[320]*3], 1, 1)]

    
    if thres is None:
        thres = 0.0
    
    ''' Masked structure '''
    mask_block = []
    has_mask = []
    for name, weight in state_dict.items():
        if 'mask.' in name:
            if weight.item() <= thres:
                mask_block.append(torch.FloatTensor([0.0]))
                has_mask.append(0)
            else:    
                mask_block.append(weight.item())
                has_mask.append(1)
        
    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0) if not m == 0 and not m == 16]
    
    has_mask[0] = 1
    
    has_mask[-1] = 1
    
    ''' Update key block-wisely '''
    new_state_dict = OrderedDict()
    
    COUNT = 0
    last_block_no = 0

    for key, value in state_dict.items():
        if 'layers' in key and 'surv' not in key:
            block_no = key.split('.')[1]
            
            if int(block_no) == 0:
                new_state_dict[key] = state_dict[key]

            elif has_mask[int(block_no)]:
                if int(block_no) != last_block_no:
                    last_block_no += 1
                    COUNT += 1
                new_key = [key.split('.')[0]] + [str(COUNT)] + key.split('.')[2:]
                new_key = '.'.join(new_key)
                new_state_dict[new_key] = state_dict[key]
            
            else:
                if int(block_no) != last_block_no:
                    last_block_no += 1
        
        elif 'surv' not in key:
            new_state_dict[key] = state_dict[key]
    
    
    ''' Padding shortcut (conv) layer '''
    '''
    num_blocks = sum(has_mask)
    
    for i in range(num_blocks):
        if f'layers.{i}.shortcut.0.weight' in list(new_state_dict.keys()):
            old_weight = new_state_dict[f'layers.{i}.shortcut.0.weight']
            
            if i == num_blocks - 1:
                ref_weight = new_state_dict['conv2.weight']
            else:
                ref_weight = new_state_dict[f'layers.{i+1}.conv1.weight']
            
            if old_weight.shape[0] != ref_weight.shape[1]:
                pad_size = int((ref_weight.shape[1]-old_weight.shape[0])/2)
                pad_weight = torch.zeros([pad_size, 
                                          old_weight.shape[1],
                                          old_weight.shape[2], 
                                          old_weight.shape[3]]).to(device)
                new_weight = torch.cat((pad_weight, old_weight), 0)
                new_weight = torch.cat((new_weight, pad_weight), 0)
                
                new_state_dict[f'layers.{i}.shortcut.0.weight'] = new_weight
                
                
                params = ['weight', 'bias', 'running_mean', 'running_var']
                
                for param in params:
                    old_weight = new_state_dict[f'layers.{i}.shortcut.1.{param}']
                    
                    new_weight = torch.cat((torch.zeros(pad_size).to(device), old_weight))
                    new_weight = torch.cat((new_weight, torch.zeros(pad_size).to(device)))
                    
                    new_state_dict[f'layers.{i}.shortcut.1.{param}'] = new_weight
                    '''
            
    ''' Load model and weights '''
    pruned_model = pruned_mobile_v2_sparse(has_mask = has_mask).to(device)

    
    print(f"Pruned / Total: {pruned_num} / {len(mask_block)}")
    print("Pruned blocks", pruned_blocks)

    
    if not args.random:
        pruned_model.load_state_dict(new_state_dict, strict = False)

    return pruned_model #sparse_model

'''
tmp = torch.ones([1, 3, 32, 32])
out = pruned_model(tmp.cuda())
'''

def resnet(args, state_dict, t, thres = None): # thres
    
    n = (56 - 2) // 6
    layers = np.arange(0, 3*n ,n)
    
    if thres is None:
        thres = 0.0
       
        
    mask_block = []
    for name, weight in state_dict.items():
        if 'mask.' in name:
            if weight.item() <= thres:
                mask_block.append(torch.FloatTensor([0.0]))
            else:    
                mask_block.append(weight.item())
    
     
    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0)]

    
    '''
    masks = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            masks.append(np.array(weight.view(-1).clone().cpu()).reshape([-1]))

    #masks_global_mean = np.mean(np.array(masks))
    
    mask_block = [] # ratios of non-zero elements
    for i in range(int(len(masks)/2)):
        mask_block.append((sum(masks[i * 2] > 0.0) + sum(masks[i * 2 + 1] > 0.0)) / (masks[i * 2].shape[0] + masks[i * 2 + 1].shape[0]))

    for i, m in enumerate(mask_block):
        if (1 - m) >= thres:
            mask_block[i] = 0.0
    
    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0)]
    

    mask_block = [] # ratios of non-zero elements
    for i in range(int(len(masks)/2)):
        mask_block.append(np.mean(np.hstack((masks[i * 2] , masks[i * 2 + 1]))))

    for i, m in enumerate(mask_block):
        if m < np.percentile(mask_block, 1 / n * grow ** epoch * 100):
            mask_block[i] = 0.0
    
    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0)]
    '''

    old_block = 0
    layer = 'layer1'
    layer_num = int(layer[-1])
    new_block = 0
    new_state_dict = OrderedDict()
    
    #COUNT = 0
    for key, value in state_dict.items():
        if 'layer' in key:
            if key.split('.')[0] != layer:
                layer = key.split('.')[0]
                layer_num = int(layer[-1])
                new_block = 0

            if key.split('.')[1] != old_block:
                old_block = key.split('.')[1]

            if mask_block[layers[layer_num-1] + int(old_block)] == 0:
                if layer_num != 1 and old_block == '0' and 'mask' in key:
                    new_block = 1
                continue

            new_key = re.sub(r'\.\d+\.', '.{}.'.format(new_block), key, 1)
            
                 
            if 'mask.' in new_key: 
                new_block += 1

            new_state_dict[new_key] = state_dict[key]
            
            '''    
            new_key = re.sub(r'\.\d+\.', '.{}.'.format(new_block), key, 1)
            
            if 'mask2' in new_key: 
                new_block += 1

                if mask_block[COUNT] == 0:
                    new_state_dict[new_key] = torch.zeros_like(state_dict[key])
                    #new_state_dict[new_key.replace('2', '1')] = torch.zeros_like(state_dict[new_key.replace('2', '1')])
                else:
                    new_state_dict[new_key] = state_dict[key]
                    
            else:
                new_state_dict[new_key] = state_dict[key]
                '''
        else:
            new_state_dict[key] = state_dict[key]
    
        #sparse_model = resnet_110_sparse(has_mask = mask_block).to(device)
        
        pruned_model = pruned_resnet_56_sparse(has_mask = mask_block).to(device)


    print(f"Pruned / Total: {pruned_num} / {len(mask_block)}")
    print("Pruned blocks", pruned_blocks)
    #print(f'Channel pruning ratio: {[1-m for m in mask_block if (1-m) > 0]}')
    
    '''
    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    print(f'=> Saving pruned model to {save_dir}')
    
    save_state_dict = {}
    save_state_dict['state_dict_s'] = new_state_dict
    save_state_dict['mask'] = mask_block
    torch.save(save_state_dict, save_dir)
    '''
    
    if not args.random:
        #sparse_model.load_state_dict(new_state_dict, strict = False)
        pruned_model.load_state_dict(new_state_dict, strict = False)

    return pruned_model #sparse_model


def densenet(args, state_dict):
    pruned_num = 0
    total = 0
    filters_list = []
    indexes_list = []
    indexes_dense = []
    indexes_trans = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.clone()
            indexes = np.array(weight_copy.gt(0.0).cpu())
            selected_index = np.squeeze(np.argwhere(indexes))

            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))
            filters = sum(indexes)

            size = weight_copy.size()[0]

            if (size - filters) == size:
                selected_index = [0]
                filters = 1

            if 'dense' in name:
                indexes_dense.append(selected_index)
            
            if 'trans' in name:
                indexes_trans.append(selected_index)

            indexes_list.append(selected_index)
            filters_list.append(filters)

            pruned_num += size - filters
            total += size

            state_dict[name] = weight_copy[selected_index]

    i = 0
    j = 0
    for name, weight in state_dict.items():
        if 'conv1' in name:
            if 'dense' in name:
                state_dict[name] = state_dict[name][:, indexes_dense[i], :, :]
                i += 1
            if 'trans' in name:
                state_dict[name] = state_dict[name][:, indexes_trans[j], :, :]
                j += 1


    model = densenet_40_sparse(filters=filters_list, indexes=indexes_list).to(args.gpus[0])
    model.load_state_dict(state_dict)

    print(f"Pruned / Total: {pruned_num} / {total}")
    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    save_state_dict = {}
    save_state_dict['state_dict_s'] = state_dict
    save_state_dict['filters'] = filters_list
    save_state_dict['indexes'] = indexes_list
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')

    return model

def vgg(args, state_dict):
    pruned_num = 0
    total = 0
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    indexes_list = [np.arange(3)]

    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.clone()
            indexes = np.array(weight_copy.gt(0.0).cpu())  # index to retrain
            selected_index = np.squeeze(np.argwhere(indexes))
            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))
            filters = sum(indexes)
            size = weight_copy.size()[0]

            if (size - filters) == size:
                selected_index = [0]
                filters = 1

            if 'features' in name:
                # get the name of conv in state_dict
                idx = int(re.findall(r"\d+", name.split('.')[1])[0])
                cfg[idx] = filters
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][selected_index, :, :, :]
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][:, indexes_list[-1], :, :]
                state_dict['features.conv{}.bias'.format(idx)] = state_dict['features.conv{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.weight'.format(idx)] = state_dict['features.norm{}.weight'.format(idx)][selected_index]
                state_dict['features.norm{}.bias'.format(idx)] = state_dict['features.norm{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.running_mean'.format(idx)] = state_dict['features.norm{}.running_mean'.format(idx)][
                    selected_index]
                state_dict['features.norm{}.running_var'.format(idx)] = state_dict['features.norm{}.running_var'.format(idx)][
                    selected_index]

            elif 'classifier' in name:
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][selected_index, :]
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][:,indexes_list[-1]]
                state_dict['classifier.linear1.bias'] = state_dict['classifier.linear1.bias'][selected_index]
                state_dict['classifier.norm1.running_mean'] = state_dict['classifier.norm1.running_mean'][selected_index]
                state_dict['classifier.norm1.running_var'] = state_dict['classifier.norm1.running_var'][selected_index]
                state_dict['classifier.norm1.weight'] = state_dict['classifier.norm1.weight'][selected_index]
                state_dict['classifier.norm1.bias'] = state_dict['classifier.norm1.bias'][selected_index]
                cfg[-1] = filters
                state_dict['classifier.linear2.weight'] = state_dict['classifier.linear2.weight'][:, selected_index]

            indexes_list.append(selected_index)
            pruned_num += size - filters
            total += size

            # change the state dict of mask
            state_dict[name] = weight_copy[selected_index]

    model = vgg_16_bn_sparse(cfg=cfg).to(args.gpus[0])
    model.load_state_dict(state_dict)

    print(f"Pruned / Total: {pruned_num} / {total}")

    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    save_state_dict = {}
    save_state_dict['state_dict_s'] = state_dict
    save_state_dict['cfg'] = cfg
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')

    return model

def googlenet(args, state_dict):
    filters = [
        [64, 128, 32, 32],
        [128, 192, 96, 64],
        [192, 208, 48, 64],
        [160, 224, 64, 64],
        [128, 256, 64, 64],
        [112, 288, 64, 64],
        [256, 320, 128, 128],
        [256, 320, 128, 128],
        [384, 384, 128, 128]
    ]
    mask = []
    arr = []
    i = 0

    for name, weight in state_dict.items():
        if 'mask' in name:
            mask.append(weight.item())

    cfg = [int(m) for m in np.argwhere(np.array(mask)==0)]
    module_name = 'inception_a3'
    index = list(range(192)) 

    i = 0
    for name, param in state_dict.items():
        if 'branch1x1.0.weight' in name or 'branch3x3.0.weight' in name or 'branch5x5.0.weight' in name or 'branch_pool.1.weight' in name:
            # get module_name
            if name.split('.')[0] != module_name:
                start = 0
                index = []

                for j, v in enumerate(mask[i * 4:i * 4 + 4]):
                    if v != 0:
                        index.extend([i for i in range(start, start + filters[i][j])]) # select index
                    start += filters[i][j]
                    
                i += 1
                module_name = name.split('.')[0]
                
            state_dict[name] = param[:, index, :, :]

        elif 'linear.weight' in name:
            fc_index = []
            fc_start = 0
            for j, v in enumerate(mask[8*4:8*4+4]):
                if v != 0:
                    fc_index.extend([i for i in range(fc_start, fc_start + filters[8][j])])
                fc_start += filters[i][j]
            state_dict[name] = param[:, fc_index]

    pruned_num = sum(m == 0 for m in mask)
    model = googlenet_sparse(has_mask=mask).to(args.gpus[0])

    print('\n---- After Prune ----\n')
    
    model_dict = model.state_dict()
    pretrained = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    save_state_dict = {}
    save_state_dict['state_dict_s'] = pretrained
    save_state_dict['mask'] = mask
    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')
    
    return model
