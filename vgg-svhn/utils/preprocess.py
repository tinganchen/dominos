import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from importlib import import_module
from model.vgg import * 
# resnet_56_sparse, densenet_40_sparse, vgg_16_bn_sparse, googlenet_sparse

import torch

from collections import OrderedDict


device = torch.device('cuda:0') 

'''
model = torch.load(args.job_dir + 'checkpoint/model_1.pt')
state_dict = model['state_dict_s']
'''


def vgg(args, state_dict, t):
    pruned_num = 0
    total = 0
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    indexes_list = [np.arange(3)]

    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.view(-1).clone()
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
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][:, indexes_list[-1]]
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
            state_dict[name] = weight_copy[selected_index].view([1, -1])
        
    
    model = vgg_16_bn_sparse(cfg = cfg, T = t).to(device)
    
    ''' without attention '''
    for key in list(state_dict.keys()):
        if key not in list(model.state_dict().keys()):
            state_dict.pop(key)
            
            
    model.load_state_dict(state_dict)
    
    
    print(f"Pruned / Total: {pruned_num} / {total} (FC layer included.)")
 
    return model.to(device)


def resnet(args, state_dict):
    # n = (56 - 2) // 6
    # layers = np.arange(0, 3*n, n)

    masks = []
    num_channel_total = 0
    nums_channel_kept = []
    
    channel_kept_idx_list = []
    
    new_state_dict = state_dict.copy()
    
    for name, weight in state_dict.items():
        if 'alpha' in name: # 'mask'
            # mask_name.append(name)
            # print('weight.data.size: {}'.format(weight.data.view([-1]).shape))
            # mask_channel.append(weight.data.view([-1])) # weight.item()
            mask_weights = weight.view(-1) # .detach().cpu()
            
            masks.append(mask_weights)
            num_channels = len(weight.view(-1))
            num_channel_total += num_channels
            
            num_channel_kept = sum(mask_weights > 0.0).item()
            nums_channel_kept.append(num_channel_kept)
            
            new_state_dict[name] = mask_weights[mask_weights > 0.0].view([1, -1, 1, 1])
        
            channel_kept_idx = torch.where(mask_weights > 0.0)[0]
            channel_kept_idx_list.append(channel_kept_idx)
            
        
    print(f'Pruned / total number of channels: {num_channel_total-sum(nums_channel_kept)}/{num_channel_total}\n')
    
    # prune filters 
    filters = [name for name, weight in state_dict.items() if 'conv' in name and 'depth_conv' not in name and 'conv1x1' not in name]

    num_layers = len(nums_channel_kept)

    new_state_dict[filters[0]] = state_dict[filters[0]][channel_kept_idx_list[0], :, :, :] 
    
    for i in range(num_layers)[1:]:
        new_state_dict[filters[i]] = state_dict[filters[i]][channel_kept_idx_list[i], :, :, :][:, channel_kept_idx_list[i-1], :, :] 
    
    # update batch normalization weights
    bns = [name for name, weight in state_dict.items() if 'bn' in name and 'num_batches_tracked' not in name]
    
    for i in range(4):
        new_state_dict[bns[i]] = state_dict[bns[i]][channel_kept_idx_list[0]]

    for i in range(int(len(bns)/4))[1:]:
        for j in range(4):
            new_state_dict[bns[i*4 + j]] = state_dict[bns[i*4 + j]][channel_kept_idx_list[i]] 

    # update fc weights
    # fc = [name for name, weight in state_dict.items() if 'fc' in name]
    
    new_state_dict['fc.weight'] = state_dict['fc.weight'][:, channel_kept_idx_list[-1]]
    
    # reshape alphas
    alphas = [name for name, weight in state_dict.items() if 'alpha' in name]
    
    for i in range(num_layers):
        new_state_dict[alphas[i]] = state_dict[alphas[i]].view([1, -1])[:, channel_kept_idx_list[i]]

    model = pruned_resnet_110_sparse(has_mask = nums_channel_kept) 
    # model = pruned_resnet_56_sparse(has_mask = list(map(len, masks))) # non-pruned model

    ordered_new_state_dict = OrderedDict()

    for name in model.state_dict().keys():
        if 'num_batches_tracked' not in name:
            ordered_new_state_dict[name] = new_state_dict[name]


    if not args.random:
        '''model.load_state_dict(ordered_new_state_dict)'''
        model.state_dict().update(ordered_new_state_dict) 
        model.load_state_dict(ordered_new_state_dict)

    return model.to(device)

def densenet(args, state_dict):
    masks = []
    
    channel_kept_idx_list = [torch.tensor(range(3)).to(device), torch.tensor(range(24)).to(device)]
    filter_kept_idx_list = []
    num_channel_total = [3, 24]

    new_state_dict = state_dict.copy()

    for name, weight in state_dict.items():
        if 'alpha' in name: 
            mask_weights = weight.view(-1) 
            
            masks.append(mask_weights)
            
            thres = 0.0
                
            num_filter_kept = sum(mask_weights > thres).item()
            
            if num_filter_kept > 0:
                #if 'trans' not in name:
                    #new_state_dict[name] = mask_weights[mask_weights > thres].view([1, -1])
                filter_kept_idx = torch.where(mask_weights > thres)[0]
                filter_kept_idx_list.append(filter_kept_idx)
            
            else:
                #if 'trans' not in name:
                    #new_state_dict[name] = mask_weights[0].view([1, -1])
                filter_kept_idx = torch.tensor([0]).to(device)
                filter_kept_idx_list.append(filter_kept_idx)
            
            if len(masks) > 2 and len(mask_weights) > len(masks[-2]) : # transition layer
                channel_kept_idx_list.append(channel_kept_idx_list[-1])
                num_channel_total.append(num_channel_total[-1])
            
            else: # dense layer
                channel_kept_idx_list.append(torch.cat((channel_kept_idx_list[-1], filter_kept_idx + len(channel_kept_idx_list[-1]))))
                num_channel_total.append(num_channel_total[-1] + mask_weights.size(0))
                

    channel_kept_idx_list.pop(-1)
    
    nums_channel_kept = list(map(len, channel_kept_idx_list))
    
    print(f'Pruned / total number of channels: {sum(num_channel_total[:-1])-sum(nums_channel_kept)}/{sum(num_channel_total[:-1])}\n')
    
    [name for name, weight in state_dict.items() if 'conv1' in name]
    
    i = 0
    #k = 0
    #l = 0
    for name, weight in state_dict.items():
        if 'conv1' in name:
            new_state_dict[name] = state_dict[name][:, channel_kept_idx_list[i], :, :]
            i += 1
        '''   
        elif 'surv.k' in name: 
            new_state_dict[name] = state_dict[name][:, filter_kept_idx_list[k]]
            k += 1
            
        elif 'surv.lam' in name:
            new_state_dict[name] = state_dict[name][:, filter_kept_idx_list[l]]
            l += 1
  
    
    i = 0
    iter_ = 0
    for name, weight in state_dict.items():
        if 'bn' in name and 'num_batches_tracked' not in name: 
            new_state_dict[name] = state_dict[name][channel_kept_idx_list[i]] 
            
            if iter_ % 4 == 3:
                i += 1        
            iter_ += 1
     '''   
    
    # update fc weights
    # fc = [name for name, weight in state_dict.items() if 'fc' in name]
    #new_state_dict['fc.weight'] = state_dict['fc.weight'][:, channel_kept_idx_list[-1]]
    
    #filters_list = [nums_channel_kept[0]] + [12] * (len(nums_channel_kept) - 1)
    
    channel_kept_idx_list.pop(0)
    nums_channel_kept.pop(0)
    
    model = pruned_densenet_40_sparse(filters = nums_channel_kept, indexes = channel_kept_idx_list) 
    
    ordered_new_state_dict = OrderedDict()

    for name in model.state_dict().keys():
        if 'num_batches_tracked' not in name:
            ordered_new_state_dict[name] = new_state_dict[name]


    if not args.random:
        #model.load_state_dict(ordered_new_state_dict)
        model.state_dict().update(ordered_new_state_dict) 
        model.load_state_dict(ordered_new_state_dict)

    return model.to(device)


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
