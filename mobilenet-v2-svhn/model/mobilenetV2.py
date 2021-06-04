import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

device = torch.device('cuda:5')

class Mask(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
        
        self.alpha = Parameter(torch.rand(self.planes).view([1, -1])) 
        

    def forward(self, input):
        alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha

    
class cal_dist(nn.Module):
    def __init__(self):
        super(cal_dist, self).__init__()
        #self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        
    def forward(self, input):
        self.local_feat = input # self.pool(input)
        out = self.local_feat.view([self.local_feat.size(0), self.local_feat.size(1), -1])
        mean = torch.mean(out, dim = 1, keepdims = True)
        dist = torch.mean((out - mean) ** 2, dim = -1)

        return dist.view([input.size(0), -1]) # [batch_size, channel_num] 
    
    
class Survival(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
        
        self.k = Parameter(torch.rand(planes).view([1, -1]))
        self.lam = Parameter(torch.rand(planes).view([1, -1]))      
        
    def forward(self, t):
        ''' Weibull distribution '''
        self.f = self.k / self.lam * (t / self.lam) ** (self.k - 1) * torch.exp((-1) * (t / self.lam) ** self.k)
        
        return self.f # [1, channel_num]


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
    
    
class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6()

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.stride == 1:
            out += self.shortcut(x)
	
        return out

class SparseBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(SparseBlock, self).__init__()
        self.stride = stride
        
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6()

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )
        
        self.mask1 = Mask(planes)
        self.mask2 = Mask(planes)
        self.mask3 = Mask(out_planes)
        
        self.mask = Mask(1)
        
        self.surv1 = Survival(planes)
        self.surv2 = Survival(planes)
        self.surv3 = Survival(out_planes)

    def forward(self, x):
        ## layer 1
        self.out1 = self.conv1(x)
        self.out1_ = self.mask1(self.out1)
        self.out1_ = self.bn1(self.out1_)
        
        out = self.relu(self.out1_)
        
        ## layer 2
        self.out2 = self.conv2(out)
        self.out2_ = self.mask2(self.out2)
        self.out2_ = self.bn2(self.out2_)
        
        out = self.relu(self.out2_)
        
        ## layer 3
        self.out3 = self.conv3(out)
        self.out3_ = self.mask3(self.out3)
        self.out3_ = self.bn3(self.out3_)
        
        
        self.out3_ = self.mask(self.out3_)
        
  
        if self.stride == 1:
            out = self.out3_ + self.shortcut(x)
        else:
            out = self.out3_
	
        return out

class PrunedBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(PrunedBlock, self).__init__()
        self.stride = stride
        self.in_planes = in_planes
        
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU6()

        self.shortcut = nn.Sequential()
        if stride == 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
            )
        
        self.mask1 = Mask(planes)
        self.mask2 = Mask(planes)
        self.mask3 = Mask(out_planes)
        
        self.mask = Mask(1)
 
    def forward(self, x):
        
        if x.shape[1] != self.in_planes:
            pad_size = int((self.in_planes - x.shape[1])/2)
            x = torch.cat((x, torch.zeros([x.shape[0], pad_size, x.shape[2], x.shape[3]]).to(device)), 1)
            x = torch.cat((torch.zeros([x.shape[0], pad_size, x.shape[2], x.shape[3]]).to(device), x), 1)
        
        ## layer 1
        self.out1 = self.conv1(x)
        self.out1_ = self.mask1(self.out1)
        self.out1_ = self.bn1(self.out1_)
        
        out = self.relu(self.out1_)
        
        ## layer 2
        self.out2 = self.conv2(out)
        self.out2_ = self.mask2(self.out2)
        self.out2_ = self.bn2(self.out2_)
        
        out = self.relu(self.out2_)
        
        ## layer 3
        self.out3 = self.conv3(out)
        self.out3_ = self.mask3(self.out3)
        self.out3_ = self.bn3(self.out3_)
        
        
        self.out3_ = self.mask(self.out3_)
        
  
        if self.stride == 1:
            out = self.out3_ + self.shortcut(x)
        else:
            out = self.out3_
	
        return out

class MobileNetV2(nn.Module):
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    

    def __init__(self, block, num_classes=10):
        
        super(MobileNetV2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace = True)
        
        self.avg_pool2d = nn.AvgPool2d(4)
        
        
    def _make_layers(self, block, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layers(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SparseMobileNetV2(nn.Module):
    
    
    def __init__(self, block, cfg=None, num_classes=10, T = 1):
        super(SparseMobileNetV2, self).__init__()
        
        self.cfg = cfg
        
        self.T = T
        
        if cfg == None:
            self.cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace = True)
        
        self.avg_pool2d = nn.AvgPool2d(4)
        
        self.softmax = nn.Softmax(dim = -1)
        
    def _make_layers(self, block, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layers(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        ## info cascade by survival analysis
        ### layer-wise distances

        distances = []

        for block in self.layers:
            dist1 = cal_dist()(block.out1_)
            dist2 = cal_dist()(block.out2_)
            dist3 = cal_dist()(block.out3_)
            distances.append(dist1)
            distances.append(dist2)
            distances.append(dist3)

        ### filters' survival distributions
        filters_weibull = []
        
        for block in self.layers:
            filters_weibull.append(block.surv1)
            filters_weibull.append(block.surv2)
            filters_weibull.append(block.surv3)

        ### cascades
        if self.T == 0:
            new_distances = distances
        
        else:
            batch_size = distances[0].size(0)
            new_distances = [distances[0]]
            self.weibull_fs = []
    
            for t in range(1, self.T + 1):
                for i in range(len(distances) - t):
                    dist_t = distances[i + t]
                    
                    weibull_f = filters_weibull[i](t)
                    self.weibull_fs.append(weibull_f.view(-1))
                    
                    weibull1 = torch.cat([weibull_f, torch.ones([1, 1]).to(device)], dim = 1)
                    
                    fi = self.softmax(weibull1.view(-1)).view([1, -1, 1]) # [1, num_filters, 1]
                    dj = dist_t.view([dist_t.size(0), 1, -1]) # [batch_size, 1, num_channels]
                    Dij = torch.matmul(fi, dj) # [batch_size, num_filters, num_channels]
                    att1tot = torch.sum(Dij[:, :-1, :], dim = -1) # [batch_size, num_filters]
                    
                    new_dist_t = Dij[:, -1, :].squeeze(1)
                    
                    new_distances[i] = (new_distances[i] + att1tot).view([batch_size, -1])
                    
                    if t == 1:       
                        new_distances.append(new_dist_t.view([batch_size, -1]))
                    else:
                        new_distances[i + t] = ((new_distances[i + t] * (t - 1) + new_dist_t) / t).view([batch_size, -1])
                        #new_distances[i + t] = new_distances[i + t] + new_dist_t
                        
        ### attentions
        new_distances_ = torch.cat(new_distances, dim = 1) # [batch_size, total_num_channels]
        
        self.att = torch.cat([self.transform(new_distances_[i, :]).view([1, -1]) for i in range(new_distances_.size(0))], 
                              dim = 0) 
        
        return out
    
    def transform(self, x):
        #return (x - torch.min(x)) * (1 - 1e-5) / (torch.max(x) - torch.min(x) + 1e-5) / torch.norm(x, 2)
        return x / torch.norm(x, 2)


class PrunedMobileNetV2(nn.Module):

    def __init__(self, block, cfg = None, num_classes = 10, has_mask = None, T = 1):
        super(PrunedMobileNetV2, self).__init__()
        
        self.cfg = cfg
        
        self.has_mask = has_mask
        
        self.T = T
        
        if cfg == None:
            self.cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
            
        if has_mask == None:
            self.has_mask = [1] * sum([i[1] for i in self.cfg])
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(block, in_planes=32, has_mask = self.has_mask)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.relu = nn.ReLU(inplace = True)
        
        self.avg_pool2d = nn.AvgPool2d(4)
        
        self.softmax = nn.Softmax(dim = -1)
    
    def _make_layers(self, block, in_planes, has_mask):
        layers = []
        '''
        in_plane_list = [32]
        for c in self.cfg:
            in_plane_list.extend([c[1]]*c[2])
        
        has_mask[0] = 1 # retain the first block
        
        link_block_no = [-1]*len(has_mask)
        
        retained_block = np.where(np.array(has_mask) == 1)[0].tolist() + [17]
        
        for i in range(len(retained_block)-1):
            link_block_no[retained_block[i]] = retained_block[i+1]
        '''
        
        COUNT_blocks = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for i, stride in enumerate(strides):
                if has_mask[COUNT_blocks + i]:
                    #shortcut_planes = in_plane_list[link_block_no[COUNT_blocks + i]]
                    #layers.append(block(in_planes, out_planes, shortcut_planes, expansion, stride))
                    layers.append(block(in_planes, out_planes, expansion, stride))

                in_planes = out_planes    
            COUNT_blocks += num_blocks
        
        return nn.Sequential(*layers)

    '''
    def _make_layers(self, block, in_planes, has_mask):
        layers = []
        COUNT_blocks = 0
        for expansion, out_planes_list, num_blocks, stride in self.cfg:
            for i in range(num_blocks):
                if has_mask[COUNT_blocks + i] or (COUNT_blocks + i) == 0:
                    layers.append(block(in_planes, out_planes_list[i][0], 
                                        out_planes_list[i][1], out_planes_list[i][2], 
                                        stride))
                in_planes = out_planes_list[i][2]
                
            COUNT_blocks += num_blocks
        return nn.Sequential(*layers)
    '''
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layers(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out

    
def mobile_v2(**kwargs):
    return MobileNetV2(Block, **kwargs)

def mobile_v2_sparse(**kwargs):
    return SparseMobileNetV2(SparseBlock, **kwargs)

def pruned_mobile_v2_sparse(**kwargs):
    return PrunedMobileNetV2(PrunedBlock, **kwargs) 
    