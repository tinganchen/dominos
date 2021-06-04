import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter #from torch.nn import Parameter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import numpy as np
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import pdb

norm_mean, norm_var = 0.0, 1.0
device = torch.device('cuda:0')

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)#, bias = False
    
def conv1x1(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 1, stride = stride,
                     padding = 1, bias = False)#, bias = False

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

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

class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        out = self.relu(out)
        return out
    
'''
class ResBasicBlock(nn.Module):
    def __init__(self,in_places, places, stride=1,downsampling=False, expansion = 4):
        super(ResBasicBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(places)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(places)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(places*self.expansion)
    

        if self.downsampling:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(places*self.expansion)
                    )

        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
'''

class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, has_mask=None, cfg=None, index=None, stride=1, downsample=None):
        super(ResBottleneck, self).__init__()
        # print("inplanes",inplanes, "planes",planes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

'''
class SparseResBasicBlock(nn.Module):
    def __init__(self,in_places, in_places2, in_places3, places, stride=1,downsampling=False, expansion = 4):
        super(SparseResBasicBlock,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.conv1 = nn.Conv2d(in_channels=in_places,out_channels=in_places2,kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(places)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_places2, out_channels=in_places3, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(places)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=in_places3, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(places*self.expansion)
    

        if self.downsampling:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(places*self.expansion)
                    )
        self.relu = nn.ReLU(inplace=True)
        
        self.mask1 = Mask(in_places2)
        self.mask2 = Mask(in_places3)
        self.mask3 = Mask(places*self.expansion)
        
        self.mask = Mask(1)
        
        self.surv1 = Survival(in_places2)
        self.surv2 = Survival(in_places3)
        self.surv3 = Survival(places*self.expansion)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.mask1(out)
        self.out1_ = self.bn1(out)
        out = self.relu1(self.out1_)
        
        out = self.conv2(out)
        out = self.mask2(out)
        self.out2_ = self.bn2(out)
        out = self.relu2(self.out2_)
        
        out = self.conv3(out)
        out = self.mask3(out)
        self.out3_ = self.bn3(out)
        
        out = self.mask(self.out3_)        
        
        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
'''
class MultiSparseResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, has_mask=None, cfg=None, index=None, stride=1, downsample=None):
        # print('in multisparse',cfg)
        super(MultiSparseResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.has_mask = has_mask
        
        self.mask1 = Mask(cfg[1])
        self.mask2 = Mask(cfg[2])
        self.mask3 = Mask(planes*self.expansion)
        
        self.mask = Mask(1)
        
        self.surv1 = Survival(cfg[1])
        self.surv2 = Survival(cfg[2])
        self.surv3 = Survival(planes*self.expansion)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.mask1(out)
        self.out1_ = self.bn1(out)
        out = self.relu(self.out1_)
        
        out = self.conv2(out)
        out = self.mask2(out)
        self.out2_ = self.bn2(out)
        out = self.relu(self.out2_)
        
        out = self.conv3(out)
        out = self.mask3(out)
        self.out3_ = self.bn3(out)
        
        out = self.mask(self.out3_)        
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

'''
class ResNet(nn.Module):
    def __init__(self, block_type, blocks, num_classes=1000, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                    
        self.layer1 = self.make_layer(block_type, in_places = 64, places= 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(block_type, in_places = 256,places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(block_type, in_places=512,places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(block_type, in_places=1024,places=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block_type, in_places, places, block, stride):
        layers = []
        layers.append(block_type(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(block_type(places*self.expansion, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
'''
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, has_mask=None, indexes=None, cfg=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if has_mask is None : has_mask = [1]*sum(layers)

        if cfg is None:
            # bottleneck 
            cfg =[[64, 64, 64], [256, 64, 64]*(layers[0]-1), [256, 128, 128], [512, 128, 128]*(layers[1]-1), [512, 256, 256], [1024, 256, 256]*(layers[2]-1), [1024, 512, 512], [2048, 512, 512]*(layers[3]-1), [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]
        
        if indexes is None:
            indexes = []
            for i in range(len(cfg)):
                indexes.append(np.arange(cfg[i]))

        start = 0
        cfg_start = 0
        cfg_end = 3*layers[0]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        remain_block = [int(m) for m in np.argwhere(np.array(has_mask))]
        layers_start = [0, 3, 7, 13]

        layers_remain = []
        for i in range(3):
            # print(len(np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0]))
            layers_remain.append(len((np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0])))
        layers_remain.append(len(np.where(np.array(remain_block) >= layers_start[3])[0]))

        # print("remain layers",layers_remain)


        self.layer1 = self._make_layer(block, 64, layers[0], has_mask=has_mask[start:layers[0]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[0:cfg_end])

        start = layers[0]
        cfg_start += 3*layers[0]
        cfg_end += 3*layers[1]

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, has_mask=has_mask[start:start+layers[1]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[1]
        cfg_start += 3*layers[1]
        cfg_end += 3*layers[2]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, has_mask=has_mask[start:start+layers[2]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[2]
        cfg_start += 3*layers[2]
        cfg_end += 3*layers[3]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,has_mask=has_mask[start:start+layers[3]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, has_mask=None, cfg=None, indexes=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if has_mask[0] == 0 and downsample is not None:
            layers.append(Downsample(downsample))
        elif not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, has_mask[0], cfg[0:3], indexes[0], stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes, has_mask[i], cfg[3*i:3*(i+1)], indexes[3*i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # 256 x 56 x 56
        x = self.layer2(x)
        # 512 x 28 x 28
        # 1024 x 14 x 14
        x = self.layer3(x)
        # 2048 x 7 x 7
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
'''


class SparseResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=1000, expansion = 4, has_mask = None, T = 1):
        super(SparseResNet,self).__init__()
        self.expansion = expansion
        
        if has_mask is None : has_mask = [1]*sum(num_blocks)
        
        self.T = T
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #self.conv1 = Conv1(in_planes = 3, places= 64)

        self.layer1 = self._make_layer(block_type, 64, 256, blocks=num_blocks[0], stride=1, has_mask=has_mask[0:num_blocks[0]])
        self.layer2 = self._make_layer(block_type, 128, 512, blocks=num_blocks[1], stride=2, has_mask=has_mask[num_blocks[0]:sum(num_blocks[:2])])
        self.layer3 = self._make_layer(block_type, 256, 1024, blocks=num_blocks[2], stride=2, has_mask=has_mask[sum(num_blocks[:2]):sum(num_blocks[:3])])
        self.layer4 = self._make_layer(block_type, 512, 2048, blocks=num_blocks[3], stride=2, has_mask=has_mask[sum(num_blocks[:3]):sum(num_blocks)])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)
        
        
        self.softmax = nn.Softmax(dim = -1)
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        
    def _make_layer(self, block_type, in_places, places, blocks, stride, has_mask):
        layers = []
        layers.append(block_type(in_places, places, places, places, stride, downsampling =True))
        for i in range(1, blocks):
            layers.append(block_type(places*self.expansion, places, places, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        ## info cascade by survival analysis
        ### layer-wise distances

        distances = []

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                dist1 = cal_dist()(block.out1_)
                dist2 = cal_dist()(block.out2_)
                dist3 = cal_dist()(block.out3_)
                distances.append(dist1)
                distances.append(dist2)
                distances.append(dist3)
         

        ### filters' survival distributions
        filters_weibull = []
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
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
        
        return x
    
    def transform(self, x):
        #return (x - torch.min(x)) * (1 - 1e-5) / (torch.max(x) - torch.min(x) + 1e-5) / torch.norm(x, 2)
        return x / torch.norm(x, 2)
    
''' 
class SparseResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, has_mask=None, indexes=None, cfg=None, T = 1):
        self.inplanes = 64
        super(SparseResNet, self).__init__()
        
        self.T = T

        if has_mask is None : has_mask = [1]*sum(layers)

        if cfg is None:
            # bottleneck 
            cfg =[[64, 64, 64], [256, 64, 64]*(layers[0]-1), [256, 128, 128], [512, 128, 128]*(layers[1]-1), [512, 256, 256], [1024, 256, 256]*(layers[2]-1), [1024, 512, 512], [2048, 512, 512]*(layers[3]-1), [2048]]
            cfg = [item for sub_list in cfg for item in sub_list]
        
        if indexes is None:
            indexes = []
            for i in range(len(cfg)):
                indexes.append(np.arange(cfg[i]))
                
        self.softmax = nn.Softmax(dim = -1)
        
        start = 0
        cfg_start = 0
        cfg_end = 3*layers[0]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        remain_block = [int(m) for m in np.argwhere(np.array(has_mask))]
        layers_start = [0, 3, 7, 13]

        layers_remain = []
        for i in range(3):
            # print(len(np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0]))
            layers_remain.append(len((np.where((np.array(remain_block)>=layers_start[i]) & (np.array(remain_block)<layers_start[i+1]))[0])))
        layers_remain.append(len(np.where(np.array(remain_block) >= layers_start[3])[0]))

        # print("remain layers",layers_remain)


        self.layer1 = self._make_layer(block, 64, layers[0], has_mask=has_mask[start:layers[0]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[0:cfg_end])

        start = layers[0]
        cfg_start += 3*layers[0]
        cfg_end += 3*layers[1]

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, has_mask=has_mask[start:start+layers[1]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[1]
        cfg_start += 3*layers[1]
        cfg_end += 3*layers[2]
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, has_mask=has_mask[start:start+layers[2]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        start += layers[2]
        cfg_start += 3*layers[2]
        cfg_end += 3*layers[3]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,has_mask=has_mask[start:start+layers[3]], cfg=cfg[cfg_start:cfg_end], indexes=indexes[cfg_start:cfg_end])

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, has_mask=None, cfg=None, indexes=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if has_mask[0] == 0 and downsample is not None:
            layers.append(Downsample(downsample))
        elif not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, has_mask[0], cfg[0:3], indexes[0], stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes, has_mask[i], cfg[3*i:3*(i+1)], indexes[3*i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # 256 x 56 x 56
        x = self.layer2(x)
        # 512 x 28 x 28
        # 1024 x 14 x 14
        x = self.layer3(x)
        # 2048 x 7 x 7
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        ## info cascade by survival analysis
        ### layer-wise distances

        distances = []

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                dist1 = cal_dist()(block.out1_)
                dist2 = cal_dist()(block.out2_)
                dist3 = cal_dist()(block.out3_)
                distances.append(dist1)
                distances.append(dist2)
                distances.append(dist3)
         

        ### filters' survival distributions
        filters_weibull = []
        
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
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
        
        return x
    
    def transform(self, x):
        #return (x - torch.min(x)) * (1 - 1e-5) / (torch.max(x) - torch.min(x) + 1e-5) / torch.norm(x, 2)
        return x / torch.norm(x, 2)


'''        

class PrunedResNet(nn.Module):
    def __init__(self, block_type, num_blocks, num_classes=1000, expansion = 4, has_mask = None, T = 1):
        super(PrunedResNet,self).__init__()
        self.expansion = expansion
        
        if has_mask is None : has_mask = [1]*sum(num_blocks)
        
        self.T = T

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        

        self.layer1 = self._make_layer(block_type, 64, 256, blocks=num_blocks[0], stride=1, has_mask=has_mask[0:num_blocks[0]])
        self.layer2 = self._make_layer(block_type, 128, 512, blocks=num_blocks[1], stride=2, has_mask=has_mask[num_blocks[0]:sum(num_blocks[:2])])
        self.layer3 = self._make_layer(block_type, 256, 1024, blocks=num_blocks[2], stride=2, has_mask=has_mask[sum(num_blocks[:2]):sum(num_blocks[:3])])
        self.layer4 = self._make_layer(block_type, 512, 2048, blocks=num_blocks[3], stride=2, has_mask=has_mask[sum(num_blocks[:3]):sum(num_blocks)])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_type, in_places, places, blocks, stride, has_mask):
        layers = []
        layers.append(block_type(in_places, places, places, places, stride, downsampling =True))
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block_type(places*self.expansion, places, places, places))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        
        return x
    
'''
'''
class PrunedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes = 1000, has_mask = None, T = 1):
        super(PrunedResNet, self).__init__()
        
        # assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        # n = (num_layers - 2) // 6
        
        if has_mask is None : has_mask = [1]*sum(num_blocks)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 7, stride = 2, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(3, 2, padding = 1)

        self.layer1 = self._make_layer(block, 64, 256, blocks=num_blocks[0], stride=1, has_mask=has_mask[0:num_blocks[0]])
        self.layer2 = self._make_layer(block, 128, 512, blocks=num_blocks[1], stride=2, has_mask=has_mask[num_blocks[0]:sum(num_blocks[:2])])
        self.layer3 = self._make_layer(block, 256, 1024, blocks=num_blocks[2], stride=2, has_mask=has_mask[sum(num_blocks[:2]):sum(num_blocks[:3])])
        self.layer4 = self._make_layer(block, 512, 2048, blocks=num_blocks[3], stride=2, has_mask=has_mask[sum(num_blocks[:3]):sum(num_blocks)])
        self.avgpool = nn.AvgPool2d(2,2,padding=1)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        self.initialize()
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes1, planes2, blocks, stride, has_mask):
        layers = nn.ModuleList()
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes2): 
            layers.append(LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes2//4, planes2//4), "constant", 0)))
        
        layers.append(block(self.inplanes, planes1, planes1, planes2, stride, True))

        self.inplanes = planes2 * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes1, planes1, planes2, False))
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
'''    
    
def resnet_50(pretrained=False, **kwargs):
    model = ResNet(ResBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        load_url = model_zoo.load_url(model_urls['resnet50'])
        model.load_state_dict(load_url)
        return model, load_url
    return model


def resnet_50_sparse(pretrained=False, **kwargs):
    model = SparseResNet(MultiSparseResBottleneck, [3, 4, 6, 3], **kwargs)
    return model

def pruned_resnet_50_sparse(pretrained=False, **kwargs):
    model = ResNet(MultiSparseResBottleneck, [3, 4, 6, 3], **kwargs)
    return model


#model_s = resnet_50_sparse()

