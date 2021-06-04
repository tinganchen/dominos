import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import pdb

norm_mean, norm_var = 0.0, 1.0
device = torch.device('cuda:2')


class ChannelSelection(nn.Module):
    def __init__(self, indexes):
        super(ChannelSelection, self).__init__()
        self.indexes = indexes

    def forward(self, input_tensor):
        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor
        
        output = input_tensor[:, self.indexes, :, :]
        return output
'''
class Mask(nn.Module):
    def __init__(self, init_value=[1]):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        # pdb.set_trace()

    def forward(self, input):
        weight = self.weight[None, :, None, None]
        return input * weight

class Mask(nn.Module):
    def __init__(self, init_value = [1]):
        super().__init__()
        
        self.alpha = Parameter(torch.Tensor(init_value).view([1, -1])) 

    def forward(self, input):
        
        alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha
'''

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
    
class DenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion = 1, growthRate = 12, dropRate = 0):
        super(DenseBasicBlock, self).__init__()
        planes = expansion * growthRate

        self.bn1 = nn.BatchNorm2d(inplanes, momentum = 0.1)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size = 3,
                               padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = F.dropout(out, p = self.dropRate, training = self.training)

        out = torch.cat((x, out), 1)

        return out

class SparseDenseBasicBlock(nn.Module):
    def __init__(self, inplanes, filters, index, expansion = 1, growthRate = 12, dropRate = 0):
        super(SparseDenseBasicBlock, self).__init__()
        planes = expansion * growthRate
        
        self.inplanes = inplanes
        self.bn1 = nn.BatchNorm2d(self.inplanes, momentum = 0.1)
        self.conv1 = nn.Conv2d(filters, growthRate, kernel_size = 3,
                               padding = 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.dropRate = dropRate

        self.select = ChannelSelection(index)
        
        #m = Uniform(torch.tensor([norm_mean]*filters), torch.tensor([norm_var]*filters)).sample()
        #self.mask = Mask(m)
        
        #m = Uniform(torch.tensor([norm_mean]*growthRate), torch.tensor([norm_var]*growthRate)).sample()
        self.mask = Mask(len(index))

        self.surv = Survival(len(index))
        
    def forward(self, x):        
        self.out = self.bn1(x)
        out = self.relu(self.out)
        out = self.select(out)
        out = self.mask(out)
        out = self.conv1(out)
        #out = self.mask(out)
        
        if self.dropRate > 0:
            out = F.dropout(out, p = self.dropRate, training = self.training)

        out = torch.cat((x, out), 1)
        
        return out


class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum = 0.1)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size = 1,
                               bias = False)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class SparseTransition(nn.Module):
    def __init__(self, inplanes, outplanes, filters, index):
        super(SparseTransition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum = 0.1)
        self.select = ChannelSelection(index)
        self.conv1 = nn.Conv2d(filters, outplanes, kernel_size = 1,
                               bias = False)
        self.relu = nn.ReLU(inplace = True)
        
        #m = Normal(torch.tensor([norm_mean]*outplanes), torch.tensor([norm_var]*outplanes)).sample()
        self.mask = Mask(len(index))

        self.surv = Survival(len(index))
        
    def forward(self, x):
        self.out = self.bn1(x)
        out = self.relu(self.out)
        out = self.select(out)
        out = self.mask(out)
        out = self.conv1(out)
        #out = self.mask(out)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    def __init__(self, depth = 40, block = DenseBasicBlock, 
        dropRate = 0, num_classes = 10, growthRate = 12, compressionRate = 2, 
        filters = None, indexes = None):
        super(DenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        transition = SparseTransition if 'Sparse' in str(block) else Transition
        if filters == None:
            filters = []
            start = growthRate * 2
            for i in range(3):
                filters.append([start + growthRate * i for i in range(n + 1)])
                start = (start + growthRate * n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, padding = 1,
                               bias = False)
        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        self.trans1 = self._make_transition(transition, compressionRate, filters[n], indexes[n])
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1])
        self.trans2 = self._make_transition(transition, compressionRate, filters[2*n+1], indexes[2*n+1])
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes, momentum = 0.1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            # print("denseblock inplanes", filters[i])
            layers.append(block(self.inplanes, filters = filters[i], index = indexes[i], growthRate = self.growthRate, dropRate = self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate)) # self.inplanes // compressionRate
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index)


    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class SparseDenseNet(nn.Module):

    def __init__(self, depth = 40, block = SparseDenseBasicBlock, 
                 dropRate = 0, num_classes = 10, growthRate = 12, compressionRate = 2, 
                 filters = None, indexes = None, T = 1):
        super(SparseDenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6
        transition = SparseTransition if 'Sparse' in str(block) else Transition
        
        if filters == None:
            filters = []
            start = growthRate * 2
            for i in range(3):
                filters.append([start + growthRate * i for i in range(n + 1)])
                start = (start + growthRate * n) // compressionRate
            filters = [item for sub_list in filters for item in sub_list]

            indexes = []
            for f in filters:
                indexes.append(np.arange(f))
        
        self.T = T
        
        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2 
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, padding = 1,
                               bias = False)
        self.dense1 = self._make_denseblock(block, n, filters[0:n], indexes[0:n])
        self.trans1 = self._make_transition(transition, compressionRate, filters[n], indexes[n])
        self.dense2 = self._make_denseblock(block, n, filters[n+1:2*n+1], indexes[n+1:2*n+1])
        self.trans2 = self._make_transition(transition, compressionRate, filters[2*n+1], indexes[2*n+1])
        self.dense3 = self._make_denseblock(block, n, filters[2*n+2:3*n+2], indexes[2*n+2:3*n+2])
        self.bn = nn.BatchNorm2d(self.inplanes, momentum = 0.1)
        self.relu = nn.ReLU(inplace = True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.inplanes, num_classes)
        
        #self.mask = Mask(growthRate * 2)
        #self.surv = Survival(growthRate * 2)
        
        self.softmax = nn.Softmax(dim = -1)

        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, filters, indexes):
        layers = []
        assert blocks == len(filters), 'Length of the filters parameter is not right.'
        assert blocks == len(indexes), 'Length of the indexes parameter is not right.'
        for i in range(blocks):
            # print("denseblock inplanes", filters[i])
            layers.append(block(self.inplanes, filters = filters[i], index = indexes[i], 
                                growthRate = self.growthRate, dropRate = self.dropRate))
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate, filters, index):
        inplanes = self.inplanes
        outplanes = int(math.floor(inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes, filters, index)


    def forward(self, x):
        x = self.conv1(x)
        #x = self.mask(x)
        
        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        
        self.x_prime = self.bn(x)
        x = self.relu(self.x_prime)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        ## info cascade by survival analysis
        ### layer-wise distances
        distances = []

        for i, layer in enumerate([self.dense1, self.trans1, self.dense2, self.trans2, self.dense3]):
            if i % 2 == 0:
                for block in layer:
                    dist = cal_dist()(block.out)
                    distances.append(dist)
            else:
                dist = cal_dist()(layer.out)
                distances.append(dist)
        
        
        ### filters' survival distributions
        filters_weibull = []
        #filters_weibull.append(self.surv)

        for i, layer in enumerate([self.dense1, self.trans1, self.dense2, self.trans2, self.dense3]):
            if i % 2 == 0:
                for block in layer:
                    filters_weibull.append(block.surv)
            else:
                filters_weibull.append(layer.surv)    
        
        ### cascades
        if self.T == 0:
            new_distances = distances

        else:
            new_distances = distances
            
            self.weibull_fs = []
            
            for t in range(1, self.T + 1):

                for i in range(len(distances) - t -1, -1, -1): # Influences computed backward 
                    dist_t = distances[i + t]
      
                    num_filters_i = filters_weibull[i](t).view(-1).size(0)
                    num_filters_i_t = filters_weibull[i + t](t).view(-1).size(0)
                    
                    weibull_f = filters_weibull[i](t)
                    self.weibull_fs.append(weibull_f.view(-1))
                    
                    weibull1 = torch.cat([weibull_f, torch.ones([1, 1]).to(device)], dim = 1)
                    
                    fi = self.softmax(weibull1.view(-1)).view([1, -1, 1]) # [1, num_filters, 1]

                    dj = dist_t.view([dist_t.size(0), 1, -1]) # [batch_size, 1, num_channels]

                    Dij = torch.matmul(fi, dj) # [batch_size, num_filters, num_channels]
                    

                    att1tot = torch.sum(Dij[:, :-1, -num_filters_i_t:], dim = -1) # [batch_size, num_filters]

                    new_dist_i = Dij[:, -1, -num_filters_i:].squeeze(1)
                    
                    new_distances[i] = torch.cat((new_distances[i][:, :-num_filters_i], new_dist_i + att1tot), dim = 1)
                    
                    if t == 1:
                        concat_info = torch.sum(Dij[:, :-1, -(num_filters_i_t + num_filters_i):-num_filters_i], dim = -1)
                        new_distances[i][: , -num_filters_i:] = new_distances[i][: , -num_filters_i:] + concat_info

        for i in range(len(new_distances)):
            num_filters = filters_weibull[i](1).view(-1).size(0)
            new_distances[i] = new_distances[i][:, -num_filters:]            

        ### attentions
        new_distances_ = torch.cat(new_distances, dim = 1) # [batch_size, total_num_channels]
        
        self.att = torch.cat([self.transform(new_distances_[i, :]).view([1, -1]) for i in range(new_distances_.size(0))], 
                              dim = 0) 
        
        return x
    
    def transform(self, x):
        #x_ = (x - torch.min(x)) * (1 - 1e-5) / (torch.max(x) - torch.min(x) + 1e-5)
        return x / torch.norm(x, 2)
    
    
        
        
def densenet_40(**kwargs):
    return DenseNet(depth = 40, block = DenseBasicBlock, compressionRate = 1, **kwargs)

def densenet_40_sparse(**kwargs):
    return SparseDenseNet(depth = 40, block = SparseDenseBasicBlock, compressionRate = 1, **kwargs)

def pruned_densenet_40_sparse(**kwargs):
    return SparseDenseNet(depth = 40, block = SparseDenseBasicBlock, compressionRate = 1, **kwargs)
