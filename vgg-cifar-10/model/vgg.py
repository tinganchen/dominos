import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from collections import OrderedDict

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]

device = torch.device('cuda:0')

'''
class ChannelSelection(nn.Module):
    def __init__(self, indexes, fc = False):
        super(ChannelSelection, self).__init__()
        self.indexes = indexes
        self.fc = fc

    def forward(self, input_tensor):
        if self.fc:
            return input_tensor[:, self.indexes]

        if len(self.indexes) == input_tensor.size()[1]:
            return input_tensor

        return input_tensor[:, self.indexes, :, :]


class Mask(nn.Module):
    def __init__(self, init_value=[1], fc=False):
        super().__init__()
        self.weight = Parameter(torch.Tensor(init_value))
        self.fc = fc

    def forward(self, input):
        if self.fc:
            weight = self.weight
        else:
            weight = self.weight[None, :, None, None]
            

        return input * weight

class Mask(nn.Module):
    def __init__(self, init_value = [1], fc = False):
        super().__init__()
        self.fc = fc
        
        self.alpha = Parameter(torch.Tensor(init_value).view([1, -1])) 

    def forward(self, input):
        if self.fc:
            alpha = self.alpha.view(-1)
        else:
            alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha


'''
class Mask(nn.Module):
    def __init__(self, planes, fc = False):
        super().__init__()
        self.planes = planes
        self.fc = fc
        
        self.alpha = Parameter(torch.rand(self.planes).view([1, -1])) 
        

    def forward(self, input):
        if self.fc:
            alpha = self.alpha.view(-1)
        else:
            alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha

class ExtractDataLayer(nn.Module):
    def __init__(self):
        super(ExtractDataLayer, self).__init__()
    
    def forward(self, x):
        self.out = x
        return x
    
class cal_dist(nn.Module):
    def __init__(self):
        super(cal_dist, self).__init__()
        #self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
    def forward(self, input):
        self.local_feat = input # self.pool(input)
        out = self.local_feat.view([self.local_feat.size(0), self.local_feat.size(1), -1])
        mean = torch.mean(out, dim = 1, keepdims = True)
        dist = torch.mean((out - mean) ** 2, dim = -1)

        return dist # [batch_size, channel_num] 
    
    
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
    
class VGG(nn.Module):
    def __init__(self, num_classes = 10, init_weights = True, is_sparse = False, cfg = None, index = None, T = 1):
        super(VGG, self).__init__()
        
        self.is_sparse = is_sparse
        
        self.features = nn.Sequential()
        
        self.T = T

        if cfg is None:
            cfg = defaultcfg

        if self.is_sparse:
            self.features = self.make_sparse_layers(cfg[:-1], True)
            #m = Normal(torch.tensor([norm_mean]*cfg[-1]), torch.tensor([norm_var]*cfg[-1])).sample()
            self.classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(cfg[-2], cfg[-1])),
                ('norm1', nn.BatchNorm1d(cfg[-1])),
                ('relu1', nn.ReLU(inplace = True)),
                ('mask', Mask(cfg[-1], fc = True)),
                #('mask', Mask(cfg[-1], fc = True)),
                ('linear2', nn.Linear(cfg[-1], num_classes)),
            ]))
            self.softmax = nn.Softmax(dim = -1)
            self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        else:
            self.features = self.make_layers(cfg[:-1], True)
            self.classifier = nn.Sequential(OrderedDict([
                ('linear1', nn.Linear(cfg[-2], cfg[-1])),
                ('norm1', nn.BatchNorm1d(cfg[-1])),
                ('relu1', nn.ReLU(inplace = True)),
                ('linear2', nn.Linear(cfg[-1], num_classes)),
            ]))
        
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm = True):
        layers = nn.Sequential()
        in_channels = 3
        for i, v in enumerate(cfg):
            if v == 'M':
                layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size = 2, stride= 2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)

                layers.add_module('conv%d' % i, conv2d)
                layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                layers.add_module('relu%d' % i, nn.ReLU(inplace = True))
                in_channels = v

        return layers

    def make_sparse_layers(self, cfg, batch_norm = True):
        in_channels = 3
        sparse_layers = nn.Sequential()
        self.filters_weibull = []
        for i, v in enumerate(cfg):
            if v == 'M':
                sparse_layers.add_module('pool%d' % i,nn.MaxPool2d(kernel_size = 2, stride = 2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
                if batch_norm:
                    sparse_layers.add_module('conv%d' % i, conv2d)
                    #m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                    sparse_layers.add_module('mask%d' % i, Mask(v))
                    sparse_layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
                    sparse_layers.add_module('out%d' % i, ExtractDataLayer())
                    sparse_layers.add_module('relu%d' % i, nn.ReLU(inplace = True))
                
                #m = Normal(torch.tensor([norm_mean]*int(v)), torch.tensor([norm_var]*int(v))).sample()
                #sparse_layers.add_module('mask%d' % i, Mask(v))

                self.filters_weibull.append(Survival(v).to(device))
                in_channels = v
                
        return sparse_layers

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        if self.is_sparse:
            ## info cascade by survival analysis
            ### layer-wise distances
    
            distances = []
            
            for name in self.features._modules:
                if 'out' in name:
                    bn_out = self.features._modules[name].out
                    dist = cal_dist()(bn_out).to(device)
                    distances.append(dist)
             
    
            ### filters' survival distributions
            filters_weibull = self.filters_weibull
           
            ### cascades
            if self.T == 0:
                new_distances = distances
            
            else:
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
                        
                        new_distances[i] = new_distances[i] + att1tot
                        
                        if t == 1:       
                            new_distances.append(new_dist_t)
                        else:
                            #new_distances[i + t] = (new_distances[i + t] * (t - 1) + new_dist_t) / t
                            new_distances[i + t] = new_distances[i + t] + new_dist_t
                        
            ### attentions
            new_distances_ = torch.cat(new_distances, dim = 1) # [batch_size, total_num_channels]
            
            self.att = torch.cat([self.transform(new_distances_[i, :]).view([1, -1]) for i in range(new_distances_.size(0))], 
                                  dim = 0) 
     
        return x
    
    def transform(self, x):
        #return (x - torch.min(x)) * (1 - 1e-5) / (torch.max(x) - torch.min(x) + 1e-5) / torch.norm(x, 2)
        return x / torch.norm(x, 2)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg_16_bn(**kwargs):
    model = VGG(**kwargs)
    return model

def vgg_16_bn_sparse(**kwargs):
    model = VGG(is_sparse = True, **kwargs)
    return model


#model = vgg_16_bn_sparse()

#tmp = torch.ones([64, 3, 32, 32])

#output = model(tmp)

#

