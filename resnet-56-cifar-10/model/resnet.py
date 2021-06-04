import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter #from torch.nn import Parameter
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

norm_mean, norm_var = 0.0, 1.0
device = torch.device('cuda:2')


def conv3x3(in_planes, out_planes, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size = 3, stride = stride,
                     padding = 1, bias = False)#, bias = False


class Mask(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.planes = planes
        
        self.alpha = Parameter(torch.rand(self.planes).view([1, -1])) 
        

    def forward(self, input):
        alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha
'''

class Mask(nn.Module):
    def __init__(self, init_value = [1]):
        super().__init__()
        
        self.alpha = Parameter(torch.Tensor(init_value).view([1, -1])) 

    def forward(self, input):
        
        alpha = self.alpha.view([1, -1, 1, 1])
        
        return input * alpha   
'''
    
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


class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                    "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class SparseResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, inplanes2, planes, stride = 1):
        super(SparseResBasicBlock, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        
        self.conv1 = conv3x3(inplanes, inplanes2, stride)
        self.bn1 = nn.BatchNorm2d(inplanes2)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(inplanes2, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                    "constant", 0))
        
        self.mask1 = Mask(inplanes2)
        self.mask2 = Mask(planes)
        
        self.mask = Mask(1)
        
        self.surv1 = Survival(inplanes2)
        self.surv2 = Survival(planes)

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
        
        self.out2_ = self.mask(self.out2_)
        
        self.out2_ += self.shortcut(x)
        out = self.relu(self.out2_)
        
        
        return out

'''
class PrunedSparseResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, inplanes2, planes, stride = 1):
        super(PrunedSparseResBasicBlock, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        
        self.conv1 = conv3x3(inplanes, inplanes2, stride)
        self.bn1 = nn.BatchNorm2d(inplanes2)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = conv3x3(inplanes2, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4),
                    "constant", 0))
        

        self.mask1 = Mask(inplanes2)
        self.mask2 = Mask(planes)

    def forward(self, x):     
        ## layer 1
        self.out1 = self.conv1(x)
        self.out1_ = self.mask2(self.out1)
        self.out1_ = self.bn1(self.out1_)
        
        out = self.relu(self.out1_)
        
        ## layer 2
        self.out2 = self.conv2(out)
        self.out2_ = self.mask2(self.out2)
        self.out2_ = self.bn2(self.out2_)
        
       
        self.out2_ += self.shortcut(x)
        out = self.relu(self.out2_)
        
        return out
'''
'''
class PrunedSparseResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, inplanes2, planes, stride = 1):
        super(PrunedSparseResBasicBlock, self).__init__()
        
        self.inplanes = inplanes
        self.planes = planes
        
        self.conv1 = conv3x3(inplanes, inplanes2, stride)
        self.bn1 = nn.BatchNorm2d(inplanes2)
        self.relu = nn.ReLU(inplace = False)
        self.conv2 = conv3x3(inplanes2, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

        self.mask1 = Mask(inplanes2)
        self.mask2 = Mask(planes)

    def forward(self, x):
        
        ## layer 1
        out = self.conv1(x)
        out1 = self.mask1(out) # layer-wise channel weighting
        out = self.bn1(out1)
                        
        out = self.relu(out)
        
        ## layer 2
        out = self.conv2(out)
        out2 = self.mask2(out)
        out = self.bn2(out2)
        
        ## shortcut
        x_pad = torch.zeros_like(out)
        
        idx0 = max((x_pad.size(0)-x.size(0)) // 2, 0)
        idx1 = max((x_pad.size(1)-x.size(1)) // 2, 0)
        
        if x.size(2) > x_pad.size(2):
            x_pad[idx0:min(idx0 + x.size(0), x_pad.size(0)), 
                  idx1:min(idx1 + x.size(1), x_pad.size(1)), 
                  (x_pad.size(2) - x.size(2) // 2) // 2:(x_pad.size(2) - x.size(2) // 2) // 2 + x.size(2) // 2, 
                  (x_pad.size(3) - x.size(3) // 2) // 2:(x_pad.size(3) - x.size(3) // 2) // 2 + x.size(3) // 2] = x[:min(x.size(0), out.size(0)), 
                                                                                                                    :min(x.size(1), out.size(1)), ::2, ::2]
        else:
            x_pad[idx0:min(idx0 + x.size(0), x_pad.size(0)), 
                  idx1:min(idx1 + x.size(1), x_pad.size(1)), 
                  (x_pad.size(2) - x.size(2)) // 2:(x_pad.size(2) - x.size(2)) // 2 + x.size(2), 
                  (x_pad.size(3) - x.size(3)) // 2:(x_pad.size(3) - x.size(3)) // 2 + x.size(3)] = x[:min(x.size(0), out.size(0)), 
                                                                                                     :min(x.size(1), out.size(1)), :, :]
        
        out += x_pad
        out = self.relu(out)
        
        return out
'''
class ResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes = 10, has_mask = None):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        
        if has_mask is None : has_mask = [1]*3*n

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(block, 16, blocks=n, stride=1, has_mask=has_mask[0:n])
        self.layer2 = self._make_layer(block, 32, blocks=n, stride=2, has_mask=has_mask[n:2*n])
        self.layer3 = self._make_layer(block, 64, blocks=n, stride=2, has_mask=has_mask[2*n:3*n])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, has_mask):
        layers = []
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes): 
            layers.append(LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)))
        if not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class SparseResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes = 10, has_mask = None, T = 1):
        super(SparseResNet, self).__init__()
        
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
                
        if has_mask is None : has_mask = [1]*3*n
        
        self.T = T

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        #self.mask = Mask(self.inplanes)

        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(block, 16, blocks = n, stride = 1, has_mask = has_mask[0:n])
        self.layer2 = self._make_layer(block, 32, blocks = n, stride = 2, has_mask = has_mask[n:2*n])
        self.layer3 = self._make_layer(block, 64, blocks = n, stride = 2, has_mask = has_mask[2*n:3*n])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()
        
        self.softmax = nn.Softmax(dim = -1)
        
        #self.surv = Survival(16)
        
        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, has_mask):
        layers = nn.ModuleList()
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes): 
            layers.append(LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)))
        
        layers.append(block(self.inplanes, planes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, planes))
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        self.x = self.conv1(x)
        self.x = self.bn1(self.x)
                
        x = self.relu(self.x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        
        ## info cascade by survival analysis
        ### layer-wise distances

        distances = []

        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                dist1 = cal_dist()(block.out1_)
                dist2 = cal_dist()(block.out2_)
                distances.append(dist1)
                distances.append(dist2)
         

        ### filters' survival distributions
        filters_weibull = []
        
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                filters_weibull.append(block.surv1)
                filters_weibull.append(block.surv2)

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
        
#model = torch.load('pretrained/resnet_110.th', map_location = 'cuda:0')
#state_dict = model['state_dict']

#[name for name, param in state_dict.items() if 'conv' in name]
class PrunedResNet(nn.Module):
    def __init__(self, block, num_layers, num_classes = 10, has_mask = None, T = 1):
        super(PrunedResNet, self).__init__()
        
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
                
        if has_mask is None : has_mask = [1]*3*n
        
        self.T = T

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)


        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(block, 16, blocks = n, stride = 1, has_mask = has_mask[0:n])
        self.layer2 = self._make_layer(block, 32, blocks = n, stride = 2, has_mask = has_mask[n:2*n])
        self.layer3 = self._make_layer(block, 64, blocks = n, stride = 2, has_mask = has_mask[2*n:3*n])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()
        
        self.softmax = nn.Softmax(dim = -1)

        self.pool = nn.MaxPool2d(kernel_size = 3, stride = 3)
        
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, has_mask):
        layers = nn.ModuleList()
        if has_mask[0] == 0 and (stride != 1 or self.inplanes != planes): 
            layers.append(LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)))
        if not has_mask[0] == 0:
            layers.append(block(self.inplanes, planes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if not has_mask[i] == 0:
                layers.append(block(self.inplanes, planes, planes))
                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        self.x = self.conv1(x)
        self.x = self.bn1(self.x)
                
        x = self.relu(self.x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
        
       

def resnet_56(**kwargs):
    return ResNet(ResBasicBlock, 56, **kwargs)

def resnet_56_sparse(**kwargs):
    return SparseResNet(SparseResBasicBlock, 56, **kwargs)

def resnet_110(**kwargs):
    return ResNet(ResBasicBlock, 110, **kwargs)

def resnet_110_sparse(**kwargs):
    return SparseResNet(SparseResBasicBlock, 110, **kwargs)

def pruned_resnet_56_sparse(**kwargs):
    ''' pruning redundant filters '''
    return PrunedResNet(SparseResBasicBlock, 56, **kwargs) 

def pruned_resnet_110_sparse(**kwargs):
    ''' pruning redundant filters '''
    return PrunedResNet(SparseResBasicBlock, 110, **kwargs) 

#model = resnet_110_sparse()
#ckpt = torch.load('pretrained/resnet_110.th')
#state_dict = ckpt[list(ckpt.keys())[0]]
#state_dict = dict((k[7:].replace('linear', 'fc'), v) for (k, v) in state_dict.items())
        
#model.load_state_dict(state_dict, strict = False)
