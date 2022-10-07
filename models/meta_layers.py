"""
Override neural network layers for meta-learning 
Conventional module:
    def forward(x):
        return self.layer(x)
Override module:
    def forward(x, params):
        return self.layer(x, params)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch 
# import numpy as np
from copy import copy
from functools import reduce
# from torch.nn.modules import padding
# from models.feature_vis import show_feature


def extract_top_level_dict(current_dict):
    """
    Build a graph dict from the passed depth_keys, value pair. Useful for dynamically passing extern params
    :param current_dict: current dictionary passed
    :return : A dict graph of the params already added to the graph
    """
    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace('layer_dict.', '')
        top_level = name.split('.')[0] ## conv, bn, etc.
        sub_level = '.'.join(name.split('.')[1:]) ## weight.0, bias.0 etc.
        
        if top_level not in output_dict:
            if sub_level == '':
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key:value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item
    return output_dict


class MetaPReLU(nn.Module):
    def __init__(self, num_filters=None):
        """
        Implementation of PReLU for Task-Driven Learning
        """
        super(MetaPReLU, self).__init__()
        if num_filters is None: ## use default activation with one unit
            self.weight = nn.Parameter(torch.Tensor([0.25]), requires_grad=True)
        else:
            self.weight = nn.Parameter(torch.ones(num_filters)*0.25, requires_grad=True)
    
    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(params)
            weight = params['weight']
        else:
            weight = self.weight
        return F.prelu(x, weight)
    
    def extra_repr(self) -> str:
        return 'MetaPReLU'

class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=True, groups=1, dilation_rate=1):
        """
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of being able to receive a parameter
        dictionary at the forward pass which allows the convolution to use external weights instead of the internal ones stored in the conv layer. Useful
        for inner loop optimization in the meta learning setting.
        :param in_channles: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: kernel size of conv kernel
        :param padding: padding of convolutional operation
        :param use_bias: whether to use a bias
        :param groups: groups
        """
        super(MetaConv2dLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        self.use_bias = use_bias
        self.groups = int(groups)
        self.dilation_rate = dilation_rate
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        
        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x, params=None):
        """
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: input image batch
        :param params: If none, then conv layer will use the stored self.weight and self.bias, if they are not none, the conv layer will use
        the passed param as its parameters.
        :return : The output of a convolutional function.
        """
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params['weight'], params['bias']
            else:
                (weight, bias) = params['weight'], None
        else:
            if self.use_bias:
                (weight, bias) = self.weight, self.bias
            else:
                (weight, bias) = self.weight, None
        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride, 
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)
        return out
    
    def extra_repr(self):
        return '{in_channels},{out_channels},kernel_size=({kernel_size}, {kernel_size}),stride={stride},padding={padding}'.format(**self.__dict__)


class MetaMAEB(nn.Module):
    def __init__(self, in_channels, num_filters, args, dilated_factors=3, negative_slope=0.2, withSE=True):
        """
        MAEB from ReMAEN: https://github.com/nnUyi/ReMAEN/blob/master/codes/ReMAEN.py
        for meta-learning
        but add residual connection 
        :param withSE: add attention mechanism or not
        """
        super(MetaMAEB, self).__init__()
        self.in_channels=in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.negative_slope = negative_slope
        self.withSE = withSE
        self.layer_dict = nn.ModuleDict()
        for i in range(self.dilated_factors):
            self.layer_dict['conv-s0-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
            self.layer_dict['conv-s1-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args, eps=1e-5, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, num_step, params=None, training=False, bkp_running_statistics=False):
        ## extract params
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s0-d{}'.format(i+1)].forward(x, params=param_dict['conv-s0-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope)
            tmp = self.layer_dict['conv-s1-d{}'.format(i+1)].forward(tmp, params=param_dict['conv-s1-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope)
            outs.append(tmp)
        out = reduce(torch.add, outs)
        out = self.layer_dict['bn'](out, num_step, params=param_dict['bn'], training=training, bkp_running_statistics=False)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        out = x[:,:self.num_filters,:,:] + out
        # show_feature(out, layer_name='maeb')
        return out

class MetaMSFUSEV3(nn.Module):
    def __init__(self, in_channels, num_filters, args, dilated_factors=3, negative_slope=0.2, withSE=True):
        """
        MAEB from ReMAEN: https://github.com/nnUyi/ReMAEN/blob/master/codes/ReMAEN.py
        for meta-learning
        but add residual connection 
        :param withSE: add attention mechanism or not
        """
        super(MetaMSFUSEV3, self).__init__()
        self.in_channels=in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.negative_slope = negative_slope
        self.withSE = withSE
        self.layer_dict = nn.ModuleDict()
        for i in range(self.dilated_factors):
            self.layer_dict['conv-s0-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
            self.layer_dict['conv-s1-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
        self.layer_dict['catconv'] = MetaConv2dLayer(num_filters*3, num_filters, 1, 1, 0, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args, eps=1e-5, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, num_step, params=None, training=False, bkp_running_statistics=False):
        ## extract params
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s0-d{}'.format(i+1)].forward(x, params=param_dict['conv-s0-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope)
            outs.append(tmp)
        fuses = [None, None, None]
        fuses[0] = reduce(torch.add, outs)
        fuses[1] = outs[0] + outs[1] - outs[2]
        fuses[2] = outs[0] - outs[1] + outs[2]
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s1-d{}'.format(i+1)].forward(fuses[i], params=param_dict['conv-s1-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope)
            outs.append(tmp)
        fuses = [None, None, None]
        fuses[0] = (outs[1]+outs[2])*0.5
        fuses[1] = (outs[0]-outs[2])*0.5
        fuses[2] = (outs[0]-outs[1])*0.5
        out = torch.cat(fuses, dim=1)
        out = self.layer_dict['catconv'](out, params=param_dict['catconv'])
        out = self.layer_dict['bn'](out, num_step, params=param_dict['bn'], training=training, bkp_running_statistics=False)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        out = x[:,:self.num_filters,:,:] + out
        return out

class MetaMS(nn.Module):
    def __init__(self, in_channels, num_filters, args, dilated_factors=3, negative_slope=0.2, withSE=True):
        """
        MAEB from ReMAEN: https://github.com/nnUyi/ReMAEN/blob/master/codes/ReMAEN.py
        for meta-learning
        but add residual connection 
        :param withSE: add attention mechanism or not
        """
        super(MetaMS, self).__init__()
        self.in_channels=in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.negative_slope = negative_slope
        self.withSE = withSE
        self.layer_dict = nn.ModuleDict()
        for i in range(self.dilated_factors):
            self.layer_dict['conv-s0-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=2*i+1, dilation_rate=2*i+1)
            self.layer_dict['conv-s1-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=2*i+1, dilation_rate=2*i+1)
        self.layer_dict['cat1'] = MetaConvBNPReLU(3*num_filters, num_filters, args, kernel_size=1, stride=1, padding=0, activation=True, withSE=False)
        self.layer_dict['catconv2'] = MetaConv2dLayer(3*num_filters, num_filters, 1, 1, 0, use_bias=False)
        self.layer_dict['catbn2'] = MetaBNLayer(num_filters, args, eps=1e-5,
                                            use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, num_step, params=None, training=False, bkp_running_statistics=False):
        ## extract params
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s0-d{}'.format(i+1)].forward(x, params=param_dict['conv-s0-d{}'.format(i+1)])
            outs.append(tmp)
        out = torch.cat(outs, dim=1)
        out = self.layer_dict['cat1'](out, num_step, params=param_dict['cat1'], training=training, bkp_running_statistics=False)
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s1-d{}'.format(i+1)].forward(out, params=param_dict['conv-s0-d{}'.format(i+1)])
            outs.append(tmp)
        out = torch.cat(outs, dim=1)
        out = self.layer_dict['catconv2'](out, params=param_dict['catconv2'])
        out = self.layer_dict['catbn2'](out, num_step, params=param_dict['catbn2'], training=training, bkp_running_statistics=False)
        out = self.layer_dict['se'](out, params=param_dict['se'])
        out = x+out
        return out

class MetaMSB(nn.Module):
    def __init__(self, in_channels, num_filters, dilated_factors=3, negative_slope=0.2, withSE=True):
        """
        Meta Multi-Scale Block
        :param in_channels: number of input channels
        :param num_filters: number of conv filters
        :param dilated_factors: number of dilated branches
        :param negative_slope: negative slope for LeakyReLU
        :param withSE: whether use SE
        """
        super(MetaMSB, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.negative_slope = negative_slope
        self.withSE = withSE
        ### initialize structure
        self.layer_dict = nn.ModuleDict()
        for i in range(dilated_factors):
            self.layer_dict['conv-s1-d{}'.format(i+1)] = MetaConv2dLayer(in_channels, num_filters, 3, 1, padding=i+1, dilation_rate=i+1)
            self.layer_dict['conv-s2-d{}'.format(i+1)] = MetaConv2dLayer(num_filters, num_filters, 3, 1, padding=i+1, dilation_rate=i+1)
        self.layer_dict['conv'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1)
        if withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, params=None):
        ### extract param_dict
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=param_dict)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        outs = []
        for i in range(self.dilated_factors):
            tmp = self.layer_dict['conv-s1-d{}'.format(i+1)].forward(x, params=param_dict['conv-s1-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, self.negative_slope)
            tmp = self.layer_dict['conv-s2-d{}'.format(i+1)].forward(tmp, params=param_dict['conv-s2-d{}'.format(i+1)])
            tmp = F.leaky_relu(tmp, self.negative_slope)
            outs.append(tmp)
        out = reduce(torch.add, outs)
        out = self.layer_dict['conv'].forward(out, params=param_dict['conv'])
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        out = x[:, :self.num_filters, :, :] + out
        return out

class MetaHierarchicalAgg(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride, padding, args=None, no_bn_learnable_params=False, 
                 device=None, negative_slope=0.2, inplace=True, use_multi_scale=False, multi_scale='maeb', withSE=True, dilated_factors=3):
        super(MetaHierarchicalAgg, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.negative_slope=negative_slope
        self.use_multi_scale = use_multi_scale
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.multi_scale = multi_scale
        self.withSE = withSE
        ## ResBlock
        if use_multi_scale:
            if multi_scale == 'msb':
                self.layer_dict['msb'] = MetaMSB(in_channels=in_channels, num_filters=num_filters, dilated_factors=dilated_factors, 
                                                 negative_slope=negative_slope, withSE=withSE)
            elif multi_scale == 'maeb':
                self.layer_dict['maeb'] = MetaMAEB(in_channels=in_channels, num_filters=num_filters, withSE=withSE, dilated_factors=dilated_factors)
            self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                    padding=padding, args=args, negative_slope=negative_slope, inplace=inplace, activation=True)
        else:
            self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                    padding=padding, args=args, negative_slope=negative_slope, inplace=inplace, activation=True)
        self.layer_dict['cbn2'] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                  padding=padding, args=args, negative_slope=negative_slope, inplace=inplace, activation=False)
        self.layer_dict['trans'] = MetaConv2dLayer(in_channels=3*num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args=args)
        if self.withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, num_step, params=None, training=False, bkp_running_statistics=False):
        ## extract params
        param_dict = dict()
        if params is not None:
            # print('keys of params in MetaHierachicalAgg: ', params.keys())
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None

        if self.use_multi_scale:
            if self.multi_scale == 'msb':
                out = self.layer_dict['msb'].forward(x, num_step, params=param_dict['msb'], training=training,
                                                        bkp_running_statistics=bkp_running_statistics)
            elif self.multi_scale == 'maeb':
                out = self.layer_dict['maeb'].forward(x, params=param_dict['maeb'])
            else:
                raise NotImplementedError
        else:
            out = x

        shortcuts = [out[:,:self.num_filters,:,:]]
        out = self.layer_dict['cbn1'].forward(out, num_step=num_step, params=param_dict['cbn1'], training=training, 
                                              bkp_running_statistics=bkp_running_statistics)
        shortcuts.append(out)
        out = self.layer_dict['cbn2'].forward(out, num_step=num_step, params=param_dict['cbn2'], training=training, 
                                              bkp_running_statistics=bkp_running_statistics)
        shortcuts.append(out)
        out = torch.cat(shortcuts, dim=1)
        
        ## trans
        out = self.layer_dict['trans'].forward(out, params=param_dict['trans'])


        out = self.layer_dict['bn'].forward(out, current_step=num_step, params=param_dict['bn'], training=training, 
                                            bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        if self.in_channels != self.num_filters:
            out = x[:,:self.num_filters, :, :] + out
        else:
            out = x + out
        out = F.leaky_relu(out, negative_slope=self.negative_slope, inplace=True)
        
        return out
    
    def restore_bkp_stats(self):
        self.layer_dict['cbn1'].restore_bkp_stats()
        self.layer_dict['cbn2'].restore_bkp_stats()
        self.layer_dict['bn'].restore_bkp_stats()


class MetaSelfCalibratedConv(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride, padding, r, args=None, use_bn=True):
        """
        Override self-calibrated convolution layer for meta-learning use
        :param in_channels: input channels
        :param out_channels: output channels or num_filters
        :param kernel_size, stride, padding: 
        :param use_bias: whether to use bias in conv
        """
        super(MetaSelfCalibratedConv, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.r = r
        self.use_bn = use_bn
        self.avg_pool = nn.AvgPool2d(kernel_size=r, stride=r)
        
        self.layer_dict = nn.ModuleDict()

        self.layer_dict['conv2'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                     padding=padding, use_bias=False)

        self.layer_dict['conv3'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                     padding=padding, use_bias=False)
        self.layer_dict['conv4'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                     padding=padding, use_bias=False)
        if self.use_bn:
            self.layer_dict['bn2'] = MetaBNLayer(num_filters, args=args)
            self.layer_dict['bn3'] = MetaBNLayer(num_filters, args=args)
            self.layer_dict['bn4'] = MetaBNLayer(num_filters, args=args)

    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        # identity = x
        # self.k2(x)
        out_2 = self.avg_pool(x)
        out_2 = self.layer_dict['conv2'].forward(out_2, params=param_dict['conv2'])
        if self.use_bn:
            out_2 = self.layer_dict['bn2'].forward(out_2, current_step=num_step, params=param_dict['bn2'], 
                                             training=training, bkp_running_statistics=bkp_running_statistics)
        # sigmoid(x+self.k2(x)UP)
        out_2 = F.interpolate(out_2, x.size()[2:])
        out_2 = torch.sigmoid(torch.add(x, out_2))
        
        ## self.k3(x)
        out_3 = self.layer_dict['conv3'].forward(x, params=param_dict['conv3'])
        if self.use_bn:
            out_3 = self.layer_dict['bn3'].forward(out_3, current_step=num_step, params=param_dict['bn3'], 
                                            training=training, bkp_running_statistics=bkp_running_statistics)
        ## self.k3(x)*sigmoid(x+self.k2(x)UP)
        out = torch.mul(out_2, out_3)
        ## self.k4(x)
        out = self.layer_dict['conv4'].forward(out, params=param_dict['conv4'])
        if self.use_bn:
            out = self.layer_dict['bn4'].forward(out, current_step=num_step, params=param_dict['bn4'], 
                                             training=training, bkp_running_statistics=bkp_running_statistics)
        return out
    
    def restore_bkp_stats(self):
        if self.use_bn:
            for i in range(3):
                self.layer_dict['bn{}'.format(i+2)].restore_bkp_stats()
    
    def extra_repr(self):
        return 'Self-Calibrated Conv ({}, {}), r: {}'.format(self.in_channels, self.num_filters, self.r)

class MetaSCBottleneck(nn.ModuleDict):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, args=None, use_bn=True, negative_slope=0.01):
        super(MetaSCBottleneck, self).__init__()
        pooling_r = 4
        self.negative_slope = negative_slope
        self.layer_dict = nn.ModuleDict()
        out_channels = in_channels//2
        self.use_bn = use_bn
        self.layer_dict['conv1_a'] = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                                                          stride=stride, padding=0, use_bias=True)
        self.layer_dict['conv2_a'] = MetaConv2dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                                          stride=stride, padding=padding, use_bias=True)
        if self.use_bn:
            self.layer_dict['bn_a'] = MetaBNLayer(out_channels, args=args, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        self.layer_dict['conv1_b'] = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, 
                                                          stride=stride, padding=0, use_bias=True)
        self.layer_dict['scconv'] =  MetaSelfCalibratedConv(in_channels=out_channels, num_filters=out_channels, kernel_size=kernel_size, stride=stride, 
                                                                padding=padding, r=pooling_r, args=args, use_bn=use_bn)
        
        self.layer_dict['conv'] = MetaConv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                                          stride=stride, padding=padding, use_bias=True)
        self.layer_dict['bn'] = MetaBNLayer(in_channels, args=args, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        residual = x
        out_a = self.layer_dict['conv1_a'].forward(x, params=param_dict['conv1_a'])
        out_a = F.leaky_relu(out_a, negative_slope=self.negative_slope, inplace=True)
        out_a = self.layer_dict['conv2_a'].forward(out_a, params=param_dict['conv2_a'])
        if self.use_bn:
            out_a = self.layer_dict['bn_a'].forward(out_a, current_step=num_step, params=param_dict['bn_a'], training=training, 
                                                    bkp_running_statistics=bkp_running_statistics)
                                                    
        out_a = F.leaky_relu(out_a, negative_slope=self.negative_slope, inplace=True)

        out_b = self.layer_dict['conv1_b'].forward(x, params=param_dict['conv1_b'])
        out_b = F.leaky_relu(out_b, negative_slope=self.negative_slope, inplace=True)
        out_b = self.layer_dict['scconv'].forward(out_b, num_step, params=param_dict['scconv'], training=training, 
                                                  bkp_running_statistics=bkp_running_statistics)
        out_b = F.leaky_relu(out_b, negative_slope=self.negative_slope, inplace=True)
        
        out = self.layer_dict['conv'].forward(torch.cat([out_a, out_b],dim=1), params=param_dict['conv'])
        out = self.layer_dict['bn'].forward(out, current_step=num_step, params=param_dict['bn'], training=training, 
                                            bkp_running_statistics=bkp_running_statistics)
        out += residual
        out = F.leaky_relu(out, negative_slope=self.negative_slope, inplace=True)
        return out

class MetaBNLayer(nn.Module):
    def __init__(self, num_features, args, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, meta_batch_norm=True,  use_per_step_bn_statistics=False):
        """
        A MetaBatchNorm layer. Applies the same functionality of a standard BatchNorm layer with the added functionality of being able to receive 
        a parameter dictionary at the forward pass which allows the convolution to use external weights instead of the internal ones stored in the
        bn layer. Useful for inner loop optimization in the meta learning setting. Also has the additional functionality of being able to store per
        step running stats and per step beta and gamma
        :param num_features: number of features
        :param args: argument
        :param eps: epsilon
        :param momentum: momentum for updating in bn
        :param affine: affine y=x_hat*gamma+beta
        :param track_running_stats: whether to track running stats
        :param meta_batch_norm:
        :param use_per_step_bn_statistics: whether to use different bn stats in each inner loop
        """
        super(MetaBNLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps

        self.affine = affine
        self.track_running_stats = track_running_stats
        self.meta_batch_norm = meta_batch_norm
        self.use_per_step_bn_statistics = use_per_step_bn_statistics
        self.args = args
        self.learnable_gamma = self.args.learnable_bn_gamma
        self.learnable_beta = self.args.learnable_bn_beta
        if self.use_per_step_bn_statistics:
            self.r_mean = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features), 
                                       requires_grad=False)
            self.r_var = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features), 
                                       requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(args.number_of_training_steps_per_iter, num_features), 
                                       requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(args.number_of_training_steps_per_iter, num_features), 
                                       requires_grad=self.learnable_gamma)
        else:
            self.r_mean = nn.Parameter(torch.zeros(num_features), requires_grad=False)
            self.r_var = nn.Parameter(torch.ones(num_features), requires_grad=False)
            self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=self.learnable_beta)
            self.weight = nn.Parameter(torch.ones(num_features), requires_grad=self.learnable_gamma)
        self.bkp_r_mean = torch.zeros(self.r_mean.shape) ## backup running mean
        self.bkp_r_var = torch.ones(self.r_var.shape)  ## backup running var
        self.momentum = momentum
    
    def forward(self, input, current_step, params=None, training=False, bkp_running_statistics=False):
        """
        Forward propagates by applying a batch norm function.
        :param input: input features
        :param current_step: The current inner loop step being taken. 
        :param params: A dict containing 'weight' and 'bias'
        :param bkp_running_statistics: whether to backup running statistics
        :return : result of batch norm
        """
        if params is not None:
            params = extract_top_level_dict(params)
            (weight, bias) = params['weight'], params['bias']
        else:
            weight, bias = self.weight, self.bias
        if self.use_per_step_bn_statistics:  ### each step use different bn statistics
            r_mean = self.r_mean[current_step]
            r_var = self.r_var[current_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[current_step]
                    weight = self.weight[current_step]
        else:
            r_mean = self.r_mean
            r_var = self.r_var

        if bkp_running_statistics and self.use_per_step_bn_statistics:
            self.bkp_r_mean.data = copy(self.r_mean.data)
            self.bkp_r_var.data = copy(self.r_var.data)

        if not training: ## for evaluation
            r_mean = self.r_mean
            r_var = self.r_var
        
        output = F.batch_norm(input, running_mean=r_mean, running_var=r_var, weight=weight, bias=bias, 
                              training=training, momentum=self.momentum,  eps=self.eps)
        return output
    
    def restore_bkp_stats(self):
        """
        Reset batch statistics to their backup values which are collected after each forward pass
        """
        if self.use_per_step_bn_statistics:
            device = self.r_mean.device
            self.r_mean = nn.Parameter(self.bkp_r_mean.to(device), requires_grad=False)
            self.r_var = nn.Parameter(self.bkp_r_var.to(device), required_grad=False)
    
    def extra_expr(self):
        return '{num_features},eps={eps},momentum={momentum},affine={affine},track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaConvBNLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, negative_slope=0.2,
                 inplace=True, groups=1, dilation_rate=1, eps=1e-5, activation=False, withSE=False):
        """
        Initialize a Meta Conv->BN->LReLU
        :param in_channels: input channels
        :param out_channels: output channels (num_filters)
        :param kernel_size: kernel size of MetaConv2d
        :param stride: stride of MetaConv2d
        :param padding: padding of MetaConv2d
        :param args: arguments for MetaBNLayer
        :param negative_slope: slope for leaky relu function
        :param inplace: inplace for leaky relu function
        :param groups: groups of conv
        :param dilation_rate: dilation rate of conv
        :param eps: eplison in bn to avoid non-zero division
        :param withSE: whether to use SEBlock
        :return:
        """
        super(MetaConvBNLReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.activation = activation
        self.withSE = withSE
        ## build block
        self.layer_dict = nn.ModuleDict()
        self.conv = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                    stride=stride, padding=padding, use_bias=False, groups=groups, dilation_rate=
                                    dilation_rate) ## do not use bias when followed by batch norm
        self.bn = MetaBNLayer(num_features=out_channels, args=args, eps=eps,
                              use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.se = MetaSEBlock(out_channels, reduction=out_channels//4)
    
    def forward(self, input, num_step, params=None, training=False, bkp_running_statistics=False):
        """
        Forward pass. If params is not None, use params for inference, else use default parameters
        :param num_step: step index of inner loop
        :param training: used for bn inference
        :param bkp_running_stats: whether to backp running statistics        
        """
        if params is not None:
            params = extract_top_level_dict(params) ## extract params from dict params
            conv_params, bn_params = params['conv'], params['bn']
            if self.withSE:
                se_params = params['se']
        else:
            conv_params, bn_params = None, None
            if self.withSE:
                se_params = None
        x = self.conv.forward(input, params=conv_params)
        x = self.bn.forward(x, current_step=num_step, params=bn_params, training=training, 
                            bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            x = self.se.forward(x, params=se_params)
        if self.activation:
            x = F.leaky_relu(input=x, negative_slope=self.negative_slope, inplace=self.inplace)
        return x
    
    def restore_bkp_stats(self):
        ### restore bn statistics
        self.bn.restore_bkp_stats()

class MetaConvBNPReLU(nn.Module):
    def __init__(self, in_channels, num_filters, args, kernel_size=3, stride=1, padding=1, activation=False, withSE=False):
        """
        Conv -> BN -> PReLU for Task-Driven Learning
        :param in_channels: input channels num
        :param num_filters: number of filters of conv
        :param use_bias: whether to use bias
        :param activation: False
        :param withSE: whether to SEBlock
        :return:
        """
        super(MetaConvBNPReLU, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.args = args
        self.activation = activation
        self.withSE = withSE
        ### build module
        self.layer_dict = nn.ModuleDict()
        self.conv = MetaConv2dLayer(in_channels, num_filters, kernel_size=kernel_size, stride=stride, padding=padding, use_bias=False)
        self.bn = MetaBNLayer(num_filters, args, eps=1e-5, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.se = MetaSEBlock(num_filters, reduction=num_filters//4)
        if activation:
            self.prelu = MetaPReLU(num_filters=1)
    
    def forward(self, inp, num_step, params=None, training=False, bkp_running_statistics=False):
        ### extract parameters
        if params is not None:
            params = extract_top_level_dict(params)
            conv_params, bn_params = params['conv'], params['bn']
            if self.withSE:
                se_params = params['se']
            if self.activation:
                prelu_params = params['prelu']
        else:
            conv_params, bn_params = None, None
            if self.withSE:
                se_params = None
            if self.activation:
                prelu_params = None
        ### forward
        out = self.conv.forward(inp, conv_params)
        out = self.bn.forward(out, num_step, params=bn_params, training=training, bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.se.forward(out, params=se_params)
        if self.activation:
            out = self.prelu.forward(out, prelu_params)
        return out
    
    def restore_bkp_stats(self):
        self.bn.restore_bkp_stats()

class MetaSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        """
        Implementation of squeeze and excitation block for meta-learning
        """
        super(MetaSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.w1 = nn.Parameter(torch.ones(in_channels//reduction, in_channels))
        self.w2 = nn.Parameter(torch.ones(in_channels, in_channels//reduction))
        self.reduction=reduction
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)
    
    def forward(self, x, params=None):
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            w1, w2 = param_dict['w1'], param_dict['w2']
        else:
            w1, w2 = self.w1, self.w2
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        z = F.linear(z, w1, None)
        z = F.relu(z, inplace=True)
        z = F.linear(z, w2, None)
        z = torch.sigmoid(z).view(b, c, 1, 1)
        out = z*x
        return out
    
    def extra_repr(self):
        return 'MetaSEBlock, reduction: {}'.format(self.reduction)

class MetaDualBlock(nn.Module):
    def __init__(self, num_filters, args, withSE=False):
        """
        Dual ResBlock for Task-Driven Learning
        """
        super(MetaDualBlock, self).__init__()
        self.withSE = withSE
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['cbr1'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
        self.layer_dict['cbr2'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
        self.layer_dict['cbr3'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)

        self.layer_dict['trans'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args, eps=1e-5,
                                            use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, num_filters//4)
    
    def forward(self, inp, num_step, params=None, training=False, bkp_running_statistics=False):
        ### extract params
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        ### do forward pass
        res = self.layer_dict['cbr1'].forward(inp, num_step, params=param_dict['cbr1'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['cbr2'].forward(res, num_step, params=param_dict['cbr2'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = torch.add(inp, out) ### residual
        out = self.layer_dict['cbr3'].forward(out, num_step, params=param_dict['cbr3'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = torch.add(out, res) ### residual
        out = self.layer_dict['trans'].forward(out, params=param_dict['trans'])
        out = self.layer_dict['bn'].forward(out, current_step=num_step, params=param_dict['bn'], training=training,
                                            bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        return out

class MetaRes(nn.Module):
    def __init__(self, num_filters, args, withSE=False):
        """
        Dual ResBlock for Task-Driven Learning
        """
        super(MetaRes, self).__init__()
        self.withSE = withSE
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['cbr1'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
        self.layer_dict['cbr2'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
        self.layer_dict['cbr3'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
        self.layer_dict['cbr4'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)

        self.layer_dict['trans'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args, eps=1e-5,
                                            use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, num_filters//4)
    
    def forward(self, inp, num_step, params=None, training=False, bkp_running_statistics=False):
        ### extract params
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        ### do forward pass
        out = self.layer_dict['cbr1'].forward(inp, num_step, params=param_dict['cbr1'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['cbr2'].forward(out, num_step, params=param_dict['cbr2'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        res = out + inp
        out = self.layer_dict['cbr3'].forward(res, num_step, params=param_dict['cbr3'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['cbr4'].forward(out, num_step, params=param_dict['cbr4'], training=training,
                                              bkp_running_statistics=bkp_running_statistics)
        out = res + out

        out = self.layer_dict['trans'].forward(out, params=param_dict['trans'])
        out = self.layer_dict['bn'].forward(out, current_step=num_step, params=param_dict['bn'], training=training,
                                            bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        
        return out


class MetaRB(nn.Module):
    def __init__(self, num_filters, args, relu_type='lrelu', withSE=False):
        """
        Residual Block for Task-Driven Learning
        :param num_filters: number of conv filters
        :param args: arguments
        :param relu_type: type of relu activation
        :param withSE: whether to use SEBlock in CBNLReLU or CBNPReLU
        """
        super(MetaRB, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.relu_type = relu_type
        self.withSE = withSE
        assert relu_type in ['lrelu', 'prelu'], 'activation: {} not implemented yet.'.format(relu_type)
        ### build block
        if self.relu_type == 'lrelu':
            self.layer_dict['cb1'] = MetaConvBNLReLU(num_filters, num_filters, 3, 1, 1, args, negative_slope=0.2,
                                                     activation=True, withSE=withSE)
            self.layer_dict['cb2'] = MetaConvBNLReLU(num_filters, num_filters, 3, 1, 1, args, negative_slope=0.2,
                                                     activation=False, withSE=withSE)
        else:
            self.layer_dict['cb1'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=True, withSE=False)
            self.layer_dict['cb2'] = MetaConvBNPReLU(num_filters, num_filters, args, activation=False, withSE=False)
        self.layer_dict['trans'] = MetaConv2dLayer(3*num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args, eps=1e-5, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se'] = MetaSEBlock(num_filters, num_filters//4)

    def forward(self, inp, num_step, params=None, training=False, bkp_running_statistics=False):
        ### extract params
        param_dict = nn.ModuleDict()
        if params is not None:
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        ### do forward pass
        residuals = [inp]
        out = self.layer_dict['cb1'].forward(inp, num_step, params=param_dict['cb1'], training=training,
                                             bkp_running_statistics=bkp_running_statistics)
        residuals.append(out)
        out = self.layer_dict['cb2'].forward(out, num_step, params=param_dict['cb2'], training=training,
                                             bkp_running_statistics=bkp_running_statistics)
        residuals.append(out)
        out = torch.cat(residuals, dim=1)
        out = self.layer_dict['trans'].forward(out, params=param_dict['trans'])
        out = self.layer_dict['bn'].forward(out, num_step, params=param_dict['bn'], training=training,
                                            bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        return out

class MetaMSRB(nn.Module):
    def __init__(self, in_channels, num_filters, args, relu_type='lrelu', msb='MARB', rb='RB', withSE=False, 
                 dilated_factors=3):
        super(MetaMSRB, self).__init__()
        ### build block
        self.num_filters = num_filters
        ### choose Multi-Scale Block
        
        if msb == 'MAEB':
            self.msb = MetaMAEB(in_channels, 48, args, dilated_factors=dilated_factors, negative_slope=0.2, withSE=withSE)
        else:
            self.msb = MetaMSB(in_channels, num_filters, dilated_factors=3, withSE=True)
        
        ### choose Residual Block
        if rb == 'RB':
            self.rb = MetaRB(num_filters, args, relu_type=relu_type, withSE=withSE)
        else:
            self.rb = MetaRes(num_filters, args, withSE=withSE)
    
    def forward(self, inp, num_step, params=None, training=False, bkp_running_statistics=False):
        ### extract params
        if params is not None:
            params = extract_top_level_dict(params)
            msb_params, rb_params = params['msb'], params['rb']
        else:
            msb_params, rb_params = None, None
        ### forward pass
        out = self.msb.forward(inp, num_step, msb_params, training=training)
        out = self.rb.forward(out, num_step, rb_params, training=training, bkp_running_statistics=bkp_running_statistics)
        out = inp[:, :self.num_filters, :, :] + out
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        return out

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.arguments import get_args
    args = get_args()
    kwargs={'Agg_input': True, 'input_channels': 3}
    msb = MetaMSB(in_channels=32, num_filters=32)
    for name, param in msb.layer_dict.named_parameters():
        print(name, param.shape)
    img = torch.rand(1, 3, 50, 50)
    input_tensor = torch.rand(1, 32, 50, 50)

    rb = MetaRB(32, args, relu_type='lrelu')
    for name, param in rb.layer_dict.named_parameters():
        if param.requires_grad:
            print(name, param.size())
    out = rb.forward(input_tensor, num_step=0)
    print('output shape: ', out.shape)

    msbrb = MetaMSRB(32, 32, args, rb='RB', msb='MAEB')
    for name, param in msbrb.named_parameters():
        if param.requires_grad:
            print(name, param.size())
