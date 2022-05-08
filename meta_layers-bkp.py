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
import numpy as np
from copy import copy
from functools import reduce

def extract_top_level_dict(current_dict):
    '''
    Build a graph dict from the passed depth_keys, value pair. Useful for dynamically passing extern params
    :param depth_keys: A list of strings making up the name of a variable. Used to make a graph for that params tree.
    :param valuue: param value
    :param key_exists: If none then assume new dict, else load existing dict and add new key->value pairs to it.
    :return : A dict graph of the params already added to the graph
    '''
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


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=True, groups=1, dilation_rate=1):
        '''
        A MetaConv2D layer. Applies the same functionality of a standard Conv2D layer with the added functionality of being able to receive a parameter
        dictionary at the forward pass which allows the convolution to use external weights instead of the internal ones stored in the conv layer. Useful
        for inner loop optimization in the meta learning setting.
        :param in_channles: Number of input channels
        :param out_channels: Number of output channels
        :param kernel_size: kernel size of conv kernel
        :param padding: padding of convolutional operation
        :param use_bias: whether to use a bias
        :param groups: groups
        '''
        super(MetaConv2dLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        num_filters = out_channels
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        self.padding = int(padding)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.dilation_rate = dilation_rate
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        # nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))
        
    def forward(self, x, params=None):
        '''
        Applies a conv2D forward pass. If params are not None will use the passed params as the conv weights and biases
        :param x: input image batch
        :param params: If none, then conv layer will use the stored self.weight and self.bias, if they are not none, the conv layer will use
        the passed param as its parameters.
        :return : The output of a convolutional function.
        '''
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

class MetaMultiScale(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size=3, dilated_factors=3):
        super(MetaMultiScale, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.layer_dict = nn.ModuleDict()
        for i in range(self.dilated_factors):
            self.layer_dict['dilate{}'.format(i+1)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, 
                                                                      kernel_size=3, stride=1, padding=2*i+1, dilation_rate=2*i+1)
        self.layer_dict['transition'] = MetaConv2dLayer(in_channels=3*num_filters, out_channels=num_filters, 
                                                        kernel_size=3, stride=1, padding=1, dilation_rate=1)
    
    def forward(self, x, params=None):
        ## extract parameters
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        dilates = []

        for i in range(self.dilated_factors):
            out = self.layer_dict['dilate{}'.format(i+1)].forward(x, params=param_dict['dilate{}'.format(i+1)])
            dilates.append(out)
            # print('out size: ', out.size())
        out = torch.cat(dilates, dim=1) ## concat
        # print('out size: ', out.size())
        out = self.layer_dict['transition'].forward(out, params=param_dict['transition'])
        return out
    
    def extra_expr(self):
        return '{in_channels},{out_channels},{dilated_factors}'.format(**self.__dict__)


class MetaMultiScaleBlock(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size=3, dilated_factors=3, args=None,negative_slope=0.2):
        """
        Multiscale block:  reference: https://github.com/Dengsgithub/DRD-Net/blob/master/DRD-Net/model.py
        """
        super(MetaMultiScaleBlock, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.negative_slope = negative_slope

        self.layer_dict = nn.ModuleDict()

        self.layer_dict['multiscale1'] = MetaMultiScale(in_channels, num_filters, kernel_size, dilated_factors)
        self.layer_dict['bn1'] = MetaBNLayer(num_filters, args=args)
        self.layer_dict['se'] = MetaSEBlock(num_filters, num_filters//4)
        # self.layer_dict['multiscale2'] = MetaMultiScale(num_filters, num_filters, kernel_size, dilated_factors)
        # self.layer_dict['bn2'] = MetaBNLayer(num_filters, args=args)
    
    def forward(self, x, num_step, params=None, training=False, bkp_running_statistics=False):
        ## extract parameters
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None

        out = x
        out = self.layer_dict['multiscale1'].forward(out, params=param_dict['multiscale1'])
        out = self.layer_dict['bn1'].forward(out, current_step=num_step, params=param_dict['bn1'], training=training, 
                                             bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        if self.in_channels != self.num_filters:
            out = x[:,:self.num_filters, :, :] + out ## add residual connection
        else:
            out = x + out
        # out = F.relu(out)
        # out = self.layer_dict['multiscale2'].forward(out, params=param_dict['multiscale2'])
        # out = self.layer_dict['bn2'].forward(out, current_step=num_step, params=param_dict['bn2'], training=training, 
        #                                     bkp_running_statistics=bkp_running_statistics)
        # out = F.relu(out)

        return out

class MetaMAEB(nn.Module):
    def __init__(self, in_channels, num_filters, dilated_factors=3, negative_slope=0.2):
        """
        MAEB from ReMAEN: https://github.com/nnUyi/ReMAEN/blob/master/codes/ReMAEN.py
        for meta-learning
        but add residual connection
        """
        super(MetaMAEB, self).__init__()
        self.in_channels=in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.negative_slope = negative_slope
        self.layer_dict = nn.ModuleDict()
        for i in range(self.dilated_factors):
            self.layer_dict['conv-s0-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
            self.layer_dict['conv-s1-d{}'.format(i+1)] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, 
                                                                         kernel_size=3, stride=1, padding=i+1, dilation_rate=i+1)
        self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, params=None):
        ## extract params
        param_dict = dict()
        if params is not None:
            # print('keys of params in MetaHierachicalAgg: ', params.keys())
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
        out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        out = x[:,:self.num_filters,:,:] + out
        return out

class MetaDilatedAggConv(nn.Module):
    def __init__(self, in_channels, num_filters, dilated_factors=3, reduction=4):
        """
        Multi-scale Dilated conv layer for meta learning
        """
        super(MetaDilatedAggConv, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.dilated_factors = dilated_factors
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight1 = nn.Parameter(torch.empty(num_filters, in_channels, 1, 1))
        self.shared_weight = nn.Parameter(torch.empty(num_filters, num_filters, 3, 3))
        self.shared_bias = nn.Parameter(torch.zeros(num_filters))

        ## squeeze and excitation
        self.w1 = nn.Parameter(torch.ones(num_filters//reduction, num_filters))
        self.w2 = nn.Parameter(torch.ones(num_filters, num_filters//reduction))

        nn.init.kaiming_normal_(self.weight1)
        nn.init.kaiming_normal_(self.shared_weight)
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)
    
    def forward(self, x, params=None):
        if params is not None:
            weight1, shared_weight, shared_bias = params['weight1'], params['shared_weight'], params['shared_bias']
            w1, w2 = params['w1'], params['w2']
        else:
            weight1, shared_weight, shared_bias = self.weight1, self.shared_weight, self.shared_bias
            w1, w2 = self.w1, self.w2
        
        zs = []
        outs = []
        b, c, _, _ = x.size()
        feature = F.conv2d(x, weight1, bias=None, stride=1, padding=0)
        for i in range(1, 1+self.dilated_factors):
            out = F.conv2d(feature, shared_weight, shared_bias, stride=1, padding=i, dilation=i)
            outs.append(out)
            z = self.avg_pool(out).view(b, c)
            zs.append(z)
        
        z = torch.stack(zs, dim=1) ## (batch_size, dilated_factors, num_filters)
        z = F.linear(z, w1, bias=None)
        z = F.linear(z, w2, bias=None) ## (batch_size, dilated_factors, num_filters)
        z = F.softmax(z, dim=1).view(b, self.dilated_factors, c, 1, 1)
        
        out = torch.stack(outs, dim=1) ## (batch_size, dilated_factors, num_filters, H, W)
        out = torch.mul(out, z)
        out = torch.sum(out, dim=1) ## (batch_size, num_filters, H, W)
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        return out
     
    def extra_expr(self):
        return 'DilatedConv, ({}, {}), {}'.format(self.in_channels, self.num_filters, self.dilated_factors)
        

class MetaHierachicalAgg(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, stride, padding, args=None, no_bn_learnable_params=False, 
                 device=None, negative_slope=0.2, inplace=True, use_multi_scale=False, multi_scale='msblock'):
        super(MetaHierachicalAgg, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.negative_slope=negative_slope
        self.use_multi_scale = use_multi_scale
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.multi_scale = multi_scale
        ## ResBlock
        if use_multi_scale:
            if multi_scale == 'msblock':
                self.layer_dict['msblock'] = MetaMultiScaleBlock(in_channels=in_channels, num_filters=num_filters, kernel_size=kernel_size, 
                                                                dilated_factors=3, args=args, negative_slope=negative_slope)
            elif multi_scale == 'maeb':
                self.layer_dict['maeb'] = MetaMAEB(in_channels=in_channels, num_filters=num_filters,)
            self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                    padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=args.device,
                                                    negative_slope=negative_slope, inplace=inplace, use_bias=False, activation=True)
        else:
            self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                    padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=args.device,
                                                    negative_slope=negative_slope, inplace=inplace, use_bias=False, activation=True)
        self.layer_dict['cbn2'] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                  padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=args.device,
                                                  negative_slope=negative_slope, inplace=inplace, use_bias=False, activation=False)
        self.layer_dict['trans'] = MetaConv2dLayer(in_channels=3*num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=False)
        self.layer_dict['bn'] = MetaBNLayer(num_filters, args=args)
        self.layer_dict['se'] = MetaSEBlock(num_filters, reduction=num_filters//4)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
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
            if self.multi_scale == 'msblock':
                out = self.layer_dict['msblock'].forward(x, num_step, params=param_dict['msblock'], training=training,
                                                        bkp_running_statistics=bkp_running_statistics)
            elif self.multi_scale == 'maeb':
                out = self.layer_dict['maeb'].forward(x, params=param_dict['maeb'])
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
    
    def restore_bkp_stats():
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


class MetaSCResBlock(nn.Module):
    def __init__(self, in_channels, num_filters, args=None, use_bn=True, negative_slope=0.01):
        super(MetaSCResBlock, self).__init__()
        self.negative_slope=negative_slope
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['cbn_in'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=num_filters, kernel_size=3, 
                                                    stride=1, padding=1, args=args, negative_slope=negative_slope)
        self.layer_dict['scblock'] = MetaSCBottleneck(in_channels=num_filters, use_bn=use_bn, negative_slope=negative_slope, args=args)
        self.layer_dict['conv_out'] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, kernel_size=3, 
                                                      stride=1, padding=1)
        self.layer_dict['bn_out'] = MetaBNLayer(num_filters, args=args, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        self.layer_dict['se_out'] = MetaSEBlock(num_filters, reduction=4)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        out = self.layer_dict['cbn_in'].forward(x, num_step, params=param_dict['cbn_in'], training=training, 
                                                bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['scblock'].forward(out, num_step, params=param_dict['scblock'], training=training, 
                                                 bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['conv_out'].forward(out, params=param_dict['conv_out'])
        out = self.layer_dict['bn_out'].forward(out, num_step, params=param_dict['bn_out'], training=training, 
                                                bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['se_out'].forward(out, params=param_dict['se_out'])
        out = F.leaky_relu(out, negative_slope=self.negative_slope, inplace=True)
        return out


class MetaBNLayer(nn.Module):
    def __init__(self, num_features, args, eps=1e-5, momentum=0.1, affine=True, 
                 track_running_stats=True, meta_batch_norm=True, 
                 no_bn_learnable_params=False, use_per_step_bn_statistics=False):
        '''
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
        :param no_bn_learnable_params: whether to set no learnable params in bn layer
        :param use_per_step_bn_statistics: whether to use different bn stats in each inner loop
        '''
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
        '''
        Forward propagates by applying a batch norm function.
        :param input: input features
        :param current_step: The current inner loop step being taken. 
        :param params: A dict containing 'weight' and 'bias'
        :param bkp_running_statistics: whether to backup running statistics
        :return : result of batch norm
        '''
        if params is not None:
            params = extract_top_level_dict(params)
            # print('keys in params of MetaBNLayer: ', params.keys())
            (weight, bias) = params['weight'], params['bias']
        else:
            weight, bias = self.weight, self.bias
        if self.use_per_step_bn_statistics:
            r_mean = self.r_mean[current_step]
            r_var = self.r_var[current_step]
            if params is None:
                if not self.args.enable_inner_loop_optimizable_bn_params:
                    bias = self.bias[num_step]
                    weight = self.weight[num_step]
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
        '''
        Reset batch statistics to their backup values which are collected after each forward pass
        '''
        if self.use_per_step_bn_statistics:
            device = self.r_mean.device
            self.r_mean = nn.Parameter(self.bkp_r_mean.to(device), requires_grad=False)
            self.r_var = nn.Parameter(self.bkp_r_var.to(device), required_grad=False)
    
    def extra_expr(self):
        return '{num_features},eps={eps},momentum={momentum},affine={affine},track_running_stats={track_running_stats}'.format(**self.__dict__)


class MetaConvBNLReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, no_bn_learnable_params=False, 
                 device=None, negative_slope=0.01, inplace=True, use_bias=True, groups=1, dilation_rate=1, eps=1e-5, activation=False):
        '''
        Initialize a Meta Conv->BN->LReLU
        :param in_channels: input channels
        :param out_channels: output channels (num_filters)
        :param kernel_size: kernel size of MetaConv2d
        :param stride: stride of MetaConv2d
        :param padding: padding of MetaConv2d
        :param use_bias: Whether to use bias in MetaConv2d
        :param args: arguments for MetaBNLayer
        :param device: device of Tensor for restoring running stats
        :param negative_slope: slope for leaky relu function
        :param inplace: inplace for leaky relu function
        :param groups: groups of conv
        :param dilation_rate: dilation rate of conv
        :param eps: eplison in bn to avoid non-zero division
        '''
        super(MetaConvBNLReLU, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.args = args
        self.negative_slope = negative_slope
        self.inplace = inplace
        self.activation = activation
        ## build block
        self.layer_dict = nn.ModuleDict()
        self.conv = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, 
                                    kernel_size=kernel_size, 
                                    stride=stride, padding=padding, use_bias=False, groups=groups, dilation_rate=
                                    dilation_rate) ## do not use bias when followed by batch norm
        self.bn = MetaBNLayer(num_features=out_channels, args=args, eps=eps, 
                              no_bn_learnable_params=no_bn_learnable_params, 
                              use_per_step_bn_statistics=args.use_per_step_bn_statistics)
    
    def forward(self, input, num_step, params=None, training=False, bkp_running_statistics=False):
        '''
        Forward pass. If params is not None, use params for inference, else use default parameters
        :param num_step: step index of inner loop
        :param training: used for bn inference
        :param bkp_running_stats: whether to backp running statistics        
        '''
        if params is not None:
            params = extract_top_level_dict(params) ## extract params from dict params
            conv_params, bn_params = params['conv'], params['bn']
        else:
            conv_params, bn_params = None, None
        x = self.conv.forward(input, params=conv_params)
        x = self.bn.forward(x, current_step=num_step, params=bn_params, training=training, 
                            bkp_running_statistics=bkp_running_statistics)
        if self.activation:
            x = F.leaky_relu(input=x, negative_slope=self.negative_slope, inplace=self.inplace)
        return x
    
    def restore_bkp_stats(self):
        '''
        restore bn statistics
        '''
        self.bn.restore_bkp_stats()
        

class MetaResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=True, groups=1, dilation_rate=1):
        '''
        ResBlock for meta learning. Contains 2 conv layers. Do 
        y = x + ReLU Conv ReLU Conv (x) if BN is allowed
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size for convolutional operation
        :param stride: stride of convolutional operation
        :param padding: padding of convolutional operation
        :param groups: groups of conv
        :param dilation_rate: dilation rate of conv
        '''
        super(MetaResBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv1'] = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
        self.layer_dict['conv2'] = MetaConv2dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
    
    def forward(self, x, params=None):
        ## print('MetaResBlock, param keys')
        param_dict = dict()
        if params is not None:
            print('keys of params in MetaResBlock: ', params.keys())
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        ini = x
        x = self.layer_dict['conv1'].forward(x, params=param_dict['conv1'])
        x = F.leaky_relu(x, inplace=True)
        x = self.layer_dict['conv2'].forward(x, params=param_dict['conv2'])
        x = F.leaky_relu(x, inplace=True)
        x = ini+x ## do residual connection
        return x

class MetaConvBNReLUDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, stages=3, no_bn_learnable_params=False, 
                 device=None, negative_slope=0.01, inplace=True, use_bias=True, groups=1, dilation_rate=1, eps=1e-5, **kwargs):
        """
        densely connected conv->bn->relu --- conv->bn->relu ---...
        """
        super(MetaConvBNReLUDenseBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.stages = stages
        self.kwargs = kwargs
        if 'Agg_input' in kwargs:
            input_c = kwargs['input_channels'] ## aggreagate input
        else:
            input_c = 0
        for i in range(stages):
            in_c = input_c + in_channels*(i+1) ## concat input
            self.layer_dict['cbn{}'.format(i+1)] = MetaConvBNLReLU(in_channels=in_c, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=device, negative_slope=negative_slope, inplace=inplace, 
                    use_bias=use_bias, groups=groups, dilation_rate=dilation_rate, eps=eps)
    
    def forward(self,  x, num_step, input_tensor=None, params=None, training=True, bkp_running_statistics=False):
        """
        forward pass
        :param x: input feature maps
        :param input_tensor: input images
        """
        param_dict = dict()
        if params is not None:
            param_dict=extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in param_dict:
                param_dict[layer_name]=None
        if 'Agg_input' in self.kwargs:
            shortcuts = [input_tensor, x]
        else:
            shortcuts = [x]
        for i in range(self.stages):
            tmp = self.layer_dict['cbn{}'.format(i+1)].forward(torch.cat(shortcuts, dim=1), num_step=num_step, params=param_dict['cbn{}'.format(i+1)], 
                          training=training, bkp_running_statistics=bkp_running_statistics)
            shortcuts.append(tmp)
        return tmp
    
    def restore_bkp_stats(self):
        for i in range(self.stages):
            self.layer_dict['cbn{}'.format(i+1)].restore_bkp_stats()



class MetaResAggBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=True, groups=1, dilation_rate=1, **kwargs):
        '''
        Meta residual aggregation block. Do
        x1 = ReLU Conv(x)
        x2 = ReLU Conv(x1)
        out = ReLU Conv([x, x1, x2])+x
        :param kwargs: dict contains 'Agg_input' and 'input_channels', when 'Agg_input', aggregate input to concatenation
        '''
        super(MetaResAggBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv1'] = MetaConv2dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
        self.layer_dict['conv2'] = MetaConv2dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
        self.kwargs = kwargs
        if 'Agg_input' in kwargs:
            input_channels = kwargs['input_channels']
        else:
            input_channels = 0
        self.layer_dict['conv3'] = MetaConv2dLayer(in_channels=3*out_channels+input_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                                   padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
    
    def forward(self, x, input=None, params=None):
        param_dict = dict()
        if params is not None:
            # print('keys of params in MetaResAggBlock: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in param_dict:
                param_dict[layer_name] = None
        shortcut_1 = x
        x = self.layer_dict['conv1'].forward(x, params=param_dict['conv1'])
        x = F.leaky_relu(x)
        shortcut_2 = x
        x = self.layer_dict['conv2'].forward(x, params=param_dict['conv2'])
        x = F.leaky_relu(x)
        shortcut_3 = x
        if 'Agg_input' in self.kwargs:
            x = torch.cat((input, shortcut_1, shortcut_2, shortcut_3),1)
        else:
            x = torch.cat((shortcut_1, shortcut_2, shortcut_3), 1)
        x = self.layer_dict['conv3'].forward(x, params=param_dict['conv3'])
        x = F.leaky_relu(x)
        x = shortcut_1 + x ## residual connection
        return x    
    
        
class ResBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, no_bn_learnable_params=False, device=None, negative_slope=0.01, 
                 use_bias=True, groups=1, dilation_rate=1, inplace=True, eps=1e-5):
        '''
        ResBNBlock for meta learning. Do
        x = x + ReLU BN Conv ReLU BN Conv(x)
        '''
        super(ResBNBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                              padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, deivce=device, 
                                              negative_slope=negative_slope, inplace=inplace, use_bias=True, groups=groups, 
                                              dilation_rate=dilation_rate, eps=eps)
        self.layer_dict['cbn2'] = MetaConvBNLReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                              padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=device, 
                                              negative_slope=negative_slope, inplace=inplace, use_bias=use_bias, groups=groups, 
                                              dilation_rate=dilation_rate, eps=eps)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys in params: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        ini = x
        x = self.layer_dict['cbn1'].forward(x, num_step=num_step, params=param_dict['cbn1'], training=training, bkp_running_statistics=bkp_running_statistics)
        x = self.layer_dict['cbn2'].forward(x, num_step=num_step, params=param_dict['cbn2'], training=training, bkp_running_statistics=bkp_running_statistics)
        return ini+x
    
    def restore_bkp_stats(self):
        self.layer_dict['cbn1'].restore_bkp_stats()
        self.layer_dict['cbn2'].restore_bkp_stats()
        

class MetaResAggBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, no_bn_learnable_params=False, device=None, negative_slope=0.01, 
                 use_bias=True, groups=1, dilation_rate=1, inplace=True, eps=1e-5, **kwargs):
        '''
        ResBNBlock for meta learning. Do
        x = x + ReLU BN Conv ReLU BN Conv(x)
        :param kwargs: when contains 'Agg_input' and 'input_channels', aggregate input images into aggregation part
        '''
        super(MetaResAggBNBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                              padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=device, 
                                              negative_slope=negative_slope, inplace=inplace, use_bias=True, groups=groups, 
                                              dilation_rate=dilation_rate, eps=eps)
        self.layer_dict['cbn2'] = MetaConvBNLReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                                              padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=device, 
                                              negative_slope=negative_slope, inplace=inplace, use_bias=use_bias, groups=groups, 
                                              dilation_rate=dilation_rate, eps=eps)
        if 'Agg_input' in kwargs:
            input_channels = kwargs['input_channels']
        else:
            input_channels = 0
        self.layer_dict['cbn3'] = MetaConvBNLReLU(in_channels=3*out_channels+input_channels, out_channels=out_channels, kernel_size=kernel_size, 
                                                  stride=stride, padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, 
                                                  device=device, negative_slope=negative_slope, inplace=inplace, use_bias=use_bias, groups=groups, 
                                                  dilation_rate=dilation_rate, eps=eps)
        self.kwargs = kwargs
    
    def forward(self, x, num_step, input=None, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys in params: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        short_cut_1 = x
        x = self.layer_dict['cbn1'].forward(x, num_step=num_step, params=param_dict['cbn1'], training=training, bkp_running_statistics=bkp_running_statistics)
        short_cut_2 = x
        x = self.layer_dict['cbn2'].forward(x, num_step=num_step, params=param_dict['cbn2'], training=training, bkp_running_statistics=bkp_running_statistics)
        short_cut_3 = x
        if 'Agg_input' in self.kwargs:
            x = torch.cat((input, short_cut_1, short_cut_2, short_cut_3), 1)
        else:
            x = torch.cat((x, short_cut_1, short_cut_2), 1)
        x = self.layer_dict['cbn3'].forward(x, num_step=num_step, params=param_dict['cbn3'], training=training, bkp_running_statistics=bkp_running_statistics)
        x = 0.5*short_cut_1 + 0.5*x ## do residual connection
        return x
    
    def restore_bkp_stats(self):
        self.layer_dict['cbn1'].restore_bkp_stats()
        self.layer_dict['cbn2'].restore_bkp_stats()
        self.layer_dict['cbn3'].restore_bkp_stats()

class MetaSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
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

class MetaDenseConvBNLReLUSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, args, stages=3, reduction=8, no_bn_learnable_params=False, 
                 device=None, negative_slope=0.01, inplace=True, use_bias=True, groups=1, dilation_rate=1, eps=1e-5, **kwargs):
        """
        densely connected conv->bn->relu->se --- conv->bn->relu->se ---...
        """
        super(MetaDenseConvBNLReLUSEBlock, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.stages = stages
        self.kwargs = kwargs
        if 'Agg_input' in kwargs:
            input_c = kwargs['input_channels'] ## aggreagate input
        else:
            input_c = 0
        for i in range(stages):
            in_c = input_c + in_channels*(i+1) ## concat input
            self.layer_dict['cbn{}'.format(i+1)] = MetaConvBNLReLU(in_channels=in_c, out_channels=out_channels, kernel_size=kernel_size, stride=stride, 
                    padding=padding, args=args, no_bn_learnable_params=no_bn_learnable_params, device=device, negative_slope=negative_slope, inplace=inplace, 
                    use_bias=use_bias, groups=groups, dilation_rate=dilation_rate, eps=eps)
            self.layer_dict['se{}'.format(i+1)] = MetaSEBlock(in_channels=out_channels, reduction=reduction)
    
    def forward(self, x, num_step, input_tensor=None, params=None, training=True, bkp_running_statistics=False):
        """
        forward pass
        :param x: input feature maps
        :param input_tensor: input images
        """
        param_dict = dict()
        if params is not None:
            param_dict=extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            layer_name = name.split('.')[0]
            if layer_name not in param_dict:
                param_dict[layer_name]=None ## if params are None, use inner params
        if 'Agg_input' in self.kwargs:
            shortcuts = [input_tensor, x]
        else:
            shortcuts = [x]
        for i in range(self.stages):
            tmp = self.layer_dict['cbn{}'.format(i+1)].forward(torch.cat(shortcuts, dim=1), num_step=num_step, params=param_dict['cbn{}'.format(i+1)], 
                          training=training, bkp_running_statistics=bkp_running_statistics)
            tmp = self.layer_dict['se{}'.format(i+1)].forward(tmp, params=param_dict['se{}'.format(i+1)])
            shortcuts.append(tmp)
        return tmp
    
    def restore_bkp_stats(self):
        for i in range(self.stages):
            self.layer_dict['cbn{}'.format(i+1)].restore_bkp_stats()

class MetaDuRB(nn.Module):
    def __init__(self, in_channels=32, num_filters=32, res_dim=32, k1_size=3, k2_size=5, use_se=True, reduction=4, args=None, negative_slope=0.1):
        """
        Implementation of DuRB-P with meta learning
        """
        super(MetaDuRB, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.negative_slope=negative_slope
        self.use_se = use_se
        self.num_filters = num_filters
        self.layer_dict['cbn1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=num_filters, kernel_size=3, stride=1, 
                                                  padding=1, use_bias=False, args=args, device=args.device, negative_slope=negative_slope)
        self.layer_dict['conv2'] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, 
                                                   padding=1, use_bias=False)
        self.layer_dict['bn2'] = MetaBNLayer(num_features=num_filters, args=args)
        self.layer_dict['up_conv'] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, kernel_size=k1_size, 
                                                     stride=1, padding=(k1_size-1)//2, use_bias=False)
        self.layer_dict['up_norm'] = MetaBNLayer(num_features=num_filters, args=args)
        self.layer_dict['down_conv'] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, kernel_size=k2_size, 
                                                       stride=1, padding=(k2_size)//2, use_bias=False)
        self.layer_dict['down_norm'] = MetaBNLayer(num_filters, args=args)
        if use_se:
            self.layer_dict['se1'] = MetaSEBlock(num_filters, reduction=reduction)
            self.layer_dict['se2'] = MetaSEBlock(num_filters, reduction=reduction) 
    
    
    def forward(self, x, res, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(params)
        for name, param in self.layer_dict.named_parameters():
            if name not in param_dict:
                param_dict[name] = None
        x_r = x
        x = self.layer_dict['cbn1'].forward(x, num_step=num_step, params=param_dict['cbn1'], training=training, bkp_running_statistics=bkp_running_statistics)
        x = self.layer_dict['conv2'].forward(x, params=param_dict['conv2'])
        x += x_r[:, -self.num_filters:, :, :]
        
        x = self.layer_dict['bn2'].forward(x, current_step=num_step, params=param_dict['bn2'], training=training, bkp_running_statistics=bkp_running_statistics)
        x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=True)
        # x = (x_r[:, -self.num_filters:, :, :] + x)

        ## T^{l}_1
        x = self.layer_dict['up_conv'].forward(x, params=param_dict['up_conv'])
        x = self.layer_dict['up_norm'].forward(x, current_step=num_step, params=param_dict['up_norm'], training=training, bkp_running_statistics=bkp_running_statistics)
        if self.use_se:
            x = self.layer_dict['se1'].forward(x, params=param_dict['se1'])
        x = x+res
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        # x = (x+res)
        res = x

        ## T^{l}_2
        x = self.layer_dict['down_conv'].forward(x, params=param_dict['down_conv'])
        x = self.layer_dict['down_norm'].forward(x, current_step=num_step, params=param_dict['down_norm'], training=training, bkp_running_statistics=bkp_running_statistics)
        if self.use_se:
            x = self.layer_dict['se2'].forward(x, params=param_dict['se2'])
        x = x_r[:, -self.num_filters:, :, :]+x

        x = F.leaky_relu(x, negative_slope=self.negative_slope, inplace=True)
        # x = (x+x_r[:, -self.num_filters:, :, :])
        return x, res
    
    def restore_bkp_stats(self):
        self.layer_dict['cbn1'].restore_bkp_stats()
        self.layer_dict['bn2'].restore_bkp_stats()
        self.layer_dict['up_norm'].restore_bkp_stats()
        self.layer_dict['down_norm'].restore_bkp_stats()






if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from utils.arguments import get_args
    args = get_args()
    kwargs={'Agg_input': True, 'input_channels': 3}
    mcbrdense = MetaConvBNReLUDenseBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1,padding=1, args=args, device=args.device, **kwargs)
    for name, param in mcbrdense.layer_dict.named_parameters():
        print(name, param.shape)
    img = torch.rand(1, 3, 50, 50)
    input_tensor = torch.rand(1, 32, 50, 50)
    out = mcbrdense.forward(x=input_tensor,num_step=0, input_tensor=img)
    print('output shape: ', out.shape)

    mcbnse = MetaDenseConvBNLReLUSEBlock(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1,args=args)
    x = torch.rand(1, 16, 50, 50)
    print(mcbnse)
    out = mcbnse.forward(x, num_step=0)
    print(out.shape)
