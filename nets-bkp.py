from meta_layers import *
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class MultiScaleResNet(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels=3, k1=2, k2=4, args=None, Agg=True):
        super(MultiScaleResNet, self).__init__()
        self.k1 = k1
        self.k2 = k2

        self.layer_dict = nn.ModuleDict()

        ## define layers
        self.Agg = Agg
        if Agg:
            agg_channel = in_channels
            self.name = 'MAEBResNetAggPA-{}-{}-{}C-SSIM{}-GDLoss{}'.format(k1, k2, num_filters, args.ssim_weight, args.channel_weight)
        else:
            agg_channel = 0
            self.name = 'MAEBResNetPA-{}-{}-{}C-SSIM{}-GDLoss{}'.format(k1, k2, num_filters, args.ssim_weight, args.channel_weight)

        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels, num_filters, kernel_size=3, stride=1, padding=1)
        for i in range(k1):
            self.layer_dict['mshier{}'.format(i+1)] = MetaHierachicalAgg(num_filters+agg_channel, num_filters, kernel_size=3, stride=1, padding=1, 
                                                                       args=args, device=args.device, use_multi_scale=True, multi_scale='maeb')
        for i in range(k2):
            self.layer_dict['hier{}'.format(i+1)] = MetaHierachicalAgg(num_filters+agg_channel, num_filters, kernel_size=3, stride=1, padding=1, 
                                                                       args=args, device=args.device)
        self.layer_dict['cbn_tail'] = MetaConvBNLReLU(num_filters, num_filters, kernel_size=3, stride=1, padding=1, 
                                                      args=args, negative_slope=0.2, activation=False)
        self.layer_dict['se_tail'] = MetaSEBlock(num_filters, num_filters//4)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, out_channels, kernel_size=3, stride=1, padding=1, use_bias=False)
        self.get_layer_index()
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        #### extract params
        #### num_step = 0
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        # residual = x
        conv_in = x
        conv_in = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        conv_in = F.leaky_relu(conv_in, negative_slope=0.2, inplace=True)

        out = conv_in
        
        for i in range(self.k1):
            if self.Agg:
                out = torch.cat([out, x], dim=1)
            out = self.layer_dict['mshier{}'.format(i+1)].forward(out, num_step, params=param_dict['mshier{}'.format(i+1)], 
                                                                   training=training, bkp_running_statistics=bkp_running_statistics)

        for i in range(self.k2):
            if self.Agg:
                out = torch.cat([out, x], dim=1)
            out = self.layer_dict['hier{}'.format(i+1)].forward(out, num_step, params=param_dict['hier{}'.format(i+1)], 
                                                                training=training, bkp_running_statistics=bkp_running_statistics)
        
        out = conv_in + out
        out = self.layer_dict['cbn_tail'].forward(out, num_step, params=param_dict['cbn_tail'], training=training, 
                                                  bkp_running_statistics=bkp_running_statistics)
        out = self.layer_dict['se_tail'].forward(out, params=param_dict['se_tail'])
        out = F.leaky_relu(out, 0.2, True)

        out = self.layer_dict['conv_tail'].forward(out, params=param_dict['conv_tail'])

        # out = F.leaky_relu(out, negative_slope=0.2)
        return x - out ## background
    
    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None
    
    def get_layer_index(self):
        """
        generate layer index of a weight
        the larger the index, the smaller the inner loop learning rate
        """
        index = [1, 1] ## conv_head weight and bias
        mshier_index_ori = [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10, 10, 11, 11]
        tail_index_ori = [1, 2, 2, 3, 3, 4] ## conv->bn->se->conv
        # assert self.k2 == 0, 'not implemented'
        for i in range(self.k1):
            mshier_index = [10*i+item for item in mshier_index_ori]
            index.extend(mshier_index)
        tail_index = [max(index)+item for item in tail_index_ori]
        index.extend(tail_index)
        print('layer indexes of weights: {}, lenght: {}'.format(index, len(index)))
        return index



class DiRes(nn.Module):
    def __init__(self, in_channels, num_filters, K=4, kernel_size=3, stride=1, padding=1, args=None):
        super(DiRes, self).__init__()
        self.name = 'DilatedRes-{}K-{}C'.format(K, num_filters)
        self.stages = k

        self.layer_dict = nn.ModuleDict()
        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, 
                                                       stride=stride, padding=padding, use_bias=True)
        for i in range(1, 1+K):
            self.layer_dict['dilate{}'.format(i)] = MetaDilatedAggConv(in_channels=num_filters, num_filters=num_filters, 
                                                                       dilated_factors=3, reduction=num_filters//8)
            self.layer_dict['resagg{}'.format(i)] = MetaHierachicalAgg(in_channels=num_filters, num_filters=num_filters, kernel_size=kernel_size, 
                                                                       stride=stride, padding=padding, args=args, device=args.device)
        
        cat_c = 2+K//2
        self.layer_dict['se'] = MetaSEBlock(in_channels=cat_c, reduction=cat_c//8)
        self.layer_dict['conv1'] = MetaConv2dLayer(in_channels=cat_c, out_channels=num_filters, kernel_size=kernel_size, 
                                                   stride=stride, padding=padding, use_bias=True)
        self.layer_dict['conv2'] = MetaConv2dLayer(in_channels=cat_c, out_channels=num_filters, kernel_size=kernel_size, 
                                                   stride=stride, padding=padding, use_bias=False)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        feature = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        shortcuts = [feature]
        out = feature

        for i in range(K):
            out = self.layer_dict['dilate{}'.format(i+1)].forward(out, params=param_dict['dilate{}'.format(i+1)])
            out = self.layer_dict['resagg{}'.format(i+1)].forward(out, num_step, params=param_dict['resagg{}'.format(i+1)], 
                                                                  training=training, bkp_running_statistics=bkp_running_statistics)
            if i%2==0 or i==K-1:
                shortcuts.append(out)
        out = torch.cat(shortcuts, dim=1)
        out = self.layer_dict['se'].forward(out, params=param_dict['se'])
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.layer_dict['conv1'].forward(out, params=param_dict['conv1'])
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.layer_dict['conv2'].forward(out, params=param_dict['conv2'])
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)

        return x+out ## x - out better
        

class ResAggKstages(nn.Module):
    def __init__(self, in_channels, num_filters, K=5, kernel_size=3, stride=1, padding=1, use_bias=True, groups=1, dilation_rate=1, **kwargs):
        '''
        contains K ResAgg blocks
        :param in_channels: channels of input images
        :param num_filters: number of filters in mid convs
        :param K: number of stages of ResAgg blocks
        :param kernel_size:
        :param stride:
        :param padding:
        :param use_bias, groups, dilation_rate:
        :param kwargs: indicates whether to aggregate input images
        '''
        self.name = 'ResAgg{}stagesDRN-{}c-{}f'.format(K, in_channels, num_filters)
        # self.kwargs = {'Agg_input': True, 'input_channels': in_channels}
        self.kwargs = kwargs
        super(ResAggKstages, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.stages = K
        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding, 
                                         use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
        for i in range(K):
            self.layer_dict['resagg%d'%(i+1)]=MetaResAggBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, padding=padding, 
                                                                use_bias=use_bias, groups=groups, dilation=dilation_rate, **self.kwargs)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(in_channels=num_filters, out_channels=in_channels, kernel_size=kernel_size, stride=stride, 
                                                       padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
    
    def forward(self, input, params=None):
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        x = input
        x = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        x = F.leaky_relu(x)
        for i in range(self.stages):
            if 'Agg_input' in self.kwargs:
                x = self.layer_dict['resagg%d'%(i+1)].forward(x, input=input, params=param_dict['resagg%d'%(i+1)])
            else:
                x = self.layer_dict['resagg%d'%(i+1)].forward(x, input=None, params=param_dict['resagg%d'%(i+1)])
        x = self.layer_dict['conv_tail'].forward(x, params=param_dict['conv_tail'])
        return input+x
    
    def zero_grad(self, set_to_none=False, params=None):
        params = self.parameters() if params is None else params
        for param in self.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

class SCNet(nn.Module):
    def __init__(self, in_channels, num_filters, stages=4, args=None, device=None, negative_slope=0.01, use_bn=True):
        super(SCNet, self).__init__()
        self.name = 'SCNetp0.2@{}-{}-{}'.format(in_channels, num_filters, stages)
        self.layer_dict = nn.ModuleDict()
        self.stages=stages
        self.negative_slope=negative_slope
        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=3, 
                                                       stride=1, padding=1, use_bias=True)
        for i in range(stages):
            in_c = num_filters*(i+1)
            self.layer_dict['scblock{}'.format(i+1)] = MetaSCResBlock(in_c, num_filters, args=args, use_bn=use_bn, negative_slope=negative_slope)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, in_channels, 3, 1, 1, use_bias=True)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        # num_step=0
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        residual = x
        out = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        out = F.leaky_relu(out, negative_slope=self.negative_slope, inplace=True)
        concats = [out]
        for i in range(self.stages):
            out = self.layer_dict['scblock{}'.format(i+1)].forward(torch.cat(concats, dim=1), num_step, params=param_dict['scblock{}'.format(i+1)], 
                                                                   training=training, bkp_running_statistics=bkp_running_statistics)
            concats.append(out)
        out = self.layer_dict['conv_tail'].forward(out, params=param_dict['conv_tail'])
        out = F.leaky_relu(out, negative_slope=self.negative_slope, inplace=True)
        out = residual + out
        return out

class MultiScaleSCNet(nn.Module):
    def __init__(self, in_channels, num_filters, stages=2, factors=3, args=None, device=None, negative_slope=0.2, use_bn=True):
        super(MultiScaleSCNet, self).__init__()
        self.name = 'MSSCNet@{}-{}-{}-{}@'.format(in_channels, num_filters, stages, factors)
        self.layer_dict = nn.ModuleDict()
        self.stages=stages
        self.negative_slope=negative_slope
        self.factors = factors

        for i in range(1, 1+factors):
            self.layer_dict['dconv{}-1'.format(i)] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=3, 
                                                                   stride=1, padding=i, use_bias=True, dilation_rate=i)
        for i in range(1, 1+factors):
            self.layer_dict['dconv{}-2'.format(i)] = MetaConv2dLayer(in_channels=num_filters, out_channels=num_filters, kernel_size=3, 
                                                                   stride=1, padding=i, use_bias=True, dilation_rate=i)
        
        self.layer_dict['se_head'] = MetaSEBlock(num_filters, reduction=num_filters//4)
        for i in range(stages):
            in_c = num_filters*(i+1)+in_channels
            self.layer_dict['scblock{}'.format(i+1)] = MetaSCResBlock(in_c, num_filters, args=args, use_bn=use_bn, negative_slope=negative_slope)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, in_channels, 3, 1, 1, use_bias=False)
    
    def forward(self, x, num_step, params=None, training=True, bkp_running_statistics=False):
        # num_step=0
        param_dict = dict()
        if params is not None:
            # print('keys() in params in ResAggKstages: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        ## forward pass
        residual = x
        multi_scales = []
        for i in range(1, 1+self.factors):
            tmp = self.layer_dict['dconv{}-1'.format(i)].forward(x, params=param_dict['dconv{}-1'.format(i)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope, inplace=True)
            tmp = self.layer_dict['dconv{}-2'.format(i)].forward(tmp, params=param_dict['dconv{}-2'.format(i)])
            tmp = F.leaky_relu(tmp, negative_slope=self.negative_slope, inplace=True)
            multi_scales.append(tmp)
        
        out = reduce(torch.add, multi_scales)
        out = self.layer_dict['se_head'].forward(out, params=param_dict['se_head'])

        concats = [residual, out]
        for i in range(self.stages):
            out = self.layer_dict['scblock{}'.format(i+1)].forward(torch.cat(concats, dim=1), num_step, params=param_dict['scblock{}'.format(i+1)], 
                                                                   training=training, bkp_running_statistics=bkp_running_statistics)
            concats.append(out)
        out = self.layer_dict['conv_tail'].forward(out, params=param_dict['conv_tail'])
        out = residual + out
        return out


class ResBNAggKstages(nn.Module):
    def __init__(self, in_channels, num_filters, K=5, kernel_size=3, stride=1, padding=1, use_bias=True, groups=1, dilation_rate=1, 
                 args=None, no_bn_learnable_params=False, device=None, negative_slope=0.01, inplace=True, eps=1e-5, **kwargs):
        '''
        Contains K ResBNAgg blocks
        :params :
        '''
        super(ResBNAggKstages, self).__init__()
        self.name = 'ResBNAgg{}stagesDRN-{}c-{}f'.format(K, in_channels, num_filters)
        self.kwargs=kwargs
        self.stages = K
        self.layer_dict = nn.ModuleDict()
        assert args is not None, 'blank arguments is not allowed.'
        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                       padding=padding, use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
        for i in range(K):
            self.layer_dict['resbnagg%d'%(i+1)] = MetaResAggBNBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                                    padding=padding,args=args,no_bn_learnable_params=no_bn_learnable_params,device=args.device,
                                                                    negative_slope=negative_slope,use_bias=use_bias,groups=groups,dilation_rate=dilation_rate,
                                                                    inplace=inplace,eps=eps,**self.kwargs)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(in_channels=num_filters, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, 
                                                       use_bias=use_bias, groups=groups, dilation_rate=dilation_rate)
    
    def forward(self, input,  num_step=0, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            # print('keys in params: ', params.keys())
            param_dict = extract_top_level_dict(params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name]=None
        x = input
        x = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        x = F.leaky_relu(x)
        for i in range(self.stages):
            if 'Agg_input' in self.kwargs:
                x = self.layer_dict['resbnagg%d'%(i+1)].forward(x, num_step=num_step, input=input, params=param_dict['resbnagg%d'%(i+1)], training=training, 
                                                                bkp_running_statistics=bkp_running_statistics)
            else:
                x = self.layer_dict['resbnagg%d'%(i+1)].forward(x, num_step=num_step, input=None, params=param_dict['resbnagg%d'%(i+1)], training=training, 
                                                                bkp_running_statistics=bkp_running_statistics)
        x = self.layer_dict['conv_tail'].forward(x, params=param_dict['conv_tail'])
        return input+x
    
    def zero_grad(self, set_to_none=False, params=None):
        params = self.parameters() if params is None else params
        for param in params:
            if param.requires_grad:
                if param.grad is not None:
                    param.grad.zero_()

    def restore_bkp_stats(self):
        for i in range(self.stages):
            self.layer_dict['resbnagg%d'%(i+1)].restore_bkp_stats()

class DenseCBNRSENet(nn.Module):
    def __init__(self, in_channels, num_filters, s1=2, s2=3, kernel_size=3, stride=1, padding=1, use_bias=True, groups=1, dilation_rate=1, 
                 args=None, no_bn_learnable_params=False, device=None, negative_slope=0.01, inplace=True, eps=1e-5, **kwargs):
        """
        Densely connected convnet
        s1 for DenseCBNR
        s2 for DenseCBNRSE
        """
        super(DenseCBNRSENet, self).__init__()
        self.name = 'DenseCBNRSENetAttenuate-{}-{}'.format(s1, s2)
        self.kwargs = kwargs
        self.stage1 = s1
        self.stage2 = s2
        self.layer_dict = nn.ModuleDict()
        assert args is not None, 'Blank arguments is not allowed'
        self.layer_dict['conv_head'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size, stride=stride, 
                                                       padding=padding, use_bias=True)
        for i in range(self.stage1):
            self.layer_dict['dcbn{}'.format(i+1)] = MetaConvBNReLUDenseBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1,
                                                                             padding=1, args=args, device=args.device, **kwargs)
                                                    
        for i in range(self.stage2):
            self.layer_dict['dcbnse{}'.format(i+1)] = MetaDenseConvBNLReLUSEBlock(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, 
                                                                                  padding=1, args=args, stages=3, device=args.device, reduction=4, **kwargs)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(in_channels=num_filters, out_channels=in_channels, kernel_size=kernel_size, stride=stride, 
                                                       padding=padding, use_bias=False)

    def forward(self, x, num_step=0, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, _ in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name]=None
        ini = x
        x = self.layer_dict['conv_head'].forward(x, params=param_dict['conv_head'])
        x = F.leaky_relu(x, inplace=True)
        for i in range(self.stage1):
            tmp = self.layer_dict['dcbn{}'.format(i+1)].forward(x, num_step=num_step, input_tensor=ini, params=param_dict['dcbn{}'.format(i+1)], 
                                                              training=training, bkp_running_statistics=bkp_running_statistics)
            x = (x+tmp)*0.5 ## residual connection
        for i in range(self.stage2):
            tmp = self.layer_dict['dcbnse{}'.format(i+1)].forward(x, num_step=num_step,input_tensor=ini, params=param_dict['dcbnse{}'.format(i+1)], 
                                                                training=training, bkp_running_statistics=bkp_running_statistics)
            x = 0.5*(x+tmp)
        x = self.layer_dict['conv_tail'].forward(x, params=param_dict['conv_tail'])
        x = (ini + x)*0.5
        return x

    def restore_bkp_stats(self):
        for i in range(self.stages1):
            self.layer_dict['dcbn{}'.format(i+1)].restore_bkp_stats()
        for i in range(self.stages2):
            self.layer_dict['dcbnse{}'.format(i+1)].restore_bkp_stats()
        
    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None


def get_inner_loop_params_dict(params, device):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if "bn" not in name:
                    param_dict[name] = param.to(device)
        return param_dict

if __name__ == '__main__':
    '''
    check networks and pass params 
    '''
    import sys
    sys.path.append('..')
    from utils.arguments import get_args
    args = get_args()
    kwargs = {'Agg_input': True, 'input_channels': 3}
    
    res5 = ResAggKstages(in_channels=3, num_filters=16, **kwargs)
    x = torch.rand(1, 3, 56, 56)
    # print(res5)
    for name, param in res5.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    out_res = res5(x)
    print('shape of output: ', out_res.size())
    # torch.onnx.export(res5, x, 'net_examples/res5_agginput.onnx', training=torch.onnx.TrainingMode)
    
    ## check resbnagg
    args = get_args()
    resbn4 = ResBNAggKstages(in_channels=3, num_filters=16, K=4, args=args, device=args.device, **kwargs)
    for name, param in resbn4.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
    out_resbn = resbn4(x, num_step=0)
    print('output shape: ', out_resbn.size())
    # torch.onnx.export(resbn4, x, 'net_examples/resbnagg4_agginput.onnx', training=torch.onnx.TrainingMode)
    
    ## check params passing
    res5_bkp = ResAggKstages(in_channels=3, num_filters=16, **kwargs)
    out_bkp = res5_bkp(x)
    param_dict = dict()
    for name, param in res5_bkp.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    print('difference between output of res5 and res5_bkp', torch.sum(out_res-out_bkp))
    out_pass = res5(x, params=param_dict)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))
    
    resbn4_bkp = ResBNAggKstages(in_channels=3, num_filters=16, K=4, args=args, device=args.device, **kwargs)
    out_bkp = resbn4_bkp.forward(x)
    param_dict = dict()
    for name, param in resbn4_bkp.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    print('difference between output of resbn4 and resbn4_bkp: ', torch.sum(out_resbn-out_bkp))
    out_pass = resbn4.forward(x, num_step=0, params=param_dict)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))
    

    ### param pass processing testing finished, test passed
    denseresnet = DenseCBNRSENet(in_channels=3, num_filters=16, args=args, device=args.device, **kwargs)
    out_bkp = denseresnet.forward(x, num_step=0)
    param_dict = dict()
    for name, param in denseresnet.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    out_pass = denseresnet.forward(x, num_step=0)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))

    # msscnet = MultiScaleSCNet(3, 32, stages=2, factors=3, args=args, device=args.device)
    args.device=torch.device('cpu')
    msscnet = MultiScaleResNet(in_channels=3, num_filters=32, out_channels=3, k1=2, k2=3, args=args)
    out_bkp = msscnet.forward(x)
    param_dict = dict()
    for name, param in msscnet.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    out_pass = msscnet.forward(x)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))
    out_param = msscnet.forward(x, params=get_inner_loop_params_dict(msscnet.named_parameters(), device=args.device))
    print('difference between out_pass and out_param', torch.sum(out_param-out_pass))
    inp = torch.rand(1, 3, 50, 50)
    out = msscnet(inp)
    torch.onnx.export(msscnet, inp, 'maebresnet.onnx', training=torch.onnx.TrainingMode.TRAINING)
