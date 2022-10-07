import torch
import torch.nn as nn
import torch.nn.functional as F
from models.meta_layers import MetaConv2dLayer, extract_top_level_dict
from functools import reduce


class MetaSDB(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size, use_bias=True):
        """
        Re-implementation of SDB in ADN
        shared parameters dilation conv module
        """
        super(MetaSDB, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = int(kernel_size)
        self.use_bias = use_bias
        self.weight_1 = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        self.weight_2 = nn.Parameter(torch.empty(num_filters, num_filters, kernel_size, kernel_size))

        nn.init.kaiming_normal_(self.weight_1)
        nn.init.kaiming_normal_(self.weight_2)
        if self.use_bias:
            self.bias_1 = nn.Parameter(torch.zeros(num_filters))
            self.bias_2 = nn.Parameter(torch.zeros(num_filters))

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
                (weight_1, bias_1) = params['weight_1'], params['bias_1']
                (weight_2, bias_2) = params['weight_2'], params['bias_2']
            else:
                (weight_1, bias_1) = params['weight_1'], None
                (weight_2, bias_2) = params['weight_2'], None
        else:
            if self.use_bias:
                (weight_1, bias_1, weight_2, bias_2) = self.weight_1, self.bias_1, self.weight_2, self.bias_2
            else:
                (weight_1, bias_1, weight_2, bias_2) = self.weight_1, None, self.weight_2, None
        outs = []
        for i in range(3):
            out = F.conv2d(input=x, weight=weight_1, bias=bias_1, stride=1, padding=i+1, dilation=i+1)
            out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
            out = F.conv2d(input=out, weight=weight_2, bias=bias_2, stride=1, padding=i+1, dilation=i+1)
            out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
            outs.append(out)
        return outs


class MetaDSB(nn.Module):
    def __init__(self, num_filters=16):
        """
        Re-implementation of DSB in ADN
        """
        super(MetaDSB, self).__init__()
        self.num_filters = num_filters
        self.weight = nn.Parameter(torch.ones(num_filters*3, num_filters*3))
        self.bias = nn.Parameter(torch.zeros(num_filters*3))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        nn.init.kaiming_normal_(self.weight)

    def forward(self, xs, params=None):
        b, c, _, _ = xs[0].size()
        if params is not None:
            weight, bias = params['weight'], params['bias']
        else:
            weight, bias = self.weight, self.bias
        inps = torch.cat(xs, dim=1)
        z = self.avg_pool(inps).view(b, 3*c)
        z = F.linear(z, weight=weight, bias=bias).view(b, 3, c)
        z = F.softmax(z, dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # (b, 3, c, 1, 1)
        feats = []
        for i in range(3):
            feats.append(xs[i]*z[:, i, :, :, :])
        feat = reduce(torch.add, feats)
        return feat


class MetaFSB(nn.Module):
    def __init__(self, in_channels, reduction=8):
        """
        Implementation of squeeze and excitation block for meta-learning
        """
        super(MetaFSB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.w1 = nn.Parameter(torch.ones(in_channels // reduction, in_channels))
        self.w2 = nn.Parameter(torch.ones(in_channels, in_channels // reduction))
        self.b1 = nn.Parameter(torch.zeros(in_channels//reduction))
        self.b2 = nn.Parameter(torch.zeros(in_channels))
        self.reduction = reduction
        nn.init.kaiming_normal_(self.w1)
        nn.init.kaiming_normal_(self.w2)

    def forward(self, x, params=None):
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
            w1, w2 = param_dict['w1'], param_dict['w2']
            b1, b2 = param_dict["b1"], param_dict['b2']
        else:
            w1, w2 = self.w1, self.w2
            b1, b2 = self.b1, self.b2
        b, c, _, _ = x.size()
        z = self.avg_pool(x).view(b, c)
        z = F.linear(z, w1, b1)
        z = F.relu(z, inplace=True)
        z = F.linear(z, w2, b2)
        z = torch.sigmoid(z).view(b, c, 1, 1)
        out = z * x
        return out

    def extra_repr(self):
        return 'MetaSEBlock, reduction: {}'.format(self.reduction)

class MetaADN(nn.Module):
    def __init__(self, in_channels, num_filters, stages=5):
        """
        Re-implementation for ADN
        :param in_channels: input channels
        :param num_filters: num_filters of conv
        :param stages: number of SDB+DSB+FSB
        :param args: arguments
        """
        super(MetaADN, self).__init__()
        # define architecture
        self.stages = stages
        self.layer_dict = nn.ModuleDict()
        self.name = 'ADN-{}'.format(num_filters)
        self.layer_dict['conv0'] = MetaConv2dLayer(in_channels=in_channels, out_channels=num_filters, kernel_size=3, stride=1,
                                                   padding=1, use_bias=True)
        for i in range(stages):
            # shared dilated block
            self.layer_dict['SDB{}'.format(i+1)] = MetaSDB(in_channels=num_filters, num_filters=num_filters,
                                                           kernel_size=3, use_bias=True)
            # dilated selection block
            self.layer_dict['DSB{}'.format(i+1)] = MetaDSB(num_filters=num_filters)
            # feature selection block
            self.layer_dict['FSB{}'.format(i+1)] = MetaFSB(in_channels=num_filters, reduction=num_filters//4)
        self.layer_dict['conv_tran'] = MetaConv2dLayer(in_channels=stages*num_filters, out_channels=num_filters, kernel_size=1,
                                                       stride=1, padding=0, use_bias=True)
        self.layer_dict['conv1'] = MetaConv2dLayer(in_channels=num_filters, out_channels=in_channels, kernel_size=3, stride=1,
                                                   padding=1, use_bias=True)

    def forward(self, inp, num_step=0, params=None):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        # do forward pass
        AggFeats = []
        out = self.layer_dict['conv0'].forward(inp, params=param_dict['conv0'])
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        for i in range(self.stages):
            out = self.layer_dict['SDB{}'.format(i+1)].forward(out, params=param_dict['SDB{}'.format(i+1)])
            out = self.layer_dict['DSB{}'.format(i+1)].forward(out, params=param_dict['DSB{}'.format(i+1)])
            out = self.layer_dict['FSB{}'.format(i+1)].forward(out, params=param_dict['FSB{}'.format(i+1)])
            AggFeats.append(out)
        feat = torch.cat(AggFeats, dim=1)
        out = self.layer_dict['conv_tran'].forward(feat, params=param_dict['conv_tran'])
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)
        out = self.layer_dict['conv1'].forward(out, params=param_dict['conv1'])
        out = inp + out
        return out

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

    def get_layer_index(self) -> list:
        """
        Generate indexes of weights for Task-Driven Learning
        Weights in same layer share same weights
        Different MSB and RB modules produce different weight indexes
        :return: list
        """
        indexes = [1, 1]  # conv
        for i in range(self.stages):
            tmp_indexes = [1, 1, 2, 2, 3, 3, 4, 5]  # conv -> conv -> FC -> Squeeze -> Excitation
            body_indexes = [j+max(indexes) for j in tmp_indexes]
            indexes.extend(body_indexes)
        tail_indexes = [1, 1, 2, 2]  # conv -> conv
        tail_indexes = [max(indexes)+j for j in tail_indexes]
        indexes.extend(tail_indexes)
        return indexes

if __name__ == '__main__':
    '''
    check networks and pass params 
    '''
    import sys

    sys.path.append('..')
    from utils.arguments import get_args
    args = get_args()

    def get_inner_loop_params_dict(params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=args.device)
                else:
                    if "bn" not in name:
                        param_dict[name] = param.to(args.device)
        return param_dict


    args = get_args()
    x = torch.rand(1, 3, 50, 50)
    args.device = torch.device('cpu')
    net = MetaADN(in_channels=3, num_filters=16, stages=5)
    print(net)
    out_bkp = net(x)
    param_dict = dict()
    for name, param in net.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    out_pass = net.forward(x)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp - out_pass))
    out_param = net.forward(x, params=get_inner_loop_params_dict(net.named_parameters()))
    print('difference between out_pass and out_param', torch.sum(out_param - out_pass))
    layer_indexes = net.get_layer_index()
    print(layer_indexes, len(layer_indexes))
    print(len(param_dict.keys()))
    param_num = 0
    for k, v in param_dict.items():
        param_num += torch.numel(v)
        print(k, torch.numel(v))
    print('Total param: ', param_num)

    # torch.onnx.export(msscnet, inp, 'msresnetAggWoMSWoSE.onnx', training=torch.onnx.TrainingMode.TRAINING)
    torch.onnx.export(net, x, 'adn.onnx', training=torch.onnx.TrainingMode.TRAINING)




