"""
Re-implement network for DDN
"""
from models.meta_layers import *
import torch.nn as nn
from utils.guided_filtering import GuidedFilter2d


class MetaDDN(nn.Module):
    def __init__(self, in_channels, num_filters, args=None):
        """
        Re-implementation for DDN
        :param in_channels: input channels
        :param num_filters: num_filters of conv
        :param blocks_num: number of blocks
        :param args: arguments
        """
        super(MetaDDN, self).__init__()
        # define architecture
        self.layer_dict = nn.ModuleDict()
        self.name = 'DDN-{}'.format(num_filters)
        self.layer_dict['guided_filter'] = GuidedFilter2d(radius=15, eps=1e-8)
        self.layer_dict['layer_1'] = MetaConvBNLReLU(in_channels=in_channels, out_channels=num_filters, kernel_size=3,
                                                     stride=1, padding=1, args=args, negative_slope=0.0, activation=True)
        for i in range(12):
            self.layer_dict['layer_{}'.format(2+2*i)] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters,
                                                                        kernel_size=3, stride=1, padding=1, args=args, negative_slope=0.0,
                                                                        activation=True)
            self.layer_dict['layer_{}'.format(3+2*i)] = MetaConvBNLReLU(in_channels=num_filters, out_channels=num_filters,
                                                                        kernel_size=3, stride=1, padding=1, args=args, negative_slope=0.0,
                                                                        activation=True)
        self.layer_dict['layer_26'] = MetaConvBNLReLU(in_channels=num_filters, out_channels=in_channels, kernel_size=3,
                                                      stride=1, padding=1, args=args, activation=False)

    def forward(self, inp, num_step=0, params=None, training=True, bkp_running_statistics=False):
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        # do forward pass
        base = self.layer_dict['guided_filter'](inp, inp)
        detail = inp - base  # detail layer
        shortcut = self.layer_dict['layer_1'].forward(detail, num_step, params=param_dict['layer_1'], training=training,
                                                      bkp_running_statistics=bkp_running_statistics)
        for i in range(12):
            out = self.layer_dict['layer_{}'.format(2+2*i)].forward(shortcut, num_step, params=param_dict['layer_{}'.format(2+2*i)],
                                                                    training=training, bkp_running_statistics=bkp_running_statistics)
            out = self.layer_dict['layer_{}'.format(3+2*i)].forward(out, num_step, params=param_dict['layer_{}'.format(3+2*i)],
                                                                    training=training, bkp_running_statistics=bkp_running_statistics)
            shortcut = torch.add(shortcut, out)
        neg_residual = self.layer_dict['layer_26'].forward(shortcut, num_step, params=param_dict['layer_26'], training=training,
                                                  bkp_running_statistics=bkp_running_statistics)
        final_out = inp + neg_residual
        return final_out

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
        indexes = [1, 2, 2]  # conv -> bn
        for i in range(12):
            tmp_indexes = [1, 2, 2, 3, 4, 4]  # conv -> bn -> relu -> conv -> bn -> relu
            body_indexes = [j+max(indexes) for j in tmp_indexes]
            indexes.extend(body_indexes)
        tail_indexes = [1, 2, 2]  # conv -> bn -> bn
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
    net = MetaDDN(in_channels=3, num_filters=16, args=args)
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
    print(net.get_layer_index())

    # torch.onnx.export(msscnet, inp, 'msresnetAggWoMSWoSE.onnx', training=torch.onnx.TrainingMode.TRAINING)
    torch.onnx.export(net, x, 'ddn.onnx', training=torch.onnx.TrainingMode.TRAINING)


