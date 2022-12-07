from models.meta_layers import *
import torch.nn as nn
import torch.nn.functional as F


class MetaMSResNet(nn.Module):
    def __init__(self, in_channels, num_filters, stages=4, args=None, Agg=True, withSE=True, dilated_factors=3, msb='MAEB',
                 rb='Dual', relu_type='lrelu'):
        """
        MSResNet for Task-Driven Learning
        :param in_channels:
        :param num_filters:
        :param stages:
        :param args:
        :param Agg:
        :param withSE:
        :param dilated_factors:
        """
        super(MetaMSResNet, self).__init__()
        self.stages = stages
        self.withSE = withSE
        self.Agg = Agg
        self.dilated_factors = dilated_factors
        self.msb_type = msb
        self.rb_type = rb
        self.relu_type = relu_type

        ### define architecture
        self.layer_dict = nn.ModuleDict()
        if self.Agg:
            Agg_channels = in_channels
            self.name = 'Agg{}-{}-{}s-SE{}-{}-0826'.format(msb, rb, stages, withSE, relu_type)
        else:
            Agg_channels = 0
            # self.name = '{}-{}-{}s-0826'.format(msb, rb, stages)
            self.name = 'MAEB-RES-Rain800-60-{}s-{}ssim'.format(stages, args.ssim_weight)
        self.layer_dict['head'] = MetaConv2dLayer(in_channels, num_filters, 3, 1, 1, use_bias=False)
        for i in range(stages):
            self.layer_dict['body{}'.format(i+1)] = MetaMSRB(num_filters+Agg_channels, num_filters, args,
                                                             relu_type=relu_type, msb=msb, rb=rb, withSE=withSE, dilated_factors=dilated_factors)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn_tail'] = MetaBNLayer(num_filters, args, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se_tail'] = MetaSEBlock(num_filters, num_filters//4)
        
        self.layer_dict['tail'] = MetaConv2dLayer(num_filters, in_channels, 3, 1, 1, use_bias=False)

    def forward(self, inp, num_step=0, params=None, training=True, bkp_running_statistics=False):
        ## please set training to True for better training
        ### extract parameters
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        ### do forward pass
        conv_in = self.layer_dict['head'].forward(inp, params=param_dict['head'])
        conv_in = F.leaky_relu(conv_in, negative_slope=0.2)

        out = conv_in
        for i in range(self.stages):
            if self.Agg:
                out = torch.cat([out, inp], dim=1)
            out = self.layer_dict['body{}'.format(i+1)].forward(out, num_step, params=param_dict['body{}'.format(i+1)],
                                                                training=training, bkp_running_statistics=bkp_running_statistics)
        residual = out
        out = self.layer_dict['conv_tail'].forward(residual, params=param_dict['conv_tail'])
        out = self.layer_dict['bn_tail'].forward(out, num_step, params=param_dict['bn_tail'], training=training,
                                                 bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se_tail'].forward(out, params=param_dict['se_tail'])
        
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)

        out = self.layer_dict['tail'].forward(out, params=param_dict['tail'])
        out = inp - out
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
        indexes = [1]
        if self.dilated_factors == 3:
            if self.msb_type == 'MAEB':
                msb_indexes = [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5]
            elif self.msb_type == 'MSB':
                msb_indexes = [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.rb_type == 'RB':
            if self.withSE:
                if self.relu_type == 'lrelu':
                    rb_indexes = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9]
                else:
                    rb_indexes = [1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11]
            else:
                if self.relu_type == 'lrelu':
                    rb_indexes = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7]
                elif self.relu_type == 'prelu':
                    rb_indexes = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 9]
        elif self.rb_type == 'Dual':
            rb_indexes = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 15]
        else:
            raise NotImplementedError
        tail_indexes = [1, 2, 2, 3, 3, 4]
        ### build indexes
        rb_indexes = [max(msb_indexes)+item for item in rb_indexes]
        body_indexes = msb_indexes
        body_indexes.extend(rb_indexes) ### indexes for body module
        index_gap = max(body_indexes) - min(body_indexes)+1
        for i in range(self.stages):
            tmp_indexes = [index_gap*i+item for item in body_indexes]
            indexes.extend(tmp_indexes)
        tail_indexes = [max(indexes)+item for item in tail_indexes]
        indexes.extend(tail_indexes)
        return indexes

class MetaMSResNetN(nn.Module):
    def __init__(self, in_channels, num_filters, stages=4, args=None, Agg=False, withSE=True, dilated_factors=3, msb='MAEB',
                 rb='Dual', relu_type='lrelu'):
        """
        MSResNet for Task-Driven Learning
        :param in_channels:
        :param num_filters:
        :param stages:
        :param args:
        :param Agg:
        :param withSE:
        :param dilated_factors:
        """
        super(MetaMSResNetN, self).__init__()
        self.stages = stages
        self.withSE = withSE
        self.Agg = Agg
        self.dilated_factors = dilated_factors
        self.msb_type = msb
        self.rb_type = rb
        self.relu_type = relu_type

        ### define architecture
        self.layer_dict = nn.ModuleDict()
        if self.Agg:
            Agg_channels = in_channels
            self.name = 'Agg{}-{}-{}s-SE{}-{}-0826'.format(msb, rb, stages, withSE, relu_type)
        else:
            Agg_channels = 0
            # self.name = '{}-{}-{}s-0826'.format(msb, rb, stages)
            self.name = 'MAEB-RES-WaterlooBSD-500-{}stages-ssim{}GRAY'.format(stages, args.ssim_weight)
        self.layer_dict['head'] = MetaConv2dLayer(in_channels+1, num_filters, 3, 1, 1, use_bias=False)
        for i in range(stages):
            self.layer_dict['body{}'.format(i+1)] = MetaMSRB(num_filters+Agg_channels, num_filters, args,
                                                             relu_type=relu_type, msb=msb, rb=rb, withSE=withSE, dilated_factors=dilated_factors)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn_tail'] = MetaBNLayer(num_filters, args, use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se_tail'] = MetaSEBlock(num_filters, num_filters//4)
        
        self.layer_dict['tail'] = MetaConv2dLayer(num_filters, in_channels, 3, 1, 1, use_bias=False)

    def forward(self, inp, num_step=0, params=None, training=True, bkp_running_statistics=False, noise_sigma=None):
        ## please set training to True for better training
        ### extract parameters
        ori = inp
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        ### do forward pass
        if not isinstance(noise_sigma, torch.Tensor):
            noise_sigma = torch.FloatTensor(noise_sigma).cuda()
        
        m = noise_sigma.repeat(1, inp.size()[-2], inp.size()[-1], 1)
        m = m.permute(3, 0, 1, 2)

        inp = torch.cat([inp, m], 1)
        conv_in = self.layer_dict['head'].forward(inp, params=param_dict['head'])
        conv_in = F.leaky_relu(conv_in, negative_slope=0.2)

        out = conv_in
        for i in range(self.stages):
            if self.Agg:
                out = torch.cat([out, inp], dim=1)
            out = self.layer_dict['body{}'.format(i+1)].forward(out, num_step, params=param_dict['body{}'.format(i+1)],
                                                                training=training, bkp_running_statistics=bkp_running_statistics)
        residual = out
        out = self.layer_dict['conv_tail'].forward(residual, params=param_dict['conv_tail'])
        out = self.layer_dict['bn_tail'].forward(out, num_step, params=param_dict['bn_tail'], training=training,
                                                 bkp_running_statistics=bkp_running_statistics)
        if self.withSE:
            out = self.layer_dict['se_tail'].forward(out, params=param_dict['se_tail'])
        
        out = F.leaky_relu(out, negative_slope=0.2, inplace=True)

        out = self.layer_dict['tail'].forward(out, params=param_dict['tail'])
        out = ori - out
        return out

    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        # if torch.sum(param.grad) > 0:
                        #     print(param.grad)
                        #     param.grad.zero_()
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
        indexes = [1]
        if self.dilated_factors == 3:
            if self.msb_type == 'MAEB':
                msb_indexes = [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5]
            elif self.msb_type == 'MSB':
                msb_indexes = [2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 4, 4, 5, 5]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if self.rb_type == 'RB':
            if self.withSE:
                if self.relu_type == 'lrelu':
                    rb_indexes = [1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 8, 8, 9, 9]
                else:
                    rb_indexes = [1, 2, 2, 3, 3, 4, 5, 6, 6, 7, 7, 8, 9, 10, 10, 11, 11]
            else:
                if self.relu_type == 'lrelu':
                    rb_indexes = [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7]
                elif self.relu_type == 'prelu':
                    rb_indexes = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 9]
        elif self.rb_type == 'Dual':
            rb_indexes = [1, 2, 2, 3, 4, 5, 5, 6, 7, 8, 8, 9, 10, 11, 11, 12, 13, 14, 14, 15, 15]
        else:
            raise NotImplementedError
        tail_indexes = [1, 2, 2, 3, 3, 4]
        ### build indexes
        rb_indexes = [max(msb_indexes)+item for item in rb_indexes]
        body_indexes = msb_indexes
        body_indexes.extend(rb_indexes) ### indexes for body module
        index_gap = max(body_indexes) - min(body_indexes)+1
        for i in range(self.stages):
            tmp_indexes = [index_gap*i+item for item in body_indexes]
            indexes.extend(tmp_indexes)
        tail_indexes = [max(indexes)+item for item in tail_indexes]
        indexes.extend(tail_indexes)
        return indexes

class AdaFM(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, args=None):
        super(AdaFM, self).__init__()
        self.layer_dict = nn.ModuleDict()
        self.nb = nb
        # sf=2
        self.name = 'AdaFM-{}c-{}nb-{}nf'.format(in_nc, nb, nf)

        # self.layer_dict['m_down'] = PixelUnShuffle(upscale_factor=2)
        self.layer_dict['fea_conv'] = MetaConv2dLayer(in_nc+1, nf, kernel_size=3, stride=2, padding=1, use_bias=True)
        for i in range(nb):
            self.layer_dict['rb{}'.format(i+1)] = MetaResBlock(nf, nf, kernel_size=3, stride=1, padding=1, args=args)
        self.layer_dict['tran_cb'] = MetaConvBNLReLU(nf, nf, 3, 1, 1, args=args, activation=False)
        self.layer_dict['up_sample'] = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer_dict['up_conv'] = MetaConv2dLayer(nf, nf, 3, 1, 1, use_bias=True)
        # self.layer_dict['up_act'] = nn.ReLU()
        self.layer_dict['tail1'] = MetaConv2dLayer(nf, nf, 3, 1, 1, True)
        self.layer_dict['tail2'] = MetaConv2dLayer(nf, out_nc, 3, 1, 1, use_bias=True)
    
    def forward(self, inp, num_step=0, params=None, training=True, noise_sigma=None, bkp_running_statistics=False):
        ### extract parameters
        param_dict = dict()
        if params is not None:
            param_dict = extract_top_level_dict(current_dict=params)
        for name, param in self.layer_dict.named_parameters():
            name = name.split('.')[0]
            if name not in param_dict:
                param_dict[name] = None
        
        ori = inp
        ### do forward pass
        h, w = inp.size()[-2:]
        paddingBottom = int(np.ceil(h/2)*2-h)
        paddingRight = int(np.ceil(w/2)*2-w)
        inp = torch.nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(inp)
        # inp = self.layer_dict['m_down'](inp)
        if not isinstance(noise_sigma, torch.Tensor):
            noise_sigma = torch.FloatTensor(noise_sigma).cuda()
        
        m = noise_sigma.repeat(1, inp.size()[-2], inp.size()[-1], 1)
        m = m.permute(3, 0, 1, 2)

        inp = torch.cat([inp, m], 1)
        out = self.layer_dict['fea_conv'](inp, params=param_dict['fea_conv'])
        residual = out
        for i in range(self.nb):
            out = self.layer_dict['rb{}'.format(i+1)](out, num_step, params=param_dict['rb{}'.format(i+1)], training=training)
        out = self.layer_dict['tran_cb'](out, num_step, params=param_dict['tran_cb'], training=training)
        out = residual + out
        out = self.layer_dict['up_sample'](out)
        out = self.layer_dict['up_conv'](out, params=param_dict['up_conv'])
        out = F.relu(out, inplace=True)
        out = self.layer_dict['tail1'](out, params=param_dict['tail1'])
        out = F.relu(out, inplace=True)
        out = self.layer_dict['tail2'](out, params=param_dict['tail2'])
        # out = ori - out[..., :h, :w]
        out = out[..., :h, :w]
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
    
    def get_layer_index(self):
        indexes = [1, 1] ## conv
        res_indexes = [2, 3, 3, 4, 5, 5]
        index_gap = max(res_indexes) - min(res_indexes) + 1
        for i in range(self.nb):
            tmp_indexes = [index_gap*i+item for item in res_indexes] ## conv -> bn -> conv -> bn
            indexes.extend(tmp_indexes)
        tail_indexes = [1, 2, 2, 3, 3, 4, 4, 5, 5]
        indexes.extend([item+max(indexes) for item in tail_indexes])
        return indexes

if __name__ == '__main__':
    '''
    check networks and pass params 
    '''
    import sys
    sys.path.append('..')
    from utils.arguments import get_args

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
    kwargs = {'Agg_input': True, 'input_channels': 3}
    x = torch.rand(1, 3, 50, 50)
    noise_sigma = torch.FloatTensor([np.random.rand()])
    
    args.device=torch.device('cpu')
    dncnn = MetaFFDNet(in_channels=3, num_filters=32, stages=5, args=args)
    out_bkp = dncnn.forward(x, noise_sigma=noise_sigma)
    param_dict = dict()
    for name, param in dncnn.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    out_pass = dncnn.forward(x, noise_sigma=noise_sigma)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))
    out_param = dncnn.forward(x, params=get_inner_loop_params_dict(dncnn.named_parameters()), noise_sigma=noise_sigma)
    print('difference between out_pass and out_param', torch.sum(out_param-out_pass))
    inp = torch.rand(1, 3, 50, 50)
    out = dncnn(inp,  noise_sigma=noise_sigma)

    ResN = MetaMSResNetNoise(in_channels=3, num_filters=16, stages=3, args=args)
    print(ResN)
    out_bkp = ResN.forward(x, noise_sigma=noise_sigma)
    param_dict = dict()
    for name, param in ResN.named_parameters():
        if param.requires_grad:
            param_dict[name] = param
    out_pass = ResN.forward(x, noise_sigma=noise_sigma)
    print('difference between out_pass and out_nkp', torch.sum(out_bkp-out_pass))
    out_param = ResN.forward(x, params=get_inner_loop_params_dict(ResN.named_parameters()), noise_sigma=noise_sigma)
    print('difference between out_pass and out_param', torch.sum(out_param-out_pass))
    print(ResN.get_layer_index())


    # torch.onnx.export(msscnet, inp, 'msresnetAggWoMSWoSE.onnx', training=torch.onnx.TrainingMode.TRAINING)
    # torch.onnx.export(dncnn, inp, 'dncnn.onnx', training=torch.onnx.TrainingMode.TRAINING)

    ### check MetaMSResNet
    '''
    msbs = ['MAEB', 'MSB']
    rbs = ['RB', 'Dual']
    SES = [True, False]
    for msb in msbs:
        for rb in rbs:
            for se in SES:
                net = MetaMSResNet(3, 32, 2, args, msb=msb, rb=rb, withSE=se)
                print(net.name)
                out = net(inp, num_step=0)
                print(out.size())
                l_indexes = net.get_layer_index()
                print(l_indexes)
                print(len(l_indexes))
                params = net.parameters()
                params = [p for p in params if p.requires_grad]
                print('number of params: ', len(params))
                torch.onnx.export(net, inp, './net_examples/{}.onnx'.format(net.name), training=torch.onnx.TrainingMode.TRAINING)
    '''


