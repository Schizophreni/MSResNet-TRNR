import torch

from models.meta_layers import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def show_feature(inp_tensor, layer_name):
    feat = inp_tensor.numpy()[0]  # [C, H, W]
    feats_num = 10
    feat = (feat - feat.min()) / (feat.max()-feat.min())
    discriminative_feat = feat.reshape((feat.shape[0], -1))
    # stds = discriminative_feat.std(axis=-1)
    stds = discriminative_feat.sum(axis=-1)
    indexes = np.argsort(stds)[::-1]
    # print(stds[indexes])
    # sort_indexes = [indexes[:feats_num//2], indexes[-feats_num//2:]]
    sort_indexes = [indexes[:feats_num]]
    indexes = np.concatenate(sort_indexes)

    # feat = (feat - feat.min()) / (feat.max()-feat.min())

    fig = plt.figure(figsize=(12, 3.2))
    plt.tight_layout()
    plt.subplots_adjust(left=None, right=None, bottom=None, top=None, hspace=0, wspace=0.05)
    rows, cols = 2, 5
    for i in range(rows):
        for j in range(cols):
            s_feat = feat[indexes[cols*i+j], :, :]
            s_feat = (1 + s_feat) / 2.0
            ax = fig.add_subplot(rows, cols, cols*i+j+1)
            # ax = plt.addsubplot(rows, cols, cols*i+j+1)
            # s_feat = (s_feat - s_feat.min()) / (s_feat.max() - s_feat.min())
            """
            if (cols*i+j+1) in [4, 6]:
                ax.spines['top'].set_color('red'), ax.spines['top'].set_linewidth(2)
                ax.spines['bottom'].set_color('red'), ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_color('red'), ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_color('red'), ax.spines['right'].set_linewidth(2)
            elif (cols*i+j+1) in [5, 8, 9]:
                ax.spines['top'].set_color('yellow'), ax.spines['top'].set_linewidth(2)
                ax.spines['bottom'].set_color('yellow'), ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_color('yellow'), ax.spines['left'].set_linewidth(2)
                ax.spines['right'].set_color('yellow'), ax.spines['right'].set_linewidth(2)
            """
            plt.xticks([])
            plt.yticks([])
            plt.imshow(s_feat) # , cmap='gray')
            print(stds[indexes[cols*i+j]], s_feat.min(), cols*i+j+1)
    plt.savefig('../feature_visualization/{}.png'.format(layer_name), bbox_inches='tight', pad_inches=0.1, dpi=400)
    #
    # plt.show()


class MetaMSResNet(nn.Module):
    def __init__(self, in_channels, num_filters, stages=4, args=None, Agg=True, withSE=True, dilated_factors=3,
                 msb='MAEB',
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
            self.name = 'MAEB-RES-WaterlooBSD-500-{}stages-ssim{}-sigma15'.format(stages, args.ssim_weight)
        self.layer_dict['head'] = MetaConv2dLayer(in_channels, num_filters, 3, 1, 1, use_bias=False)
        for i in range(stages):
            self.layer_dict['body{}'.format(i + 1)] = MetaMSRB(num_filters + Agg_channels, num_filters, args,
                                                               relu_type=relu_type, msb=msb, rb=rb, withSE=withSE,
                                                               dilated_factors=dilated_factors)
        self.layer_dict['conv_tail'] = MetaConv2dLayer(num_filters, num_filters, 3, 1, 1, use_bias=False)
        self.layer_dict['bn_tail'] = MetaBNLayer(num_filters, args,
                                                 use_per_step_bn_statistics=args.use_per_step_bn_statistics)
        if self.withSE:
            self.layer_dict['se_tail'] = MetaSEBlock(num_filters, num_filters // 4)

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
            out = self.layer_dict['body{}'.format(i + 1)].forward(out, num_step,
                                                                  params=param_dict['body{}'.format(i + 1)],
                                                                  training=training,
                                                                  bkp_running_statistics=bkp_running_statistics)
            # show_feature(inp_tensor=out, layer_name='body{}'.format(i + 1))
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
        rb_indexes = [max(msb_indexes) + item for item in rb_indexes]
        body_indexes = msb_indexes
        body_indexes.extend(rb_indexes)  ### indexes for body module
        index_gap = max(body_indexes) - min(body_indexes) + 1
        for i in range(self.stages):
            tmp_indexes = [index_gap * i + item for item in body_indexes]
            indexes.extend(tmp_indexes)
        tail_indexes = [max(indexes) + item for item in tail_indexes]
        indexes.extend(tail_indexes)
        return indexes

if __name__ == '__main__':
    from utils.arguments import  get_args
    from PIL import Image
    import numpy as np
    import torch
    from metaunit import MetaUnit

    args = get_args()
    args.device = torch.device('cpu')

    net = MetaMSResNet(in_channels=3, num_filters=48, stages=4, args=args, Agg=False)
    model = MetaUnit(args=args, net=net)
    # ckp = torch.load('results/MSResNet-Rain100L-100-TRNR/best.tar', map_location='cpu')
    # model.load_state_dict(ckp['net'])
    ckp = torch.load('Ablation/results/MSResNet-Rain100L-Full-RIS/best.pth', map_location='cpu')
    model.net.load_state_dict(ckp)

    img = Image.open('D:/MetaLIP/data/Rain100L/train/rain-97.png')
    img = np.array(img).transpose(2, 0, 1) / 255.0
    x = torch.FloatTensor(img[None, ...])
    with torch.no_grad():
        model.net.forward(x, num_step=0, params=None, training=False)
