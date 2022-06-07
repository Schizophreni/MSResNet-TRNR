import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
import sys
sys.path.append('..')
from utils.metrics import PSNR

psnr = PSNR(max_val=1.0)

class GradCam:
    def __init__(self, model, layer_name='se_tail'):
        self.model = model
        self.feature = None
        self.gradient = None
        self.layer_name = layer_name

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.requires_grad=False

    def save_gradient(self, grad):
        self.gradient = grad
    
    def __call__(self, x, y):
        image_size = (x.size(-1), x.size(-2))  ### size of image (W, H)
        datas = torch.Tensor(x)
        datas.requires_grad = True ## set gradient is True

        heat_maps = [] ## heat maps
        body_idx = 0
        
        for i in range(datas.size(0)):
            img = datas[i].data.cpu().numpy()

            feature = datas[i].unsqueeze(0)
            for name, module in self.model.layer_dict.named_children():
                if name == 'head':
                    feature = module(feature)
                    feature = F.leaky_relu(feature, 0.2, inplace=True)
                    conv_in = feature ## residual
                elif 'body' in name:
                    body_idx += 1
                    tmp = feature
                    for sub_name, sub_module in module.named_children():
                        if sub_name == 'rb':
                            feature = sub_module(feature, num_step=0, training=False)
                            feature = feature+tmp
                            feature = F.leaky_relu(feature, 0.2, True)
                        else:
                            feature = torch.cat([feature, x], dim=1)
                            feature = sub_module(feature)
                        if self.layer_name == sub_name+str(body_idx):
                            print('[!!!] Register hook')
                            feature.register_hook(self.save_gradient)
                            self.feature = feature
                    if name == 'body4':
                        feature = feature+conv_in

                else:
                    print(name)
                    if name == 'bn_tail':
                        feature = module(feature, current_step=0, training=False)
                    elif name == 'se_tail':
                        feature = module(feature)
                        feature = F.leaky_relu(feature, 0.2, True)
                    else:
                        feature = module(feature)
                    if name == 'tail':
                        feature = x - feature
                    if self.layer_name == name:
                        print('[!!!] Register hook')
                        feature.register_hook(self.save_gradient)
                        self.feature = feature

            out = torch.clamp(feature, 0.0, 1.0)
            print('psnr value: ', psnr.calc_psnr(out, y))
            self.model.zero_grad(params=None)
            loss = F.l1_loss(out, y, reduction='mean')
            loss.backward()

            weight = self.gradient.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
            mask = F.relu((weight * self.feature).sum(dim=1)).squeeze(0)
            mask = cv2.resize(mask.data.cpu().numpy(), image_size)
            mask = mask - np.min(mask)
            if np.max(mask) != 0:
                mask = mask / np.max(mask)
            # mask = 1-mask
            heat_map = np.float32(cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET))
            cam = heat_map + np.float32((np.uint8(img.transpose((1, 2, 0)) * 255)))
            cam = cam - np.min(cam)
            if np.max(cam) != 0:
                cam = cam / np.max(cam)
            heat_maps.append(transforms.ToTensor()(cv2.cvtColor(np.uint8(255 * cam), cv2.COLOR_BGR2RGB)))
        heat_maps = torch.stack(heat_maps)
        return heat_maps
