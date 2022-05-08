import sys
sys.path.append('..')
from nets import MetaMSResNet
from utils.metrics import SSIM, PSNR
from utils.arguments import get_args
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import os
import torch
import numpy as np
import glob
from PIL import Image


psnr = PSNR(max_val=1.0)
ssim = SSIM()
args = get_args()

def get_img_pair(norain_img_name):
    norain_img = Image.open(norain_img_name)
    rain_img = Image.open(norain_img_name.replace('norain', 'rain'))
    # plt.imshow(rain_img)
    # plt.show()

    norain_img = np.array(norain_img).transpose(2, 0, 1)[None, ...]
    rain_img = np.array(rain_img).transpose(2, 0, 1)[None, ...]

    norain_img = torch.from_numpy(norain_img/255.0).float()
    rain_img = torch.from_numpy(rain_img/255.0).float()
    return norain_img, rain_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    random.seed(0)
    np.random.seed(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    img_path = '/home/rw/Public/Rain100L-2/test'
    ckp_path = 'results/MAEB-RES-WithMS-withSE-Rain100L-full-4stages-ssim0.0/models/{}-iterModel.tar'.format(args.total_epochs)

    model = MetaMSResNet(3, 48, stages=4, args=args, Agg=False, withSE=True, msb='MAEB', rb='Dual', relu_type='lrelu', dilated_factors=3)
    ckp = torch.load(ckp_path, map_location='cuda')
    model.load_state_dict(ckp['net'])
    model = model.cuda()

    test_norain_imgs = glob.glob(os.path.join(img_path, 'norain-*.png'))
    psnrs = []
    ssims = []
    with torch.no_grad():
        for norain_img_name in test_norain_imgs:
            norain_img, rain_img = get_img_pair(norain_img_name)
            rain_img = rain_img.cuda()
            norain_img = norain_img.cuda()
        
            derain_img = model.forward(rain_img, num_step=0, training=False)
            derain_img = torch.clamp(derain_img, 0.0, 1.0)
            psnr_val = psnr.calc_psnr(derain_img, norain_img)
            ssim_val = ssim.ssim(derain_img, norain_img)
            print('=== test {}, psnr: {}, ssim: {}'.format(norain_img_name.replace('norain', 'rain'), psnr_val, ssim_val))
            psnrs.append(psnr_val)
            ssims.append(ssim_val)
        print('Average psnr: {}, ssim: {}'.format(np.array(psnrs).mean(), np.array(ssims).mean()))

    

