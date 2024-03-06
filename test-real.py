from utils.arguments import get_args
from models.nets import MetaMSResNet
from metaunit import MetaUnit
from utils.metrics import SSIM, PSNR
import cv2
from PIL import Image
import os
import numpy as np
import torch


ssim = SSIM()
psnr = PSNR(max_val=1.0)
args = get_args()

def parse_imgs(imgs_dir):
    """
    return all test images path 
    """
    assert os.path.exists(imgs_dir), '{} folder does not exists'.format(imgs_dir)
    imgs = os.listdir(imgs_dir)
    imgs = [os.path.join(imgs_dir, img) for img in imgs if img.endswith('png') or img.endswith('jpg')]
    return imgs

net = MetaMSResNet(3, 48, stages=4, args=args, Agg=False, withSE=True, msb='MAEB', rb='Dual', relu_type='lrelu')
model = MetaUnit(args=args, net=net)
model.load_state_dict(torch.load(args.test_model)['net'])

test_psnrs, test_ssims = [], []
base_H, base_W = 1000, 1000
imgs_dir = '/home/rw/Public/datasets/derain/Practical'
imgs = parse_imgs(imgs_dir)

for img in imgs:
    print('process ', img)
    noise_img = Image.open(img)
    noise_img = np.array(noise_img, dtype=np.float32).transpose((2, 0, 1))/255.0
    noise_img = torch.from_numpy(noise_img[None, :3, :, :]).cuda()
    if noise_img.size(2)>1000 or noise_img.size(3) >1000:
        H = noise_img.size(2)
        W = noise_img.size(3)
        denoise_img = torch.zeros_like(noise_img).cuda()
        with torch.no_grad():
            for idx_i in range(H//base_H+1):
                for idx_j in range(W//base_W+1):
                    end_H = min(H, (idx_i+1)*base_H)
                    end_W = min(W, (idx_j+1)*base_W)
                    crop_rain = noise_img[:,:, idx_i*base_H:end_H, idx_j*base_W:end_W]
                    crop_derain = model.test_with_attenuate(crop_rain, verbose=False)
                    denoise_img.data[:,:,idx_i*base_H:end_H, idx_j*base_W:end_W] = crop_derain.data
    else:
        with torch.no_grad():
            denoise_img = model.test_with_attenuate(noise_img, verbose=False)
    denoise_img = torch.clamp(denoise_img, min=0.0, max=1.0)
    denoise_img = denoise_img[0].detach().cpu().numpy()
    denoise_img = denoise_img.transpose(1, 2, 0)*255
    cv2.imwrite('derained/Practical_TRNR/derain-{}'.format(img.split('/')[-1]),denoise_img.astype('uint8')[..., ::-1])
    
