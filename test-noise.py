from utils.arguments import get_args
from models.nets import MetaMSResNetN, AdaFM
from models.metaunit_noise import MetaUnit
from utils.metrics import SSIM, PSNR
import cv2
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import numpy as np
import torch


transformation = transforms.Compose(
            [lambda x: Image.open(x), #.convert('L'),
            # lambda x: np.expand_dims(np.array(x), -1),
            lambda x: np.transpose(x, (2, 0, 1)), ## change to (c, h, w)
            
            lambda x: x/255.]
        )

ssim = SSIM()

psnr = PSNR(max_val=1.0)
np.random.seed(0)
args = get_args()

sigma = 25

def parse_imgs(imgs_dir):
    """
    return all test images path 
    """
    assert os.path.exists(imgs_dir), '{} folder does not exists'.format(imgs_dir)
    imgs = os.listdir(imgs_dir)
    imgs = [os.path.join(imgs_dir, img) for img in imgs if img.endswith('png') or img.endswith('jpg') or img.endswith("tif")]
    return imgs


args.device=torch.device('cuda')
os.environ['CUDA_VISIBLE_DEVICES']='0'
net = net = MetaMSResNetN(in_channels=1, num_filters=64, stages=6, args=args, withSE=True)
# net = AdaFM(in_nc=3, out_nc=3, args=args)
model = MetaUnit(args=args, net=net)
model.load_state_dict(torch.load(args.model))

test_psnrs, test_ssims = [], []

imgs_dir = '/home/rw/Public/datasets/denoise/Kodak'
test_imgs = parse_imgs(imgs_dir=imgs_dir)

noise_level = sigma/255.0
print('noise level: ', noise_level)
test_noise_sigma = torch.FloatTensor([noise_level]).cuda()
test_imgs = sorted(test_imgs)
for img in test_imgs:
    clean_img = transformation(img)
    img_name = img.split('/')[-1]
    noise_img = clean_img + noise_level*np.random.randn(*clean_img.shape)
    # print(noise_img[0, 100, 100])

    clean_img = torch.from_numpy(clean_img).cuda().unsqueeze(0).float()
    noise_img = torch.from_numpy(noise_img).cuda().unsqueeze(0).float()
    with torch.no_grad():
        # denoise_img = model.test_with_attenuate(noise_img, verbose=False, training=False)
        denoise_img = model.net.forward(noise_img, num_step=0, training=False, noise_sigma=test_noise_sigma)
        denoise_img = torch.clamp(denoise_img, min=0.0, max=1.0)
    psnr_val = psnr.calc_psnr(denoise_img, clean_img)
    ssim_val = ssim.ssim(denoise_img, clean_img)
    test_psnrs.append(psnr_val)
    test_ssims.append(ssim_val)
    
    denoise_img = denoise_img[0].detach().cpu().numpy()
    denoise_img = denoise_img.transpose(1, 2, 0)*255

    noise_img.clamp_(0.0, 1.0)
    noise_img = noise_img[0].detach().cpu().numpy()
    noise_img = noise_img.transpose(1, 2, 0)*255
    # cv2.imwrite('denoised/denoise-{}-{}'.format(sigma, img_name), denoise_img.astype('uint8')[..., ::-1])
    # cv2.imwrite('denoised/noise-{}'.format(img_name), noise_img.astype('uint8')[..., ::-1])
    print('index: {}, psnr: {}, ssim: {}, shape: {}'.format(img_name, psnr_val, ssim_val, clean_img.shape))
test_psnr = np.array(test_psnrs).mean().astype(np.float32)
test_ssim = np.array(test_ssims).mean().astype(np.float32)
print('average psnr: {}, ssim: {}'.format(test_psnr, test_ssim))
