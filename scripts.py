import sys
sys.path.append('..')
from utils.arguments import get_args
from metaunit import MetaUnit
from nets import DenseCBNRSENet, ResBNAggKstages
import torch
from ImgLIPDset import TestDataset
from utils.metrics import SSIM, PSNR
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from ImgLIPNfreqsKshot import ImgLIPNfreqsKshot

args = get_args()
kwargs = {'Agg_input': 3, 'input_channels': 3}
net = DenseCBNRSENet(in_channels=3, num_filters=32, s1=2, s2=3, args=args, **kwargs)
# net = ResBNAggKstages(in_channels=3, num_filters=36, K=4, args=args, device=args.device, **kwargs)
print(net)
ssim = SSIM()
psnr = PSNR(max_val=1.0)

ckp = 'results/derain/DenseCBNRSENet-2-3-8-16/models/23000-iterModel.tar'
meta_learner = MetaUnit(args=args, net=net)
meta_learner.load_model(ckp)

meta_param_dict = dict()
for name, param in meta_learner.named_parameters():
    if name not in meta_param_dict:
        if param.requires_grad:
            name = name.replace('layer_dict.', '')
            name = name.replace('net.','')
            meta_param_dict[name] = param
    else:
        raise Exception

net_param_dict = dict()
for name, param in meta_learner.net.named_parameters():
    if param.requires_grad:
        if name not in net_param_dict:
            name = name.replace('layer_dict.', '')
            net_param_dict[name] = param
        else:
            raise Exception

for key in meta_param_dict.keys():
    if key in net_param_dict:
        diff = torch.sum(meta_param_dict[key]-net_param_dict[key])
        print(key, diff)
'''
for name, param in meta_learner.named_parameters():
    if 'r_mean' in name:
        print('r_mean: ', param)
    elif 'r_var' in name:
        print('r_var: ', param)
        '''


test_kwargs={
    'dataset': 'Rain100L-test',
    'type': 'rain',
    # 'clean_dir': '../MetaLIP/data/RawData/Rain100L/test/clean',
    # 'noise_dir': '../MetaLIP/data/RawData/Rain100L/test/rain',
    'clean_dir': '/home/wran/Public/datasets/derain/Rain100L-origin/test/norain',
    'noise_dir': '/home/wran/Public/datasets/derain/Rain100L-origin/test/rain'
    }
dataset_config = {
    'Rain100L': ('norain-{}.png', 'rain-{}.png', 50, 1),
    'Rain100L-test': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1)

}
db_test =  TestDataset(dataset_config, **test_kwargs)

clean_img, noise_img = db_test[0]
'''
plt.subplot(1, 2, 1)
plt.imshow(clean_img.transpose(1, 2, 0))
plt.subplot(1, 2, 2)
plt.imshow(noise_img.transpose(1, 2, 0))
plt.show()
'''

clean_img = Image.open('../MetaLIP/data/ProcessData/Rain100L-50/train/clean/cp-0/patch-0.png')
noise_img = Image.open('../MetaLIP/data/ProcessData/Rain100L-50/train/rain/cp-0/patch-0.png')
'''
plt.subplot(1, 2, 1)
plt.imshow(clean_img)
plt.subplot(1, 2, 2)
plt.imshow(noise_img)
plt.show()
'''
clean_img = np.array(clean_img).astype('f4').transpose(2, 0, 1)/255.0
noise_img = np.array(noise_img).astype('f4').transpose(2, 0, 1)/255.0


clean_img = torch.FloatTensor(clean_img[None, :, :50, :50]).cuda()
noise_img = torch.FloatTensor(noise_img[None, :, :50, :50]).cuda()

database = ImgLIPNfreqsKshot(root_dir='../MetaLIP/data/ProcessData/Rain100L-50', batch_size=8, n_freqs=16, k_shot=1, k_query=1, patch_size=50, 
            sigmas='5,15')

        

spt_x, spt_y, qry_x, qry_y = database.next(mode='train')
spt_x = torch.FloatTensor(spt_x).cuda()[0]
spt_y = torch.FloatTensor(spt_y).cuda()[0]
# loss, ssim_val, psnr_val = meta_learner.net_forward(noise_img, clean_img, params=None, training=False, calc_metrics=True, bkp_running_statistics=False, num_step=0

denoise_img = meta_learner.net.forward(spt_x, num_step=0, params=None, training=False)

ssim_val = ssim.ssim(denoise_img, spt_y)
print(torch.max(spt_y), torch.max(noise_img))
psnr_val = psnr.calc_psnr(denoise_img, spt_y)
print(ssim_val, psnr_val)
print(torch.sum(denoise_img-clean_img))


test_psnrs, test_ssims = [], []
for i in range(len(db_test)):
    clean_img , noise_img = db_test[i]
    clean_img = torch.from_numpy(clean_img[None, ...]).cuda()
    noise_img = torch.from_numpy(noise_img[None, ...]).cuda()
    with torch.no_grad():
        denoise_img = meta_learner.net.forward(noise_img, training=False,num_step=0)
        denoise_img = torch.clamp(denoise_img, min=0.0, max=1.0)
        test_psnrs.append(psnr.calc_psnr(denoise_img, clean_img))
        test_ssims.append(ssim.ssim(denoise_img, clean_img))
test_psnr = np.array(test_psnrs).mean().astype(np.float16)
test_ssim = np.array(test_ssims).mean().astype(np.float16)
msg = 'Test: ssim: {}, psnr: {}'.format(test_ssim, test_psnr)
print(msg)