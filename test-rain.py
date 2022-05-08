from utils.arguments import get_args
from ImgLIPNfreqsKshot import ImgLIPNfreqsKshot
from nets import *
from metaunit import MetaUnit
from ImgLIPDset import TestDataset
args = get_args()
from utils.metrics import SSIM, PSNR
import cv2
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

ssim = SSIM()

psnr = PSNR(max_val=1.0)

kwargs = {'Agg_input': True, 'input_channels': 3}
test_kwargs={
'dataset': 'Rain100L-t',
'type': 'rain',
'clean_dir': '/home/rw/Public/datasets/derain/Rain100L-origin/test/norain',
'noise_dir': '/home/rw/Public/datasets/derain/Rain100L-origin/test/rain',
}

Rain100H_test_kwargs={
'dataset': 'Rain100H-t',
'type': 'rain',
'clean_dir': '/home/rw/Public/datasets/derain/Rain100H/test',
'noise_dir': '/home/rw/Public/datasets/derain/Rain100H/test',
}

Rain800_test_kwargs={
'dataset': 'Rain800-t',
'type': 'rain',
'clean_dir': '/home/rw/Public/datasets/derain/Rain800/test/norain',
'noise_dir': '/home/rw/Public/datasets/derain/Rain800/test/rain',
}

Rain100L_test_kwargs = {
    'dataset': 'Rain100L-t',
    'type': 'rain',
    'clean_dir': '/home/rw/Public/datasets/derain/Rain100L/norain',
    'noise_dir': '/home/rw/Public/datasets/derain/Rain100L/rain',
}

dataset_config = {
'Rain100L': ('norain-{}.png', 'rain-{}.png', 50, 1),
'Rain100L-test': ('norain-{}.png', 'rain-{}.png', 30, 1),
'Rain100L-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
'Rain100H-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
'Rain800-t': ('norain-{:03d}.png', 'rain-{:03d}.png', 100, 1),
}
#torch.manual_seed(0)
# np.random.seed(0)
db_test =  TestDataset(dataset_config, **Rain100L_test_kwargs)
net = MetaMSResNet(3, 48, stages=4, args=args, Agg=False, withSE=True, msb='MAEB', rb='Dual', relu_type='lrelu')
model = MetaUnit(args=args, net=net)
model.load_model(model_save_path='results/derain/MAEB-RES-Rain100L64-100-4stages-ssim5.0-5-12/models/{}-iterModel.tar'.format(args.total_epochs))
# model.load_model(model_save_path='Ablation/results/dataSize/PA/MSResNetAggPA-4-0-48C-Rain100L-10-multi-merge-1-32/models/{}-iterModel.tar'.format(args.total_epochs))
# model.load_model(model_save_path='results/derain/MPR/MSResNetAggPA-4-0-48C-SSIM5.0-GDLoss0.0-4-30-40imgs/models/{}-iterModel.tar'.format(args.total_epochs))

test_psnrs, test_ssims = [], []
base_H, base_W = 1000, 1000
for i in range(len(db_test)):
    clean_img , noise_img = db_test[i]
    clean_img = torch.from_numpy(clean_img[None, ...]).cuda()
    noise_img = torch.from_numpy(noise_img[None, ...]).cuda()
    if clean_img.size(2)>1000 or clean_img.size(3) >1000:
        H = clean_img.size(2)
        W = clean_img.size(3)
        denoise_img = torch.zeros_like(clean_img).cuda()
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
            denoise_img = model.test_with_attenuate(noise_img, verbose=True if i==0 else False)
    denoise_img = torch.clamp(denoise_img, min=0.0, max=1.0)
    psnr_val = psnr.calc_psnr(denoise_img, clean_img)
    ssim_val = ssim.ssim(denoise_img, clean_img)
    test_psnrs.append(psnr_val)
    test_ssims.append(ssim_val)
    denoise_img = denoise_img[0].detach().cpu().numpy()
    denoise_img = denoise_img.transpose(1, 2, 0)*255
    cv2.imwrite('derained/derain{}.png'.format(i+1),denoise_img.astype('uint8')[..., ::-1])
    print('index: {}, psnr: {}, ssim: {}'.format(i+1, psnr_val, ssim_val))
test_psnr = np.array(test_psnrs).mean().astype(np.float32)
test_ssim = np.array(test_ssims).mean().astype(np.float32)
print('average psnr: {}, ssim: {}'.format(test_psnr, test_ssim))
