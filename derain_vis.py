from utils.arguments import get_args
from ImgLIPNfreqsKshot import ImgLIPNfreqsKshot
from nets import MetaMSResNet
from metaunit import MetaUnit
from ImgLIPDset import TestDataset
args = get_args()
from utils.metrics import SSIM, PSNR
import cv2
import glob
from AttentionVis.grad_cam import GradCam

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


db_test =  TestDataset(dataset_config, **Rain100L_test_kwargs)

net = MetaMSResNet(3, 48, stages=4, args=args, Agg=True, withSE=False, msb='MAEB', rb='Dual', relu_type='lrelu')
model = MetaUnit(args=args, net=net)
model.load_model(model_save_path='results/derain/AggMAEB-Dual-4s-SEFalse-lrelu-Rain100L-80-4-24/models/{}-iterModel.tar'.format(args.total_epochs), 
                 map_location='cpu')
model = model.cpu()

test_psnrs, test_ssims = [], []
layer_name = 'rb4_tail'

grad_cam = GradCam(model=model.net, layer_name=layer_name)

# for i in range(len(db_test)):
clean_img , noise_img = db_test[0]
clean_img = torch.from_numpy(clean_img[None, ...])
noise_img = torch.from_numpy(noise_img[None, ...])

feature_img = grad_cam(noise_img, clean_img)
print(feature_img.size())
feature_img = feature_img[0].detach().numpy()
feature_img = np.transpose(feature_img, (1, 2, 0))
print(np.max(feature_img))
cv2.imwrite('AttentionVis/{}_vis.png'.format(layer_name), (feature_img*255).astype('uint8'))
