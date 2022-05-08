"""
generate noisy patches from clean images
noisy patches contain: noise, rainstreaks, haze etc.
"""
import glob
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import os


def clean2noisy(x, sigmas):
    """
    generate noisy image from clean image
    :param x: shape of [setsz, c, h, w]
    :sigmas: tuple of sigma like (10, 15, 20)
    :return: noisy + x
    """
    tmp = np.zeros_like(x)
    for idx in range(x.shape[0]):
        sigma = np.random.choice(sigmas)
        tmp[idx, ...] = np.random.randn(x.shape[1], x.shape[2], x.shape[3])*(sigma/255.0)
    return tmp+x

def clean2rain(x, rainstreaks_dir, patch_size=40):
    """
    generate rainy image from clean patches
    :param x: clean patches
    :param rainstreaks: rainstreaks_dir 
    rainstreaks are chozen via downsampling
    """
    rain_transformation = transforms.Compose([
        lambda x: Image.open(x),
        transforms.RandomCrop(patch_size),
        # transforms.Resize(patch_size, interpolation=2),
        lambda x: np.transpose(x, (2, 0, 1)),
        lambda x: x/255.
    ])
    batch_size = x.shape[0]
    rainstreaks = glob.glob(os.path.join(rainstreaks_dir, '*.png'))
    rainstreaks = np.random.choice(rainstreaks, batch_size, replace=False) ## choose 
    rain = np.zeros_like(x)
    ## print('max of x: ', np.max(x))
    for i in range(batch_size):
        # print('use rainstreaks: ', rainstreaks[i])
        rain[i, ...] = rain_transformation(rainstreaks[i])
        '''
        ch = np.random.rand() ## augmentation
        if ch < 0.25:
            rain[i, ...] = rain[i, ::-1, :]
        elif ch<0.5:
            rain[i, ...] = rain[i, :, ::-1]
        elif ch<0.75:
            rain[i, ...] = rain[i, ::-1, ::-1]
        '''
    rain = x+rain
    rain = np.maximum(rain, 0)
    rain = np.minimum(rain, 1.0)
    return rain ## no cutoff


def main():
    import cv2
    rain_dir = r'E:/datasets/Rain100L/rainstreaks'
    img_dir = r'E:\MetaLIP-update\data\ProcessData\train\cp-119'
    imgs = glob.glob(os.path.join(img_dir, '*.png'))
    imgs = sorted(imgs, key=lambda x: int(x.split('-')[-1].split('.')[0]))
    x = np.zeros((len(imgs), 3, 40, 40))
    for i in range(x.shape[0]):
        img = Image.open(imgs[i])
        x[i] = np.transpose(img, (2, 0, 1))
    x = x/255.
    rains = clean2rain(x, rain_dir)
    rains = np.minimum(rains, 1)
    # rains = np.maximum(rains, 0)
    rains = (255*rains).astype('uint8')
    
    save_dir = r'E:/MetaLIP/data/applyData/patchRain'
    for i in range(rains.shape[0]):
        save_path = os.path.join(save_dir, '{}.png'.format(i))
        cv2.imwrite(save_path, np.transpose(rains[i], (1, 2, 0))[..., ::-1])

if __name__ == '__main__':
    main()