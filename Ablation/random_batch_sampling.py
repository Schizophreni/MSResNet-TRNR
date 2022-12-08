"""
Implement random batch sampling strategy I and II 
"""

import torch.utils.data as data
import os
import glob
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as tvF
import torch


class RandomBatchDataset(data.Dataset):
    def __init__(self, root_dir, patch_size=50, process_type='rain', dataset_name='Rain100L'):
        super(RandomBatchDataset, self).__init__()
        self.patch_size = patch_size
        self.process_type = process_type
        self.dataset_name = dataset_name
        if process_type == 'rain':
            if dataset_name in ['Rain100L', 'Rain100H', 'Rain800']:
                self.clean_imgs = glob.glob(os.path.join(root_dir, 'norain/norain-*.png'))
                self.clean_imgs = sorted(self.clean_imgs) # make sure same results
                print('Load data from {}, total examples: {}'.format(root_dir, len(self.clean_imgs)))
            elif dataset_name in ['Rain200L', 'Rain200H']:
                self.clean_imgs = glob.glob(os.path.join(root_dir, 'norain/norain-*.png'))
                self.clean_imgs = sorted(self.clean_imgs) # make sure same results
                print('Load data from {}, total examples: {}'.format(root_dir, len(self.clean_imgs)))
            elif dataset_name == 'Rain14000':
                self.rain_imgs = glob.glob(os.path.join(root_dir, 'rain/*.jpg'))
                self.rain_imgs = sorted(self.rain_imgs) # make sure same results
                print('Load data from {}, total examples: {}'.format(root_dir, len(self.rain_imgs)))
            elif dataset_name == 'Few':
                # for few images learning 
                self.clean_imgs = glob.glob(os.path.join(root_dir, 'clean/norain-*.png'))
                self.clean_imgs = sorted(self.clean_imgs)
        else: ## for noise
            self.clean_imgs = glob.glob(os.path.join(root_dir, '*.bmp'))
            self.clean_imgs.extend(glob.glob(os.path.join(root_dir, '*.jpg')))
            print('Load data from {}, total examples: {}'.format(root_dir, len(self.clean_imgs)))
        

    def __getitem__(self, idx):
        if self.process_type == 'rain':
            if self.dataset_name in ['Rain100L', 'Rain100H', 'Rain800', 'Few']:
                clean_img = self.clean_imgs[idx]
                rain_img = clean_img.replace('clean', 'rain').replace('norain', 'rain')
            elif self.dataset_name in ['Rain200L', 'Rain200H']:
                clean_img = self.clean_imgs[idx]
                rain_img = clean_img.replace('train/norain', 'train/rain').replace('test/norain', 'test/rain/X2').replace('.png', 'x2.png')
            elif self.dataset_name == 'Rain14000':
                rain_img = self.rain_imgs[idx]
                clean_img = '{}.jpg'.format(rain_img.split('_')[0].replace('/rain', '/norain'))
                # print(rain_img, clean_img)
            return self._tuple_trans_rain(clean_img, rain_img)
        else:
            clean_img = self.clean_imgs[idx]
            noise_sigma = np.random.uniform(0, 55)/255.0
            clean_img, noise_img = self._tuple_trans_noise(clean_img, noise_sigma)
            return clean_img, noise_img, noise_sigma
    
    def traverse_noise(self, idx, noise_sigma=None):
        clean_img = Image.open(self.clean_imgs[idx]).convert('RGB')# .convert('L') ## 'L' for gray
        clean_img = np.array(clean_img).astype('f4')/255.0
        if clean_img.ndim == 2:
            clean_img = np.expand_dims(clean_img, axis=-1)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        assert noise_sigma < 0.5, 'Invalid noise_sigma' 
        noise_img = np.random.randn(*clean_img.shape)*noise_sigma + clean_img
        clean_img = clean_img[None, ...]
        noise_img = noise_img[None, ...]
        return clean_img, noise_img
    
    def traverse_rain(self, idx):
        if self.dataset_name in ['Rain100L', 'Rain100H', 'Rain800']:
            clean_img = Image.open(self.clean_imgs[idx])
            rain_img = self.clean_imgs[idx].replace('clean', 'rain').replace('norain', 'rain')
            clean_img = np.transpose(clean_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
            rain_img = Image.open(rain_img)
            rain_img = np.transpose(rain_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
        elif self.dataset_name in ['Rain200L', 'Rain200H']:
            clean_img = Image.open(self.clean_imgs[idx])
            rain_img = self.clean_imgs[idx].replace('train/norain', 'train/rain').replace('test/norain', 'test/rain/X2').replace('.png', 'x2.png')
            clean_img = np.transpose(clean_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
            rain_img = Image.open(rain_img)
            rain_img = np.transpose(rain_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
        elif self.dataset_name == 'Rain14000':
            rain_img = Image.open(self.rain_imgs[idx])
            clean_img = '{}.jpg'.format(self.rain_imgs[idx].split('_')[0].replace('/rain', '/norain'))
            rain_img = np.transpose(rain_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
            clean_img = Image.open(clean_img)
            clean_img = np.transpose(clean_img, (2, 0, 1))[None, ...].astype('f4') / 255.0
        return clean_img, rain_img
    
    def __len__(self):
        if self.dataset_name == 'Rain14000':
            return len(self.rain_imgs)
        else:
            return len(self.clean_imgs)
    
    def _tuple_trans_rain(self, clean_img, rain_img):
        clean_img = Image.open(clean_img) 
        rain_img = Image.open(rain_img)
        ## crop coordinates
        i, j, h, w = transforms.RandomCrop.get_params(clean_img, (self.patch_size, self.patch_size))
        ## crop clean image
        clean_img = tvF.crop(clean_img, i, j, h, w)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        clean_img = clean_img/255.0
        ## crop rainy image
        rain_img = tvF.crop(rain_img, i, j, h, w)
        rain_img = np.transpose(rain_img, (2, 0, 1))
        rain_img = rain_img/255.0

        data_aug = np.random.choice(range(4))
        if data_aug == 1:
            rain_img = rain_img[:,::-1,:].copy()
            clean_img = clean_img[:,::-1,:].copy()
        elif data_aug == 2:
            rain_img = rain_img[:,:,::-1].copy()
            clean_img = clean_img[:,:,::-1].copy()
        elif data_aug == 3:
            rain_img = rain_img[:, ::-1, ::-1].copy()
            clean_img = clean_img[:, ::-1, ::-1].copy()
        return clean_img, rain_img
    
    def _tuple_trans_noise(self, clean_img, noise_sigma=None):
        clean_img = Image.open(clean_img).convert('RGB')# .convert('RGB')  ## gray 'L'
        ## crop coordinates
        i, j, h, w = transforms.RandomCrop.get_params(clean_img, (self.patch_size, self.patch_size))
        ## crop clean image
        clean_img = tvF.crop(clean_img, i, j, h, w)
        clean_img = np.array(clean_img).astype('f4')
        if (clean_img.ndim) == 2:
            clean_img = np.expand_dims(clean_img, axis=-1)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        clean_img = clean_img/255.0

        data_aug = np.random.choice(range(4))
        if data_aug == 1:
            clean_img = clean_img[:,::-1,:].copy()
        elif data_aug == 2:
            clean_img = clean_img[:,:,::-1].copy()
        elif data_aug == 3:
            clean_img = clean_img[:, ::-1, ::-1].copy()
        assert noise_sigma < 0.5, 'Invalid noise sigma' 
        noise_img = clean_img+np.random.randn(*clean_img.shape).astype('f4')*noise_sigma
        return clean_img, noise_img


class RandomBatchSamplingI:
    def __init__(self, root_dir, patch_size=50, channels=3, process_type='rain', dataset_name='Rain100L'):
        """
        Random batch sampling strategy I: put back sampled images
        """
        self.dset = RandomBatchDataset(root_dir, patch_size, process_type=process_type, dataset_name=dataset_name)
        self.channels = channels
        self.patch_size = patch_size
        self.process_type = process_type
        self.dataset_name = dataset_name
        print('=== Find {} samples'.format(len(self.dset)))
    
    def next_batch(self, batch_size):
        """
        Sample a data batch for training or testing
        """
        if batch_size >= len(self.dset):
            batch_size = len(self.dset)
        
        cleans = np.empty(shape=(batch_size, self.channels, self.patch_size, self.patch_size))
        rains = np.empty(shape=cleans.shape)
        sigmas = np.empty(shape=(batch_size,))
        indexes = np.random.choice(len(self.dset), batch_size, replace=False)
        for i, index in enumerate(indexes):
            if self.process_type == 'noise':
                cleans[i], rains[i], sigmas[i] = self.dset[index]
            else:
                cleans[i], rains[i] = self.dset[index]
            # print(i, index, sigmas[i])
        if self.process_type == 'noise':
            return cleans, rains, sigmas
        else:
            return cleans, rains
    
    def next_example(self, idx, noise_sigma=15):
        clean, noise = self.dset.traverse_noise(idx, noise_sigma)
        return clean, noise

"""
Random batch sampling II can be simply implemented with DataLoader, will not implement here
"""

class RandomPatch(data.Dataset):
    def __init__(self, root_dir, mode='train', dataset_name='Few', process_type='rain'):
        """
        Random patch sampling strategy: directly sample image patches from datasets
        """
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.process_type = process_type
        self.parse_dir()
        self.transformation = transforms.Compose(
            [lambda x: Image.open(x),
            lambda x: np.transpose(x, (2, 0, 1)), ## change to (c, h, w)
            lambda x: x.astype('f4'),
            lambda x: x/255.]
        )
        print('=== Find {} examples for {}ing'.format(len(self.clean_patches), mode))
    
    def parse_dir(self):
        if 'trainPatches' in os.listdir(self.root_dir):
            self.clean_patches = glob.glob(os.path.join(self.root_dir, '{}Patches'.format(self.mode),'clean/*.png'))
        else:
            sub_dirs = os.listdir(self.root_dir)
            print(sub_dirs)
            self.clean_patches = []
            for sub_dir in sub_dirs:
                clean_patches = glob.glob(os.path.join(self.root_dir, sub_dir, '{}Patches'.format(self.mode),'clean/*.png'))
                clean_patches = sorted(clean_patches)
                self.clean_patches.extend(clean_patches)


    def __getitem__(self, idx):
        clean_patch = self.clean_patches[idx]
        rain_patch = clean_patch.replace('clean', 'rain')
        clean_patch, rain_patch = self.transformation(clean_patch), self.transformation(rain_patch)
        data_aug = np.random.choice(range(4))
        if data_aug == 1:
            rain_patch = rain_patch[:, ::-1, :].copy()
            clean_patch = clean_patch[:, ::-1, :].copy()
        elif data_aug == 2:
            rain_patch = rain_patch[:, :, ::-1].copy()
            clean_patch = clean_patch[:, :, ::-1].copy()
        elif data_aug == 3:
            rain_patch = rain_patch[:, ::-1, ::-1].copy()
            clean_patch = clean_patch[:, ::-1, ::-1].copy()
        return clean_patch, rain_patch

    def __len__(self):
        return len(self.clean_patches)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test RandomBatchSamplingI
    root_dir = '/home/wran/Public/datasets/derain/forTPAFI/Rain100L64-80/RawData/Rain100L-80-multi/train'
    dset = RandomBatchDataset(root_dir=root_dir)
    print('total samples in dset: ', len(dset))
    
    rbsI = RandomBatchSamplingI(root_dir)
    batch = rbsI.next_batch(batch_size=2)
    print(batch[0].shape)
    cleans, rains = batch
    ## show
    plt.subplot(2, 2, 1)
    plt.imshow(cleans[0].transpose(1, 2, 0))
    plt.subplot(2, 2, 2)
    plt.imshow(rains[0].transpose(1, 2, 0))
    plt.subplot(2, 2, 3)
    plt.imshow(cleans[1].transpose(1, 2, 0))
    plt.subplot(2, 2, 4)
    plt.imshow(rains[1].transpose(1, 2, 0))
    plt.show()

    ### test RandomBatchSamplingII
    dataloader = data.DataLoader(dset, batch_size=20)
    for batch in dataloader:
        print(len(batch))
        cleans, rains = batch
        cleans = cleans.numpy()
        rains = rains.numpy()
        print(cleans.shape)
        ## show
        plt.subplot(2, 2, 1)
        plt.imshow(cleans[0].transpose(1, 2, 0))
        plt.subplot(2, 2, 2)
        plt.imshow(rains[0].transpose(1, 2, 0))
        plt.subplot(2, 2, 3)
        plt.imshow(cleans[1].transpose(1, 2, 0))
        plt.subplot(2, 2, 4)
        plt.imshow(rains[1].transpose(1, 2, 0))
        plt.show()
    

    ### test RandomBatch sampling
    print('=== test random patch sampling')
    root_dir = '/home/wran/Public/datasets/derain/forTPAFI/Rain100L64-80/RawData/Rain100L-80-multi/'
    patchset = RandomPatch(root_dir=root_dir)
    print(len(patchset))
    patchLoader = data.DataLoader(patchset, batch_size=128)
    for patches in patchLoader:
        cleans, rains = patches
        cleans = cleans.numpy()
        rains = rains.numpy()
        print(cleans.shape)
        ## show
        plt.subplot(2, 2, 1)
        plt.imshow(cleans[0].transpose(1, 2, 0))
        plt.subplot(2, 2, 2)
        plt.imshow(rains[0].transpose(1, 2, 0))
        plt.subplot(2, 2, 3)
        plt.imshow(cleans[1].transpose(1, 2, 0))
        plt.subplot(2, 2, 4)
        plt.imshow(rains[1].transpose(1, 2, 0))
        plt.show()



