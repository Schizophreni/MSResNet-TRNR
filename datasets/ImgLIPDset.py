"""
meta learning based low-level img process datasets
return images path and labels (label means frequency)
"""
import torch.utils.data as data
import os
from PIL import Image
import numpy as np

class ImgLIPDset(data.Dataset):
    def __init__(self, root_dir, mode='train'):
        ## assert mode in ['train', 'test'], 'Invalid mode: {}'.format(mode)
        self.root_dir = root_dir
        self.all_items = traverse_freqs(root_dir, mode) ## item: [img_name, root_dir]
        self.idx_freqs = index_freqs(self.all_items)  ## use idx_freqs to index different freqs
        ## print(self.idx_freqs)
    
    def __getitem__(self, index):
        item = self.all_items[index]
        img = os.path.join(item[1], item[0])
        freq = self.idx_freqs[item[1].strip('/').split('/')[-1]]
        return img, freq  ## note that img is image path, freq is n-th frequencies
    
    def __len__(self):
        return len(self.all_items)
    

def traverse_freqs(root_dir, mode='train', token='clean'): ## 'clean' for derain, '' for denoise
    ret = []
    dset_dir = os.path.join(root_dir, mode, token)
    # print(dset_dir)
    # print('dset dir: ', dset_dir)
    for (root, dirs, files) in os.walk(dset_dir):
        for f in files:
            if (f.endswith('png')): ## means file is image, dirs = []
                ret.append([f, root])
                ## print(root, f)
    print('== Found %d items for %s' % (len(ret), mode))
    # print(sorted(ret, key=lambda x: x[1].split('\\')[-1]))
    return sorted(ret, key=lambda x: x[1])

def index_freqs(items):
    idx = {}
    for item in items:
        freq_name = item[1].strip('/').split('/')[-1]
        if freq_name not in idx:
            idx[freq_name] = len(idx) ## index freqs with filtering names
    print('== Found %d freqs' % len(idx))
    return idx

class TestDataset(data.Dataset):
    def __init__(self, dataset_config, **kwargs):
        self.type = kwargs['type']
        if self.type != 'noise':
            self.clean_dir, self.noise_dir = kwargs['clean_dir'], kwargs['noise_dir']
        else:
            self.clean_dir = kwargs['clean_dir']
            self.sigmas = kwargs['sigmas']
        self.name = kwargs['dataset']
        self.clean_format, self.noise_format, self.total_sampels, self.begin_idx = dataset_config[kwargs['dataset']]
    
    def __getitem__(self, index):
        if self.type != 'noise':
            if 'Rain1400' not in self.name:
                clean_img = os.path.join(self.clean_dir, self.clean_format.format(index+self.begin_idx))
                noise_img = os.path.join(self.noise_dir, self.noise_format.format(index+self.begin_idx))
                # print(clean_img, noise_img)
            else:
                img_idx = index // 14
                aug_idx = index % 14 + 1 
                clean_img = os.path.join(self.clean_dir, self.clean_format.format(img_idx+self.begin_idx))
                noise_img = os.path.join(self.noise_dir, self.noise_format.format(img_idx+self.begin_idx, aug_idx))
            clean_img, noise_img = Image.open(clean_img), Image.open(noise_img)
            clean_img = np.transpose(clean_img, (2, 0, 1))/255.0
            noise_img = np.transpose(noise_img, (2, 0, 1))/255.0

        else:
            clean_img = os.path.join(self.clean_dir, self.clean_format.format(index+self.begin_idx))
            clean_img = Image.open(clean_img)
            clean_img = np.transpose(clean_img, (2, 0, 1))/255.0
            sigma = np.random.choice(self.sigmas)
            noise_img = clean_img+np.random.randn(clean_img.shape)*(sigma/255.0)
        clean_img = clean_img.astype('f4')
        noise_img = noise_img.astype('f4')
        return clean_img, noise_img
    
    def __len__(self):
        return self.total_sampels

            
