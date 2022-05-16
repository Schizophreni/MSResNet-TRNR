"""
files for N-freqs K-shot meta learning
"""
from ImgLIPDset import ImgLIPDset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
import sys
sys.path.append('..')
from utils.clean2noise import clean2rain


class ImgLIPNfreqsKshot:
    """
    N-freqs K-shot meta learning
    """
    def __init__(self, root_dir, batch_size, n_freqs, k_shot, k_query, patch_size, imchannel=3, sigmas='5, 10, 15, 20'):
        """
        @params:
        root_dir: root_dir for image path (pass to ImgLIPDset)
        batch_size: batch_size for meta learning
        n_freqs: n_way in classification problem
        k_shot: each freqs k_shot images
        k_query: same as k_shot but for query set
        patch_size: image patch_size
        imchannel: image channels
        """
        imgDset_spt = ImgLIPDset(root_dir, mode='train')
        imgDset_qry = ImgLIPDset(root_dir, mode='test')
        dict_spt = {} ## (label1: img1, img2, img3, ...)
        dict_qry = {}
        for (img, label) in imgDset_spt:
            if label in dict_spt.keys():
                dict_spt[label].append(img)
            else:
                dict_spt[label] = [img]
        for (img, label) in imgDset_qry:
            if label in dict_qry.keys():
                dict_qry[label].append(img)
            else:
                dict_qry[label] = [img]
        
        self.dict_spt = dict_spt
        self.dict_qry = dict_qry

        self.batch_size = batch_size
        self.n_freqs = n_freqs
        self.k_shot = k_shot
        self.k_query = k_query
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.imchannel = imchannel
        self.sigmas = tuple(int(item) for item in sigmas.split(','))
        self.transformation = transforms.Compose(
            [lambda x: Image.open(x),
            # lambda x: np.expand_dims(x, -1), ## for gray-scale
            lambda x: np.transpose(x, (2, 0, 1)), ## change to (c, h, w)
            lambda x: x/255.]
        )
        
        ## save pointer of current read batch
        self.indexes = {'train': 0, 'test': 0} ## record batch index in cache
        self.datasets = {'train': self.dict_spt, 'test': self.dict_qry}
        print('DB: train', len(imgDset_spt), self.patch_size, 'test', len(imgDset_qry), self.patch_size)
        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train'], mode='train'), 
                                'test': self.load_data_cache(self.datasets['test'], mode='test')}
        self.total_train_samples = len(imgDset_spt)
        self.total_test_samples = len(imgDset_qry)
        
    def load_data_cache(self, dset, mode='train'):
        """
        store several batches for N-freqs learning
        :param dset: dict of (label: img1, img2, ...)
        :return: a list with [spt_set_x, spt_set_y, qry_set_x, qry_set_y]
        :store 10 batches data for meta learning training or testing 
        """
        setsz = self.k_shot*self.n_freqs
        qyrsz = self.k_query*self.n_freqs
        data_cache = []
        
        for episode in range(20):
            spts_x, spts_y, qrys_x, qrys_y = [], [], [], []
            for batch in range(self.batch_size):
                spt_y = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_y = np.zeros((qyrsz, self.imchannel, *self.patch_size)) ## one batch clean data
                spt_x = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_x = np.zeros((qyrsz, self.imchannel, *self.patch_size)) ## one batch clean data
                ### use another strategy to sample support and query data
                selected_freqs = np.random.choice(len(dset), self.n_freqs+1, False)  ## choose n_freqs*2, n_freqs for support, n_freqs for query
                # print(selected_freqs)
                for label, freq in enumerate(selected_freqs[:-1]):
                    imgs_path = dset[freq]
                    selected_imgs = np.random.choice(imgs_path, self.k_shot+self.k_query, False)
                    selected_imgs = list(selected_imgs)
                    # print('selected_imgs: ', selected_imgs[0])
                    assert self.k_shot+self.k_query<=len(imgs_path), 'not enough images for division'
                    for index, img in enumerate(selected_imgs[:self.k_shot]):
                        spt_y[label*self.k_shot+index, ...] = self.transformation(img)
                        # spt_x[label*self.k_shot+index, ...] = self.transformation(img.replace('clean', 'rain'))
                    if label < (self.n_freqs-1):
                        for index, img in enumerate(selected_imgs[self.k_shot:]):
                            qry_y[label*self.k_query+index, ...] = self.transformation(img)
                            # qry_x[label*self.k_query+index, ...] = self.transformation(img.replace('clean', 'rain'))
                remain_imgs = np.random.choice(dset[selected_freqs[-1]], self.k_query, False)
                label = self.n_freqs-1
                for index, img in enumerate(remain_imgs):
                    qry_y[label*self.k_query+index, ...] = self.transformation(img)
                    # qry_x[label*self.k_query+index, ...] = self.transformation(img.replace('clean', 'rain'))
                

                # spt_x[:self.n_freqs//2, ...] = clean2noisy(spt_y[:self.n_freqs//2, ...], self.sigmas)
                # qry_x[:self.n_freqs//2, ...] = clean2noisy(qry_y[:self.n_freqs//2, ...], self.sigmas)
                # rain_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain100-new/train/Rainstreaks'
                # spt_x[self.n_freqs//2:, ...] = clean2rain(spt_y[self.n_freqs//2:, ...], rain_dir, patch_size=self.patch_size)
                # qry_x[self.n_freqs//2:, ...] = clean2rain(qry_y[self.n_freqs//2:, ...], rain_dir, patch_size=self.patch_size)
                # spt_x = spt_x + np.random.randn(*spt_x.shape)*5/255.0
                ## print(np.max(spt_x), np.min(spt_x))
                # random_noise = np.random.randn(*spt_x.shape)*5/255.0
                # spt_x = spt_x+random_noise
                # spt_y = spt_y + random_noise

                spts_x.append(spt_x)
                spts_y.append(spt_y)
                qrys_x.append(qry_x)
                qrys_y.append(qry_y)
            
            spts_x = np.array(spts_x).astype('f4') ## [b, setsz, c, h, w]
            spts_y = np.array(spts_y).astype('f4')
            qrys_x = np.array(qrys_x).astype('f4')
            qrys_y = np.array(qrys_y).astype('f4')
            data_cache.append([spts_x, spts_y, qrys_x, qrys_y])
        return data_cache
    
    def next(self, mode='train'):
        """
        Gets next batch from the dataset with mode
        :param mode: train or test
        """
        if self.indexes[mode]>=len(self.datasets_cache[mode]):
            ## need new cache
            self.indexes[mode] = 0  ## re-initialize index
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch

def clean2noisy(x, sigmas):
    """
    generate noisy image from clean image
    :param x: shape of [setsz, c, h, w]
    :sigmas: tuple of sigma like (5, 10, 15, 20)
    :return: noisy + x
    """
    tmp = np.zeros_like(x)
    
    for idx in range(x.shape[0]):
        sigma = np.random.choice(sigmas)
        tmp[idx, ...] = np.random.randn(x.shape[1], x.shape[2], x.shape[3])*(sigma/255.0)
    return tmp+x


def main():
    import time
    import torch
    import matplotlib.pyplot as plt
    root_dir = '/home/wran/Documents/deepModels/MetaLIP/data/ProcessData/BSDdenoise'
    bsz = 4
    n_freqs = 5
    k_shot = 1
    k_query = 1
    db = ImgLIPNfreqsKshot(root_dir=root_dir, batch_size=bsz, n_freqs=n_freqs, k_shot=k_shot, k_query=k_query, patch_size=50)
    spts_x, spts_y, qrys_x, qrys_y = db.next(mode='train')
    print(spts_x.shape, spts_y.shape)
    print(qrys_x.shape, qrys_y.shape)
    for j in range(bsz):
        for i in range(n_freqs):
            plt.figure(1)
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)
            plt.sca(ax1)
            plt.imshow(spts_x[j, i].transpose(1, 2, 0))
            plt.sca(ax2)
            plt.imshow(spts_y[j, i].transpose(1, 2, 0))
            plt.sca(ax3)
            plt.imshow(qrys_x[j, i].transpose(1, 2, 0))
            plt.sca(ax4)
            plt.imshow(qrys_y[j, i].transpose(1, 2, 0))
            plt.show()


if __name__ == '__main__':
    main()
        



        