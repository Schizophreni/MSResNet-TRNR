"""
files for N-freqs K-shot meta learning
"""
from ImgLIPDset import ImgLIPDset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
import sys
import torchvision.transforms.functional as tvF
import random
import torch
import matplotlib.pyplot as plt


class ImgLIPNfreqsKshot:
    """
    N-freqs K-shot meta learning
    """
    def __init__(self, root_dir, batch_size, n_freqs, k_shot, k_query, patch_size, imchannel=3):
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
        self.sigma_interval = (0, 55)  ### noise sigma interval

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
        self.transformation = transforms.Compose(
            [lambda x: Image.open(x),
            lambda x: np.transpose(x, (2, 0, 1)), ## change to (c, h, w)
            lambda x: x/255.0,
            ]
        )
        
        ## save pointer of current read batch
        self.indexes = {'train': 0, 'test': 0} ## record batch index in cache
        self.datasets = {'train': self.dict_spt, 'test': self.dict_qry}
        print('DB: train', len(imgDset_spt), self.patch_size, 'test', len(imgDset_qry), self.patch_size)
        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train'], mode='train'), # 'test': None}
                                 'test': self.load_data_cache(self.datasets['test'], mode='test')}
        self.total_train_samples = len(imgDset_spt)
    
    def init_cache(self):
        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train'], mode='train'), # 'test': None}
                                 'test': self.load_data_cache(self.datasets['test'], mode='test')}

    def _tuple_trans(self, clean_img):
        noise_sigma = np.random.uniform(*self.sigma_interval)/255.0

        clean_img = Image.open(clean_img).convert('L') ## this accords with cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) but not with cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        clean_img = np.array(clean_img)
        
        data_aug = np.random.choice(7)

        if data_aug == 0:
            clean_img =  np.flipud(np.rot90(clean_img))
        elif data_aug == 1:
            clean_img = np.flipud(clean_img)
        elif data_aug == 2:
            clean_img = np.rot90(clean_img, k=3)
        elif data_aug == 3:
            clean_img = np.flipud(np.rot90(clean_img, k=2))
        elif data_aug == 4:
            clean_img = np.rot90(clean_img)
        elif data_aug == 5:
            clean_img = np.rot90(clean_img, k=2)
        elif data_aug == 6:
            clean_img = np.flipud(np.rot90(clean_img, k=3))

        clean_img = clean_img[..., None]
        clean_img = np.transpose(clean_img, (2, 0, 1))
        clean_img = clean_img/255.0
        ### add noise to clean images
        noise_img = np.random.randn(*clean_img.shape)*noise_sigma + clean_img
        return clean_img, noise_img, noise_sigma

    def load_data_cache(self, dset, mode='train'):
        """
        store several batches for N-freqs learning
        :param dset: dict of (label: img1, img2, ...)
        :return: a list with [spt_set_x, spt_set_y, qry_set_x, qry_set_y]
        :store 10 batches data for meta learning training or testing 
        """
        setsz = self.k_shot*self.n_freqs
        qrysz = self.k_query*self.n_freqs
        data_cache = []
        
        for episode in range(20):
            spts_x, spts_y, qrys_x, qrys_y = [], [], [], []
            spts_sigma, qrys_sigma = [], []
            selected_freqs_each_eposide = np.random.choice(len(dset), (self.n_freqs+1)*self.batch_size, replace=False)
            for batch in range(self.batch_size): ## batch_size -> task num each episode
                spt_y = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_y = np.zeros((qrysz, self.imchannel, *self.patch_size)) ## one batch clean data
                spt_x = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_x = np.zeros((qrysz, self.imchannel, *self.patch_size)) ## one batch clean data
                spt_sigma = np.empty((setsz, ))
                qry_sigma = np.empty((qrysz, ))
                selected_freqs = selected_freqs_each_eposide[batch*(self.n_freqs+1):(batch+1)*(self.n_freqs+1)]
                ### use another strategy to sample support and query data
                for label, freq in enumerate(selected_freqs[:-1]):
                    imgs_path = dset[freq]
                    selected_imgs = np.random.choice(imgs_path, self.k_shot+self.k_query, False)
                    selected_imgs = list(selected_imgs)

                    assert self.k_shot+self.k_query<=len(imgs_path), 'not enough images for division'
                    for index, img in enumerate(selected_imgs[:self.k_shot]):
                        spt_y[label*self.k_shot+index, ...], spt_x[label*self.k_shot+index, ...], spt_sigma[label*self.k_shot+index] = self._tuple_trans(clean_img=img)
                        # spt_x[label*self.k_shot+index, ...] = self.transformation(img.replace('clean', 'rain'))
                    if label < self.n_freqs-1:
                        for index, img in enumerate(selected_imgs[self.k_shot:]):
                            qry_y[label*self.k_query+index, ...],qry_x[label*self.k_query+index, ...], qry_sigma[label*self.k_query+index] = self._tuple_trans(clean_img=img)
                                # qry_x[label*self.k_query+index, ...] = self.transformation(img.replace('clean', 'rain'))
                
                remain_imgs = np.random.choice(dset[selected_freqs[-1]], self.k_query, False)
                label = self.n_freqs-1
                for index, img in enumerate(remain_imgs):
                    qry_y[label*self.k_query+index, ...],qry_x[label*self.k_query+index, ...], qry_sigma[label*self.k_query+index]= self._tuple_trans(clean_img=img)
                    # qry_x[label*self.k_query+index, ...] = self.transformation(img.replace('clean', 'rain'))
                
                spts_x.append(spt_x)
                spts_y.append(spt_y)
                qrys_x.append(qry_x)
                qrys_y.append(qry_y)
                spts_sigma.append(spt_sigma)
                qrys_sigma.append(qry_sigma)
            
            spts_x = np.array(spts_x).astype('f4') ## [b, setsz, c, h, w]
            spts_y = np.array(spts_y).astype('f4')
            qrys_x = np.array(qrys_x).astype('f4')
            qrys_y = np.array(qrys_y).astype('f4')
            spts_sigma = np.array(spts_sigma).astype('f4')
            qrys_sigma = np.array(qrys_sigma).astype('f4')
            data_cache.append([spts_x, spts_y, qrys_x, qrys_y, spts_sigma, qrys_sigma])
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


def main():
    import time
    import torch
    import matplotlib.pyplot as plt
    root_dir = '/home/wran/Public/datasets/denoise/forTPAFI/BSD500-150/ProcessData'
    bsz = 4
    n_freqs = 5
    k_shot = 1
    k_query = 1
    db = ImgLIPNfreqsKshot(root_dir=root_dir, batch_size=bsz, n_freqs=n_freqs, k_shot=k_shot, k_query=k_query, patch_size=64)
    spts_x, spts_y, qrys_x, qrys_y, spts_sigma, qrys_sigma = db.next(mode='test')
    print('support set input, label and sigma: ', spts_x.shape, spts_y.shape, spts_sigma.shape)
    print('query set input, label and sigma: ', qrys_x.shape, qrys_y.shape, qrys_sigma.shape)
    print(spts_x[0, 0, :, 0, 0])
    for j in range(bsz):
        for i in range(n_freqs):
            plt.figure(1)
            ax1 = plt.subplot(2, 2, 1)
            ax2 = plt.subplot(2, 2, 2)
            ax3 = plt.subplot(2, 2, 3)
            ax4 = plt.subplot(2, 2, 4)
            print(spts_sigma[j, i], qrys_sigma[j, i])
            plt.sca(ax1)
            plt.imshow(spts_x[j, i, :, :, :].transpose(1, 2, 0))
            plt.title(spts_sigma[j, i])
            plt.sca(ax2)
            plt.imshow(spts_y[j, i, :, :, :].transpose(1, 2, 0))
            plt.sca(ax3)
            plt.imshow(qrys_x[j, i, :, ...].transpose(1, 2, 0))
            plt.title(qrys_sigma[j, i])
            plt.sca(ax4)
            plt.imshow(qrys_y[j, i].transpose(1, 2, 0))
            plt.show()


if __name__ == '__main__':
    main()
        



        
