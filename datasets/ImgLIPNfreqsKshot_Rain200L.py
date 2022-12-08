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
import torchvision.transforms.functional as tvF
import random
import torch


class ImgLIPNfreqsKshot:
    """
    N-freqs K-shot meta learning
    """

    def __init__(self, root_dir, batch_size, n_freqs, k_shot, k_query, patch_size, imchannel=3, sigmas='5, 10, 15, 20',
                 dataset_name='Rain200L'):
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
        dict_spt = {}  ## (label1: img1, img2, img3, ...)
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
             lambda x: np.transpose(x, (2, 0, 1)),  ## change to (c, h, w)
             lambda x: x / 255.0,
             ]
        )

        if dataset_name in ['Rain200L', 'Rain200H']:
            self.repeat_times = 6
        elif dataset_name == 'Rain14000':
            self.repeat_times = 14
        else:
            self.repeat_times = 1

        ## save pointer of current read batch
        self.indexes = {'train': 0, 'test': 0}  ## record batch index in cache
        self.datasets = {'train': self.dict_spt, 'test': self.dict_qry}
        print('DB: train', len(imgDset_spt), self.patch_size, 'test', len(imgDset_qry), self.patch_size)
        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train'], mode='train'),  # 'test': None}
                               'test': self.load_data_cache(self.datasets['test'], mode='test')}
        self.total_train_samples = len(imgDset_spt)

    def init_cache(self):
        self.datasets_cache = {'train': self.load_data_cache(self.datasets['train'], mode='train'),  # 'test': None}
                               'test': self.load_data_cache(self.datasets['test'], mode='test')}

    def parse_root_dir(self):
        """
        Extract multi training dataset from root_dir, for an example
        ++++++++++++++
        Rain100L-multi:
                      |- Rain100L-50-stride-0
                                            |- cp-0
                                            |- cp- 1
                                            | ...
                      |- Rain100L-50-stride-12
                                            |- cp-0
                                            |- cp-1
                                            |...
        ++++++++++++++
        """
        self.root_dirs = []
        if 'cp-0' in os.listdir(self.root_dir)[0]:
            self.root_dirs = [self.root_dir]
        else:
            self.root_dirs = os.listdir(self.root_dir)

    def _tuple_trans(self, clean_img, noise_img):
        clean_img = Image.open(clean_img)# .convert('L') ## this accords with cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) but not with cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        noise_img = Image.open(noise_img)
        data_aug = np.random.choice(4)
        if data_aug == 1:
            clean_img = np.array(clean_img)[::-1, ::-1, :]
            # clean_img = np.array(clean_img)
            noise_img = np.array(noise_img)[::-1, ::-1, :]
        elif data_aug == 2:
            clean_img = np.array(clean_img)[::-1, :, ...]
            noise_img = np.array(noise_img)[::-1, :, ...]
        elif data_aug == 3:
            clean_img = np.array(clean_img)[:, ::-1, ...]
            noise_img = np.array(noise_img)[:, ::-1, ...]
        else:
            clean_img = np.array(clean_img)
            noise_img = np.array(noise_img)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        clean_img = clean_img/255.0
        noise_img = np.transpose(noise_img, (2, 0, 1))
        noise_img = noise_img/255.0
        return clean_img, noise_img

    def load_data_cache(self, dset, mode='train'):
        """
        store several batches for N-freqs learning
        :param dset: dict of (label: img1, img2, ...)
        :return: a list with [spt_set_x, spt_set_y, qry_set_x, qry_set_y]
        :store 10 batches data for meta learning training or testing
        """
        setsz = self.k_shot * self.n_freqs
        qyrsz = self.k_query * self.n_freqs
        data_cache = []

        for episode in range(20):
            spts_x, spts_y, qrys_x, qrys_y = [], [], [], []
            selected_freqs_each_eposide = np.random.choice(len(dset), (self.n_freqs + 1) * self.batch_size,
                                                           replace=False)
            for batch in range(self.batch_size):
                spt_y = np.zeros((setsz, self.imchannel, *self.patch_size))  ## one batch clean data
                qry_y = np.zeros((qyrsz, self.imchannel, *self.patch_size))  ## one batch clean data
                spt_x = np.zeros((setsz, self.imchannel, *self.patch_size))  ## one batch clean data
                qry_x = np.zeros((qyrsz, self.imchannel, *self.patch_size))  ## one batch clean data
                selected_freqs = selected_freqs_each_eposide[
                                 batch * (self.n_freqs + 1):(batch + 1) * (self.n_freqs + 1)]
                ### use another strategy to sample support and query data
                # selected_freqs = np.random.choice(len(dset), self.n_freqs+1, False)  ## choose n_freqs*2, n_freqs for support, n_freqs for query
                # print(selected_freqs)
                # aug_idx = np.random.choice(self.repeat_times)
                # print(aug_idx)
                for label, freq in enumerate(selected_freqs[:-1]):
                    imgs_path = dset[freq]
                    selected_imgs = np.random.choice(imgs_path, self.k_shot + self.k_query, False)
                    selected_imgs = list(selected_imgs)

                    assert self.k_shot + self.k_query <= len(imgs_path), 'not enough images for division'
                    for index, img in enumerate(selected_imgs[:self.k_shot]):
                        aug_idx = np.random.choice(self.repeat_times)
                        noise_img = img.replace('clean', 'rain').replace('.png', 'x{}.png'.format(aug_idx+1))
                        spt_y[label * self.k_shot + index, ...], spt_x[
                            label * self.k_shot + index, ...] = self._tuple_trans(clean_img=img, noise_img=noise_img)

                    if label < self.n_freqs - 1:
                        for index, img in enumerate(selected_imgs[self.k_shot:]):
                            aug_idx = np.random.choice(self.repeat_times)
                            noise_img = img.replace('clean', 'rain').replace('.png', 'x{}.png'.format(aug_idx + 1))
                            qry_y[label * self.k_query + index, ...], qry_x[
                                label * self.k_query + index, ...] = self._tuple_trans(clean_img=img,
                                                                                       noise_img=noise_img)

                remain_imgs = np.random.choice(dset[selected_freqs[-1]], self.k_query, False)
                label = self.n_freqs - 1
                for index, img in enumerate(remain_imgs):
                    aug_idx = np.random.choice(self.repeat_times)
                    noise_img = img.replace('clean', 'rain').replace('.png', 'x{}.png'.format(aug_idx + 1))
                    qry_y[label * self.k_query + index, ...], qry_x[
                        label * self.k_query + index, ...] = self._tuple_trans(clean_img=img,
                                                                               noise_img=noise_img)

                spts_x.append(spt_x)
                spts_y.append(spt_y)
                qrys_x.append(qry_x)
                qrys_y.append(qry_y)

            spts_x = np.array(spts_x).astype('f4')  ## [b, setsz, c, h, w]
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
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            ## need new cache
            self.indexes[mode] = 0  ## re-initialize index
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1
        return next_batch
