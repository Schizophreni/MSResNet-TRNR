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
from PAsampling import traverse_folder, gen_sampling_sequence
import glob
import cv2


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
        # self.root_dir = root_dir
        # imgDset_spt = ImgLIPDset(root_dir, mode='train')
        # imgDset_qry = ImgLIPDset(root_dir, mode='test')

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
            transforms.RandomCrop(self.patch_size), ## randomly crop
            # lambda x: np.expand_dims(x, -1), ## for gray-scale
            lambda x: np.transpose(x, (2, 0, 1)), ## change to (c, h, w)
            lambda x: x/255.]
        )

        self.use_trainset_index = 0 ## initially use first trainset

        ## generate sampling sequence (class & sample important sampling)
        self.init_hard_task()
        self.initialize_dataset(root_dir)
        self.important_sampling(mode='train') ## gennerate sampling sequences
        self.important_sampling(mode='test') ## generate sampling sequences

        
        ## save pointer of current read batch
        self.indexes = {'train': 0, 'test': 0} ## record batch index in cache
        
        self.datasets_cache = {'train': self.load_data_cache(mode='train'), # 'test': None}
                                 'test': self.load_data_cache(mode='test')}
    def _parse_root_dir(self, root_dir):
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
        if 'train' in os.listdir(root_dir):
            self.root_dirs = [root_dir]
        else:
            sub_sets = os.listdir(root_dir)
            self.root_dirs = [os.path.join(root_dir, item) for item in sub_sets]
        self.use_trainset_num = len(self.root_dirs) ## how many trainsets are used 
        print(self.root_dirs)
    
    def _generate_dict_from_dset(self, dset):
        img_dict = dict()
        for img, label in dset:
            if label in img_dict:
                img_dict[label].append(img)
            else:
                img_dict[label] = [img]
        return img_dict
    
    def initialize_dataset(self, root_dir):
        """
        generate directories for support and query data
        """
        self._parse_root_dir(root_dir) ## generate data dirs
        self.dict_spts = []
        self.dict_qry = None
        self.total_train_samples = 0
        for data_dir in self.root_dirs:
            ### generate data for support sets
            spt_set = ImgLIPDset(data_dir, mode='train')
            self.total_train_samples += len(spt_set)
            self.dict_spts.append(self._generate_dict_from_dset(spt_set))
            ### generate data for query set
            print('DB: train', len(spt_set), self.patch_size)
            if 'cp-0' in os.listdir(os.path.join(data_dir, 'test/clean')): ## Found datasets
                qry_set = ImgLIPDset(data_dir, mode='test')
                self.dict_qry = self._generate_dict_from_dset(qry_set)
                print('DB: test', len(qry_set), self.patch_size)


    def _tuple_trans(self, clean_img, noise_img, p=0.0):
        # clean_img = Image.open(clean_img)
        clean_img = cv2.imread(clean_img)[..., ::-1]
        # sigmas = range(0, 25)
        ratio = np.random.rand()
        # i, j, h, w = transforms.RandomCrop.get_params(clean_img, self.patch_size)
        # clean_img = tvF.crop(clean_img, i, j, h, w)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        clean_img = clean_img/255.0
        noise_img = cv2.imread(noise_img)[..., ::-1]
        # noise_img = tvF.crop(noise_img, i, j, h, w)
        noise_img = np.transpose(noise_img, (2, 0, 1))
        noise_img = noise_img/255.0
        if ratio < p: ## change -1 to p for noise examples
            """
            noise is the same as clean
            """
            # sigma = np.random.choice(sigmas)
            # np.random.seed(seed=random.choice(range(99999)))
            # noise_img=clean_img+np.random.randn(*clean_img.shape)*sigma/255.0
            if np.random.rand() < 0.5:
                ## flip horizontal
                clean_img = clean_img[:, :, ::-1]
                noise_img = noise_img[:, :, ::-1]
            else:
                clean_img = clean_img[:, ::-1, :]
                noise_img = noise_img[:, ::-1, :]
        return clean_img, noise_img
    
    def important_sampling(self, mode='train'):
        """
        Generate important sampling sequence for N-frequency-K-shot learning problem
        Notice that if mode='train': there maybe multi datasets
        Function: generate sampling sequence, generate sampling dataset 
        """
        if mode=='train':
            root_dir = self.root_dirs[self.use_trainset_index]
            train_class_samples = traverse_folder(folder_path=os.path.join(root_dir,'train/clean'))
            self.train_sampling_seq = gen_sampling_sequence(class_samples=train_class_samples, hard_task_indexes=self.hard_tasks, 
                                                            class_per_sampling=self.n_freqs+1)
            random.shuffle(self.train_sampling_seq) ## randomly shuffle

            ## test
        elif mode == 'test':
            for data_dir in self.root_dirs:
                if 'cp-0' in os.listdir(os.path.join(data_dir, 'test/clean')):
                    test_class_samples = traverse_folder(folder_path=os.path.join(data_dir, 'test/clean'))
                    self.test_sampling_seq = gen_sampling_sequence(class_samples=test_class_samples, hard_task_indexes=[], 
                                                        class_per_sampling=self.n_freqs+1)
            random.shuffle(self.test_sampling_seq) ## randomly shuffle
            
    
    def init_hard_task(self):
        ## initialize hard tasks
        pass
        ## 
        self.hard_tasks = []

    def load_data_cache(self, mode='train'):
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
            # selected_freqs_each_eposide = np.random.choice(len(dset), (self.n_freqs+1)*self.batch_size, replace=False)
            for batch in range(self.batch_size):
                spt_y = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_y = np.zeros((qyrsz, self.imchannel, *self.patch_size)) ## one batch clean data
                spt_x = np.zeros((setsz, self.imchannel, *self.patch_size)) ## one batch clean data
                qry_x = np.zeros((qyrsz, self.imchannel, *self.patch_size)) ## one batch clean data
                # selected_freqs = selected_freqs_each_eposide[batch*(self.n_freqs+1):(batch+1)*(self.n_freqs+1)]
                ### use another strategy to sample support and query data
                # selected_freqs = np.random.choice(len(dset), self.n_freqs+1, False)  ## choose n_freqs*2, n_freqs for support, n_freqs for query
                # print(selected_freqs)
                if mode == 'train':
                    if len(self.train_sampling_seq) == 0:
                        self.use_trainset_index = (self.use_trainset_index+1) % self.use_trainset_num ## automatic choose trainset
                        self.important_sampling(mode='train')
                    selected_freqs = self.train_sampling_seq.pop()
                    dset = self.dict_spts[self.use_trainset_index]
                elif mode == 'test':
                    if len(self.test_sampling_seq) == 0:
                        self.important_sampling(mode='test')
                    # print('max in test: ', np.max(np.array(self.test_sampling_seq)))
                    selected_freqs = self.test_sampling_seq.pop()
                    dset = self.dict_qry
                else:
                    raise Exception('not this way')
 
                for label, freq in enumerate(selected_freqs[:-1]):
                    imgs_path = dset[freq]
                    selected_imgs = np.random.choice(imgs_path, self.k_shot+self.k_query, False)
                    selected_imgs = list(selected_imgs)

                    assert self.k_shot+self.k_query<=len(imgs_path), 'not enough images for division'
                    # seed = np.random.randint(999999)
                    # prob = np.random.rand()*0.3 + 0.1
                    # np.random.seed(seed) ## set seed
                    for index, img in enumerate(selected_imgs[:self.k_shot]):
                        spt_y[label*self.k_shot+index, ...], spt_x[label*self.k_shot+index, ...]  = self._tuple_trans(clean_img=img, noise_img=img.replace('clean', 'rain'),p=0.5)
                    if label < self.n_freqs-1:
                    # np.random.seed(seed) ## set same seed for query set
                        for index, img in enumerate(selected_imgs[self.k_shot:]):
                            qry_y[label*self.k_query+index, ...],qry_x[label*self.k_query+index, ...] = self._tuple_trans(clean_img=img, noise_img=img.replace('clean', 'rain'),p=0.5)
                
                remain_imgs = np.random.choice(dset[selected_freqs[-1]], self.k_query, False)
                label = self.n_freqs-1
                for index, img in enumerate(remain_imgs):
                    qry_y[label*self.k_query+index, ...],qry_x[label*self.k_query+index, ...]= self._tuple_trans(clean_img=img, noise_img=img.replace('clean', 'rain'), p=0.5)
                    # qry_x[label*self.k_query+index, ...] = self.transformation(img.replace('clean', 'rain'))
                
                # spt_x = clean2noisy(spt_y, self.sigmas)
                # qry_x = clean2noisy(qry_y, self.sigmas)
                # print(np.max(qry_y))
                # rain_dir = r'E:\Dustbin\data\RawData\Rain100L-small\train\Rainstreaks'
                # rain_dir = '../data/RawData/Rain100-new/train/Rainstreaks'
                # spt_x = clean2rain(spt_y, rain_dir, patch_size=self.patch_size)
                # qry_x = clean2rain(qry_y, rain_dir, patch_size=self.patch_size)
                # spt_x = spt_x + np.random.randn(*spt_x.shape)*5/255.0
                ## print(np.max(spt_x), np.min(spt_x))
                # random_noise = np.random.randn(*spt_x.shape)*5/255.0
                # spt_x = spt_x+random_noise
                # spt_y = spt_y + random_noise

                ## randomly shuffle strategy
                '''
                spt_rains = spt_x - spt_y
                qry_rains = qry_x - qry_y
                # print(type(spt_x), type(spt_y), type(qry_x), type(qry_y))
                mask = np.random.randint(0, 2, setsz).reshape((setsz, 1, 1, 1))
                
                prob = np.random.rand()*0.3
                spt_x = spt_y + spt_rains*(1-prob) + prob*(spt_rains*mask + (1-mask)*qry_rains)
                # qry_x = qry_y + spt_rains*(1-mask) + mask*qry_rains
                '''
                
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
            self.datasets_cache[mode] = self.load_data_cache(mode=mode)
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
    root_dir = '../MetaLIP/data/ProcessData/Rain100L-50-multi-merge'
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
        



        
