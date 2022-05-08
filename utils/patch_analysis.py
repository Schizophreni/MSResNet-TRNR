import cv2
import numpy as np
import glob
import os
import random
from scipy.stats import wasserstein_distance as emd_dis ## 
import time

class PatchAnalysis:
    def __init__(self, patches_dir,flag=cv2.IMREAD_COLOR, threshold=1000):
        """
        :param patches_dir: patches dir
        """
        self.patches_dir = patches_dir
        self.flag = flag
        # self.patchInfo = [] ## patches information (patch1, info1, info2, ...)
        self.classifier = {} ## class1: [patch1, patch2, ...], class2: [patch1, patch2, ...]
        self.patchCenter = {} ## class1: centerP1, class2: centerP2, ...
        self.threshold = threshold
    
    def _obtain_Infos(self, patch):
        """
        obtain informations for image patches
        """
        pass

    def _calc_avg_info(self, patchInfo):
        """
        mapping: infos -> real number (for calculating similarity)
        """
        pass

    def _calc_histogram_distance(self, patch1, patch2):
        """
        calculate distance between two patches. use histogram absolute distance
        """
        dis = np.abs(patch1-patch2).sum()/256
        return dis
    
    def _calc_KL_divergence(self, source_patch, target_patch, eps=1e-5, channel_wise=True):
        """
        calculate KL divergence of two patch distribution KL(p||q)
        :param eps: used for non-zero division
        """
        if channel_wise:
            pb = np.histogram(source_patch[..., 0], bins=51, range=(0, 255))[0]
            pg = np.histogram(source_patch[..., 1], bins=51, range=(0, 255))[0]
            pr = np.histogram(source_patch[..., 2], bins=51, range=(0, 255))[0]

            qb = np.histogram(target_patch[..., 0], bins=51, range=(0, 255))[0]
            qg = np.histogram(target_patch[..., 1], bins=51, range=(0, 255))[0]
            qr = np.histogram(target_patch[..., 2], bins=51, range=(0, 255))[0]

            p = np.concatenate([pb, pg, pr], axis=0)
            q = np.concatenate([qb, qg, qr], axis=0)
            p = p+eps
            q = q+eps
            p = p/p.sum()
            q = q/q.sum()
            KL_divergence = (p*np.log(p/q)).sum()/np.log(2)
            return KL_divergence
        else:
            p = np.histogram(source_patch, bins=51, range=(0, 255))[0]
            q = np.histogram(target_patch, bins=51, range=(0, 255))[0]
            p = p+eps
            q = q+eps
            p = p / p.sum()
            q = q / q.sum()
            KL_divergence = (p*np.log(p/q)).sum()/np.log(2)
            return KL_divergence
    
    def _calc_JS_divergence(self, source_patch, target_patch, channel_wise=False, eps=1e-5):
        p = np.histogram(source_patch, bins=51, range=(0, 255))[0]
        q = np.histogram(target_patch, bins=51, range=(0, 255))[0]
        p = p+eps
        q = q+eps
        p = p / p.sum()
        q = q / q.sum()
        r = (p+q)/2
        KL_pr = (p*np.log(p/r)).sum()/np.log(2)
        KL_qr = (q*np.log(q/r)).sum()/np.log(2)
        return (KL_pr+KL_qr)/2.0
    
    def _calc_emd_distance(self, source_patch, target_patch, channel_wise=True):
        """
        calculate wasserstein distance between two patches
        :param source_patch: source patch compared with
        :param target_patch: target patch compare 
        """
        if channel_wise:
            pb = np.histogram(source_patch[..., 0], bins=51, range=(0, 255))[0]
            pg = np.histogram(source_patch[..., 1], bins=51, range=(0, 255))[0]
            pr = np.histogram(source_patch[..., 2], bins=51, range=(0, 255))[0]

            qb = np.histogram(target_patch[..., 0], bins=51, range=(0, 255))[0]
            qg = np.histogram(target_patch[..., 1], bins=51, range=(0, 255))[0]
            qr = np.histogram(target_patch[..., 2], bins=51, range=(0, 255))[0]

            # p = np.concatenate([pb, pg, pr], axis=0)
            # print('p.shape: ', p.shape)
            # q = np.concatenate([qb, qg, qr], axis=0)
            a = np.arange(pb.shape[0])
            emd_r = emd_dis(a, a, pb, qb)
            emd_g = emd_dis(a, a, pg, qg)
            emd_b = emd_dis(a, a, pr, qr)
            return (emd_b+emd_g+emd_r)/3.0
        else:
            p = np.histogram(source_patch, bins=51, range=(0, 255))[0]
            q = np.histogram(target_patch, bins=51, range=(0, 255))[0]

            p = p/p.sum()
            q = q/q.sum()
            a = np.arange(p.shape[0])
            return emd_dis(a, a, p, q)

    def _classify(self, patch, patch_path, metric='KL', isFull=False):
        """
        classify patch into a class.
        :param patch: np.array
        :param threshold: if distance between a patch and center is less than threshold, then patch is classified into this class
        """
        assert metric in ['KL', 'L1', 'MED'], 'metric {} not implemented yet'.format(metric)
        min_L1_Dis = np.prod(patch.shape)*10
        min_KL_Dis = 100
        current_class = len(self.patchCenter)
        current_patch = None
        if len(self.patchCenter) == 0:
            self.patchCenter[0] = patch ## update patch
            self.classifier[0] = [patch_path]
            return
    
        for patch_class, patch_center in self.patchCenter.items():
            if len(patch_center)>=1:
                L1_dis = self._calc_histogram_distance(patch_center, patch)
                KL_dis = self._calc_KL_divergence(patch_center, patch, channel_wise=True)
                if L1_dis < min_L1_Dis and KL_dis < min_KL_Dis:
                    # min_L1_Dis = L1_dis //  comment to only use KL distance
                    min_KL_Dis = KL_dis
                    current_class = patch_class
                    current_patch = patch_center
        if (min_KL_Dis < self.threshold and min_L1_Dis<5e6) or isFull: ## less than threshold
            if(min_KL_Dis < 1e-6):
                print(min_KL_Dis)
            self.classifier[current_class].append(patch_path)
            if not isFull:
                self._update_patch_center(current_patch, patch, current_class)
        else: ## new a class
            current_class = len(self.patchCenter)
            self.patchCenter[current_class] = patch
            self.classifier[current_class] = [patch_path]

    def _update_patch_center(self, current_patch, patch, current_class, momentum=0.05):
        """
        update patch center when new patch comes
        :param current_patch: patch center
        :param patch: new patch classified to this classs
        """
        self.patchCenter[current_class] = momentum*patch+(1-momentum)*current_patch
    
    def _traverse_patches(self, patches_dir, maxClass=300):
        """
        traverse folder to construct classifier
        """
        all_patches = glob.glob(os.path.join(patches_dir, "*.png"))  ## find all patches
        random.shuffle(all_patches) ## shuffle

        print('Find {} patches to classify'.format(len(all_patches)))
        isFull = False
        start = time.time()
        for iter, patch_path in enumerate(all_patches):
            if len(self.patchCenter) >= maxClass:
                isFull=True ## reach max classes
            patch = cv2.imread(patch_path, self.flag)
            self._classify(patch, patch_path, isFull=isFull)
            if (iter+1) % 500 == 0:
                end = time.time()
                print('===finished 500, time consuming: {} (seconds), total classes: {}'.format(end-start, len(self.patchCenter)))
                start = end
                
        print('===coarsed dispart finished')
        
        for c, imgs in self.classifier.items():
            if len(imgs)==1: ## reclass
                patch = cv2.imread(imgs[0], self.flag)
                self.patchCenter[c] = []
                self._classify(patch, imgs[0], isFull=True)
                self.classifier[c] = []


    def build_dataset(self, save_dir, maxClass=300):
        """
        build dataset from self.classfier
        """
        self._traverse_patches(self.patches_dir, maxClass=maxClass)
        # print(self.classifier)
        save_idx = 0
        threshold = 1
        for patch_class, patches in self.classifier.items():
            if len(patches) > threshold: ## at least 2 patches
                sub_dir = os.path.join(save_dir, 'cp-{}'.format(save_idx))
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                for idx, patch_path in enumerate(patches):
                    patch = cv2.imread(patch_path, self.flag)
                    save_img_path = os.path.join(sub_dir, 'patch-%d.png' % idx)
                    cv2.imwrite(save_img_path, patch)
                save_center_path = os.path.join(sub_dir, 'center.png')
                cv2.imwrite(save_center_path, self.patchCenter[patch_class])
                save_idx += 1

        print('=== Build dataset finished ===')
        for patch_class, patches_path in self.classifier.items():
            if len(patches_path) != 0:
                print('class: {}, total items: {}'.format(patch_class, len(patches_path)))

class PatchPairAnalysis:
    def __init__(self, patches_dir, flag, token, threshold=0.1):
        """
        patch analysis with image pairs
        use clean patches for classification
        :param patches_dir: patches dir
        :param token: indicates specific task
        """
        self.patches_dir = patches_dir
        self.flag = flag
        self.token=token
        self.threshold = threshold
        # self.patchInfo = [] ## patches information (patch1, info1, info2, ...)
        self.classifier = {} ## class1: [patch1,0, 
        self.patchCenter = {}
        self.attach_classifier = {}

    def _calc_avg_info(self, patchInfo):
        pass

    def _calc_histogram_distance(self, patch1, patch2):
        """
        calculate distance between two patches. use histogram absolute distance
        """
        dis = np.abs(patch1-patch2).sum()/256
        return dis
    
    def _calc_KL_divergence(self, source_patch, target_patch, eps=1e-5, channel_wise=True):
        """
        calculate KL divergence of two patch distribution KL(p||q)
        :param eps: used for non-zero division
        """
        if channel_wise:
            pb = np.histogram(source_patch[..., 0], bins=51, range=(0, 255))[0]
            pg = np.histogram(source_patch[..., 1], bins=51, range=(0, 255))[0]
            pr = np.histogram(source_patch[..., 2], bins=51, range=(0, 255))[0]

            qb = np.histogram(target_patch[..., 0], bins=51, range=(0, 255))[0]
            qg = np.histogram(target_patch[..., 1], bins=51, range=(0, 255))[0]
            qr = np.histogram(target_patch[..., 2], bins=51, range=(0, 255))[0]

            p = np.concatenate([pb, pg, pr], axis=0)
            q = np.concatenate([qb, qg, qr], axis=0)
            p = p+eps
            q = q+eps
            p = p/p.sum()
            q = q/q.sum()
            KL_divergence = (p*np.log(p/q)).sum()/np.log(2)
            return KL_divergence
        else:
            p = np.histogram(source_patch, bins=51, range=(0, 255))[0]
            q = np.histogram(target_patch, bins=51, range=(0, 255))[0]
            p = p+eps
            q = q+eps
            p = p / p.sum()
            q = q / q.sum()
            KL_divergence = (p*np.log(p/q)).sum()/np.log(2)
            return KL_divergence
    
    def _clean2noise_path(self, patch_path):
        """
        clean patch path to noise patch path
        """
        if self.token == 'rain':
            return patch_path.replace('clean', 'rain')
        else:
            raise NotImplementedError
    
    def _noise2clean_path(self, patch_path):
        """
        noise patch to clean patch path
        """
        if self.token == 'clean':
            patch_path = patch_path.replace('rain', 'clean') ## 'train' will be transposed to 'tclean'
            return patch_path.replace('tclean', 'train')
        else:
            raise NotImplementedError

    def _classify(self, patch, noise_patch, metric='KL', isFull=False):
        """
        classify patch into a class.
        :param patch: np.array
        :param threshold: if distance between a patch and center is less than threshold, then patch is classified into this class
        :param noise_path: correspond noisy patch 
        """
        assert metric in ['KL', 'L1'], 'metric {} not implemented yet'.format(metric)
        min_KL_Dis = 100
        current_class = len(self.patchCenter)
        current_patch = None

        if len(self.patchCenter) == 0:
            # deal with first patch when patchCenter and classifier is empty
            self.patchCenter[0] = patch ## update patch
            self.classifier[0] = [patch]
            self.attach_classifier[0] = [noise_patch]
            return
    
        for patch_class, patch_center in self.patchCenter.items():
            if len(patch_center) >= 1:
                # L1_dis = self._calc_histogram_distance(patch_center, patch)
                KL_dis = self._calc_KL_divergence(patch_center, patch, channel_wise=True)
                if KL_dis < min_KL_Dis:
                    min_KL_Dis = KL_dis
                    current_class = patch_class
                    current_patch = patch_center
        if min_KL_Dis < self.threshold or isFull: ## less than threshold
            self.classifier[current_class].append(patch)
            self.attach_classifier[current_class].append(noise_patch)
            if not isFull:
                self._update_patch_center(current_patch, patch, current_class)
        else: ## new a class
            current_class = len(self.patchCenter)
            self.patchCenter[current_class] = patch
            self.classifier[current_class] = [patch]
            self.attach_classifier[current_class] = [noise_patch]

    def _update_patch_center(self, current_patch, patch, current_class):
        """
        update patch center when new patch comes
        :param current_patch: current patch center
        :param patch: new patch classified to this classs
        """
        momentum = 0.05 ## can be adaptive
        self.patchCenter[current_class] = momentum*patch+(1-momentum)*current_patch
    
    def _traverse_patches(self, patches_dir, maxClass=1000, token='clean'):
        """
        traverse folder to construct classifier
        """
        all_patches = glob.glob(os.path.join(patches_dir, "*.png"))  ## find all patches
        random.shuffle(all_patches) ## shuffle

        print('Find {} patches to classify'.format(len(all_patches)))
        isFull = False
        start = time.time()
        for iter, patch_path in enumerate(all_patches):
            if len(self.patchCenter) >= maxClass:
                isFull=True ## reach max classes
            patch = cv2.imread(patch_path, self.flag)
            if token=='clean':
                reverse_patch = cv2.imread(self._clean2noise_path(patch_path), self.flag)
                assert reverse_patch is not None, '{} patch not exists'.format(self._clean2noise_path(patch_path))
            elif token==self.token:
                reverse_patch = cv2.imread(self._noise2clean_path(patch_path), self.flag)
                assert reverse_patch is not None, '{} patch not exists'.format(self._noise2clean_path(patch_path))
            self._classify(patch, noise_patch=reverse_patch, isFull=isFull)
            if (iter+1) % 500 == 0:
                end = time.time()
                print('===finished 500, time consuming: {} (seconds), total classes: {}'.format(end-start, len(self.patchCenter)))
                start = end
        
        for c, imgs in self.classifier.items():
            if len(imgs)==1: ## reclass
                
                patch = imgs[0]
                reverse_patch = self.attach_classifier[c][0]
                self.classifier[c] = []
                self.attach_classifier[c] = []
                self._classify(patch, reverse_patch, isFull=True)
                


    def build_dataset(self, save_dir, maxClass=1000, token='clean'):
        """
        build dataset from self.classfier
        classify with token
        """
        if token=='clean':
            reverse_token = self.token
        elif token == self.token:
            reverse_token = 'clean'
        else:
            raise NotImplementedError

        patches_dir = os.path.join(self.patches_dir, token)
        self._traverse_patches(patches_dir, maxClass=maxClass, token=token)
        # print(self.classifier)
        # print('attach classifier: ', self.attach_classifier)
        save_idx = 0
        token_dir = os.path.join(save_dir, token)
        reverse_token_dir = os.path.join(save_dir, reverse_token)
        for patch_class, patches in self.classifier.items():
            if len(patches) > 1:
                sub_token_dir = os.path.join(token_dir, 'cp-{}'.format(save_idx))
                sub_reverse_token_dir = os.path.join(reverse_token_dir, 'cp-{}'.format(save_idx))
                if not os.path.exists(sub_token_dir):
                    os.mkdir(sub_token_dir)
                if not os.path.exists(sub_reverse_token_dir):
                    os.mkdir(sub_reverse_token_dir)
                for idx, patch in enumerate(patches):
                    save_img_path = os.path.join(sub_token_dir, 'patch-%d.png' % idx)
                    cv2.imwrite(save_img_path, patch)

                    reverse_patch =  self.attach_classifier[patch_class][idx]
                    save_reverse_path = os.path.join(sub_reverse_token_dir, 'patch-%d.png' % idx)
                    cv2.imwrite(save_reverse_path, reverse_patch)
                save_idx += 1
        print('=== Build dataset finished ===')
        for patch_class, patches_path in self.classifier.items():
            print('class: {}, total items: {}'.format(patch_class, len(patches_path)))


def main():
    '''
    ### for not-paired patches
    ## for color image, T =0.45, gray image, T=0.25
    # patches_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain100L-origin/trainPatches/rain'
    patches_dir = '../data/RawData/BSDdenoisegray/testpatches/'
    PA = PatchAnalysis(patches_dir, threshold=0.25, flag=cv2.IMREAD_GRAYSCALE) ## 0.15 for KL (channel: False), 0.55 for KL (channel: True)
    save_dir = '../data/ProcessData/BSDdenoisegray/test'
    PA.build_dataset(save_dir=save_dir, maxClass=800)
    
    '''
    '''
    imgs = glob.glob(os.path.join(patches_dir, '*.png'))
    imgs = sorted(imgs)
    # idx = 1
    idx = np.random.choice(len(imgs))
    print('choose img: ', imgs[idx])
    s = cv2.imread(imgs[idx])
    KL = []
    inverse_KL = []
    L1 = []
    for i, img in enumerate(imgs):
        if i==idx:
            KL.append(100)
            L1.append(10000)
            continue
        t = cv2.imread(img)
        KL.append(PA._calc_KL_divergence(s, t, channel_wise=False))
        l1_dis = PA._calc_histogram_distance(s, t)
        L1.append(l1_dis)
        # if l1_dis < 2000:
        #     print(img, l1_dis)
        
        # inverse_KL.append(PA._calc_KL_divergence(t, s))
    
    # print(KL)
    print('mean EMD: {}, max EMD: {}, min EMD: {}'.format(np.mean(np.array(KL)), max(KL), min(KL)))
    KL = np.array(KL)
    L1 = np.array(L1)
    print('number of KL (channel: True) less than 0.15: ', (KL<0.15).sum())
    print('number of KL (channel: True) less than 0.25: ', (KL<0.25).sum())
    idxs = np.arange(len(imgs))[KL<0.45]
    #for i in idxs:
     #   print(imgs[i], KL[i], L1[i])# inverse_KL[i])
    '''


    
    
    ### for paired patches
    patches_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain100L-multi/Rain100L-start-12/testPatches'
    save_dir = '/home/wran/Documents/deepModels/MetaLIP/data/ProcessData/Rain100L-multi/Rain100L-start-12/test'
    PA = PatchPairAnalysis(patches_dir, flag=cv2.IMREAD_COLOR, threshold=0.45, token='rain')
    PA.build_dataset(save_dir, maxClass=350, token='clean')
    
    '''
    patches_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain100L-origin/trainPatches/rain'
    PA = PatchPairAnalysis(patches_dir, flag=cv2.IMREAD_COLOR, threshold=0.1, token='rain')
    
    imgs = glob.glob(os.path.join(patches_dir, '*.png'))
    idx = 1
    s = cv2.imread(imgs[idx])
    KL = []
    for i, img in enumerate(imgs):
        if i==idx:
            continue
        t = cv2.imread(img)
        KL.append(PA._calc_KL_divergence(s, t, channel_wise=True))
    
    # print(KL)
    print('mean KL: {}, max KL: {}, min KL: {}'.format(np.mean(np.array(KL)), max(KL), min(KL)))
    KL = np.array(KL)
    print('number of KL less than 0.55: ', (KL<0.55).sum())
    print('number of KL less than 0.45: ', (KL<0.45).sum())
    '''
    
    

if __name__ == '__main__':
    main()
