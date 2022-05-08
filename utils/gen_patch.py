import cv2
import numpy as np
import glob
import os
import random
import shutil

class Patches:
    """
    generate patch
    """
    def __init__(self, img_dir, save_dir, flag=cv2.IMREAD_COLOR):
        """
        :param: flag, imread mode of cv2
        """
        self.img_dir = img_dir
        self.flag = flag
        self.save_dir = save_dir
        self.imgs = self._traverse()
    
    def _traverse(self):
        imgs = glob.glob(os.path.join(self.img_dir, '*.png'))
        imgs = sorted(imgs) ## sorted to keep sequence the same everytime
        return imgs
    
    def _gen_patch_from_one_img(self, img, patch_size, stride):
        """
        generate patch from one image
        :param img: np.ndarray
        :param patch_size: patch size 
        :param strdie: stride for neighbour patches
        """
        patches = []
        h, w = img.shape[0], img.shape[1]
        for begin_h in range(0, h-patch_size, stride):
            for begin_w in range(0, w-patch_size, stride):
                patch = img[begin_h: begin_h+patch_size, begin_w: begin_w+patch_size, ...]
                patches.append(patch)
        return np.array(patches) ## patches
    
    def _parse_img(self, img):
        """
        obtain img name
        :param: img is the path of image
        """
        img = img.split('/')[-1]
        img = img.split('\\')[-1]
        img = img.split('.')[0]
        return img
    
    def _read_img(self, img):
        """
        read image from memory
        :param: img is image path
        """
        return cv2.imread(img, self.flag)

    def gen_patches(self, patch_size, stride):
        for img_path in self.imgs:
            img = self._read_img(img_path)
            patches = self._gen_patch_from_one_img(img, patch_size, stride)
            img_name = self._parse_img(img_path)
            for idx in range(patches.shape[0]):
                save_img_dir = os.path.join(self.save_dir, '%s-patch-%04d.png'% (img_name, idx))
                cv2.imwrite(save_img_dir, patches[idx])
            print('generate {} patches from image: {} finished'.format(patches.shape[0], img_path))
    
def split_patches(patches_dir, split_ratio=0.25, save_dir=r'../data/ProcessData/Rain'):
    """
    split patches for training and testing
    :@param patches_dir: file path of image patches
    :@param split_ratio: split ratio for test
    "@param save_dir: save father path
    """
    patches = glob.glob(os.path.join(patches_dir, '*.png'))
    patch_num = len(patches)
    print('=== find {} patches to split, split ratio is: {} for test'.format(patch_num, split_ratio))
    test_patch_num = int(patch_num*split_ratio)
    random.shuffle(patches)
    save_train_folder = os.path.join(save_dir, 'trainPatches', 'clean')
    save_test_folder = os.path.join(save_dir, 'testPatches', 'clean')
    for patch in patches[test_patch_num:]:
        patch_name = patch.split('\\')[-1]
        patch_name = patch_name.split('/')[-1]
        save_patch = os.path.join(save_train_folder, patch_name)
        shutil.copy(patch, save_patch)
    for patch in patches[:test_patch_num]:
        patch_name = patch.split('\\')[-1]
        patch_name = patch_name.split('/')[-1]
        save_patch = os.path.join(save_test_folder, patch_name)
        shutil.copy(patch, save_patch)
    
    print('=== split all patches finished.')

def transfer_patch(rain_dir, clean_dir, save_dir):
    """
    transfer correspond rain patches with clean patches generated from func split_patches
    :@param rain_dir: rain patches (contains patches for training and testing)
    :@param clean_dir: clean patches(either train or test)
    :@param save_dir: save rain patches path
    """
    clean_patches = glob.glob(os.path.join(clean_dir, '*.png'))
    for clean_patch in clean_patches:
        patch_name = clean_patch.split('\\')[-1]
        patch_name = patch_name.split('/')[-1]
        rain_patch_name = patch_name.replace('clean', 'rain')
        rain_patch = os.path.join(rain_dir, rain_patch_name)
        shutil.copy(rain_patch, os.path.join(save_dir, rain_patch_name))
    print('=== handle correspond rain patches finished')

    


if __name__ == '__main__':
    
    
    img_dir = '../data/RawData/Rain100L-origin/test/clean/'
    save_dir = '../data/RawData/Rain100L-50//testPatches/clean'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    patch_size = 50
    stride = 25
    p = Patches(img_dir, save_dir)
    p.gen_patches(patch_size, stride)
    
    
    '''
    ## for deraining
    # patches_dir = r'E:\Dustbin\data\RawData\Rain100L-origin\Allpatches\clean'
    # save_dir = r'E:\Dustbin\data\RawData\Rain100L-origin'
    patches_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain100-new/Allpatches/clean'
    save_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800'
    # split_patches(patches_dir=patches_dir, save_dir=save_dir)
    
    # build train dataset
    rain_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/Allpatches/rain'
    clean_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/trainPatches/clean' 
    save_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/trainPatches/rain'
    transfer_patch(rain_dir=rain_dir, clean_dir=clean_dir, save_dir=save_dir)
    # build test dataset
    rain_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/Allpatches/rain'
    clean_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/testPatches/clean' 
    save_dir = '/home/wran/Documents/deepModels/MetaLIP/data/RawData/Rain800/testPatches/rain'
    transfer_patch(rain_dir=rain_dir, clean_dir=clean_dir, save_dir=save_dir)
    '''
    
    


    
