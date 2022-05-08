"""
metrics for image processing
ssim, psnr etc.
"""

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

class PSNR:
    def __init__(self, max_val=255):
        self.max_val = torch.Tensor([max_val])
    
    def compute_mse(self, img1, img2):
        diff = img1-img2
        mse = torch.mean(diff**2)
        return mse
    
    def compute_rmse(self, img1, img2):
        diff = img1 - img2
        rmse = torch.sqrt(torch.mean(diff**2))
        return rmse
    
    def compute_psnr(self, img1, img2):
        max_val = self.max_val.to(img1.device)
        mse = self.compute_mse(img1, img2)
        if mse == 0:
            return 100.0
        psnr = 10*torch.log10(torch.pow(max_val, 2)/mse)
        return psnr
    
    def calc_psnr(self, img1, img2):
        max_val = self.max_val.to(img1.device)
        assert img1.shape == img2.shape, 'shape of im1 and im2 violate'
        assert img1.dim() == 4, 'not batch of data, should call compute_psnr'
        batch = img1.size(0)
        diff = (img1 - img2).reshape(batch, -1)
        mse = torch.mean(torch.pow(diff, 2), dim=-1) ## sz of [batch]
        psnrs = 10*torch.log10(torch.pow(max_val, 2)/mse)
        psnr_mean = torch.mean(psnrs)
        return psnr_mean.item()

"""
Copy from github: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
change forward path to 1-ssim(labels, targets)
"""
def gaussian(window_size, sigma):
    """
    generate a gaussian distribution of fixed size
    """
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    """
    create a gaussian filter kernel
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    """
    main function to compute ssim metric
    use default settings if no specific aim
    """
    mu1 = F.conv2d(img1, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = 0, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = 0, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    """
    max_val = 1.0
    """
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, eps=1e-5):
        """
        forward path to compute ssim loss: 
        use 1-ssim(labels, targets) as a loss evalutation
        """
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        ssim_term = _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return 1-ssim_term

    def ssim(self, img1, img2, window_size = 11, size_average = True):
        (_, channel, _, _) = img1.size()
        window = create_window(window_size, channel)
        
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        
        return _ssim(img1, img2, window, window_size, channel, size_average).item()

class ChannelStatisticalLoss(torch.nn.Module):
    """
    To constrain the information distribuiton loss across channel
    """
    def __init__(self):
        super(ChannelStatisticalLoss, self).__init__()
    
    def forward(self, img1, img2):
        cmap1 = torch.std(img1, dim=1, keepdim=True)
        cmap2 = torch.std(img2, dim=1, keepdim=True)
        return F.l1_loss(cmap1, cmap2, size_average=True)

class GradientLoss(torch.nn.Module):
    """
    To constrain the gradients between ground truth and restore image
    """
    def __init__(self, type='L1'):
        super(GradientLoss, self).__init__()
        self.type = type
    
    def forward(self, img1, img2):
        diff =  img1-img2
        gra_h = diff[:,:,1:,:] - diff[:,:,:-1,:]
        gra_w = diff[:,:,:,1:] - diff[:,:,:,:-1]
        if self.type == 'L1':
            loss_val = torch.abs(gra_h).mean() + torch.abs(gra_w).mean()
        else:
            loss_val = torch.pow(gra_h, 2).mean() + torch.pow(gra_w, 2).mean()
        return loss_val
    
    def extra_repr(self):
        return 'Gradient Loss Layer of type {}'.format(self.type)

'''
class SSIM:
    def __init__(self, max_val=255):
        ## hyper-params
        self.k1 = 0.01
        self.k2 = 0.03
        self.L = torch.Tensor([max_val]).double()
        self.c1 = torch.Tensor([(self.k1*self.L)**2]).double()
        self.c2 = torch.Tensor([(self.k2*self.L)**2]).double()
    
    def _fspecial_gaussian(self, window_size, sigma, check=False):
        gau = torch.Tensor([np.exp(-(x-window_size//2)**2/(2*sigma**2)) for x in range(window_size)])
        gau = gau.unsqueeze(1)
        check_res = True
        if check:
            check_res = self._check_gaussian(gau, window_size, sigma, eps=1e-7)
        if check_res:        
            return gau/gau.sum()
        else:
            raise Exception('check fspecial gaussian kernel failed')
    
    def _check_gaussian(self, gau, window_size, sigma):
        ker = cv2.getGaussianKernel(window_size, sigma, eps=1e-7)
        gau = gau.numpy()
        if np.abs(ker-gau).sum() < eps:
            print('check fspecial gaussian kernel passed')
            return True
        else:
            return False
    
    def _create_window(self, window_size, sigma, channel, check=False):
        _1D_window = self._fspecial_gaussian(window_size, sigma, check)
        _2D_window = _1D_window.mm(_1D_window.t()).double().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    
    def calc_ssim(self, img1, img2, size=11, sigma=1.5, check=False):
        assert img1.shape == img2.shape, 'shape of im1 and im2 violate'
        assert img1.dim()==4, 'not batch of data, should call compute_ssim'
        channel = img1.size(1)
        img1 = img1.double()
        img2 = img2.double()
        window = self._create_window(size, sigma, channel).to(img1.device)
        mu1 = F.conv2d(img1, window, groups=channel)
        mu2 = F.conv2d(img2, window, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, groups=channel)-mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, groups=channel)-mu2_sq
        sigma12 = F.conv2d(img1*img2, window, groups=channel)-mu1_mu2
        c1 = self.c1.to(img1.device)
        c2 = self.c2.to(img2.device)

        ssim_map = ((2*mu1_mu2+c1)*(2*sigma12+c2))/((mu1_sq+mu2_sq+c1)*(sigma1_sq+sigma2_sq+c2))

        return ssim_map.mean().item()
'''

def call_ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
  # calculate SSIM
  # the same outputs as MATLAB's
  # img1, img2: [0, 255]
  
  if not img1.shape == img2.shape:
    raise ValueError('Input images must have the same dimensions.')
  if img1.ndim == 2:
    return call_ssim(img1, img2)
  elif img1.ndim== 3:
    if img1.shape[2] == 3:
      ssims = []
      for i in range(3):
        ssims.append(call_ssim(img1, img2))
      return np.array(ssims).mean()
    elif img1.shape[2] == 1:
      return call_ssim(np.squeeze(img1), np.squeeze(img2))
  else:
    raise ValueError('Wrong input image dimensions.')


def main():
    psnr = PSNR(max_val=255.0)
    ssim = SSIM()
    im1 = cv2.imread('imgs_test_metrics/rain-001.png')/255.
    im2 = cv2.imread('imgs_test_metrics/norain-001.png')/255.
    print('ssim: ', calculate_ssim(im1, im2))
    im1 = im1.transpose((2, 0, 1))
    im2 = im2.transpose((2, 0, 1))
    im1 = torch.Tensor(im1).unsqueeze(0)
    im2 = torch.Tensor(im2).unsqueeze(0)
    print(im1.shape, im2.shape)

    print('psnr: %.4f, ssim: %.4f'% (psnr.calc_psnr(im1, im2), ssim.ssim(im1, im2)))

if __name__ == '__main__':
    main()






