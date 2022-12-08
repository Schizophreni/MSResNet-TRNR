import sys
sys.path.append('..')
from models.nets import AdaFM, MetaMSResNetN
from utils.metrics import SSIM, PSNR
from utils.logconf import get_logger
from utils.arguments import get_args
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import os
import torch
import numpy as np
import random
from tqdm import tqdm


psnr = PSNR(max_val=1.0)
ssim = SSIM()

def train(train_data, test_data, batch_size=128, total_iter=200000, args=None):
    # model = AdaFM(in_nc=3, out_nc=3, args=args)
    model = MetaMSResNetN(in_channels=3, num_filters=64, stages=6, args=args, withSE=True)
    model.name='MAEB-RES-3c-6s'
    # model.name = 'MAEB-RES-WaterlooBSDSigma055-4stages-ssim{}'.format(args.ssim_weight)
    noise_interval = [0, 55]
    
    '''
    for name, param in model.named_parameters():
        print(name, param.size())
    '''

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    train_params = sum(map(lambda x: np.prod(x.shape), tmp))
    print('total params: ', train_params)
    model = model.cuda()

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0)

    total_epochs = (total_iter*batch_size)//len(train_data)+1

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)
 
    ## save checkpoint settings
    save_dir = 'results/dataSize/Noise/{}'.format(model.name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_model_dir = os.path.join(save_dir, 'models')
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    
    logger = get_logger(os.path.join(save_dir, 'log.log'))
    best_psnr = 0
    start_iter = 0

    ## checkpoint loading 
    if os.path.exists(os.path.join(save_model_dir, 'latest.tar')):
        ckp = torch.load(os.path.join(save_model_dir, 'latest.tar'))
        start_iter = ckp['iter']+1
        best_psnr = ckp['best_psnr']
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['lr_scheduler'])
        print('load model at {}-th iteration'.format(start_iter))
    else:
        logger.info(model)
        logger.info(args)

    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    current_epoch = start_iter*batch_size//len(train_data)
    current_iter_in_epoch = start_iter - current_epoch*len(train_data)//batch_size
    for epoch in range(current_epoch, total_epochs):
        with tqdm(train_loader, initial=current_iter_in_epoch, total=len(train_data)//batch_size, ncols=120) as pbar_train:
            for datas in pbar_train:
                data, noise_data, noise_sigma = datas
                
                clean_batch = data.cuda()
                noise_batch = noise_data.cuda()
                noise_sigma = noise_sigma.cuda().float()
                
                model.zero_grad()
            
                derain_batch = model.forward(noise_batch, num_step=0, training=True, noise_sigma=noise_sigma)
                loss = F.l1_loss(clean_batch, derain_batch, size_average=True)# +args.ssim_weight*ssim.forward(derain_batch, clean_batch)
                optimizer.zero_grad()
                loss.backward()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(name)
                        else:
                            param.grad.clamp_(-10, 10)
                optimizer.step()

                pbar_train.set_description('[Epoch: {} | Iter: {}]'.format(epoch, start_iter))
                pbar_train.set_postfix(loss=loss.detach().cpu().float())

                if start_iter % 50 == 0:
                    ssim_val = ssim.ssim(clean_batch, derain_batch)
                    psnr_val = psnr.calc_psnr(clean_batch, derain_batch)
                    logger.info('[step: {}], loss: {}, psnr: {}, ssim: {}'.format(start_iter, loss.item(), psnr_val, ssim_val))
                start_iter += 1
        ## evaluation
        scheduler.step()
        psnr_list = []
        ssim_list = []
        test_noise_simga=25/255.0
        for i in range(len(test_data.dset)):
            test_clean_batch, test_noise_batch = test_data.next_example(i, noise_sigma=test_noise_simga)
            test_clean_batch = torch.FloatTensor(test_clean_batch).cuda()
            test_noise_batch = torch.FloatTensor(test_noise_batch).cuda()
            test_noise_sigma = torch.FloatTensor([test_noise_simga]).cuda()
            with torch.no_grad():
                denoise_batch = model(test_noise_batch, num_step=0, training=False, noise_sigma=test_noise_sigma)
                denoise_batch = torch.clamp_(denoise_batch, 0.0, 1.0)
            psnr_list.append(psnr.calc_psnr(denoise_batch, test_clean_batch))
            ssim_list.append(ssim.ssim(denoise_batch, test_clean_batch))
        mean_psnr = np.array(psnr_list).mean()
        mean_ssim = np.array(ssim_list).mean()
        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            model_path = os.path.join(save_model_dir, 'bestckp.tar')
            torch.save({
                'net': model.state_dict(),
            }, model_path)
        logger.info('[===test: psnr: {}, ssim: {}'.format(mean_psnr, mean_ssim))
        model_path = os.path.join(save_model_dir, 'latest.tar'.format(start_iter))
        torch.save({
            'iter': start_iter, 
            'ssim': mean_ssim, 
            'best_psnr': best_psnr, 
            'net': model.state_dict(),
            'optim': optimizer.state_dict(),
            'lr_scheduler': scheduler.state_dict(),
        }, model_path)
    

if __name__ == '__main__':
    args = get_args()
    ### RBSI
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from random_batch_sampling import RandomBatchSamplingI, RandomBatchDataset
    iterations = 500000
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    train_data = RandomBatchDataset(root_dir='/home/rw/Public/datasets/denoise/WaterlooBSD/', patch_size=64, process_type='noise')
    test_data = RandomBatchSamplingI(root_dir='/home/rw/Public/datasets/denoise/BSD68', patch_size=64, process_type='noise', channels=1)
    train(train_data, test_data, total_iter=iterations, args=args, batch_size=64)
    

    
