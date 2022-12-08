import sys
sys.path.append('..')
from models.nets import MetaMSResNet
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

psnr = PSNR(max_val=1.0)
ssim = SSIM()

def train(train_data, test_data, batch_size=32, total_iter=50000, mode='RBSI', args=None):
    process_type = train_data.process_type
    dataset_name = train_data.dataset_name

    if process_type == 'rain':
        model = MetaMSResNet(3, 48, stages=4, args=args, Agg=False, withSE=True, msb='MAEB', rb='Dual', relu_type='lrelu', dilated_factors=3)
        model.name = 'MAEB-RES-Rain100L-100-SSIM{}'.format(args.ssim_weight)
    else:
        pass
    model = model.cuda()
    # for name, param in model.named_parameters():
    #     print(name, param.size())
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    train_params = sum(map(lambda x: np.prod(x.shape), tmp))
    print('total params: ', train_params)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=4e-5)

    assert mode in ['RBSI', 'RBSII', 'RPS'], 'invalid sampling mode'
    
    if mode=='RBSI':
        total_epochs = (batch_size*total_iter) // len(train_data.dset) + 1
    else:
        total_epochs = (total_iter*batch_size) // len(train_data) + 1
    start_iter = 0 ## initalize start iteration
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-4)
 
    ## save checkpoint settings
    save_dir = 'results/dataSize/RPS/{}'.format(model.name)
    # save_dir = 'resutls/dataSize/RCS/{}'.format(model_name) # for ablation 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    save_model_dir = os.path.join(save_dir, 'models')
    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    
    logger = get_logger(os.path.join(save_dir, 'log.txt'))
    best_psnr = 0

    ## checkpoint loading 
    if os.path.exists(os.path.join(save_model_dir, 'latest.tar')):
        ckp = torch.load(os.path.join(save_model_dir, 'latest.tar'))
        start_iter = ckp['iter']+1
        best_psnr = ckp['psnr']
        model.load_state_dict(ckp['net'])
        optimizer.load_state_dict(ckp['optim'])
        scheduler.load_state_dict(ckp['lr_scheduler'])
        print('load model at {}-th iteration'.format(start_iter))
    else:
        logger.info(model)
        logger.info(args)

    if mode != 'RBSI':
        ### case for no put back sampling
        train_data = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    if mode == 'RBSI':
        while start_iter < total_iter:
            data_batch = train_data.next_batch(batch_size=batch_size)
            clean_batch, rain_batch = data_batch

            clean_batch = torch.FloatTensor(clean_batch).cuda()
            rain_batch = torch.FloatTensor(rain_batch).cuda()
            
            derain_batch = model.forward(rain_batch, num_step=0, training=True)

            loss = F.l1_loss(clean_batch, derain_batch, size_average=True)+args.ssim_weight*ssim.forward(derain_batch, clean_batch)
            optimizer.zero_grad()
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(name)
                    else:
                        param.grad.clamp_(-10, 10)
            
            optimizer.step()

            if start_iter % 50 == 0:
                ssim_val = ssim.ssim(clean_batch, derain_batch)
                psnr_val = psnr.calc_psnr(clean_batch, derain_batch)
                print('[step: {}], loss: {}, psnr: {}, ssim: {}'.format(start_iter, loss.item(), psnr_val, ssim_val))
                logger.info('[step: {}], loss: {}, psnr: {}, ssim: {}'.format(start_iter, loss.item(), psnr_val, ssim_val))
            if start_iter % 500 == 0:
                # scheduler.step()
                psnr_list = []
                ssim_list = []
                if process_type == 'rain':
                    for test_idx in range(len(test_data)):
                        clean_img, rain_img = test_data.traverse_rain(test_idx)
                        if clean_img.shape[-2] > 1000 or clean_img.shape[-1] > 1000:
                            continue # do not test large image for fast test in training
                        rain_img =  torch.FloatTensor(rain_img).cuda()
                        clean_img = torch.FloatTensor(clean_img)
                        with torch.no_grad():
                            derain_img = model(rain_img, num_step=0, training=False).clamp_(0.0, 1.0).cpu()
                            psnr_list.append(psnr.calc_psnr(derain_img, clean_img))
                            ssim_list.append(ssim.ssim(derain_img, clean_img))
                else:
                    test_noise_simga = 25 / 255.0
                    for i in range(len(test_data)):
                        test_clean_batch, test_noise_batch = test_data.traverse_noise(i, noise_sigma=test_noise_simga)
                        test_clean_batch = torch.FloatTensor(test_clean_batch)
                        test_noise_batch = torch.FloatTensor(test_noise_batch).cuda()
                        test_noise_sigma = torch.FloatTensor([test_noise_simga]).cuda()
                        with torch.no_grad():
                            denoise_batch = model(test_noise_batch, num_step=0, training=False, noise_sigma=test_noise_sigma)
                            denoise_batch = torch.clamp_(denoise_batch, 0.0, 1.0).cpu()
                        psnr_list.append(psnr.calc_psnr(denoise_batch, test_clean_batch))
                        ssim_list.append(ssim.ssim(denoise_batch, test_clean_batch))

                mean_psnr = np.array(psnr_list).mean()
                mean_ssim = np.array(ssim_list).mean()
                if mean_psnr > best_psnr:
                    best_psnr = mean_psnr
                    torch.save(model.state_dict(), os.path.join(save_model_dir, 'best.pth'))
                print('[===test: psnr: {}, ssim: {}'.format(mean_psnr, mean_ssim))
                logger.info('[===test: psnr: {}, ssim: {}'.format(mean_psnr, mean_ssim))
                model_path = os.path.join(save_model_dir, 'latest.tar'.format(start_iter))
                torch.save({
                    'iter': start_iter, 
                    'ssim': mean_ssim, 
                    'psnr': mean_psnr, 
                    'net': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()
                }, model_path)
            start_iter += 1
            current_epoch = (start_iter * batch_size) // len(train_data.dset)
            scheduler.step(current_epoch)
    else:
        while start_iter < total_iter:
            for epoch in range(total_epochs):
                for data_batch in train_data:
                    clean_batch, rain_batch = data_batch

                    clean_batch = torch.FloatTensor(clean_batch).cuda()
                    rain_batch = torch.FloatTensor(rain_batch).cuda()
                    
                    derain_batch = model(rain_batch,num_step=0)

                    loss = F.l1_loss(clean_batch, derain_batch, size_average=True)+args.ssim_weight*ssim.forward(derain_batch, clean_batch)

                    optimizer.zero_grad()
                    loss.backward()

                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                print(name)
                            else:
                                param.grad.clamp_(-10, 10)
                    optimizer.step()

                    if start_iter % 50 == 0:
                        ssim_val = ssim.ssim(clean_batch, derain_batch)
                        psnr_val = psnr.calc_psnr(clean_batch, derain_batch)
                        print('[step: {}], loss: {}, psnr: {}, ssim: {}'.format(start_iter, loss.item(), psnr_val, ssim_val))
                        logger.info('[step: {}], loss: {}, psnr: {}, ssim: {}'.format(start_iter, loss.item(), psnr_val, ssim_val))
                    if start_iter % 500 == 0:
                        psnr_list = []
                        ssim_list = []
                        if process_type == 'rain':
                            for test_idx in range(len(test_data)):
                                clean_img, rain_img = test_data.traverse_rain(test_idx)
                                if clean_img.shape[-2] > 1000 or clean_img.shape[-1] > 1000:
                                    continue # do not test large image for fast test in training
                                rain_img =  torch.FloatTensor(rain_img).cuda()
                                clean_img = torch.FloatTensor(clean_img)
                                with torch.no_grad():
                                    derain_img = model(rain_img, num_step=0, training=False).clamp_(0.0, 1.0).cpu()
                                    psnr_list.append(psnr.calc_psnr(derain_img, clean_img))
                                    ssim_list.append(ssim.ssim(derain_img, clean_img))
                        else:
                            test_noise_simga = 25 / 255.0
                            for i in range(len(test_data)):
                                test_clean_batch, test_noise_batch = test_data.traverse_noise(i, noise_sigma=test_noise_simga)
                                test_noise_batch = torch.FloatTensor(test_noise_batch).cuda()
                                test_noise_sigma = torch.FloatTensor([test_noise_simga]).cuda()
                                with torch.no_grad():
                                    denoise_batch = model(test_noise_batch, num_step=0, training=False, noise_sigma=test_noise_sigma)
                                    denoise_batch = torch.clamp_(denoise_batch, 0.0, 1.0).cpu()
                                psnr_list.append(psnr.calc_psnr(denoise_batch, test_clean_batch))
                                ssim_list.append(ssim.ssim(denoise_batch, test_clean_batch))

                        mean_psnr = np.array(psnr_list).mean()
                        mean_ssim = np.array(ssim_list).mean()
                        if mean_psnr > best_psnr:
                            best_psnr = mean_psnr
                            torch.save(model.state_dict(), os.path.join(save_model_dir, 'best.pth'))
                        print('[===test: psnr: {}, ssim: {}, num: {}'.format(mean_psnr, mean_ssim, len(psnr_list)))
                        logger.info('[===test: psnr: {}, ssim: {}'.format(mean_psnr, mean_ssim))
                        model_path = os.path.join(save_model_dir, 'latest.tar'.format(start_iter))
                        torch.save({
                            'iter': start_iter, 
                            'ssim': mean_ssim, 
                            'psnr': best_psnr, 
                            'net': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'lr_scheduler': scheduler.state_dict()
                        }, model_path)
                    start_iter += 1
                scheduler.step()

if __name__ == '__main__':
    args = get_args()
    ### RBSI
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from random_batch_sampling import RandomBatchSamplingI, RandomPatch, RandomBatchDataset
    iterations = 60000
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    ## training for RBS
    # torch.use_deterministic_algorithms(True)
    
    """
    train_data = RandomBatchSamplingI(root_dir='/home/rw/Public/datasets/derain/Rain800-200-multi/train', patch_size=64, dataset_name='Few', process_type='rain')
    test_data = RandomBatchDataset(root_dir='/home/rw/Public/datasets/derain/Rain800/test', patch_size=64, dataset_name='Rain800', process_type='rain')
    train(train_data, test_data, total_iter=iterations, mode='RBSI', args=args)
    """

    ### train for RPS
    train_data = RandomPatch(root_dir='/home/rw/Public/datasets/derain/Rain100L64-100/RawData', mode='train', dataset_name='Rain100L', process_type='rain')
    # test_data = RandomPatch(root_dir='/home/cv/TEMP/data/Rain100L-multi', mode='test')
    test_data = RandomBatchDataset(root_dir='/home/rw/Public/datasets/derain/Rain100L/test', patch_size=64, dataset_name='Rain100L', process_type='rain')
    train(train_data, test_data, total_iter=iterations, mode='RPS', args=args)
    

    
