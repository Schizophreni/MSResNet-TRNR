import tqdm
import os
import numpy as np
import sys
import time
import torch
sys.path.append('..')
from utils.logconf import get_logger, build_folder, load_checkpoint
from utils.metrics import SSIM, PSNR

class MergeAllIngredients:
    def __init__(self, args=None, database=None, test_dataset=None, model=None):
        """
        Initialzes an full training (validation, testing) process of metalip
        :param args: arguments
        :param database: data base
        :param model: model
        """
        self.ssim = SSIM()
        self.psnr = PSNR(max_val=1.0)
        
        self.args, self.device = args, args.device
        self.model = model
        self.database = database
        self.test_dataset = test_dataset
        save_folder = os.path.join(args.save_folder, args.process_mode)

        build_folder(save_folder) ## build save_folder
        save_experiments_folder = os.path.join(save_folder, '{}-{}-{}'.format(model.net.name, args.task_num, args.freqn))
        
        build_folder(save_experiments_folder) ## build save_experiments folder
        ## check log
        if not os.path.exists(os.path.join(save_experiments_folder, 'log.txt')):
            self.save_log = get_logger(os.path.join(save_experiments_folder, 'log.txt'))
            print('=== training begin')
            self.save_log.info(model) ## model architecture
            self.save_log.info(args) ## arguments
        else:
            self.save_log = get_logger(os.path.join(save_experiments_folder, 'log.txt'))
            print('=== Training')
        
        self.save_model_folder = os.path.join(save_experiments_folder, 'models')
        build_folder(self.save_model_folder)
        self.iters_per_epoch = database.total_train_samples//args.freqn
        
        self.state = dict() ## statement
        self.total_losses = dict()
        self.state['best_val_psnr'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.use_bn = True if 'BN' in self.model.net.name else False
        self.model = model.to(args.device)
        
        self.start_iter = 0
        
        ## load checkpoint if exists
        if args.load_ckp_mode == 'latest':
            load_kwargs = {'mode': 'latest'}
        elif args.load_ckp_mode == 'iteration':
            load_kwargs = {'mode': 'iteration', 'iter': args.load_begin_iter}
        checkpoint, model_save_path = load_checkpoint(ckp_folder=self.save_model_folder, **load_kwargs)
        if checkpoint is not None:
            self.state['best_val_psnr'] = 0.0 if 'best_val_psnr' not in checkpoint else checkpoint['best_val_psnr']
            self.state['best_val_iter'] = checkpoint['best_val_iter']
            self.state['current_iter'] = checkpoint['current_iter']+1
            self.start_iter = checkpoint['current_iter']+1
            print('[!!!] load model from iteration {}'.format(model_save_path))
            self.save_log.info('[!!!] load model from iteration {}'.format(model_save_path))
            self.model.load_model(model_save_path)
        
    
    def build_iter_string(self, summary_losses):
        '''
        Builds a progress bar summary string given summary losses dictionary
        :param summary_losses: current summary losses
        :return: A summary string
        '''
        loss = summary_losses['loss']
        psnrs = np.around(summary_losses['psnrs'], 4)
        ssims = np.around(summary_losses['ssims'], 5)
        return 'step:{}, {:.6}, {}, {}'.format(self.state['current_iter'], loss, psnrs, ssims)
    
    def train_iteration(self, train_batch, current_iter, pbar_train, calc_metrics=False):
        '''
        runs a training iteration, updates the progress bar and returns the total and current epoch train losses
        :param train_sample: training data batch sampled from database
        :param current_iter: the current iteration
        :param pbar_train: tqdm bar for training
        '''
        current_epoch = current_iter // self.iters_per_epoch
        losses = self.model.run_train_iter(data_batch=train_batch, epoch=current_epoch, use_bn=self.use_bn, calc_metrics=calc_metrics)
        current_iter+=1
        pbar_train.update(1)
        if calc_metrics:
            msg = self.build_iter_string(summary_losses=losses)
            self.save_log.info(msg)
            pbar_train.set_description('Epoch:{}'.format(current_epoch))
            pbar_train.set_postfix(hist=msg)
        return current_iter
    
    def test_iteration(self, test_batch, pbar_test, calc_metrics=True):
        '''
        Runs a validation iteration, updates the progress bar and returns the total losses
        '''
        losses = self.model.run_validation_iter(data_batch=test_batch, use_bn=self.use_bn, calc_metrics=calc_metrics)
        pbar_test.update(1)
        pbar_test.set_description('Test')
        msg = self.build_iter_string(summary_losses=losses)
        pbar_test.set_postfix(hist=msg)
        return losses['psnrs']
    
    def test_imgs(self, pbar):
        """
        test images
        """
        test_psnrs = []
        test_ssims = []
        for i in range(len(self.test_dataset)):
            clean_img , noise_img = self.test_dataset[i]
            clean_img = torch.from_numpy(clean_img[None, ...]).to(self.args.device)
            noise_img = torch.from_numpy(noise_img[None, ...]).to(self.args.device)
            with torch.no_grad():
                denoise_img = self.model.test_with_attenuate(noise_img, verbose=True if i==0 else False)
                denoise_img = torch.clamp(denoise_img, min=0.0, max=1.0)
                test_psnrs.append(self.psnr.calc_psnr(denoise_img, clean_img))
                test_ssims.append(self.ssim.ssim(denoise_img, clean_img))
        test_psnr = np.array(test_psnrs).mean().astype(np.float16)
        test_ssim = np.array(test_ssims).mean().astype(np.float16)
        msg = 'Test: ssim: {}, psnr: {}'.format(test_ssim, test_psnr)
        self.save_log.info(msg)
        pbar.write(msg)
        return test_ssim, test_psnr
    
    def save_models(self, model_path, state):
        """
        save model
        :param model_path: save models path
        :param state: state dict
        """
        self.model.save_model(model_path, state)
    
    def train_metalip(self):
        '''
        Runs a full training process
        '''
        total_epochs = self.args.total_iteration//self.iters_per_epoch
        begin_epoch = self.state['current_iter'] // self.iters_per_epoch
        for epoch in range(begin_epoch, total_epochs):
            start_iter = self.state['current_iter']
            end_iter = (start_iter // self.iters_per_epoch+1)*self.iters_per_epoch
            with tqdm.tqdm(initial=start_iter, total=self.iters_per_epoch, ncols=150) as pbar_train:
                for iteration in range(start_iter, end_iter):
                    train_batch = self.database.next(mode='train')
                    # self.model.SGD_forward(data_batch=train_batch)
                    if self.state['current_iter'] % self.args.iters_every_bar == 0:
                        self.state['current_iter'] = self.train_iteration(train_batch=train_batch, current_iter=self.state['current_iter'], 
                                                                        pbar_train=pbar_train, calc_metrics=True)
                        
                    else:
                        self.state['current_iter'] = self.train_iteration(train_batch=train_batch, current_iter=self.state['current_iter'], 
                                                                        pbar_train=pbar_train, calc_metrics=False)
                    if self.state['current_iter'] % self.args.iters_every_test == 0:
                        ## run test
                        avg_psnrs = []
                        with tqdm.tqdm(total=self.args.num_evaluation_tasks//self.args.task_num+1) as pbar_val:
                            for _ in range(self.args.num_evaluation_tasks//self.args.task_num+1):
                                test_batch = self.database.next(mode='test')
                                psnr_vals = self.test_iteration(test_batch=test_batch, pbar_test=pbar_val, calc_metrics=True)
                                avg_psnrs.append(np.array(psnr_vals).mean())
                            avg_psnr = np.array(avg_psnrs).mean()
                        if (avg_psnr >= self.state['best_val_psnr']):
                            self.save_log.info('best psnr: {}'.format(avg_psnr))
                            self.state['best_val_psnr'] = avg_psnr
                            self.state['best_val_iter'] = self.state['current_iter']
                        self.save_models(model_path=os.path.join(self.save_model_folder, '{}-iterModel.tar'.format(self.state['current_iter'])), 
                                         state=self.state)
                        self.save_models(model_path=os.path.join(self.save_model_folder, 'latest.tar'), state=self.state)
                        self.model.init_attenuate_weight()
                    
if __name__ == '__main__':
    from utils.arguments import get_args
    # from ImgLIPNfreqsKshot import ImgLIPNfreqsKshot
    from ImportantSampling import ImgLIPNfreqsKshot
    from nets import *
    from metaunit import MetaUnit
    from ImgLIPDset import TestDataset
    args = get_args()
    torch.manual_seed(1)
    kwargs = {'Agg_input': True, 'input_channels': 3}
    test_kwargs={
    'dataset': 'Rain100L-test',
    'type': 'rain',
    'clean_dir': '../MetaLIP/data/RawData/Rain100L/test/clean',
    'noise_dir': '../MetaLIP/data/RawData/Rain100L/test/rain',
    }
    dataset_config = {
    'Rain100L': ('norain-{}.png', 'rain-{}.png', 50, 1),
    'Rain100L-test': ('norain-{}.png', 'rain-{}.png', 30, 1)
    
    }
    #torch.manual_seed(0)
    # np.random.seed(0)
    db_test =  TestDataset(dataset_config, **test_kwargs)
    # net = ResBNAggKstages(in_channels=3, num_filters=36, K=4, args=args, device=args.device, **kwargs)
    # net = DenseCBNRSENet(in_channels=3, num_filters=36, s1=2, s2=4, args=args, device=args.device, **kwargs)
    # net = DuRBP(args=args, device=args.device, stages=3, negative_slope=0.01)
    # net = SCNet(in_channels=3, num_filters=32, stages=4, args=args, device=args.device, negative_slope=0.2)
    # net = MultiScaleSCNet(in_channels=3, num_filters=36, args=args, device=args.device, negative_slope=0.2, stages=4)
    # net = DiRes(in_channels=3, num_filters=32, out_channels=3, args=args, K=4)
    net = MultiScaleResNet(in_channels=3, num_filters=48, out_channels=3, k1=4, k2=0, args=args, Agg=True)
    model = MetaUnit(args=args, net=net)

    tmp = filter(lambda x: x.requires_grad, model.parameters())
    train_params = sum(map(lambda x: np.prod(x.shape), tmp))
    print('total params: ', train_params)

    database = ImgLIPNfreqsKshot(root_dir=args.root_dir, batch_size=args.batch_size, n_freqs=args.freqn, k_shot=args.k_shot, k_query=args.k_query, patch_size=args.patch_size, 
            sigmas='5,15,25')
    experiment = MergeAllIngredients(args=args, database=database, test_dataset=db_test, model=model)
    experiment.train_metalip()
       
            
            
        
        
