import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append('..')
from utils.metrics import SSIM, PSNR, GradientLoss

ssim = SSIM()
psnr = PSNR(max_val=1.0)


def set_torch_seed(seed):
    '''
    Sets the pytorch seeds for current training process
    '''
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)
    return rng


class MetaUnit(nn.Module):
    def __init__(self, args=None, net=None):
        '''
        Initialize a MAML few shot learning system
        :param args: arguments
        :param net: meta neural network
        :param inner_optimizer: inner loop optimizer
        '''
        super(MetaUnit, self).__init__()
        self.args = args
        self.device = args.device
        self.batch_size = args.batch_size
        self.current_epoch=0
        
        self.rng = set_torch_seed(seed=args.seed)
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.net = net
        # self.channelLayer = ChannelStatisticalLoss()
        self.gradientLoss = GradientLoss()

        self.ssim_weight = args.ssim_weight
        self.channel_weight = args.channel_weight

        names_weights_copy = self.get_inner_loop_params_dict(self.net.named_parameters())
        self.init_update_lr()
        ## Attenuator for L2F
        if args.attenuate:
            num_layers = len(names_weights_copy.keys())
            self.attenuator = nn.Sequential(
                nn.Linear(num_layers, num_layers),
                nn.ReLU(inplace=True),
                nn.Linear(num_layers, num_layers),
                nn.Sigmoid()
            ).to(self.device)
            ## use attenuate
            self.attenuate_weight = torch.zeros(num_layers).to(self.device)

        task_name_params = self.get_inner_loop_params_dict(self.named_parameters())
        print('Inner Loop parameters')
        for k, v in task_name_params.items():
            print(k, v.shape)
        print('Outer Loop parameters')
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        self.meta_optim = optim.Adam(self.trainable_params(), lr=self.meta_lr, amsgrad=False, weight_decay=0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.meta_optim, 
                          T_max=self.args.total_iteration//self.args.iters_every_test, eta_min=self.args.min_learning_rate)
        self.to(self.device)
    
    def init_update_lr(self):
        max_update_lr = self.args.init_inner_loop_learning_rate
        min_update_lr = self.args.init_inner_loop_learning_rate/10
        layer_index = self.net.get_layer_index()
        rate = (max_update_lr-min_update_lr)/(max(layer_index)-1)
        layer_update_lr = [max_update_lr - rate*(idx-1) for idx in layer_index] ## per layer learning rate
        print('initial inner loop learning rate per layer: ', layer_update_lr)
        self.update_lr = nn.Parameter(torch.FloatTensor(layer_update_lr), requires_grad=False)
    
    def fill_update_lr(self):
        layer_index = self.net.get_layer_index()
        self.update_lr = nn.Parameter(torch.ones(len(layer_index))*self.args.init_inner_loop_learning_rate, requires_grad=False)
    
    def get_per_step_loss_importances(self):
        '''
        Generate a tensor of size: num_inner_loop_steps indicating the importance of each step's target loss func
        :return : A tensor to be used to compute the weighted average of loss
        '''
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter))/self.args.number_of_training_steps_per_iter
        decay_rate = 1.0/self.args.number_of_training_steps_per_iter/self.args.multi_step_loss_num_epochs
        min_value_for_non_final_loss = 0.03/self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights)-1):
            loss_weights[i] = np.maximum(loss_weights[i]-decay_rate*self.current_epoch, min_value_for_non_final_loss)
        loss_weights[-1] = 1-loss_weights[:-1].sum()
        loss_weights = torch.FloatTensor(loss_weights).to(self.device)
        return loss_weights
        
    def get_inner_loop_params_dict(self, params):
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "bn" not in name:
                        param_dict[name] = param.to(self.device)
        return param_dict
    
    def get_task_embeddings(self, spt_x, spt_y, names_weights_copy,use_bn=True):
        ### use gradients as task embeddings
        support_loss, _, _ = self.net_forward(x=spt_x, y=spt_y, params=names_weights_copy, 
                                              bkp_running_statistics=True, training=True, num_step=0,use_bn=use_bn)
        self.net.zero_grad(names_weights_copy)
        grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=False)
        layerwise_mean_grads = []
        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())
        layerwise_mean_grads = torch.stack(layerwise_mean_grads)
        return layerwise_mean_grads
    
    def attenutate_init(self, task_embeddings, names_weights_copy):
        gamma = 0.3*self.attenuator(task_embeddings) + 0.7
        updated_weights = list(map(
            lambda current_params, gamma: ((gamma)*current_params.to(self.device)), 
                names_weights_copy.values(), gamma)
        )
        updated_names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))
        self.attenuate_weight += gamma.data*(1/(self.args.task_num*self.args.iters_every_test))
        return updated_names_weights_copy

    def trainable_params(self):
        '''
        returns an iterator over the trainable params
        '''
        for param in self.parameters():
            if param.requires_grad:
                yield param
    
    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step):
        '''
        Applies an inner loop update given current step
        '''
        self.net.zero_grad(names_weights_copy) ## zero_grad
        grads = torch.autograd.grad(loss, names_weights_copy.values(), 
                                    create_graph=use_second_order)
        updated_weights = list(map(
            lambda current_params, learning_rates, grads: current_params-learning_rates*grads, 
            names_weights_copy.values(), self.update_lr, grads
        ))
        names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))
        return names_weights_copy
    
    def get_accross_task_loss_metrics(self, total_losses):
        losses = dict()
        losses['loss'] = torch.mean(torch.stack(total_losses))
        return losses
    
    def init_attenuate_weight(self):
        if self.args.attenuate:
            self.attenuate_weight *= 0.0
    
    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, 
                training_phase, use_bn=True, calc_metrics=False, init_attenuate_weight=False):
        '''
        Runs a forward outer loop pass on the batch of tasks
        :param data_batch: a batch of data containing x_spt, y_spt, x_qry, y_qry
                x_spt of shape: [task_num, batch_sz, img_c, img_h, img_w], batch_sz = n_freq*k_shot
        :param use_bn: whether to use bn in network
        '''
        ## of shape: (task_num, n_freqs*k_shot, c, h, w)
        x_spt, y_spt, x_qry, y_qry, sigma_spt, sigma_qry = data_batch
        task_num, batch_sz, _, _, _ = x_spt.shape

        total_losses = []
        total_psnrs = np.zeros(self.args.number_of_training_steps_per_iter)
        total_ssims = np.zeros(self.args.number_of_training_steps_per_iter)
        self.net.zero_grad()

        for i in range(task_num):
            task_losses = []
            per_step_loss_importances = self.get_per_step_loss_importances()
            names_weights_copy = self.get_inner_loop_params_dict(self.net.named_parameters()) ## inner params
            
            ## Atenuate the initialization for L2F
            if self.args.attenuate:
                task_embeddings = self.get_task_embeddings(spt_x=x_spt[i],spt_y=y_spt[i],
                                                           names_weights_copy=names_weights_copy, use_bn=use_bn)
                names_weights_copy = self.attenutate_init(task_embeddings=task_embeddings, 
                                                          names_weights_copy=names_weights_copy)

            for num_step in range(self.args.number_of_training_steps_per_iter):
                spt_loss, _, _ = self.net_forward(x=x_spt[i], y=y_spt[i], 
                                                  params=names_weights_copy, 
                                                  bkp_running_statistics=False, 
                                                  training=True, num_step=num_step, noise_sigma=sigma_spt[i],
                                                  use_bn=use_bn, calc_metrics=False)
                
                names_weights_copy = self.apply_inner_loop_update(loss=spt_loss, 
                                                                  names_weights_copy=names_weights_copy, 
                                                                  use_second_order=use_second_order, 
                                                                  current_step=num_step)
                if use_multi_step_loss_optimization and training_phase and epoch<self.args.multi_step_loss_epochs:
                    '''
                    use multi step loss
                    '''
                    qry_loss, ssim_val, psnr_val = self.net_forward(x=x_qry[i], y=y_qry[i], 
                                                                    params=names_weights_copy, 
                                                                    bkp_running_statistics=False, 
                                                                    training=True, num_step=num_step, noise_sigma=sigma_qry[i],
                                                                    use_bn=use_bn, calc_metrics=calc_metrics)
                    task_losses.append(per_step_loss_importances[num_step]*qry_loss)
                    if calc_metrics:
                        total_ssims[num_step] += ssim_val
                        total_psnrs[num_step] += psnr_val
                else:
                    if num_step != (self.args.number_of_training_steps_per_iter-1):
                        _, ssim_val, psnr_val = self.net_forward(x=x_qry[i], y=y_qry[i], 
                                                                            params=names_weights_copy, 
                                                                            bkp_running_statistics=False, 
                                                                            training=True, num_step=num_step, noise_sigma=sigma_qry[i],
                                                                            use_bn=use_bn, calc_metrics=calc_metrics)
                    else:
                        qry_loss, ssim_val, psnr_val = self.net_forward(x=x_qry[i], y=y_qry[i], 
                                                                            params=names_weights_copy, 
                                                                            bkp_running_statistics=False, 
                                                                            training=True, num_step=num_step, noise_sigma=sigma_qry[i],
                                                                            use_bn=use_bn, calc_metrics=calc_metrics)
                        task_losses.append(qry_loss)
                    if calc_metrics:
                        total_psnrs[num_step] += psnr_val
                        total_ssims[num_step] += ssim_val
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            '''
            if not training_phase:
                self.net.restore_bkp_stats()
            '''
        losses = self.get_accross_task_loss_metrics(total_losses=total_losses)
        if calc_metrics:
            losses['psnrs'] = total_psnrs/task_num
            losses['ssims'] = total_ssims/task_num
        losses['loss_importances'] = per_step_loss_importances.detach().cpu().numpy()
        return losses
    
    def loss_func(self, x, y):
        '''
        Compute loss with x and y as input: target is y
        '''
        # return F.l1_loss(x, y, reduction='mean')*2*np.sqrt(1e-3) + self.ssim_weight*ssim.forward(x, y)+0.03*self.channelLayer(x, y)
        # return F.l1_loss(x, y, size_average=True)+self.ssim_weight*ssim.forward(x, y)+0.5*self.channelLayer(x, y)
        # print(F.l1_loss(x,y), ssim.forward(x, y), self.gradientLoss(x, y))
        return F.l1_loss(x, y) + ssim.forward(x, y)*self.ssim_weight # + self.gradientLoss(x, y)*self.channel_weight
        # return F.mse_loss(x, y)
    
    def net_forward(self, x, y, params, bkp_running_statistics, training, num_step, noise_sigma, use_bn=True, calc_metrics=False):
        '''
        A base model forward pass on some data points x. Using the params in params
        :param x: of shape [batch_sz, img_c, img_h, img_w]
        :param calc_metrcs: whether to calculate metrics
        '''
        if use_bn:
            # print('[===] {}', training)
            preds = self.net.forward(x, params=params, training=training, noise_sigma=noise_sigma,
                                     bkp_running_statistics=bkp_running_statistics, num_step=num_step)
        else:
            preds = self.net.forward(x, params=params, num_step=num_step, noise_sigma=noise_sigma)
        loss = self.loss_func(preds, y)
        if calc_metrics:
            with torch.no_grad():
                ssim_val = ssim.ssim(preds, y)
                psnr_val = psnr.calc_psnr(preds, y)
        else:
            ssim_val, psnr_val = None, None
        return loss, ssim_val, psnr_val
    
    def train_forward_prop(self, data_batch, epoch, use_bn=True, calc_metrics=False):
        losses = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=self.args.use_second_order and 
                              epoch>self.args.first_order_to_second_order_epoch, use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization, 
                              training_phase=True, use_bn=use_bn, calc_metrics=calc_metrics)
        return losses
    
    def evaluation_forward_prop(self, data_batch, epoch, use_bn=True, calc_metrics=True):
        losses = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False, use_multi_step_loss_optimization=False, 
                              training_phase=False, use_bn=use_bn, calc_metrics=calc_metrics)  ## training_phase needs discussion
        return losses
    
    def meta_update(self, loss):
        '''
        Applies meta update process
        '''
        self.meta_optim.zero_grad()
        loss.backward()
        '''
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                param.grad.data.clamp_(-10, 10)  ## clamp gradient
        '''
        self.meta_optim.step()
    
    def run_train_iter(self, data_batch, epoch, use_bn=True, calc_metrics=False):
        if self.current_epoch != epoch:
            self.current_epoch = epoch
        if not self.training:
            self.train()
        x_spt, y_spt, x_qry, y_qry, sigma_spt, sigma_qry = data_batch
        x_spt = torch.FloatTensor(x_spt).to(self.device)
        y_spt = torch.FloatTensor(y_spt).to(self.device)
        x_qry = torch.FloatTensor(x_qry).to(self.device)
        y_qry = torch.FloatTensor(y_qry).to(self.device)
        sigma_spt = torch.FloatTensor(sigma_spt).to(self.device)
        sigma_qry = torch.FloatTensor(sigma_qry).to(self.device)
        
        data_batch = (x_spt, y_spt, x_qry, y_qry, sigma_spt, sigma_qry)
        losses = self.train_forward_prop(data_batch=data_batch, epoch=epoch, use_bn=use_bn, calc_metrics=calc_metrics)
        self.meta_update(loss=losses['loss'])
        losses['learning_rate'] = self.scheduler.get_last_lr()[0]
        self.meta_optim.zero_grad()
        # if schedule_lr:
           #  print('schedule lr: schedule_lr')
        self.scheduler.step(epoch=epoch)
        return losses
    
    def run_validation_iter(self, data_batch, use_bn=True, calc_metrics=True):
        '''
        if self.training:
            self.eval()
            '''
        x_spt, y_spt, x_qry, y_qry, sigma_spt, sigma_qry = data_batch
        x_spt = torch.FloatTensor(x_spt).to(self.device)
        y_spt = torch.FloatTensor(y_spt).to(self.device)
        x_qry = torch.FloatTensor(x_qry).to(self.device)
        y_qry = torch.FloatTensor(y_qry).to(self.device)
        sigma_spt = torch.FloatTensor(sigma_spt).to(self.device)
        sigma_qry = torch.FloatTensor(sigma_qry).to(self.device)
        
        data_batch = (x_spt, y_spt, x_qry, y_qry, sigma_spt, sigma_qry)
        losses = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch, use_bn=use_bn, calc_metrics=calc_metrics)
        return losses
    
    def save_model(self, model_save_dir, state):
        state['net'] = self.state_dict()
        state['optim'] = self.meta_optim.state_dict()
        state['lr_scheduler'] = self.scheduler.state_dict()
        if self.args.attenuate:
            state['attenuate_weight'] = self.attenuate_weight
        torch.save(state, f=model_save_dir)
    
    def load_model(self, model_save_path, map_location='cuda'):
        ckp = torch.load(model_save_path, map_location=map_location)
        self.meta_optim.load_state_dict(ckp['optim'])
        self.scheduler.load_state_dict(ckp['lr_scheduler'])
        self.load_state_dict(ckp['net'])
        if self.args.attenuate:
            self.attenuate_weight = ckp['attenuate_weight']
        if torch.max(self.update_lr) < self.args.update_lr:
            self.update_lr.data = self.update_lr.data + self.args.update_lr
    

    def test_with_attenuate(self, x, training=False, noise_sigma=None, bkp_running_statistics=False, verbose=False):
        names_weights_copy = self.get_inner_loop_params_dict(self.net.named_parameters())
        if not self.args.attenuate:
            out = self.net.forward(x, num_step=0, params=names_weights_copy, training=False, noise_sigma=noise_sigma,
                                   bkp_running_statistics=bkp_running_statistics)
        else:
            updated_weights = list(map(
                lambda current_params, gamma: ((gamma)*current_params.to(self.device)), names_weights_copy.values(), self.attenuate_weight
            ))
            names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))
            if verbose:
                print(self.attenuate_weight)
                n = list(names_weights_copy.keys())[0]
                p = names_weights_copy[n][0, 0, ...]
                print(p)
            with torch.no_grad():
                out = self.net.forward(x, num_step=0, params=names_weights_copy, training=False, bkp_running_statistics=False)
        return out
            
        # return ckp


if __name__ == '__main__':
    
    ### test class MetaUnit
    import sys
    sys.path.append('..')
    from utils.arguments import get_args
    from nets import *
    from optimizers import GradientDescentLearningRule
    from ImgLIPNfreqsKshot import ImgLIPNfreqsKshot
    args = get_args()
    kwargs = {'Agg_input': True, 'input_channels': 3}
    database = ImgLIPNfreqsKshot(root_dir='../MetaLIP/data/ProcessData/Rain100-new/', batch_size=8, n_freqs=16, k_shot=1, k_query=1, patch_size=50)
    torch.manual_seed(1)
    np.random.seed(1)
    
    ### test ResAggKstages model
    net = ResAggKstages(in_channels=3, num_filters=16, **kwargs)
    # print(net)
    meta_learner = MetaUnit(args=args, net=net)
    data_batch = database.next(mode='train')
    ## test net_forward function
    x_spt, y_spt, x_qry, y_qry = data_batch
    x_spt = torch.FloatTensor(x_spt).to(args.device)
    y_spt = torch.FloatTensor(y_spt).to(args.device)
    x_qry = torch.FloatTensor(x_qry).to(args.device)
    y_qry = torch.FloatTensor(y_qry).to(args.device)
    loss, s_val, p_val = meta_learner.net_forward(x_spt[0], y_spt[0], params=None, use_bn=False, bkp_running_statistics=False, num_step=0, training=True
                             , calc_metrics=True)
    print('loss: {}, ssim: {}, psnr: {}'.format(loss, s_val, p_val))
    print('=== test net_forward function passed.')
    ## test forward function
    data_batch = (x_spt, y_spt, x_qry, y_qry)
    losses = meta_learner.forward(data_batch, epoch=0, use_second_order=False, use_multi_step_loss_optimization=False, 
                                  training_phase=True, use_bn=False, calc_metrics=True)
    print('loss: ', losses['loss'])
    print('loss importances: ', losses['loss_importances'])
    print('inner loop metrics: ')
    print('ssims: {}, psnrs: {}'.format(losses['ssims'], losses['psnrs']))
    print('=== test forward function passed.')
    ### test train_forword_prop function
    losses = meta_learner.train_forward_prop(data_batch=data_batch, epoch=0, use_bn=False, calc_metrics=True)
    print('[iter {}] loss: {}'.format(1, losses['loss']))
    print('[iter {}] loss importances: {}'.format(1, losses['loss_importances']))
    print('inner loop metrics: {}')
    print('[iter {}] ssims: {}, psnrs: {}'.format(1, losses['ssims'], losses['psnrs']))
    print('=== test train_forward_prop function passed.')
    ## test run_train_iter function
    for i in range(10):
        data_batch = database.next(mode='train')
        losses = meta_learner.run_train_iter(data_batch, epoch=0, use_bn=False,calc_metrics=True)
        print('[iter {}] loss: {}'.format(i+1, losses['loss']))
        print('[iter {}] loss importances: {}'.format(i+1, losses['loss_importances']))
        print('[iter {}] ssims: {}, psnrs: {}'.format(i+1, losses['ssims'], losses['psnrs']))
    print('=== test run_train_iter function passed.')
    print('=== test ResAggKstages class passed.')
    state_dict = meta_learner.state_dict()
    for k, v in state_dict.items():
        print(k)
    
    
    
    """
    ## test ResBNAggKstages class
    resbnagg4 = ResBNAggKstages(in_channels=3, num_filters=16, K=4, no_bn_learnable_params=False, args=args, device=args.device, **kwargs)
    metabn_learner = MetaUnit(args=args, net=resbnagg4)
    data_batch = database.next(mode='train')
    ## test net_forward function
    x_spt, y_spt, x_qry, y_qry = data_batch
    x_spt = torch.FloatTensor(x_spt).to(args.device)
    y_spt = torch.FloatTensor(y_spt).to(args.device)
    x_qry = torch.FloatTensor(x_qry).to(args.device)
    y_qry = torch.FloatTensor(y_qry).to(args.device)
    loss, s_val, p_val = metabn_learner.net_forward(x_spt[0], y_spt[0], params=None, use_bn=True, bkp_running_statistics=False, num_step=0, training=True
                             , calc_metrics=True)
    print('loss: {}, ssim: {}, psnr: {}'.format(loss, s_val, p_val))
    print('=== test net_forward function passed.')
    ## test forward function
    data_batch = (x_spt, y_spt, x_qry, y_qry)
    losses = metabn_learner.forward(data_batch=data_batch, epoch=0, use_second_order=False, use_multi_step_loss_optimization=False, 
                                    training_phase=True, use_bn=True, calc_metrics=True)
    print('loss: ', losses['loss'])
    print('loss importances: ', losses['loss_importances'])
    print('inner loop metrics: ')
    print('ssims: {}, psnrs: {}'.format(losses['ssims'], losses['psnrs']))
    print('=== test forward function passed.')
    ## test train_forward_prop function
    losses = metabn_learner.train_forward_prop(data_batch=data_batch, epoch=0, use_bn=True,calc_metrics=True)
    print('[iter {}] loss: {}'.format(1, losses['loss']))
    # print('[iter {}] loss importances: {}'.format(1, losses['loss_importances']))
    print('inner loop metrics: {}')
    print('[iter {}] ssims: {}, psnrs: {}'.format(1, losses['ssims'], losses['psnrs']))
    print('=== test train_forward_prop function passed.')
    ## test run_train_iter function
    '''
    for i in range(10):
        data_batch = database.next(mode='train')
        losses = metabn_learner.run_train_iter(data_batch, epoch=0, use_bn=True)
        print('[iter {}] loss: {}'.format(i+1, losses['loss']))
        print('[iter {}] loss importances: {}'.format(i+1, losses['loss_importances']))
        print('inner loop metrics: {}')
        print('[iter {}] ssims: {}, psnrs: {}'.format(i+1, losses['ssims'], losses['psnrs']))
    print('=== test run_train_iter function passed.')
    print('=== test ResAggKstages class passed.')
    '''
    """
