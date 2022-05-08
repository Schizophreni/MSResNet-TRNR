'''
Arguments
'''
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_bn_learnable_params', type=bool, help='use learnable params for bn', default=False)
    parser.add_argument('--learnable_bn_gamma', type=bool, help='learn gamma in bn', default=True)
    parser.add_argument('--learnable_bn_beta', type=bool, help='learn beta in bn', default=True)
    parser.add_argument('--number_of_training_steps_per_iter', type=int, help='inner loop steps', default=5)
    parser.add_argument('--use_per_step_bn_statistics', type=int, help='use bn in each inner loop', default=False)
    parser.add_argument('--device', type=torch.device, help='cpu or gpu device', default=torch.device('cuda'))
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    parser.add_argument('--update_lr', type=float, help='inner loop lr', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='outer loop lr', default=0.001)
    parser.add_argument('--multi_step_loss_num_epochs', type=int, help='multi step loss epochs count', default=100)
    parser.add_argument('--multi_step_loss_epochs', type=int, help='how many epochs use MSL strategy', default=100)
    parser.add_argument('--use_second_order', type=bool, help='use second order to build computation graph', default=False)
    parser.add_argument('--first_order_to_second_order_epoch', type=int, help='begin epoch from first order to second order', default=80)
    parser.add_argument('--total_epochs', type=int, help='total training epochs', default=200)
    parser.add_argument('--min_learning_rate', type=float, help='minimum learning rate of meta learner', default=1e-4)
    parser.add_argument('--use_multi_step_loss_optimization', type=bool, help='whether to use MSL strategy', default=False)
    parser.add_argument('--save_folder', type=str, help='save log and model folder', default='./results')
    parser.add_argument('--process_mode', type=str, help='image processing mode', default='derain')
    parser.add_argument('--task_num', type=int, help='number of tasks per iter', default=8)
    parser.add_argument('--freqn', type=int, help='number of freqs per iter', default=16)
    parser.add_argument('--load_begin_iter', type=int, help='load from iterations', default=0)
    parser.add_argument('--load_ckp_mode', type=str, help='mode of loading checkpoint', default='latest')
    parser.add_argument('--iters_every_bar', type=int, help='number of iters every bar', default=50)
    parser.add_argument('--total_iteration', type=int, help='total iterations', default=50000)
    parser.add_argument('--iters_every_test', type=int, help='iterations per testing', default=500)
    parser.add_argument('--num_evaluation_tasks', type=int, help='number of testing tasks', default=100)
    parser.add_argument('--k_shot', type=int, help='K shot', default=1)
    parser.add_argument('--k_query', type=int, help='K query', default=1)
    parser.add_argument('--root_dir', type=str, help='root img dir', default='/home/rw/Public/datasets/derain/Rain800-80-multi-merge')
    parser.add_argument('--patch_size', type=int, help='image patch size', default=50)
    parser.add_argument('--enable_inner_loop_optimizable_bn_params', type=bool, default=True)
    parser.add_argument('--init_inner_loop_learning_rate', type=float, help='initial inner loop learning rate', default=0.001)
    parser.add_argument('--learnable_per_layer_per_step_inner_loop_lr', type=bool, default=False)
    parser.add_argument('--attenuate', type=bool, help='whether to use attentuation', default=False)
    parser.add_argument('--ssim_weight', type=float, help='ssim weight', default=5.0)
    parser.add_argument('--channel_weight', type=float, help='channel loss weight', default=0.0)
    args = parser.parse_args()
    return args
