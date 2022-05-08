"""
parser: parse log file to obtain useful info which contains
avg_psnrs, avg_ssims
"""
import matplotlib.pyplot as plt
import os
import numpy as np

def parse_log(log_file, print_steps=50):
    """
    parse log file
    :@print_steps: steps between two print information
    """
    if not os.path.exists(log_file):
        raise Exception('{} file does not exists'.format(log_file))
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    lines = [line.split('INFO]')[-1].strip() for line in lines]
    res = dict()
    for line in lines:
        if line[:4] == 'step':
            infos = line.split(',')
            ## infos: ['step: step', 'loss_val: loss_val', 'avg_psnrs: psnr list', 'avg_ssims:ssim list']
            assert len(infos)==4, infos
            ## handle step
            step = int(infos[0].split(':')[-1].split('/')[0])//print_steps
            ## handle loss_val
            loss_val = float(infos[1].split(':')[-1])
            ## handle psnr
            psnrs = infos[2].split('[')[-1].split(']')[0] ## psnrs: 'p1, p2, p3, p4'
            psnrs = psnrs.split()
            psnrs = [float(psnr) for psnr in psnrs]
            ## handle ssim
            ssims = infos[3].split('[')[-1].split(']')[0] 
            ssims = ssims.split()
            ssims = [float(ssim) for ssim in ssims]

            info_list = [loss_val, psnrs, ssims]
            res[step] = info_list
    
    return res

def parse_info(info_dict):
    total_item = len(info_dict)
    losses = np.zeros(total_item)
    total_steps = len(info_dict[0][2])
    print(' ===== total items: ', total_item)
    print(' ===== total steps: ', total_steps)
    psnrs = np.zeros((total_item, total_steps))
    ssims = np.zeros((total_item, total_steps))

    for k, v in info_dict.items():
        losses[k] = v[0]
        psnrs[k] = np.array(v[1])
        ssims[k] = np.array(v[2])
    return (losses, psnrs, ssims)


def plot_loss(losses):
    plt.plot(losses[1:], 'r', linewidth='1')
    plt.xlabel('Epoch/50')
    plt.ylabel('loss value')
    plt.title('loss curve')
    plt.legend()
    plt.show()

def plot_metrics(metric, name='psnr'):
    items, steps = metric.shape
    plt.plot(np.arange(items-1)[::5], metric[1::5, 0], label='before update', linewidth='0.5')
    for i in range(1, steps, 2):
        plt.plot(np.arange(items-1)[::5], metric[1::5, i], label='{}-th update'.format(i), linewidth='0.5')
    plt.xlabel('Iter/50')
    plt.ylabel('{} value'.format(name))
    plt.title('{} curve for meta learning'.format(name))
    plt.legend()
    plt.show()


if __name__ == '__main__':
    log_file = r'../results/denoise/ResAggSSIM5stages@16-8-1-1/log.txt'
    log_dict = parse_log(log_file)
    print(log_dict[0])
    print(log_dict[1])
    (losses, psnrs, ssims) = parse_info(log_dict)
    plot_loss(losses)
    plot_metrics(psnrs)
    plot_metrics(ssims, name='ssim')


            
