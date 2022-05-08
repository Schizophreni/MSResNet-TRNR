import os
import sys
import time

cmd = 'screen python train.py --ssim_weight 0'

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().strip().split('|')
    gpu_status = [g for g in gpu_status if g!='' and g!='\n']
    gpu_num = len(gpu_status) // 3
    
    gpu_memories = []
    gpu_powers = []
    
    for i in range(gpu_num):
        # print(gpu_status[i*3].split())
        power = gpu_status[i*3].split()[-3]
        # print(power)
        power = int(power[:-1])
        mem = gpu_status[i*3+1].strip().split()[0]
        mem = int(mem[:-3])

        gpu_powers.append(power)
        gpu_memories.append(mem)
    return gpu_powers, gpu_memories

def setup(interval=2):
    
    while True:
        gpu_powers, gpu_memories = gpu_info()
        min_mem = min(gpu_memories)
        idx = gpu_memories.index(min_mem)
        if min_mem < 1000:
            break
        msg = 'Index: {} | power: {}MiB | power: {}W | monitoring >>>\n'.format(idx, 
               gpu_memories[idx], gpu_powers[idx])
        sys.stdout.write('\r'+msg)
        sys.stdout.flush()
        time.sleep(interval)
    print('running command: ', cmd)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(idx)
    os.system(cmd)

if __name__ == '__main__':
    setup()



