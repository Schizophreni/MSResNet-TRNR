from utils.arguments import  get_args
from nets import MultiScaleResNet, MetaMSResNet
import numpy as np
import torch
import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(0)
args = get_args()

net1 = MultiScaleResNet(3, 48, 3, k1=2, k2=0, args=args)
set_seed(0)
net2 = MetaMSResNet(3, 48, 2, args, withSE=False)

p1 = list(net1.named_parameters())
p2 = list(net2.named_parameters())

for idx, item in enumerate(p1):
    diff = p1[idx][1] - p2[idx][1]
    diff = torch.sum(diff)
    if diff != 0:
        print(p1[idx][0], diff)

x = torch.rand(1, 3, 50, 50)
'''
out1 = net1.layer_dict['conv_head'](x)
out1 = F.leaky_relu(out1, 0.2, True)
out1 = torch.cat([out1, x], dim=1)

out2 = net2.layer_dict['head'](x)
out2 = F.leaky_relu(out2, 0.2, True)
out2 = torch.cat([out2, x], dim=1)

print(torch.sum(out1-out2), 'diff for head')

shortcut1 = out1
shortcut2 = out2

out1 = net1.layer_dict['mshier1'](out1, 0)
out2 = net2.layer_dict['body1'](out2, 0)
print(torch.sum(out1-out2), 'diff for body')

out1 = torch.cat([out1, x], dim=1)
out2 = torch.cat([out2, x], dim=1)
'''

out1 = net1(x, 0, training=True)
out2 = net2(x, 0, training=True)
print(torch.sum(out1-out2), 'diff for nets')


