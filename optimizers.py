'''
Optimizers for Meta learning inner loop
'''
import torch.nn as nn
import torch

class GradientDescentLearningRule(nn.Module):
    def __init__(self, lr=1e-3):
        '''
        Simple stochastic gradient descent
        p = p -\alpha*dp
        '''
        super(GradientDescentLearningRule, self).__init__()
        assert lr>0., 'learning rate should be positive'
        self.lr = lr
    
    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step=0):
        '''
        Applies a single gradient descent update to all params.
        :param names_grads_wrt_params_dict: A list of gradients of the scalar loss func with respect to the params
        '''
        update_names_weights_dict = dict()
        for name in names_weights_dict.keys():
            update_names_weights_dict[name] = names_weights_dict[name]-self.lr*names_grads_wrt_params_dict[name]
        return update_names_weights_dict


class LSLRGradientDescentLearningRule(nn.Module):
    def __init__(self, total_num_inner_loop_steps, lr=1e-3):
        super(LSLRGradientDescentLearningRule, self).__init__()
        self.lrs = [lr for i in range(total_num_inner_loop_steps)]
    
    def update_params(self, names_weights_dict, names_grads_wrt_params_dict, num_step):
        update_names_weights_dict = dict()
        lr = self.lrs[num_step]
        for name in names_weights_dict.keys():
            update_names_weights_dict[name] = names_weights_dict[name] - lr*names_grads_wrt_params_dict[name]
        return update_names_weights_dict