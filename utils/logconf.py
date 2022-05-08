import logging
import os
import torch

def get_logger(log_file, mode='a', verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    #formatter = logging.Formatter(
    #    "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    #)
    formatter = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(log_file, mode='a+')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    '''
    sh with output log contents to cmd
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    ''' 
    return logger

def build_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


def load_checkpoint(ckp_folder, **kwargs):
    """
    load checkpoint from saved logs
    :param ckp_path: path of checkpoint
    :param **kwargs: kwargs (contains keys:)
    """
    if kwargs['mode'] == 'latest':
        ## load from latest model
        ckp = os.path.join(ckp_folder, 'latest.tar')
        if os.path.exists(ckp):
            state = torch.load(ckp)
        else:
            state=None
    elif kwargs['mode'] == 'iteration':
        ## load from iteration
        ckp = os.path.join(ckp_folder, '{}-iterModel.tar'.format(kwargs['iter']))
        if os.path.exists(ckp):
            state = torch.load(ckp)
        else:
            state=None
    else:
        raise NotImplementedError
    return state, ckp
            
