import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def traverse_folder(folder_path, folder_format='cp-{}', traverse_format='cp-*', img_token='*.png', verbose=False):
    """
    generate information of all classes
    :param folder_path: path of image folder
    :param folder_format: folder name format
    :param traverse_format: 
    :return : class_samples, sample count of each class
    """
    sampling_sequence = []
    all_classes = glob.glob(os.path.join(folder_path, traverse_format))
    class_num = len(all_classes)
    print('traverse folder: {}, total classes: {}'.format(folder_path, class_num))
    
    class_samples = []
    for i in range(class_num):
        imgs_folder = os.path.join(folder_path, folder_format.format(i))
        imgs_num = len(glob.glob(os.path.join(imgs_folder, img_token)))
        class_samples.append(imgs_num)
    class_indexes = list(range(class_num))
    ## plot
    print(len(class_samples))
    if verbose:
        plt.bar(class_indexes, class_samples, width=6.0)
        plt.xlabel('Class Index')
        plt.ylabel('Image Count')
        plt.title('Image count in different classes')
        plt.show()
    
    print('total samples: ', np.sum(np.array(class_samples)))
    '''
    for index, num in enumerate(class_samples):
        print('({}, {})'.format(index, num))
    '''
    return class_samples

def gen_sampling_sequence(class_samples, hard_task_indexes, class_per_sampling):
    """
    generate sampling sequence 
    :param class_samples: sample count in each class
    :param hard_task_indexes: hard classes indexes
    :param class_per_sampling: sample class count each sampling
    :return: sampling sequence of an epoch
    """
    sampling_sequence = []
    class_samples = np.array(class_samples)
    # manipulate sampling 
    low_sample_class_indexes = class_samples <= 5
    class_samples[low_sample_class_indexes] = class_samples[low_sample_class_indexes]*3
    
    class_indexes = list(range(len(class_samples)))
    while len(class_indexes) + len(hard_task_indexes) >= class_per_sampling: 
## can do sample
        class_indexes = [j for j in class_indexes if class_samples[j] > 0] ## indexes of classes containing samples
        if len(class_indexes) < class_per_sampling and len(class_indexes)+len(hard_task_indexes)>=class_per_sampling:
            ### use hard task to help sampling
            auxillary = np.random.choice(hard_task_indexes, class_per_sampling-len(class_indexes), replace=False)
            selected_classes = class_indexes+list(auxillary)
            selected_classes.sort()
            for j in class_indexes:
                class_samples[j] -= 1 ## sample one
            for j in hard_task_indexes:
                if class_samples[j] <= 0:
                    class_samples[j] = 0 ## don't drop out hard task samples
            sampling_sequence.append(selected_classes)
            # print('Hard sampling: ', selected_classes)
        elif len(class_indexes) >= class_per_sampling: ## use base classes for sampling directly
            selected_classes = np.random.choice(class_indexes, class_per_sampling, replace=False)
            selected_classes.sort()
            sub = [1 if j in selected_classes else 0 for j in range(len(class_samples))]
            class_samples = class_samples - np.array(sub)
            sampling_sequence.append(selected_classes)
            # print('Base sampling: ', selected_classes)
    print('sampling sequence length: ', len(sampling_sequence))
    return sampling_sequence


if __name__ == '__main__':
    folder_path = '../MetaLIP/data/ProcessData/Rain100L-50/train/clean'
    class_samples = traverse_folder(folder_path, verbose=False)
    # class_samples = [5, 10, 9, 7, 6, 4, 7, 9, 11, 6]
    # hard_task_indexes = [2, 4, 7, 10, 6, 9, 11, 100, 230, 300, 150]
    hard_task_indexes = [2, 4, 9, 0, 11, 100]
    seq = gen_sampling_sequence(class_samples, hard_task_indexes, 16)
    print('sampling samples num: , ', 16*len(seq))
    # sample_times = sampling_analysis(seq, len(class_samples), verbose=False)
    '''
    for index, t in enumerate(sample_times):
        print('({}, {})'.format(index, t))
    '''





