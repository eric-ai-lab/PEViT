import numpy as np
import os
import json
import glob

import matplotlib.pyplot as plt

datasets_checked=['caltech-101',
 'cifar-10',
 'cifar-100',
 'country211',
 'dtd',
 'eurosat_clip',
 'fer-2013',
 'fgvc-aircraft-2013b-variants102',
 'food-101',
 'gtsrb',
 'hateful-memes',
 'kitti-distance',
 'mnist',
 'oxford-flower-102',
 'oxford-iiit-pets',
 'patch-camelyon',
#  'ping-attack-on-titan-plus',
#  'ping-whiskey-plus',
 'rendered-sst2',
 'resisc45_clip',
 'stanford-cars',
 'voc-2007-classification',
#  'imagenet-1k'
 ]

two_lr = ['', 'two_lr']




def read_json(log_path, dataset_name='', file_prefix=''):
    
    datasets, accs, num_para = [], [], []

    log_path = os.path.join(log_path, 'predictions/finetuning_5')
    file_filter = f'*.txt'
    txt_path = os.path.join(log_path, file_filter)

    files = glob.glob(txt_path, recursive = True)




    for file in files:
        
        data = ''
        # multiple dict-like string in the file
        try: 
            Lines = open(file, 'r').readlines()   
            texts = open(file, 'r').read() 
            data =  Lines[-1].strip()
            data = data.split(',')[-2].replace('%', '').split(':')[1]
            if dataset_name in file:

                parameter_data = texts.strip().split('n_trainable_params is:')[-1].split(',')[0]
                num_para.append(parameter_data)


                acc_data = texts.strip().split('best acc is:')[-1].split(',')[0]
                accs.append(acc_data)
        except:

            # print(f"Failed at {file}")
            continue
    
    # print(accs)
    return accs, num_para



# finetuning evaluation
def extract_finetune_results(proj_path, dataset_name, num_samples_per_class, rs):
    training_mode = ['finetuning']  # ['finetuning', 'linear_probe']
    # training_mode = ['linear_probe']
    accs = np.zeros([len(training_mode), len(num_samples_per_class)])
    for j in range(len(training_mode)):
        for i in range(len(num_samples_per_class)):
            file_prefix = training_mode[j] + '_' +  num_samples_per_class[i] + '_' 
    
            clip_results, num_para = read_json(proj_path, dataset_name, file_prefix)                

            try:
                print(num_para[-1])
            except:
                print('[]')

    return accs


def extract_finetune_results_para(proj_path, dataset_name, num_samples_per_class, rs):
    training_mode = ['finetuning']  # ['finetuning', 'linear_probe']
    # print(proj_path)
    # training_mode = ['linear_probe']
    accs = np.zeros([len(training_mode), len(num_samples_per_class)])
    for j in range(len(training_mode)):
        for i in range(len(num_samples_per_class)):
            file_prefix = training_mode[j] + '_' +  num_samples_per_class[i] + '_' 
            
            clip_results, num_para = read_json(proj_path, dataset_name, file_prefix)
            try:
                print(clip_results[-1])
            except:
                print('[]')

    return accs

proj_path="vision_benchmark/vitb16"


num_samples_per_class = ['5'] # ['5', '20', '50', 'full']

# random_seeds =  ['log_random_0', 'log_random_1', 'log_random_2'] # , 'random_3_sgd','random_4_sgd'
# random_seeds =  ['log_random_0']
random_seeds =  ['0']

accs_per_dataset_rs = []
for rs in random_seeds:
    proj_path_rs = os.path.join(proj_path, rs, 'vitb16_CLIP', 'log')
    accs_per_dataset = []

    # print number of parameters
    for dataset_name in datasets_checked:
        accs = extract_finetune_results(proj_path_rs, dataset_name, num_samples_per_class, rs)
        accs_per_dataset.append(accs)
    accs_per_dataset_rs.append(accs_per_dataset)

    # print accuracy
    for dataset_name in datasets_checked:
        para = extract_finetune_results_para(proj_path_rs, dataset_name, num_samples_per_class, rs)

