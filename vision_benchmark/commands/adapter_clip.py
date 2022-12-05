
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging

import numpy as np
import os
import random

from vision_datasets import DatasetTypes
from vision_benchmark.common.constants import get_dataset_hub
from vision_benchmark.utils import comm, create_logger
from vision_benchmark.evaluation import construct_dataloader
from vision_benchmark.evaluation.adapter_tuning_clip import *
from vision_benchmark.config import config, update_config
# These 2 lines are a walk-around for "Too many open files error". Refer: https://github.com/pytorch/pytorch/issues/11201
import torch.multiprocessing
from vision_benchmark.common.utils import log_arg_env_config, submit_predictions

torch.multiprocessing.set_sharing_strategy('file_system')

MULTILABEL_DATASETS = {"chestx-ray8"}


def add_finetuning_args(parser):
    parser.add_argument('--ds', required=False, help='Evaluation dataset configure file name.', type=str)
    parser.add_argument('--model', required=True, help='Evaluation model configure file name', type=str)
    parser.add_argument('--submit-predictions', help='submit predictions and model info to leaderboard.', default=False, action='store_true')
    parser.add_argument('--submit-by', help='Person who submits the results.', type=str)
    parser.add_argument('--no-tuning', help='No hyperparameter-tuning.', default=False, type=lambda x:x.lower()=="true")
    parser.add_argument('--l2', help='(Inverse) L2 regularization strength. This option is only useful when option --no-tuning is True.', default=0.316, type=float)
    parser.add_argument('--lr', help='Test with a specific learning rate. This option is only useful when option --no-tuning is True.', default=0.001, type=float)
    parser.add_argument('--run', help='Run id', default=1, type=int)
    parser.add_argument('--fix_seed', help='Fix the random seed. [-1] not fixing the seeds', default=0, type=int)
    parser.add_argument('--save-predictions', help='save predictions logits for analysis.', default=True, action='store_true')

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)    


def main():
    parser = argparse.ArgumentParser(description='Test a classification model, with finetuning.')
    add_finetuning_args(parser)
    args = parser.parse_args()

    args.cfg = args.ds
    update_config(config, args)
    args.cfg = args.model
    update_config(config, args)
    config.defrost()
    config.NAME = ''
    config.freeze()

    if args.submit_predictions:
        assert args.submit_by

    if args.fix_seed != -1:
        random.seed(args.fix_seed)
        np.random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        torch.cuda.manual_seed_all(args.fix_seed)

    n_samples = str(config.DATASET.NUM_SAMPLES_PER_CLASS) if config.DATASET.NUM_SAMPLES_PER_CLASS > 0 else 'full'
    exp_name = 'finetuning_' + n_samples
    if config.TRAIN.TWO_LR: exp_name += '_two_lr'
    final_output_dir = create_logger(config, exp_name)

    if config.DATASET.NUM_SAMPLES_PER_CLASS == 1:
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 2
        config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN = False
        config.freeze()

    if comm.is_main_process():
        log_arg_env_config(args, config, final_output_dir)

    if config.DATASET.DATASET == 'patch-camelyon' and config.DATASET.NUM_SAMPLES_PER_CLASS == -1:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logging.info(f'Detecting large dataset with {config.DATASET.NUM_SAMPLES_PER_CLASS}-shot.')
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = 10000
        config.freeze()
        logging.info(f'Used the subset ({config.DATASET.NUM_SAMPLES_PER_CLASS}-shot) to train the model.')

    logging.info(f'{config.DATASET.DATASET} is a dataset.')
    train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)

    # Run full model finetuning
    prediction_folder = os.path.join(config.OUTPUT_DIR, 'predictions', exp_name)
    logging.info(f'Finetuning with full model. This may take several minutes to hours depending on the size of your data.')
    path = os.path.join(prediction_folder, f'seed{config.DATASET.RANDOM_SEED_SAMPLING}_{config.DATASET.DATASET}.json' )
    logging.info(f'Path is {path}.')

    logging.info(f'seed{config.DATASET.RANDOM_SEED_SAMPLING}_{config.DATASET.DATASET}.json')
    best_acc, model_info = adapt_clip(train_dataloader, val_dataloader, test_dataloader, args.no_tuning, args.lr, args.l2, config)



    test_predictions = model_info['best_logits']

    if args.save_predictions:

    
        import json

        # a hack to control the json dump float accuracy
        # if you find the accuracy is not enough, pleae consider increasing `prec`.
        def json_prec_dump(data, prec=6):
            return json.dumps(json.loads(json.dumps(data), parse_float=lambda x: round(float(x), prec)))

        results_dict = {
            'model_name': config.MODEL.NAME,
            'dataset_name': config.DATASET.DATASET,
            'num_trainable_params': model_info.get('n_trainable_params', None),
            'num_params': model_info.get('n_params', None),
            'num_visual_params': model_info.get('n_visual_params', None),
            'num_backbone_params': model_info.get('n_backbone_params', None),
            'n_shot': config.DATASET.NUM_SAMPLES_PER_CLASS,
            'rnd_seeds': [config.DATASET.RANDOM_SEED_SAMPLING],
            'predictions': [test_predictions.tolist()],
        }
        json_string = json_prec_dump(results_dict)

        prediction_folder = os.path.join(config.OUTPUT_DIR, 'predictions', exp_name)
        os.makedirs(prediction_folder, exist_ok=True)


        with open(os.path.join(prediction_folder, f'seed{config.DATASET.RANDOM_SEED_SAMPLING}_{config.DATASET.DATASET}.json' ) , 'w') as outfile:
            outfile.write(json_string)

if __name__ == '__main__':
    main()
