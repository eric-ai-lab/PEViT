import logging
import pprint

from torch.utils.collect_env import get_pretty_env_info


def log_arg_env_config(args, config, output_dir):
    logging.info("=> collecting env info (might take some time)")
    logging.info("\n" + get_pretty_env_info())
    logging.info(pprint.pformat(args))
    logging.info(config)
    logging.info(f'=> saving logging info into: {output_dir}')


def submit_predictions(prediction_list, submit_by, config, track, task):
    from vision_benchmark.commands.submit_predictions import submit_predictions_to_leaderboard, submit_model_to_leaderboard

    submission = {
        'dataset_name': config.DATASET.DATASET,
        'model_name': config.MODEL.NAME,
        'track': track,
        'task': task,
        'created_by': submit_by,
        'predictions': [prediction_list]
    }

    logging.info('Submit model and predictions to leaderboard.')
    submit_predictions_to_leaderboard(submission)

    model_info = {
        "name": config.MODEL.NAME,
        "author": config.MODEL.AUTHOR,
        "num_params_in_millions": config.MODEL.NUM_PARAMS_IN_M,
        "pretrained_data": config.MODEL.PRETRAINED_DATA,
        "creation_time": config.MODEL.CREATION_TIME
    }

    submit_model_to_leaderboard(model_info)
