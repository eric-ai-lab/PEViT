import os.path as op
import yaml
from yacs.config import CfgNode as CN

from ..utils.comm import comm

_C = CN()

_C.BASE = ['']
_C.NAME = ''
_C.DATA_DIR = ''
_C.DIST_BACKEND = 'nccl'
_C.GPUS = (0,)
# _C.LOG_DIR = ''
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.OUTPUT_DIR = ''
_C.PIN_MEMORY = True
_C.PRINT_FREQ = 20
_C.RANK = 0
_C.VERBOSE = True
_C.WORKERS = 4

_C.AMP = CN()
_C.AMP.ENABLED = False
_C.AMP.MEMORY_FORMAT = 'nchw'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'cls_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_PARAMS_IN_M = 0.0
_C.MODEL.AUTHOR = ''
_C.MODEL.PRETRAINED_DATA = ''
_C.MODEL.CREATION_TIME = ''

_C.MODEL.CLIP_FP32 = False

_C.MODEL.PRETRAINED_LAYERS = ['*']
_C.MODEL.NUM_CLASSES = 1000
_C.MODEL.SPEC = CN(new_allowed=True)
_C.MODEL.SPEC.TEXT = CN(new_allowed=True)
_C.MODEL.SPEC.TEXT.CONTEXT_LENGTH = 77

_C.MODEL.STATS = CN(new_allowed=True)

_C.KNOWLEDGE = CN(new_allowed=True)
_C.KNOWLEDGE.WORDNET = CN(new_allowed=True)
_C.KNOWLEDGE.WORDNET.USE_HIERARCHY = False
_C.KNOWLEDGE.WORDNET.USE_DEFINITION = False

_C.LOSS = CN()
_C.LOSS.LABEL_SMOOTHING = 0.0
_C.LOSS.LOSS = 'softmax'
_C.LOSS.FOCAL = CN()
_C.LOSS.FOCAL.NORMALIZE = True
_C.LOSS.FOCAL.ALPHA = 1.0
_C.LOSS.FOCAL.GAMMA = 0.5

# DATASET related params
_C.DATASET = CN(new_allowed=True)
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'imagenet'
_C.DATASET.IMAGE_SIZE = (224,)
_C.DATASET.CENTER_CROP = True
_C.DATASET.NUM_CLASSES = 0
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VAL_SET = ''
_C.DATASET.TEST_SET = 'val'
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.LABELMAP = ''
_C.DATASET.TRAIN_TSV_LIST = []
_C.DATASET.TEST_TSV_LIST = []
_C.DATASET.COCO = CN(new_allowed=True)
_C.DATASET.COCO.SCALES = ['m', 'l']
_C.DATASET.COCO.BALANCE_DATA = True
_C.DATASET.NUM_SAMPLES_PER_CLASS = -1 # -1 indicate the full dataset, other it indicates the number of samplers per class in few-shot learning
_C.DATASET.RANDOM_SEED_SAMPLING = 0 # The random seed used to sample the a subset of dataset to few-shot learning
_C.DATASET.MERGE_TRAIN_VAL_FINAL_RUN = True # merge the train and val in the final run

_C.KNOWLEDGE = CN(new_allowed=True)
_C.KNOWLEDGE.WORDNET = CN(new_allowed=True)
_C.KNOWLEDGE.WORDNET.USE_HIERARCHY = False
_C.KNOWLEDGE.WORDNET.USE_DEFINITION = False
_C.KNOWLEDGE.WIKITIONARY = CN(new_allowed=True)
_C.KNOWLEDGE.WIKITIONARY.USE_DEFINITION = False
_C.KNOWLEDGE.WIKITIONARY.WIKI_DICT_PATH = 'resources/knowledge/external'
_C.KNOWLEDGE.GPT3 = CN(new_allowed=True)
_C.KNOWLEDGE.GPT3.USE_GPT3 = False
_C.KNOWLEDGE.GPT3.GPT3_DICT_PATH = 'resources/knowledge/gpt3'
_C.KNOWLEDGE.AGGREGATION = CN(new_allowed=True)
_C.KNOWLEDGE.AGGREGATION.MEHTOD = 'WIKI_AND_GPT3' # 'WIKI_THEN_GPT3', 'WIKI_AND_GPT3'
_C.KNOWLEDGE.AGGREGATION.NUM_GPT3_ITEMS = 1 # Range from 1 to 5

# Used by ClassAwareTargetSizeSampler. Set to the desired dataset size
# Or by default, sample all available data.
_C.DATASET.TARGET_SIZE = -1

# training data augmentation
_C.INPUT = CN()
_C.INPUT.MEAN = [0.485, 0.456, 0.406]
_C.INPUT.STD = [0.229, 0.224, 0.225]

# data augmentation
_C.AUG = CN()
_C.AUG.RANDOM_CENTER_CROP = False
_C.AUG.SCALE = (0.08, 1.0)
_C.AUG.RATIO = (3.0 / 4.0, 4.0 / 3.0)
_C.AUG.COLOR_JITTER = [0.4, 0.4, 0.4, 0.1, 0.0]
_C.AUG.GRAY_SCALE = 0.0
_C.AUG.GAUSSIAN_BLUR = 0.0
_C.AUG.DROPBLOCK_LAYERS = [3, 4]
_C.AUG.DROPBLOCK_KEEP_PROB = 1.0
_C.AUG.DROPBLOCK_BLOCK_SIZE = 7
_C.AUG.MIXUP_PROB = 0.0
_C.AUG.MIXUP = 0.0
_C.AUG.MIXCUT = 0.0
_C.AUG.MIXCUT_MINMAX = []
_C.AUG.MIXUP_SWITCH_PROB = 0.5
_C.AUG.MIXUP_MODE = 'batch'
_C.AUG.MIXCUT_AND_MIXUP = False
_C.AUG.TIMM_AUG = CN(new_allowed=True)
_C.AUG.TIMM_AUG.USE_LOADER = False
_C.AUG.TIMM_AUG.USE_TRANSFORM = False

_C.SWA = CN()
_C.SWA.ENABLED = False
_C.SWA.DEVICE = 'cpu'
_C.SWA.BEGIN_EPOCH = -1
_C.SWA.LR_RATIO = 0.5
_C.SWA.ANNEAL_EPOCHS = 10
_C.SWA.ANNEAL_STRATEGY = 'cos'
_C.SWA.FROZEN_BN = False

# train
_C.TRAIN = CN()

_C.TRAIN.AUTO_RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.LR_SCHEDULER = CN(new_allowed=True)
_C.TRAIN.SCHEDULE = []
# _C.TRAIN.LR_FACTOR = 0.1
# _C.TRAIN.LR_STEP = [30, 60, 90]
_C.TRAIN.LR = 0.001

_C.TRAIN.SEARCH_WD_LOG_LOWER = -6
_C.TRAIN.SEARCH_WD_LOG_UPPER = 6

_C.TRAIN.FREEZE_IMAGE_BACKBONE = False
_C.TRAIN.TWO_LR = False # if two lr is used, one is for backbone, the other for head
_C.TRAIN.USE_CHANNEL_BN = True # if the channel bn should be used
_C.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER = False # if linear head is initialized by the output of the text encoder
_C.TRAIN.LOGIT_SCALE_INIT = 'none' # how is the logit scale initialized
_C.TRAIN.TRAINABLE_LOGIT_SCALE = False # if the logit scale is trainable
_C.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ = False # if linear head is merged with the last projection layer in visual encoder
_C.TRAIN.NORMALIZE_VISUAL_FEATURE = False # normalize the feature output from the visual encoder
_C.TRAIN.SEARCH_RESULT_ON_LAST_EPOCH = False # use the last epoch accuracy for the hyperparameter search
_C.TRAIN.OPTIMIZER = 'sgd'
_C.TRAIN.OPTIMIZER_ARGS = CN(new_allowed=True)
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.WD_SEARCH_LEFT = False # WD_SEARCH_LEFT is used in the inital release, whereas we later find WD_SEARCH_IDX to be more stable.


# _C.TRAIN.WD_SEARCH_LEFT = True



_C.TRAIN.WITHOUT_WD_LIST = []
_C.TRAIN.NESTEROV = True
# for adam
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 100
_C.TRAIN.EXTRA_FINAL_TRAIN_EPOCH = 0

_C.TRAIN.EMULATE_ZERO_SHOT = False

_C.TRAIN.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TRAIN.BATCH_SIZE_PER_GPU = 32
_C.TRAIN.SHUFFLE = True

_C.TRAIN.EMA_DECAY = 0.0
_C.TRAIN.EVAL_BEGIN_EPOCH = 0

_C.TRAIN.LARC = False

_C.TRAIN.DETECT_ANOMALY = False

_C.TRAIN.CLIP_GRAD_NORM = 0.0

_C.TRAIN.LOADER = 'blobfuse'  # available options: "blobfuse" and "azcopy"
_C.TRAIN.SAMPLER = 'default'  # available options: 'default', 'class_aware', 'class_aware_target_size', 'chunk'
_C.TRAIN.NUM_SAMPLES_CLASS = 'average'  # 'average', 'median' or any integer

_C.TRAIN.SAVE_ALL_MODELS = False

# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 32
_C.TEST.CENTER_CROP = True
_C.TEST.IMAGE_SIZE = [224, 224]  # width * height, ex: 192 * 256
_C.TEST.INTERPOLATION = 2
_C.TEST.MODEL_FILE = ''
_C.TEST.REAL_LABELS = False
_C.TEST.VALID_LABELS = ''
_C.TEST.METRIC = ''

_C.FINETUNE = CN()
_C.FINETUNE.FINETUNE = False
_C.FINETUNE.USE_TRAIN_AUG = False
_C.FINETUNE.BASE_LR = 0.003
_C.FINETUNE.BATCH_SIZE = 512
_C.FINETUNE.EVAL_EVERY = 3000
# _C.FINETUNE.MODEL_FILE = ''
_C.FINETUNE.FROZEN_LAYERS = []

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False

_C.USE_DEEPSPEED = False

_C.DEEPSPEED = CN(new_allowed=True)


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, op.join(op.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    config.merge_from_list(args.opts)
    config.TRAIN.LR *= comm.world_size
    file_name, _ = op.splitext(op.basename(args.cfg))
    config.NAME = file_name + config.NAME
    config.RANK = comm.rank

    if hasattr(config.TRAIN.LR_SCHEDULER, "METHOD"):
        if 'timm' == config.TRAIN.LR_SCHEDULER.METHOD:
            config.TRAIN.LR_SCHEDULER.ARGS.epochs = config.TRAIN.END_EPOCH

    if 'timm' == config.TRAIN.OPTIMIZER:
        config.TRAIN.OPTIMIZER_ARGS.lr = config.TRAIN.LR

    aug = config.AUG
    if aug.MIXUP > 0.0 or aug.MIXCUT > 0.0 or aug.MIXCUT_MINMAX:
        aug.MIXUP_PROB = 1.0
    config.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
