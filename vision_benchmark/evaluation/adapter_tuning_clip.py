import os
from re import L
import time
import logging
import pickle
import numpy as np
import sys, json
import random

from torch import nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from .feature import FeatureData
from ..optim import build_optimizer
from ..evaluation.metric import get_metric

from ..common.constants import get_dataset_hub, VISION_DATASET_STORAGE
from ..models import *
from ..datasets import class_map, template_map

from vision_benchmark.datasets import SimpleTokenizer, HFPTTokenizer
from vision_benchmark.evaluation import clip_zeroshot_evaluator, construct_dataloader

import pdb

from tqdm import tqdm
from vision_datasets import ManifestDataset
from nltk.corpus import wordnet as wn
import nltk
from nltk.tokenize import word_tokenize
from .feature import extract_text_features

import gc
from ..evaluation.clip_load import *
from ..evaluation.clip_load import adapter_load

nltk.download('punkt')
nltk.download('wordnet')


MULTILABEL_DATASETS = {"voc-2007-classification","chestx-ray8"}

def gpu_gc():
    gc.collect()
    torch.cuda.empty_cache()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_cls_model(config, feature_type='image'):
    # if config.MODEL.CLIP_FP32:
    #     import clip_vlp as clip
    # else:
    #     import clip
    # import clip_vlp as clip
    # import clip
    # model, _ = clip.load(config.MODEL.NAME, jit=False)
    # model, _ = load(config.MODEL.NAME, jit=False)
    model, _ = adapter_load(config.MODEL.NAME, jit=False)
    if feature_type == 'image':
        model.forward = model.encode_image
    elif feature_type == 'text':
        model.forward = model.encode_text
    else:
        raise Exception('Incorrect model type.')
    return model

class Classifier(torch.nn.Module):
    """
    Linear classifier.
    """

    def __init__(self, config, l2_lambda):
        super(Classifier, self).__init__()

        feature_type="image"


        self.backbone = get_cls_model(config, feature_type=feature_type)

        if config.MODEL.NAME.startswith('vit_'):
            self.backbone.head = self.backbone.head_dist = None

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
            if name.startswith('text') or name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final') \
                or name.startswith('positional_embedding') or name.startswith('logit_scale'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
            # if name.startswith('text') or name.startswith('transformer') or name.startswith('token_embedding') or name.startswith('ln_final'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                param.requires_grad = False

            if config.TRAIN.FREEZE_IMAGE_BACKBONE:
                # freeze for supervised ViT under linear probing settings
                if config.MODEL.NAME.startswith('vit_'):
                    param.requires_grad = False

                if name.startswith('visual.conv1') or name.startswith('visual.ln_pre') or name.startswith('visual.transformer') or name.startswith('visual'): # image encoder names: [visual ResidualAttentionBlock]; text encoder names: [transformers ResidualAttentionBlock]
                    param.requires_grad = False

            if 'adapter' in name:
                param.requires_grad = True


        input_dim, output_dim = config.MODEL.SPEC.EMBED_DIM, config.DATASET.NUM_CLASSES
        self.optim = None
        self.l2_lambda = l2_lambda
        self.channel_bn = torch.nn.BatchNorm1d(
            input_dim,
            affine=False,
        )
        self.layers = torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))

        if config.TRAIN.INIT_HEAD_WITH_TEXT_ENCODER:
            if config.MODEL.SPEC.TEXT.TOKENIZER == 'clip':
                tokenizer = SimpleTokenizer()
            elif 'hf_' in config.MODEL.SPEC.TEXT.TOKENIZER:
                tokenizer = HFPTTokenizer(pt_name=config.MODEL.SPEC.TEXT.TOKENIZER[3:])
            else:
                tokenizer = None

            zeroshot_weights = extract_text_features(config, tokenizer, model=self.backbone, return_numpy=False)
            self.layers[0].weight.data = zeroshot_weights.T.to(self.layers[0].weight.dtype).to(self.layers[0].weight.device).contiguous()
            self.layers[0].bias.data.fill_(0.0)

        if config.TRAIN.MERGE_ENCODER_AND_HEAD_PROJ and self.backbone.visual.proj is not None:
            encoder_proj = self.backbone.visual.proj
            head_proj = self.layers[0].weight.data
            head_bias = self.layers[0].bias.data
            self.backbone.visual.proj = None
            encoder_ic, encoder_oc = encoder_proj.shape
            self.channel_bn = torch.nn.BatchNorm1d(
                encoder_ic,
                affine=False,
            )
            self.layers = torch.nn.Sequential(torch.nn.Linear(encoder_ic, output_dim))
            self.layers[0].weight.data = head_proj @ encoder_proj.T.to(head_proj.dtype).to(head_proj.device)
            self.layers[0].bias.data = head_bias

        self.logit_scale = nn.Parameter(torch.ones([]))
        self.logit_scale.requires_grad = config.TRAIN.TRAINABLE_LOGIT_SCALE
        if config.TRAIN.LOGIT_SCALE_INIT == 'pretrained':
            self.logit_scale.data = self.backbone.logit_scale.data.to(self.logit_scale.dtype).to(self.logit_scale.device)
        elif config.TRAIN.LOGIT_SCALE_INIT == 'ln_cls':
            self.logit_scale.data *= np.log(np.log(config.DATASET.NUM_CLASSES))
        elif config.TRAIN.LOGIT_SCALE_INIT == 'clip':
            self.logit_scale.data *= np.log(1 / 0.07)
        else:
            self.logit_scale.data *= 0

        self.normalize_visual_output = config.TRAIN.NORMALIZE_VISUAL_FEATURE

        if not config.TRAIN.USE_CHANNEL_BN:
            self.channel_bn = nn.Identity()

    def forward(self, img):
        pdtype = img.dtype
        feature = self.backbone(img).to(pdtype)
        outputs = self.channel_bn(feature)

        if self.normalize_visual_output:
            outputs = F.normalize(outputs)

        outputs = self.layers(outputs)
        return outputs


def hyperparameter_sweep(train_dataloader, val_dataloader, config):
    logging.info(f"=> Learning rate {config.TRAIN.LR}: tuning l2 regularization strength.")
    start = time.time()
    l2_lambda_list = np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=97).tolist()
    # l2_lambda_list = np.logspace(-3, 3, num=97).tolist()
    l2_lambda_init_idx = [i for i, val in enumerate(l2_lambda_list) if val in set(np.logspace(config.TRAIN.SEARCH_WD_LOG_LOWER, config.TRAIN.SEARCH_WD_LOG_UPPER, num=7))]
    peak_idx = -1
    peak_score = 0
    iter_num = 0
    for idx in l2_lambda_init_idx:
        config.defrost()
        config.TRAIN.WD = l2_lambda_list[idx]

        # best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        try:
            best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
        except:
            best_score_ = 0.0
            continue       

        if best_score_ > peak_score:
            peak_idx = idx
            peak_score = best_score_
    logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")

    step_span = 8
    while step_span > 0:
        left, right = max(peak_idx - step_span, 0), min(peak_idx + step_span, len(l2_lambda_list) - 1)
        search_idx = []
        if left != peak_idx:
            search_idx.append(left)
        if right != peak_idx:
            search_idx.append(right)
        for idx in search_idx:
            config.TRAIN.WD = l2_lambda_list[left]
            
            # best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            try:
                best_score_ = train_task(train_dataloader, val_dataloader, config, sweep_run=True)
            except:
                best_score_ = 0.0
                continue

            if best_score_ > peak_score:
                peak_idx = idx
                peak_score = best_score_
        iter_num += 1
        logging.info(f"Iteration {iter_num}: l2_lambda: {l2_lambda_list[peak_idx]}, best score {best_score_}")
        step_span //= 2

    logging.info(f"=> Learning rate {config.TRAIN.LR}: The best l2 lambda is {l2_lambda_list[peak_idx]}")
    logging.info('=> Learning rate {}: l2 regularization strength tuning duration time: {:.2f}s'.format(config.TRAIN.LR, time.time() - start))
    return l2_lambda_list[peak_idx], peak_score


def train_task(train_dataloader, test_dataloader, config, sweep_run=False):
    best_acc1 = 0

    model = Classifier(config, 0)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'Number of trainable params: {pytorch_total_params / 1000000}M.')
    logging.info(f"=> model detail is {model}.")


    gpu = config.GPUS

    if len(gpu) == 1:
        torch.cuda.set_device(gpu[0])
        model = model.cuda(gpu[0])

    # define loss function (criterion) and optimizer
    if config.DATASET.DATASET in MULTILABEL_DATASETS:
        criterion = torch.nn.BCEWithLogitsLoss().cuda(gpu)
    else:
        criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = build_optimizer(config, model)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC

    # Generate model statistics
    model_info = {}
    visual_backbone = model.backbone.visual if hasattr(model.backbone, 'visual') and model.backbone.visual is not None else model.backbone
    model_info['n_trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info['n_visual_params'] = sum(p.numel() for p in visual_backbone.parameters())
    model_info['n_backbone_params'] = sum(p.numel() for p in model.backbone.parameters())
    model_info['n_params'] = sum(p.numel() for p in model.parameters())

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        adjust_learning_rate(optimizer, epoch, config)

        # train for one epoch
        if not config.TRAIN.EMULATE_ZERO_SHOT:
            train_one(train_dataloader, model, criterion, optimizer, epoch, config)

        # evaluate on validation set
        acc1, logits = validate(test_dataloader, model, criterion, epoch, config, return_logits=True)

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            model_info['best_logits'] = logits
        best_acc1 = max(acc1, best_acc1)

    logging.info(f'=> Learning rate {config.TRAIN.LR}, L2 lambda {config.TRAIN.WD}: Best score: Acc@1 {best_acc1:.3f}')

    if sweep_run and config.TRAIN.SEARCH_RESULT_ON_LAST_EPOCH:
        return acc1

    del model, criterion, optimizer
    gpu_gc()

    if sweep_run:
        return best_acc1
    else:
        return best_acc1, model_info



def train_one(train_loader, model, criterion, optimizer, epoch, config):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    end = time.time()
    for _,  batch in enumerate(train_loader):

        images, target = batch[:2]

        # measure data loading time
        data_time.update(time.time() - end)

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)

        if images.shape[0] == 1: continue # TODO: check this fix on batch left is size-1
        if target.shape[-1] == 1: target = target[:,0]
        target = target.cuda(config.GPUS[0], non_blocking=True)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        output = model.forward(images)

        # pdb.set_trace()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), images.size(0))

        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100. * metric(labels, logits)
    except:
        metric_result = 0.
    logging.info(f'[Epoch {epoch}] Train: {metric_name} {metric_result:.3f}')


@torch.no_grad()
def validate(val_loader, model, criterion, epoch, config, return_logits=False):
    batch_time = AverageMeter()
    metric = get_metric(config.TEST.METRIC)
    metric_name = metric.__name__

    outputs = []
    targets = []

    model.eval()
    end = time.time()
    for batch in val_loader:
        images, target = batch[:2]

        if len(config.GPUS) == 1:
            images = images.cuda(config.GPUS[0], non_blocking=True)
        target = target.cuda(config.GPUS[0], non_blocking=True)
        if target.shape[-1] == 1: target = target[:,0]

        # compute output
        output = model(images)
        outputs.append(output)
        targets.append(target)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    outputs = torch.cat(outputs, dim=0)
    targets = torch.cat(targets, dim=0)
    logits = outputs.softmax(-1).data.cpu().numpy()
    labels = targets.data.cpu().numpy()
    # TODO: this try except block is used for addressing NaNs on metrics like mAP.
    try:
        metric_result = 100. * metric(labels, logits)
    except:
        metric_result = 0.
    logging.info(f'[Epoch {epoch}] Val: {metric_name} {metric_result:.3f}')

    if return_logits:
        return metric_result, logits
    else:
        return metric_result


def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate based on schedule"""
    lr = config.TRAIN.LR
    for milestone in config.TRAIN.SCHEDULE:
        lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def hyperparameter_sweep_lr(train_dataloader, val_dataloader, config):
    logging.info("=> Start hyperparameter tuning.")
    start = time.time()
    learning_rate_list = np.logspace(-6, -1, num=6).tolist()
    best_score = 0
    best_lr = 0
    best_l2_lambda = 0
    for lr_one in learning_rate_list:
        config.defrost()
        config.TRAIN.LR = lr_one
        config.freeze()
        l2_lambda, best_score_one = hyperparameter_sweep(train_dataloader, val_dataloader, config)
        logging.info(f"=> Learning rate: {lr_one}, best_score {best_score_one}")
        if best_score < best_score_one:
            best_score = best_score_one
            best_lr = lr_one
            best_l2_lambda = l2_lambda
    logging.info(f"Hyper parameter tuning result: learning rate {best_lr}, l2_lambda {best_l2_lambda}")
    logging.info('=> Hyperparameter tuning duration time: {:.2f}s'.format(time.time() - start))
    logging.info('=> Finished hyperparameter tuning.')
    return best_lr, best_l2_lambda


def merge_trainval_loader(train_loader, val_loader):
    # TODO: DataLoader from feature.py get_dataloader()
    trainset, valset = train_loader.dataset, val_loader.dataset
    fullset = trainset.dataset
    assert trainset.dataset is valset.dataset
    assert len(fullset) == len(trainset) + len(valset)

    trainval_loader = torch.utils.data.DataLoader(
        fullset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        sampler=None,
        drop_last=False,
    )
    return trainval_loader


def adapt_clip(train_dataloader, val_dataloader, test_dataloader, no_hyperparameter_tuning, lr, l2, config):
    # no_hyperparameter_tuning = True
    if no_hyperparameter_tuning:
        best_lr = lr
        best_l2_lambda = l2
    else:
        best_lr, best_l2_lambda = hyperparameter_sweep_lr(train_dataloader, val_dataloader, config)

    logging.info("=> The final classifier is on training ...")
    logging.info(f"Hyperparameters: learning_rate = {best_lr}, l2_lambda = {best_l2_lambda}")
    config.defrost()
    config.TRAIN.LR = best_lr
    config.TRAIN.WD = best_l2_lambda
    config.TRAIN.END_EPOCH += config.TRAIN.EXTRA_FINAL_TRAIN_EPOCH
    config.freeze()

    if config.DATASET.DATASET == 'patch-camelyon' and config.DATASET.NUM_SAMPLES_PER_CLASS == 10000:
        # deal with patch camelyon large dataset (search using 10000-shot subset, final run with the full dataset)
        logging.info(f'Used the subset to train the model, regenerating the full set for final run.')
        config.defrost()
        config.DATASET.NUM_SAMPLES_PER_CLASS = -1
        config.freeze()
        logging.info(f'Old: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.')
        train_dataloader, val_dataloader, test_dataloader = construct_dataloader(config)
        logging.info(f'Generated: len(train)={len(train_dataloader.dataset)}, len(val)={len(val_dataloader.dataset)}, len(test)={len(test_dataloader.dataset)}.')

    if config.DATASET.MERGE_TRAIN_VAL_FINAL_RUN:
        trainval_dataloader = merge_trainval_loader(train_dataloader, val_dataloader)
        logging.info(f'Using the full trainval set to train final model. len(dataset)={len(trainval_dataloader.dataset)}')
    else:
        trainval_dataloader = train_dataloader
        logging.info(f'Using the train set only to train final model. len(dataset)={len(trainval_dataloader.dataset)}')
    return train_task(trainval_dataloader, test_dataloader, config)

