import dataclasses
import datetime
import logging
import math
import pathlib
from typing import List

from .data_class_base import DataClassBase
from .constants import VISION_DATASET_STORAGE
from vision_datasets import DatasetTypes, DatasetHub, Usages, DatasetManifest


class Tasks:
    IC_MULTILABEL = DatasetTypes.IC_MULTILABEL
    IC_MULTICLASS = DatasetTypes.IC_MULTICLASS
    OBJECT_DETECTION = DatasetTypes.OD

    VALID_TYPES = [IC_MULTILABEL, IC_MULTICLASS, OBJECT_DETECTION]

    @staticmethod
    def is_valid(task):
        return task in Tasks.VALID_TYPES


class Tracks:
    LINEAR_PROBING = 'linear_probing'
    TRANSFER_LEARNING = 'transfer_learning'
    ZERO_SHOT = 'zero_shot'

    VALID_TYPES = [LINEAR_PROBING, TRANSFER_LEARNING, ZERO_SHOT]

    @staticmethod
    def is_valid(task, track):
        if track not in Tracks.VALID_TYPES:
            return False

        if task in [Tasks.IC_MULTICLASS, Tasks.IC_MULTILABEL]:
            return True

        if task == Tasks.OBJECT_DETECTION:
            return track != Tracks.LINEAR_PROBING

        return False


@dataclasses.dataclass(frozen=True)
class PredictionSubmission(DataClassBase):
    dataset_name: str
    model_name: str
    created_by: str
    task: str
    track: str
    predictions: List

    def validate(self):
        vision_dataset_json = (pathlib.Path(__file__).resolve().parents[1] / 'resources' / 'datasets' / 'vision_datasets.json').read_text()
        hub = DatasetHub(vision_dataset_json)
        dataset_names = set([x['name'] for x in hub.list_data_version_and_types()])

        self._check_value('dataset_name', lambda x: x and x in dataset_names)
        self._check_value('model_name', lambda x: x)
        self._check_value('created_by', lambda x: x)
        self._check_value('task', lambda x: Tasks.is_valid(x))
        self._check_value('track', lambda x: Tracks.is_valid(self.task, x))
        self._check_value('predictions', lambda x: x)
        dataset_manifest = hub.create_dataset_manifest(VISION_DATASET_STORAGE, None, self.dataset_name, usage=Usages.TEST_PURPOSE)[0]
        logging.info(f'Created test set manifest for {self.dataset_name}')
        for fold_idx, predictions in enumerate(self.predictions):
            PredictionSubmission.validate_predictions(dataset_manifest, predictions, fold_idx)

    @staticmethod
    def validate_predictions(dataset_manifest: DatasetManifest, predictions, fold_idx):
        assert predictions, f'fold {fold_idx}, empty predictions.'
        assert len(predictions) == len(dataset_manifest.images), f'fold {fold_idx}, Number of predictions does not match number of images.'

        if dataset_manifest.data_type in [DatasetTypes.IC_MULTICLASS, DatasetTypes.IC_MULTILABEL]:
            for i, probs in enumerate(predictions):
                if dataset_manifest.data_type == DatasetTypes.IC_MULTICLASS:
                    sum_probs = sum(probs)
                    assert math.isclose(sum_probs, 1.0, rel_tol=1e-3), f'fold {fold_idx}, Sum of predicted prob vector for image {i}: {sum_probs}, should be 1.0.'

                assert all([0.0 <= prob <= 1.0 for prob in probs]), f'fold {fold_idx}, Predicted prob for image {i} not in [0, 1]: {probs}'

        if dataset_manifest.data_type == DatasetTypes.OD:
            # [[[class_index, conf, L, T, R, B], [class_index, conf, L, T, R, B], ..., []], [...], ..., [...]]
            for i, img_wise_bboxes in enumerate(predictions):
                for bbox_pred in img_wise_bboxes:
                    assert PredictionSubmission.is_valid_box(bbox_pred, len(dataset_manifest.labelmap)), f'fold {fold_idx}, Invalid predicted bbox for image {i}: {bbox_pred}'

    @staticmethod
    def is_valid_box(bbox_pred, num_classes):
        return len(bbox_pred) == 6 and (0 <= bbox_pred[0] < num_classes) and (0.0 <= bbox_pred[1] <= 1.0) and all([x >= 0 for x in bbox_pred[2:]]) and (bbox_pred[2] <= bbox_pred[4]) \
            and (bbox_pred[3] <= bbox_pred[5])


@dataclasses.dataclass(frozen=True)
class ModelInfoSubmission(DataClassBase):
    name: str
    author: str
    num_params_in_millions: int
    pretrained_data: str
    creation_time: str

    def validate(self):
        self._check_value('name', lambda x: x)
        self._check_value('author', lambda x: x)
        self._check_value('num_params_in_millions', lambda x: x > 0)
        self._check_value('pretrained_data', lambda x: x)
        self._check_value('creation_time', lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
