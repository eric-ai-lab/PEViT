import logging

from sklearn.metrics import roc_auc_score
import vision_evaluation.evaluators as v_eval


def accuracy(y_label, y_pred):
    """ Compute Top1 accuracy
    Args:
        y_label: the ground truth labels. Shape (N,)
        y_pred: the prediction of a model. Shape (N,)
    """
    evaluator = v_eval.TopKAccuracyEvaluator(1)
    evaluator.add_predictions(predictions=y_pred, targets=y_label)
    return evaluator.get_report()['accuracy_top1']


def map_11_points(y_label, y_pred_proba):
    evaluator = v_eval.MeanAveragePrecisionNPointsEvaluator(11)
    evaluator.add_predictions(predictions=y_pred_proba, targets=y_label)
    return evaluator.get_report()[evaluator._get_id()]


def balanced_accuracy_score(y_label, y_pred):
    evaluator = v_eval.BalancedAccuracyScoreEvaluator()
    evaluator.add_predictions(y_pred, y_label)
    return evaluator.get_report()[evaluator._get_id()]


def roc_auc(y_true, y_score):
    if y_score.shape[1] == 2:
        return roc_auc_score(y_true, y_score[:, 1])
    return roc_auc_score(y_true, y_score)


def get_metric(metric_name):
    if metric_name == "accuracy":
        return accuracy
    if metric_name == "mean-per-class":
        return balanced_accuracy_score
    if metric_name == "11point_mAP":
        return map_11_points
    if metric_name == "roc_auc":
        return roc_auc

    logging.error("Undefined metric.")
