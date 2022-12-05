"""
CLIP zeroshot evaluation
"""
import torch
import torch.nn.functional as F
from .metric import get_metric


def clip_zeroshot_evaluator(image_features, text_features, image_labels, config):
    metric = get_metric(config.TEST.METRIC)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_features = torch.from_numpy(image_features).to(device)
    text_features = torch.from_numpy(text_features).to(device)
    image_labels = torch.from_numpy(image_labels).to(device)

    # Normalize image_features
    image_features = F.normalize(image_features)

    # Compute logits
    logits = (100. * image_features @ text_features).softmax(dim=-1)
    result = metric(image_labels.squeeze().cpu().detach().numpy(), logits.cpu().detach().numpy())
    return result, logits, metric.__name__
