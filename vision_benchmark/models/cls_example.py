from torch import nn


class Example(nn.Module):
    def forward_features():
        """
        This method is called to extract features for evaluation.
        """
        pass


def get_cls_model(config, **kwargs):
    """
    Specify your model here
    """
    model = Example()
    return model
