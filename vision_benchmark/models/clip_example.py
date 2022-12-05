from torch import nn


class Example(nn.Module):
    def encode_image():
        """
        This method is called to extract image features for evaluation.
        """
        pass

    def encode_text():
        """
        This method is called to extract text features for evaluation.
        """
        pass


def get_zeroshot_model(config, **kwargs):
    """
    Specify your model here
    """
    model = Example()
    return model
