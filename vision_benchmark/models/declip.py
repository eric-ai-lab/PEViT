import torch
import logging

from .declip_model import declip as _declip
from .declip_model import slip as _slip
from .declip_model import filip as _filip

def get_model(config):
    if config.MODEL.NAME in ['filip_vitb32', 'defilip_vitb32']:
        model = _filip.filip_vitb32(**config.MODEL.SPEC.DECLIP)
    elif config.MODEL.NAME == 'slip_vitb32':
        model = _slip.slip_vitb32(**config.MODEL.SPEC.DECLIP)
    else:
        model = _declip.declip_clip_vitb32(**config.MODEL.SPEC.DECLIP)

    model_file = config.TEST.MODEL_FILE
    logging.info(f'=> load model file: {model_file}')

    if model_file.startswith('http'):
        checkpoint = torch.hub.load_state_dict_from_url(model_file, progress=False, map_location="cpu")
    else:
        checkpoint = torch.load(model_file, map_location="cpu")

    # rename moco pre-trained keys
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]

    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        logging.warning('Missing keys: {}'.format(', '.join(incompatible.missing_keys)))
    if incompatible.unexpected_keys:
        logging.warning('Unexpected keys: {}'.format(', '.join(incompatible.unexpected_keys)))

    return model
