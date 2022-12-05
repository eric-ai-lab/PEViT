from .clip import (  # noqa: F401
    clip_vitb32
)

from .declip import declip_vitb32

from .filip import filip_vitb32

from .slip import slip_vitb32

from .defilip import defilip_vitb32



def model_entry(config):
    return globals()[config['type']](**config['kwargs'])
