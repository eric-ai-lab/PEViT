from collections import OrderedDict
from typing import Tuple, Union
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformers import AutoModel

from timm.models.layers import DropPath, trunc_normal_


from vision_benchmark.utils.comm import comm
from .cls_swin import SwinTransformer
from vision_benchmark.datasets.languages.build import build_tokenizer


logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 attn_mask: torch.Tensor = None,
                 drop_path: float = 0.0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self,
                 context_length: int,
                 vocab_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 drop_path: float = 0.0):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, width)

        self.context_length = context_length
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, width)
        )

        self.width = width
        self.layers = layers
        attn_mask = self.build_attention_mask()
        dpr = [x.item() for x in torch.linspace(0, drop_path, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, dpr[i])
                for i in range(layers)
            ]
        )

        self.ln_final = LayerNorm(width)

        trunc_normal_(self.positional_embedding, std=.02)
        # nn.init.normal_(self.token_embedding, std=.02)
        trunc_normal_(self.token_embedding.weight, std=.02)
        self.apply(self._init_weights)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # if comm.is_main_process():
            #     logger.info('=> init weight of Linear/Conv2d from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # if comm.is_main_process():
                #     logger.info('=> init bias of Linear/Conv2d to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'positional_embedding',
            'token_embedding',
        }

    def forward(self, text: torch.Tensor):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x


class CLIP(nn.Module):
    def __init__(self, config: dict,):
        super().__init__()

        spec_text = config['MODEL']['SPEC']['TEXT']
        assert spec_text['TOKENIZER'] == 'clip', 'Only support clip tokenizer'
        self.tokenizer_style = spec_text['TOKENIZER']
        self.tokenizer = build_tokenizer(spec_text['TOKENIZER'])

        self.text = Transformer(
            context_length=spec_text['CONTEXT_LENGTH'],
            vocab_size=self.tokenizer.get_vocab_size(),
            width=spec_text['WIDTH'],
            layers=spec_text['LAYERS'],
            heads=spec_text['HEADS']
        )

        embed_dim = config['MODEL']['SPEC']['EMBED_DIM']
        self.text_projection = nn.Parameter(
            torch.empty(spec_text['WIDTH'], embed_dim)
        )

        spec_vision = config['MODEL']['SPEC']['VISION']
        self.visual = SwinTransformer(
            img_size=config['TRAIN']['IMAGE_SIZE'][0],
            patch_size=spec_vision['PATCH_SIZE'],
            in_chans=spec_vision['IN_CHANS'],
            num_classes=0,
            embed_dim=spec_vision['EMBED_DIM'],
            depths=spec_vision['DEPTHS'],
            num_heads=spec_vision['NUM_HEADS'],
            window_size=spec_vision['WINDOW_SIZE'],
            mlp_ratio=spec_vision['MLP_RATIO'],
            qkv_bias=spec_vision['QKV_BIAS'],
            qk_scale=spec_vision.get('QK_SCALE', None),
            drop_rate=spec_vision['DROP_RATE'],

            ape=spec_vision['APE'],
            patch_norm=spec_vision['PATCH_NORM'],
            use_checkpoint=False,
            layer_scale=spec_vision.get('LAYER_SCALE', False)
        )
        width = spec_vision['EMBED_DIM'] * 2 ** (len(spec_vision['DEPTHS']) - 1)
        self.vision_projection = nn.Parameter(
            torch.empty(width, embed_dim)
        )
        # self.logit_scale = nn.Parameter(torch.FloatTensor([np.log(1 / 0.07)]))
        self.logit_scale = nn.Parameter(torch.ones([]))

        trunc_normal_(self.text_projection, std=.02)
        trunc_normal_(self.vision_projection, std=.02)

    def init_weights(self, pretrained='', pretrained_layers=[], verbose=True):
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained, map_location='cpu')
            logger.info(f'=> loading pretrained model {pretrained}')
            model_dict = self.state_dict()
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict.keys()
            }
            need_init_state_dict = {}
            for k, v in pretrained_dict.items():
                need_init = (
                        k.split('.')[0] in pretrained_layers
                        or pretrained_layers[0] is '*'
                )
                if need_init:
                    if verbose:
                        logging.info(f'=> init {k} from {pretrained}')
                    need_init_state_dict[k] = v
            self.load_state_dict(need_init_state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = {'logit_scale'}
        for k in self.text.no_weight_decay():
            no_weight_decay.add('text.'+k)

        for k in self.visual.no_weight_decay():
            no_weight_decay.add('visual.'+k)

        return no_weight_decay

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, norm=True):
        x = self.visual.forward_features(image)
        x = x @ self.vision_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def encode_text(self, text, norm=True):
        x = self.text(text)
        x = x @ self.text_projection

        if norm:
            x = x / x.norm(dim=-1, keepdim=True)

        return x

    def forward(self, image, text):
        features_image = self.encode_image(image)
        features_text = self.encode_text(text)

        # cosine similarity as logits
        T = self.logit_scale.exp()

        return features_image, features_text, T



def get_zeroshot_model(config, **kwargs):
    model = CLIP(config)

    if config['MODEL']['INIT_WEIGHTS']:
        model.init_weights(
            config['MODEL']['PRETRAINED'],
            config['MODEL']['PRETRAINED_LAYERS'],
            config['VERBOSE']
        )

    return model
