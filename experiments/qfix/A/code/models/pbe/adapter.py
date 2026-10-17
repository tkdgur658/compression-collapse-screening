"""Adapter for the proposed model. Ablations are configurations of this one package."""
import torch

from .src.matching import matching_loss
from .src.net import PBENet

NAME = 'pbe'
DEFAULT_CFG = {'c_list': [128, 192, 256, 384, 512], 'skip': 'predictive', 'block': 'ebpb',
               'match': 'cos', 'lambda_s': 0.1, 'band_kernel': 3}
TRAIN_ONLY = ('base.gt_conv', 'shead')


def build(cfg):
    return PBENet(cfg['c_list'], skip=cfg['skip'], block=cfg['block'])


def forward_train(model, img):
    out, side, heads = model(img, return_aux=True)
    return {'out': out, 'side': side, 'heads': heads}


def extra_loss(o, mask, cfg):
    if cfg['match'] == 'none' or cfg['lambda_s'] == 0:
        return torch.zeros((), device=mask.device)
    return cfg['lambda_s'] * matching_loss(o['heads'], mask, cfg['match'], cfg['band_kernel'])


def infer(model, img):
    return model(img)


def is_train_only(param_name):
    return param_name.startswith(TRAIN_ONLY)
