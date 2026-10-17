"""Adapter for the proposed model. Ablations are configurations of this one package."""
import torch

from .src.matching import matching_loss, off_band_energy_ratio
from .src.net import PBENet

NAME = 'pbe'
DEFAULT_CFG = {'c_list': [128, 192, 256, 384, 512], 'skip': 'predictive', 'block': 'ebpb',
               'match': 'cos', 'lambda_s': 0.1, 'band_kernel': 3}
TRAIN_ONLY = ('base.gt_conv', 'shead')


def build(cfg):
    # optional keys (absent in DEFAULT_CFG so the main-row config hash is unchanged):
    #   band_stride: pooling stride r of the eBPB (default 4)
    #   match_loc:   'skip' (default), 'output_band' or 'output_mask' (loss-location ablation)
    return PBENet(cfg['c_list'], skip=cfg['skip'], block=cfg['block'], r=cfg.get('band_stride', 4))


def forward_train(model, img):
    out, side, heads = model(img, return_aux=True)
    return {'out': out, 'side': side, 'heads': heads, 'skips': model.last_skips}


def _match_term(o, mask, cfg):
    if cfg['match'] == 'none' or cfg['lambda_s'] == 0:
        return torch.zeros((), device=mask.device)
    loc = cfg.get('match_loc', 'skip')
    if loc == 'skip':
        return cfg['lambda_s'] * matching_loss(o['heads'], mask, cfg['match'], cfg['band_kernel'])
    target = {'output_band': 'band', 'output_mask': 'mask'}[loc]
    return cfg['lambda_s'] * matching_loss([o['out']], mask, cfg['match'], cfg['band_kernel'], target=target)


def extra_loss(o, mask, cfg):
    loss = _match_term(o, mask, cfg)
    # optional candidate term (key absent in DEFAULT_CFG, so protocol-v1 rows are unchanged)
    if cfg.get('flat_weight', 0) > 0:
        loss = loss + cfg['flat_weight'] * off_band_energy_ratio(o['skips'], mask, cfg['band_kernel'])
    return loss


def infer(model, img):
    return model(img)


def is_train_only(param_name):
    return param_name.startswith(TRAIN_ONLY)
